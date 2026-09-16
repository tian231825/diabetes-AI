# -*- encoding: utf-8 -*-
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataloader2 import DiabetesDataset, collate_fn
from retrieval_model.models.person_pair import PersonPairSimilarityInference

from .encoder import Encoder
from .decoder import Decoder


class LengthPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.d_person * 2 + cfg.d_bg

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(getattr(cfg, "dropout", 0.1)),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.LayerNorm(cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(getattr(cfg, "dropout", 0.1)),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, person_value, person_mask, initial_bg):
        features = torch.cat([person_value, person_mask, initial_bg], dim=-1)
        length_logits = self.predictor(features).squeeze(-1)
        return F.softplus(length_logits) + 1.0


class Mapper(nn.Module):
    def __init__(
        self,
        cfg,
        c_pep_dim: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = cfg.d_model
        self.dropout = nn.Dropout(dropout)

        self.c_pep_proj = nn.Linear(c_pep_dim, self.hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, ff_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.residual_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x_in, refer_person_emb, pred_person_emb):
        # x_in: [B, K, D] or [B, D]
        # refer_person_emb: [B, K, D] or [B, D]
        # pred_person_emb: [B, D]
        if x_in.dim() == 3 and pred_person_emb.dim() == 2:
            pred_person_emb = pred_person_emb.unsqueeze(1).expand(-1, x_in.shape[1], -1)
        stage1_fusion = x_in + refer_person_emb
        x = torch.cat([stage1_fusion, pred_person_emb], dim=-1)
        x = self.residual_proj(x)

        for layer in self.encoder_layers:
            x = layer(x) + x

        return self.output_layer(self.dropout(x))


class TopKMapAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        top_k = max(1, int(getattr(cfg, "top_k", 1)))
        dropout = getattr(cfg, "dropout", 0.1)
        self.temperature = max(float(getattr(cfg, "memory_fusion_temperature", 1.0)), 1e-6)
        self.use_retrieval_bias = bool(getattr(cfg, "use_retrieval_bias", 1))
        self.use_attention_fusion = bool(getattr(cfg, "use_attention_fusion", 0))
        if self.use_attention_fusion:
            self.attention = nn.MultiheadAttention(
                d_model,
                num_heads=max(1, int(getattr(cfg, "fusion_nhead", 4))),
                dropout=dropout,
                batch_first=True,
            )

        self.score_proj = nn.Linear(d_model + 1, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(top_k * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, map_k, similarity):
        # map_k: [B, K, D]
        # similarity: [B, K]
        scaled_similarity = similarity / self.temperature
        similarity_feat = scaled_similarity.unsqueeze(-1)  # [B, K, 1]
        x = torch.cat([map_k, similarity_feat], dim=-1)  # [B, K, D+1]
        x = self.score_proj(x)  # [B, K, D]
        x = self.fuse(x)  # [B, K, D]
        if self.use_retrieval_bias:
            bias = torch.softmax(scaled_similarity, dim=1).unsqueeze(-1)
            x = x + bias * map_k
        if self.use_attention_fusion and x.shape[1] > 1:
            x, _ = self.attention(x, x, x, need_weights=False)
        x = x.reshape(x.shape[0], -1)  # [B, K*D]
        return self.output_proj(x)  # [B, D]


class TwoStageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device = torch.device(cfg.device if hasattr(cfg, "device") else "cuda:0")

        self.stage1 = Encoder(cfg)
        self.stage2 = Decoder(cfg)
        self.bg_Mapper = Mapper(cfg)
        self.tr_Mapper = Mapper(cfg)
        memory_gate_logit = float(getattr(cfg, "memory_mapper_init_gate_logit", -2.0))
        self.bg_memory_map_gate_logit = nn.Parameter(torch.tensor(memory_gate_logit))
        self.tr_memory_map_gate_logit = nn.Parameter(torch.tensor(memory_gate_logit))
        self.bg_map_aggregator = TopKMapAggregator(cfg)
        self.tr_map_aggregator = TopKMapAggregator(cfg)
        self.bg_state_summary = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(getattr(cfg, "dropout", 0.1)),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.tr_state_summary = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(getattr(cfg, "dropout", 0.1)),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.simPredictor = PersonPairSimilarityInference(device=device)

        self.retrieval_top_k = max(1, int(getattr(cfg, "top_k", 1)))
        self._knowledge_base = []
        self._knowledge_id_to_index = {}
        self._knowledge_person_value = None
        self._knowledge_person_mask = None
        self._knowledge_person_emb = None
        self._knowledge_emb_device = None
        self._build_knowledge_base()

        self.use_length_predictor = getattr(cfg, "use_length_predictor", False)
        if self.use_length_predictor:
            self.length_predictor = LengthPredictor(cfg)
            logging.info("Using length predictor")

    def _build_knowledge_base(self):
        dataset = DiabetesDataset(self.cfg, mode="train")
        self._knowledge_base = [deepcopy(sample) for sample in dataset.fetch_database()]
        self._knowledge_id_to_index = {
            sample["check_id"]: idx for idx, sample in enumerate(self._knowledge_base)
        }
        self._knowledge_person_value = torch.stack(
            [sample["person_value"].detach().cpu().float() for sample in self._knowledge_base],
            dim=0,
        )
        self._knowledge_person_mask = torch.stack(
            [sample["person_mask"].detach().cpu().float() for sample in self._knowledge_base],
            dim=0,
        )
        self._knowledge_person_emb = None
        self._knowledge_emb_device = None
        logging.info(
            f"Knowledge base ready: {len(self._knowledge_base)} patients, top_k={self.retrieval_top_k}"
        )

    def _ensure_knowledge_embeddings(self, device):
        if self._knowledge_person_emb is not None and self._knowledge_emb_device == str(device):
            return

        with torch.no_grad():
            person_value = self._knowledge_person_value.to(device)
            person_mask = self._knowledge_person_mask.to(device)
            self._knowledge_person_emb = self.simPredictor.model.encode_person(person_value, person_mask).detach()
        self._knowledge_emb_device = str(device)

    def _score_against_knowledge_base(self, query_value, query_mask):
        device = self.simPredictor.device
        self._ensure_knowledge_embeddings(device)

        with torch.no_grad():
            if query_value.dim() == 1:
                query_value = query_value.unsqueeze(0)
            if query_mask.dim() == 1:
                query_mask = query_mask.unsqueeze(0)

            query_value = query_value.to(device)
            query_mask = query_mask.to(device)
            query_emb = self.simPredictor.model.encode_person(query_value, query_mask)
            query_emb = query_emb.expand(self._knowledge_person_emb.shape[0], -1)
            cosine_sim = torch.sum(query_emb * self._knowledge_person_emb, dim=-1)
            pair_feat = torch.cat(
                [
                    torch.abs(query_emb - self._knowledge_person_emb),
                    query_emb * self._knowledge_person_emb,
                ],
                dim=-1,
            )
            sim_residual = self.simPredictor.model.sim_head(pair_feat).squeeze(-1)
            residual_scale = float(getattr(self.simPredictor.cfg, "residual_scale", 0.05))
            scores = cosine_sim + residual_scale * torch.tanh(sim_residual)
        return scores.detach()

    def _retrieve_topk_indices(self, batch):
        batch_size = batch["person_value"].shape[0]
        topk = min(self.retrieval_top_k, len(self._knowledge_base))
        topk_indices = []
        topk_scores = []

        for i in range(batch_size):
            scores = self._score_against_knowledge_base(batch["person_value"][i], batch["person_mask"][i])
            if "check_id" in batch:
                current_id = batch["check_id"][i]
                current_index = self._knowledge_id_to_index.get(current_id)
                if current_index is not None:
                    scores[current_index] = float("-inf")

            row_scores, row_indices = torch.topk(scores.detach().cpu(), k=topk, largest=True)
            topk_indices.append(row_indices)
            topk_scores.append(row_scores)

        return torch.stack(topk_indices, dim=0), torch.stack(topk_scores, dim=0)

    def _build_reference_batch(self, topk_indices):
        selected_samples = []
        for row in topk_indices.tolist():
            for index in row:
                selected_samples.append(deepcopy(self._knowledge_base[index]))
        return collate_fn(selected_samples)

    @staticmethod
    def _move_batch_to_device(batch_like, device):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch_like.items()
        }

    @staticmethod
    def _weighted_sum(features, scores):
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (features * weights).sum(dim=1)

    @staticmethod
    def _gather_last_valid_state(states, real_lengths):
        # states: [B, K, T, D], real_lengths: [B, K]
        last_indices = torch.clamp(real_lengths.long() - 1, min=0)
        gather_index = last_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, states.shape[-1])
        return torch.gather(states, dim=2, index=gather_index).squeeze(2)

    @staticmethod
    def _masked_mean_state(states, real_lengths):
        # states: [B, K, T, D], real_lengths: [B, K]
        max_t = states.shape[2]
        time_index = torch.arange(max_t, device=states.device).view(1, 1, max_t, 1)
        valid_mask = (time_index < real_lengths.unsqueeze(-1).unsqueeze(-1)).float()
        denom = valid_mask.sum(dim=2).clamp_min(1.0)
        return (states * valid_mask).sum(dim=2) / denom
    
    def _encode_reference_batch(self, reference_batch, batch_size, topk):
        outputs = self.stage1(reference_batch)
        insulin_values = reference_batch["insulin1"]
        if insulin_values.dim() == 4:
            insulin_values = insulin_values.flatten(start_dim=2)
        return {
            "bg_states": outputs["bg_states"].reshape(
                batch_size, topk, outputs["bg_states"].shape[1], outputs["bg_states"].shape[2]
            ),  # [B, K, T, D]
            "tr_states": outputs["tr_states"].reshape(
                batch_size, topk, outputs["tr_states"].shape[1], outputs["tr_states"].shape[2]
            ),  # [B, K, T, D]
            "person_emb": outputs["person_emb"].reshape(batch_size, topk, outputs["person_emb"].shape[1]),  # [B, K, D]
            "real_lengths": outputs["real_lengths"].reshape(batch_size, topk),  # [B, K]
            "bg_values": reference_batch["bg1"].reshape(
                batch_size, topk, reference_batch["bg1"].shape[1], reference_batch["bg1"].shape[2]
            ),
            "insulin_values": insulin_values.reshape(
                batch_size, topk, insulin_values.shape[1], insulin_values.shape[2]
            ),
        }

    def _build_reference_memory(self, reference_outputs, topk_scores, topk_indices, device):
        return {
            "bg_states": reference_outputs["bg_states"],
            "tr_states": reference_outputs["tr_states"],
            "real_lengths": reference_outputs["real_lengths"],
            "bg_values": reference_outputs.get("bg_values"),
            "insulin_values": reference_outputs.get("insulin_values"),
            "topk_scores": topk_scores.to(device),
            "topk_indices": topk_indices.to(device),
        }

    @staticmethod
    def _map_reference_state_sequence(states, refer_person_emb, current_person_emb, mapper, gate_logit=None):
        # Map every reference time step into the target patient's representation space.
        batch_size, topk, seq_len, hidden_dim = states.shape
        states_flat = states.reshape(batch_size, topk * seq_len, hidden_dim)
        refer_person_flat = (
            refer_person_emb.unsqueeze(2)
            .expand(batch_size, topk, seq_len, hidden_dim)
            .reshape(batch_size, topk * seq_len, hidden_dim)
        )
        mapped_flat = mapper(states_flat, refer_person_flat, current_person_emb)
        if gate_logit is not None:
            gate = torch.sigmoid(gate_logit).to(dtype=mapped_flat.dtype, device=mapped_flat.device)
            mapped_flat = states_flat + gate * (mapped_flat - states_flat)
        return mapped_flat.reshape(batch_size, topk, seq_len, hidden_dim)

    def calculate_sim(self, person_value_a, person_mask_a, person_value_b, person_mask_b):
        person_a = {"person_value": person_value_a, "person_mask": person_mask_a}
        person_b = {"person_value": person_value_b, "person_mask": person_mask_b}
        return self.simPredictor.predict_similarity(person_a, person_b)

    def forward(self, batch, tf_ratio, mode="train"):
        device = batch["person_value"].device
        batch_size = batch["person_value"].shape[0]  # [B]

        topk_indices, topk_scores = self._retrieve_topk_indices(batch)  # [B, K], [B, K]
        topk = topk_indices.shape[1]  # K

        reference_batch = self._build_reference_batch(topk_indices)
        reference_batch = self._move_batch_to_device(reference_batch, device)
        current_person_emb = self.stage1.encode_person(
            batch["person_value"],
            batch["person_mask"],
        )  # [B, D]

        reference_outputs = self._encode_reference_batch(reference_batch, batch_size, topk)
        refer_person_emb = reference_outputs["person_emb"]  # [B, K, D]
        mapped_bg_states = self._map_reference_state_sequence(
            reference_outputs["bg_states"],
            refer_person_emb,
            current_person_emb,
            self.bg_Mapper,
            self.bg_memory_map_gate_logit,
        )
        mapped_tr_states = self._map_reference_state_sequence(
            reference_outputs["tr_states"],
            refer_person_emb,
            current_person_emb,
            self.tr_Mapper,
            self.tr_memory_map_gate_logit,
        )
        reference_memory = self._build_reference_memory(
            {
                **reference_outputs,
                "bg_states": mapped_bg_states,
                "tr_states": mapped_tr_states,
            },
            topk_scores,
            topk_indices,
            device,
        )

        # New structure: decoder inputs come from target-side observations, not mapped reference summaries.
        map_bg_final = None
        map_tr_final = None

        predicted_length = None
        if self.use_length_predictor:
            initial_bg = batch["bg1"][:, 0, :]
            predicted_length = self.length_predictor(
                batch["person_value"],
                batch["person_mask"],
                initial_bg,
            )

        if self.training:
            outputs = self.stage2(
                batch,
                map_bg_final,
                map_tr_final,
                reference_memory,
                mode=mode,
                tf_ratio=tf_ratio,
                max_len=None,
                predicted_length=predicted_length,
            )
        else:
            outputs = self.stage2(
                batch,
                map_bg_final,
                map_tr_final,
                reference_memory,
                mode=mode,
                max_len=self.cfg.max_seq_len,
                enable_early_stop=True,
                early_stop_threshold=0.01,
                patience=3,
                predicted_length=predicted_length,
            )

        result = [outputs["insulin_preds"], outputs["bg_preds"]]
        if predicted_length is not None:
            result.append(predicted_length)
        if "actual_lengths" in outputs and predicted_length is None:
            result.append(outputs["actual_lengths"])
        return tuple(result)
