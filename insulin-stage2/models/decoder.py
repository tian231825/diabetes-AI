# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

from .encoder import SimpleTransformer, ResidualConnection, cfg_get


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dropout = cfg_get(cfg, "dropout", 0.1)
        self.bg_min_value = cfg_get(cfg, "bg_min_value", 2.0)
        self.bg_max_value = cfg_get(cfg, "bg_max_value", 35.0)
        self.insulin_min_value = cfg_get(cfg, "insulin_min_value", 0.0)
        self.insulin_max_value = cfg_get(cfg, "insulin_max_value", 55.0)
        self.transfer4_flag_mode = int(cfg.d_insulin) == 5
        self.register_buffer(
            "insulin_max_values",
            torch.full((int(cfg.d_insulin),), float(self.insulin_max_value)),
        )
        self.insulin_presence_threshold = cfg_get(cfg, "effective_dose_threshold", 2.0)
        self.memory_fusion_temperature = max(float(cfg_get(cfg, "memory_fusion_temperature", 1.0)), 1e-6)
        self.bg_residual_delta_scale = float(cfg_get(cfg, "bg_residual_delta_scale", 6.0))
        self.insulin_residual_delta_scale = float(cfg_get(cfg, "insulin_residual_delta_scale", 30.0))

        self.bg_decoder = SimpleTransformer(cfg, cfg.d_model, is_decoder=True)
        self.tr_decoder = SimpleTransformer(cfg, cfg.d_model, is_decoder=True)

        self.tr_transform = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.bg_transform = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.bg_insulin_condition_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layernorm_tr = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.layernorm_bg = nn.LayerNorm(cfg.d_model, eps=1e-5)

        self.bg_value_encoder = nn.Sequential(
            nn.Linear(cfg.d_bg * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.bg_initial_value_encoder = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.bg_initial_index_embedding = nn.Embedding(cfg.d_bg + 1, cfg.d_model)
        self.insulin_encoder = nn.Sequential(
            nn.Linear(cfg.d_insulin * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(cfg.d_drug, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.time_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        nn.init.normal_(self.time_embedding.weight, mean=0.0, std=0.02)

        self.bg_output = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model // 2, cfg.d_bg),
            nn.Softplus(),
        )
        self.bg_delta_output = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model // 2, cfg.d_bg),
        )
        if self.transfer4_flag_mode:
            self.regimen_embedding = nn.Embedding(2, cfg.d_model)
            self.regimen_context_gate_logit = nn.Parameter(
                torch.tensor(float(cfg_get(cfg, "regimen_context_gate_init_logit", -2.0)))
            )
            self.insulin_flag_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cfg.d_model // 2, 1),
            )
            self.insulin_context_fuse = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.insulin_dose_output = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cfg.d_model // 2, cfg.d_insulin - 1),
            )
            self.insulin_dose_delta_output = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cfg.d_model // 2, cfg.d_insulin - 1),
            )
            self.register_buffer(
                "insulin_reference_prior_slot_scale",
                torch.ones(int(cfg.d_insulin) - 1),
            )
            self.register_buffer(
                "transfer4_mask_template",
                torch.tensor([0.0] + [1.0] * (int(cfg.d_insulin) - 1)),
            )
        else:
            self.regimen_embedding = None
            self.regimen_context_gate_logit = None
            self.insulin_context_fuse = None
            self.insulin_flag_head = None
            self.insulin_dose_output = None
            self.insulin_dose_delta_output = None
            self.insulin_output = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cfg.d_model // 2, cfg.d_insulin),
                nn.Softplus(),
            )
            self.insulin_delta_output = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cfg.d_model // 2, cfg.d_insulin),
            )
        residual_mix_logit = float(cfg_get(cfg, "residual_output_mix_init_logit", 0.0))
        self.bg_residual_mix_logit = nn.Parameter(torch.tensor(residual_mix_logit))
        self.insulin_residual_mix_logit = nn.Parameter(torch.tensor(residual_mix_logit))
        self.bg_reference_prior_mix_logit = nn.Parameter(
            torch.tensor(float(cfg_get(cfg, "bg_reference_prior_mix_init_logit", -1.0)))
        )
        self.insulin_reference_prior_mix_logit = nn.Parameter(
            torch.tensor(float(cfg_get(cfg, "insulin_reference_prior_mix_init_logit", -0.5)))
        )
        self.bg_reference_correction_head = nn.Linear(cfg.d_model, cfg.d_bg)
        self.insulin_reference_correction_head = nn.Linear(cfg.d_model, cfg.d_insulin)
        self.bg_reference_dynamic_logit_scale = float(cfg_get(cfg, "bg_reference_dynamic_logit_scale", 0.35))
        self.insulin_reference_dynamic_logit_scale = float(cfg_get(cfg, "insulin_reference_dynamic_logit_scale", 0.18))
        nn.init.zeros_(self.bg_reference_correction_head.weight)
        nn.init.zeros_(self.bg_reference_correction_head.bias)
        nn.init.zeros_(self.insulin_reference_correction_head.weight)
        nn.init.zeros_(self.insulin_reference_correction_head.bias)

        self.residual_bg = ResidualConnection(cfg.d_model, dropout)
        self.residual_tr = ResidualConnection(cfg.d_model, dropout)

    @staticmethod
    def _expand_topk(x, topk):
        return x.unsqueeze(1).expand(-1, topk, -1)

    @staticmethod
    def _ensure_topk_query(x, topk):
        if x.dim() == 3:
            return x
        return x.unsqueeze(1).expand(-1, topk, -1)

    @staticmethod
    def _flatten_memory(memory):
        bsz, topk, seq_len, hidden = memory.shape
        return memory.reshape(bsz * topk, seq_len, hidden)

    @staticmethod
    def _flatten_memory_padding_mask(memory_lengths, memory_len):
        if memory_lengths is None:
            return None
        lengths = memory_lengths.long().clamp(min=1, max=memory_len)
        steps = torch.arange(memory_len, device=lengths.device).view(1, 1, memory_len)
        return (steps >= lengths.unsqueeze(-1)).reshape(lengths.shape[0] * lengths.shape[1], memory_len)

    @staticmethod
    def _flatten_query(query):
        bsz, topk, hidden = query.shape
        return query.reshape(bsz * topk, hidden).unsqueeze(1)

    @staticmethod
    def _restore_hidden(hidden, batch_size, topk):
        return hidden.reshape(batch_size, topk, hidden.shape[-1])

    def _aggregate_topk(self, pred_k, scores):
        weights = torch.softmax(scores / self.memory_fusion_temperature, dim=1).unsqueeze(-1)
        return (pred_k * weights).sum(dim=1)

    def _reference_step_values(self, reference_memories, key, step, feature_dim, device, dtype):
        values = reference_memories.get(key, None)
        if values is None:
            return None
        values = values.to(device=device, dtype=dtype)
        lengths = reference_memories.get("real_lengths", None)
        if lengths is None:
            indices = torch.full(values.shape[:2], min(step, values.shape[2] - 1), device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device).long().clamp(min=1, max=values.shape[2])
            indices = torch.minimum(
                torch.full_like(lengths, step),
                lengths - 1,
            )
        gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, values.shape[-1])
        prior = torch.gather(values, dim=2, index=gather_index).squeeze(2)
        if prior.shape[-1] > feature_dim:
            prior = prior[..., :feature_dim]
        elif prior.shape[-1] < feature_dim:
            prior = torch.nn.functional.pad(prior, (0, feature_dim - prior.shape[-1]))
        return prior

    def _decode_topk(self, query_k, memory_k, decoder, norm_layer, batch_size, topk, memory_lengths=None):
        query_k = norm_layer(query_k)
        query_flat = self._flatten_query(query_k)
        memory_flat = self._flatten_memory(memory_k)
        memory_mask = self._flatten_memory_padding_mask(memory_lengths, memory_flat.shape[1])
        hidden = decoder(query_flat, memory=memory_flat, mem_mask=memory_mask).squeeze(1)
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
        hidden = torch.clamp(hidden, min=-50.0, max=50.0)
        return self._restore_hidden(hidden, batch_size, topk)

    @staticmethod
    def _broadcast_limit(limit, target):
        if torch.is_tensor(limit):
            limit = limit.to(device=target.device, dtype=target.dtype)
            view_shape = [1] * target.dim()
            view_shape[-1] = limit.shape[-1]
            return limit.view(*view_shape)
        return torch.tensor(float(limit), device=target.device, dtype=target.dtype)

    def _clamp_prediction(self, value, min_value, max_value):
        min_tensor = self._broadcast_limit(min_value, value)
        max_tensor = self._broadcast_limit(max_value, value)
        return torch.minimum(torch.maximum(value, min_tensor), max_tensor)

    def _clamp_insulin_prediction(self, pred):
        if self.transfer4_flag_mode:
            flag = pred[..., :1].clamp(min=0.0, max=1.0)
            doses = self._clamp_prediction(pred[..., 1:], self.insulin_min_value, self.insulin_max_values[1:])
            return torch.cat([flag, doses], dim=-1)
        return self._clamp_prediction(pred, self.insulin_min_value, self.insulin_max_values)

    def _build_insulin_mask_from_prediction(self, insulin_pred):
        if self.transfer4_flag_mode:
            return self.transfer4_mask_template.to(device=insulin_pred.device, dtype=insulin_pred.dtype).view(1, -1).expand_as(insulin_pred)
        return torch.ones_like(insulin_pred)

    def _regimen_context_from_flag_scalar(self, flag_scalar):
        if not self.transfer4_flag_mode:
            return torch.zeros(*flag_scalar.shape[:-1], self.cfg.d_model, device=flag_scalar.device, dtype=flag_scalar.dtype)
        premix_prob = torch.clamp(flag_scalar, min=0.0, max=1.0)
        probs = torch.cat([1.0 - premix_prob, premix_prob], dim=-1)
        regimen_context = torch.matmul(probs, self.regimen_embedding.weight)
        gate = torch.sigmoid(self.regimen_context_gate_logit).to(
            device=flag_scalar.device,
            dtype=flag_scalar.dtype,
        )
        return gate * regimen_context

    def _blend_residual_prediction(
        self,
        absolute_pred,
        delta_raw,
        base_value,
        min_value,
        max_value,
        delta_scale,
        mix_logit,
        reference_prior=None,
        reference_mix_logit=None,
    ):
        if base_value is None:
            base_value = torch.zeros_like(absolute_pred)
        base_value = base_value.to(device=absolute_pred.device, dtype=absolute_pred.dtype)
        delta = torch.tanh(delta_raw) * delta_scale
        residual_pred = self._clamp_prediction(base_value + delta, min_value, max_value)
        residual_mix = torch.sigmoid(mix_logit).to(device=absolute_pred.device, dtype=absolute_pred.dtype)
        pred = residual_mix * residual_pred + (1.0 - residual_mix) * absolute_pred
        if reference_prior is not None and reference_mix_logit is not None:
            reference_prior = reference_prior.to(device=absolute_pred.device, dtype=absolute_pred.dtype)
            reference_prior = self._clamp_prediction(reference_prior, min_value, max_value)
            reference_mix = torch.sigmoid(reference_mix_logit).to(device=absolute_pred.device, dtype=absolute_pred.dtype)
            pred = (1.0 - reference_mix) * pred + reference_mix * reference_prior
        return self._clamp_prediction(pred, min_value, max_value)

    def _apply_state_update(
        self,
        delta_raw,
        base_value,
        min_value,
        max_value,
        delta_scale,
        reference_value=None,
        correction_logit=None,
        correction_scale=None,
        correction_context=None,
        correction_head=None,
    ):
        if base_value is None:
            base_value = torch.zeros_like(delta_raw)
        base_value = base_value.to(device=delta_raw.device, dtype=delta_raw.dtype)
        delta = torch.tanh(delta_raw) * delta_scale
        updated_value = base_value + delta
        if reference_value is not None and correction_logit is not None:
            reference_value = reference_value.to(device=delta_raw.device, dtype=delta_raw.dtype)
            scale_tensor = self._broadcast_limit(delta_scale, updated_value).abs().clamp(min=1e-6)
            reference_delta = torch.tanh((reference_value - base_value) / scale_tensor) * scale_tensor
            correction_gate = torch.sigmoid(correction_logit).to(device=delta_raw.device, dtype=delta_raw.dtype)
            if correction_context is not None and correction_head is not None:
                dynamic_logit = correction_head(correction_context).to(device=delta_raw.device, dtype=delta_raw.dtype)
                if correction_head is self.bg_reference_correction_head:
                    dynamic_scale = self.bg_reference_dynamic_logit_scale
                else:
                    dynamic_scale = self.insulin_reference_dynamic_logit_scale
                dynamic_logit = torch.tanh(dynamic_logit) * dynamic_scale
                correction_gate = torch.sigmoid(correction_logit + dynamic_logit).to(
                    device=delta_raw.device, dtype=delta_raw.dtype
                )
            if correction_scale is not None:
                correction_gate = correction_gate * self._broadcast_limit(correction_scale, updated_value)
            alignment = torch.sigmoid(2.0 * delta * reference_delta / (scale_tensor * scale_tensor + 1e-6))
            correction_gate = correction_gate * (0.5 + 0.5 * alignment)
            updated_value = updated_value + correction_gate * reference_delta
        return self._clamp_prediction(updated_value, min_value, max_value)

    def _project_bg_prediction(self, hidden_state, previous_bg, reference_bg=None):
        absolute_bg = torch.clamp(
            self.bg_output(hidden_state),
            min=self.bg_min_value,
            max=self.bg_max_value,
        )
        return self._blend_residual_prediction(
            absolute_bg,
            self.bg_delta_output(hidden_state),
            previous_bg,
            self.bg_min_value,
            self.bg_max_value,
            self.bg_residual_delta_scale,
            self.bg_residual_mix_logit,
            reference_bg,
            self.bg_reference_prior_mix_logit,
        )

    def _build_bg_insulin_condition(self, insulin_hidden):
        # Preserve the forward insulin->BG coupling while avoiding BG loss
        # directly reshaping the insulin decoder latent.
        return self.bg_insulin_condition_proj(insulin_hidden.detach())

    def _project_insulin_prediction(self, hidden_state, previous_insulin=None, reference_insulin=None):
        if self.transfer4_flag_mode:
            flag_prob = torch.sigmoid(self.insulin_flag_head(hidden_state))
            regimen_context = self._regimen_context_from_flag_scalar(flag_prob)
            fused_hidden = self.insulin_context_fuse(torch.cat([hidden_state, regimen_context], dim=-1))
            dose_base = previous_insulin[..., 1:] if previous_insulin is not None else None
            dose_prior = reference_insulin[..., 1:] if reference_insulin is not None else None
            absolute_dose = self._clamp_prediction(
                torch.nn.functional.softplus(self.insulin_dose_output(fused_hidden)),
                self.insulin_min_value,
                self.insulin_max_values[1:],
            )
            dose_values = self._blend_residual_prediction(
                absolute_dose,
                self.insulin_dose_delta_output(fused_hidden),
                dose_base,
                self.insulin_min_value,
                self.insulin_max_values[1:],
                self.insulin_residual_delta_scale,
                self.insulin_residual_mix_logit,
                dose_prior,
                self.insulin_reference_prior_mix_logit,
            )
            pred = torch.cat([flag_prob, dose_values], dim=-1)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=self.insulin_max_value, neginf=0.0)
            pred = self._clamp_insulin_prediction(pred)
            pred_mask = self._build_insulin_mask_from_prediction(pred)
            return pred, pred_mask

        absolute_pred = self._clamp_prediction(
            self.insulin_output(hidden_state),
            self.insulin_min_value,
            self.insulin_max_values,
        )
        pred = self._blend_residual_prediction(
            absolute_pred,
            self.insulin_delta_output(hidden_state),
            previous_insulin,
            self.insulin_min_value,
            self.insulin_max_values,
            self.insulin_residual_delta_scale,
            self.insulin_residual_mix_logit,
            reference_insulin,
            self.insulin_reference_prior_mix_logit,
        )
        pred = torch.nan_to_num(pred, nan=0.0, posinf=self.insulin_max_value, neginf=0.0)
        pred = self._clamp_insulin_prediction(pred)
        pred_mask = self._build_insulin_mask_from_prediction(pred)
        return pred, pred_mask

    def forward(
        self,
        batch,
        map_bg,
        map_tr,
        reference_memories,
        person_emb=None,
        mode="train",
        tf_ratio=0.9,
        max_len=None,
        enable_early_stop=False,
        early_stop_threshold=0.01,
        patience=3,
        predicted_length=None,
    ):
        if mode == "train":
            return self._forward_train(batch, map_bg, map_tr, reference_memories, tf_ratio, predicted_length)
        return self._forward_inference(batch, map_bg, map_tr, reference_memories, mode, max_len, predicted_length)

    def _prepare_sequence_targets(self, batch):
        bg_true = batch["bg2"]
        bg_mask = batch["bg2_mask"]
        insulin_true = batch["insulin2"]
        insulin_true_mask = batch["insulin2_mask"]
        drug_true = batch["drug2"]
        if insulin_true.dim() == 4:
            insulin_true = insulin_true.flatten(start_dim=2)
        if insulin_true_mask.dim() == 4:
            insulin_true_mask = insulin_true_mask.flatten(start_dim=2)
        return bg_true, bg_mask, insulin_true, insulin_true_mask, drug_true

    def _calculate_real_lengths(self, bg_mask):
        bsz = bg_mask.shape[0]
        device = bg_mask.device
        real_lengths = torch.zeros(bsz, dtype=torch.long, device=device)
        for b in range(bsz):
            valid_timesteps = (bg_mask[b].sum(dim=1) > 0).nonzero(as_tuple=False)
            real_lengths[b] = valid_timesteps[-1].item() + 1 if len(valid_timesteps) > 0 else 1
        return real_lengths

    @staticmethod
    def build_initial_bg(bg, bg_mask):
        bsz, _, bg_dim = bg.shape
        device = bg.device
        initial_bg = torch.zeros(bsz, bg_dim, device=device, dtype=bg.dtype)
        initial_bg_mask = torch.zeros(bsz, bg_dim, device=device, dtype=bg_mask.dtype)
        initial_bg_value = torch.zeros(bsz, 1, device=device, dtype=bg.dtype)
        initial_bg_index = torch.full((bsz,), bg_dim, dtype=torch.long, device=device)
        for b in range(bsz):
            valid_steps = (bg_mask[b].sum(dim=1) > 0).nonzero(as_tuple=False)
            if len(valid_steps) > 0:
                t0 = valid_steps[0].item()
                first_idx = bg_mask[b, t0].nonzero(as_tuple=False)
                if len(first_idx) > 0:
                    k = first_idx[0].item()
                    initial_bg[b, k] = bg[b, t0, k]
                    initial_bg_mask[b, k] = 1.0
                    initial_bg_value[b, 0] = bg[b, t0, k]
                    initial_bg_index[b] = k
        return initial_bg, initial_bg_mask, initial_bg_value, initial_bg_index

    def _build_initial_bg_embedding(self, initial_bg, initial_bg_mask, initial_bg_value, initial_bg_index, time_emb):
        sparse_bg_emb = self.bg_value_encoder(torch.cat([initial_bg, initial_bg_mask], dim=-1))
        known_flag = (initial_bg_index != self.cfg.d_bg).float().unsqueeze(-1)
        known_value_emb = self.bg_initial_value_encoder(torch.cat([initial_bg_value, known_flag], dim=-1))
        known_index_emb = self.bg_initial_index_embedding(initial_bg_index)
        return sparse_bg_emb + known_value_emb + known_index_emb + time_emb

    def _build_admission_start_initial_bg(self, batch, bg_true, bg_mask):
        s2_init = self.build_initial_bg(bg_true, bg_mask)
        bg1 = batch.get("bg1")
        bg1_mask = batch.get("bg1_mask")
        if bg1 is None or bg1_mask is None:
            return s2_init

        s1_init = self.build_initial_bg(bg1, bg1_mask)
        s1_valid = s1_init[1].sum(dim=1).gt(0)
        has_s1_history = batch.get("has_s1_history")
        if torch.is_tensor(has_s1_history):
            s1_valid = s1_valid & has_s1_history.to(s1_valid.device).view(-1).gt(0.5)
        choose_s1 = s1_valid.view(-1, 1)
        choose_s1_index = s1_valid
        return (
            torch.where(choose_s1, s1_init[0], s2_init[0]),
            torch.where(choose_s1, s1_init[1], s2_init[1]),
            torch.where(choose_s1, s1_init[2], s2_init[2]),
            torch.where(choose_s1_index, s1_init[3], s2_init[3]),
        )

    def _forward_train(self, batch, map_bg, map_tr, reference_memories, tf_ratio, predicted_length):
        device = batch["bg2"].device
        batch_size = batch["bg2"].shape[0]

        bg_true, bg_mask, insulin_true, insulin_true_mask, drug_true = self._prepare_sequence_targets(batch)
        if predicted_length is not None:
            real_lengths = torch.clamp(torch.round(predicted_length).long(), min=1, max=self.cfg.max_seq_len)
        else:
            real_lengths = batch.get("real_lengths", self._calculate_real_lengths(bg_mask))

        actual_t = min(real_lengths.max().item(), bg_true.shape[1])
        positions = torch.arange(actual_t, device=device).unsqueeze(0).expand(batch_size, -1)
        tf_mode = str(cfg_get(self.cfg, "teacher_forcing_mode", "prefix")).lower()
        max_prev_steps = max(actual_t - 1, 0)
        teacher_prefix_steps = int(round(float(tf_ratio) * max_prev_steps))
        teacher_prefix_steps = max(0, min(max_prev_steps, teacher_prefix_steps))

        bg_memory = reference_memories["bg_states"]
        tr_memory = reference_memories["tr_states"]
        topk_scores = reference_memories["topk_scores"].to(device)
        memory_lengths = reference_memories.get("real_lengths", None)
        if memory_lengths is not None:
            memory_lengths = memory_lengths.to(device)
        topk = bg_memory.shape[1]

        initial_bg, initial_bg_mask, initial_bg_value, initial_bg_index = self._build_admission_start_initial_bg(
            batch, bg_true, bg_mask
        )
        bg_initial_h = self._build_initial_bg_embedding(
            initial_bg,
            initial_bg_mask,
            initial_bg_value,
            initial_bg_index,
            torch.zeros(batch_size, self.cfg.d_model, device=device),
        )
        bg_initial_h_k = self._expand_topk(bg_initial_h, topk)
        i_h_0 = torch.zeros(batch_size, self.cfg.d_model, device=device)
        i_h_0_k = self._expand_topk(i_h_0, topk)

        prev_bg_pred = torch.clamp(initial_bg, min=self.bg_min_value, max=self.bg_max_value)
        prev_insulin_pred = torch.zeros(batch_size, self.cfg.d_insulin, device=device, dtype=bg_true.dtype)
        prev_insulin_mask = torch.ones_like(prev_insulin_pred)

        bg_history = []
        tr_history = []
        bg_preds = []
        insulin_preds = []
        bg_h_k = None
        insulin_h_k = None

        for t in range(actual_t):
            idx = positions[:, t].clamp(max=self.time_embedding.num_embeddings - 1)
            time_emb = self.time_embedding(idx)
            drug_current = drug_true[:, t] if t < drug_true.shape[1] else drug_true[:, -1]
            drug_emb = self.drug_encoder(drug_current) + time_emb
            drug_emb_k = self._expand_topk(drug_emb, topk)
            if t > 0:
                tr_teacher = insulin_true[:, t - 1]
                tr_mask_teacher = insulin_true_mask[:, t - 1]
                bg_teacher = bg_true[:, t - 1]
                bg_mask_teacher = bg_mask[:, t - 1]
                if tf_mode == "stepwise":
                    use_teacher = (torch.rand(batch_size, device=device) < tf_ratio).unsqueeze(1)
                else:
                    use_teacher = torch.full(
                        (batch_size, 1),
                        t - 1 < teacher_prefix_steps,
                        dtype=torch.bool,
                        device=device,
                    )
            else:
                tr_teacher = prev_insulin_pred
                tr_mask_teacher = prev_insulin_mask
                bg_teacher = prev_bg_pred
                bg_mask_teacher = torch.ones_like(prev_bg_pred, device=device)
                use_teacher = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

            teacher_mask_insulin = use_teacher.expand_as(prev_insulin_pred)
            teacher_mask_insulin_mask = use_teacher.expand_as(prev_insulin_mask)
            insulin_for_bg = torch.where(teacher_mask_insulin, tr_teacher, prev_insulin_pred)
            insulin_mask_for_bg = torch.where(teacher_mask_insulin_mask, tr_mask_teacher, prev_insulin_mask)
            insulin_for_bg = torch.nan_to_num(insulin_for_bg, nan=0.0, posinf=100.0, neginf=0.0)
            regimen_context = self._regimen_context_from_flag_scalar(insulin_for_bg[..., :1])
            insulin_emb = self.insulin_encoder(torch.cat([insulin_for_bg, insulin_mask_for_bg], dim=-1)) + time_emb + regimen_context
            insulin_emb = torch.clamp(insulin_emb, min=-50.0, max=50.0)
            insulin_emb_k = self._expand_topk(insulin_emb, topk)

            if t == 0:
                tr_query_k = torch.cat([bg_initial_h_k, insulin_emb_k, drug_emb_k, i_h_0_k], dim=-1)
            else:
                tr_query_k = torch.cat([bg_h_k, insulin_emb_k, drug_emb_k, insulin_h_k], dim=-1)

            tr_query_k = self.tr_transform(tr_query_k)
            tr_query_k = torch.clamp(tr_query_k, min=-50.0, max=50.0)
            insulin_h_k = self._decode_topk(
                tr_query_k,
                tr_memory,
                self.tr_decoder,
                self.layernorm_tr,
                batch_size,
                topk,
                memory_lengths=memory_lengths,
            )
            insulin_base_k = self._expand_topk(insulin_for_bg, topk)
            insulin_prior_k = self._reference_step_values(
                reference_memories,
                "insulin_values",
                t,
                self.cfg.d_insulin,
                device,
                insulin_h_k.dtype,
            )
            tr_pred_k, tr_pred_mask_k = self._project_insulin_prediction(
                insulin_h_k,
                insulin_base_k,
                insulin_prior_k,
            )
            tr_pred = self._aggregate_topk(tr_pred_k, topk_scores)
            tr_pred_mask = self._aggregate_topk(tr_pred_mask_k, topk_scores)
            tr_pred = torch.nan_to_num(
                tr_pred,
                nan=self.insulin_min_value,
                posinf=self.insulin_max_value,
                neginf=self.insulin_min_value,
            )
            tr_pred = self._clamp_insulin_prediction(tr_pred)
            prev_insulin_mask = self._build_insulin_mask_from_prediction(tr_pred_mask)

            if t == 0:
                bg_emb = self._build_initial_bg_embedding(
                    initial_bg,
                    initial_bg_mask,
                    initial_bg_value,
                    initial_bg_index,
                    time_emb,
                )
                bg_input = initial_bg
            else:
                teacher_mask_bg = use_teacher.expand_as(prev_bg_pred)
                bg_input = torch.where(teacher_mask_bg, bg_teacher, prev_bg_pred)
                bg_input = torch.nan_to_num(
                    bg_input,
                    nan=self.bg_min_value,
                    posinf=self.bg_max_value,
                    neginf=self.bg_min_value,
                )
                final_bg_mask = torch.where(
                    teacher_mask_bg,
                    bg_mask_teacher,
                    torch.ones_like(prev_bg_pred, device=device),
                )
                bg_emb = self.bg_value_encoder(torch.cat([bg_input, final_bg_mask], dim=-1)) + time_emb
            bg_emb = torch.clamp(bg_emb, min=-50.0, max=50.0)
            bg_emb_k = self._expand_topk(bg_emb, topk)

            bg_insulin_condition_k = self._build_bg_insulin_condition(insulin_h_k)
            if t == 0:
                bg_query_k = torch.cat([bg_initial_h_k, bg_emb_k, bg_insulin_condition_k], dim=-1)
            else:
                bg_query_k = torch.cat([bg_h_k, bg_emb_k, bg_insulin_condition_k], dim=-1)

            bg_query_k = self.bg_transform(bg_query_k)
            bg_query_k = torch.clamp(bg_query_k, min=-50.0, max=50.0)
            bg_h_k = self._decode_topk(
                bg_query_k,
                bg_memory,
                self.bg_decoder,
                self.layernorm_bg,
                batch_size,
                topk,
                memory_lengths=memory_lengths,
            )
            bg_base_k = self._expand_topk(bg_input, topk)
            bg_prior_k = self._reference_step_values(
                reference_memories,
                "bg_values",
                t,
                self.cfg.d_bg,
                device,
                bg_h_k.dtype,
            )
            bg_pred_k = self._project_bg_prediction(bg_h_k, bg_base_k, bg_prior_k)
            bg_pred = self._aggregate_topk(bg_pred_k, topk_scores)
            bg_pred = torch.nan_to_num(
                bg_pred,
                nan=self.bg_min_value,
                posinf=self.bg_max_value,
                neginf=self.bg_min_value,
            )

            insulin_preds.append(tr_pred)
            bg_preds.append(bg_pred)
            tr_history.append(self._aggregate_topk(insulin_h_k, topk_scores))
            bg_history.append(self._aggregate_topk(bg_h_k, topk_scores))
            prev_insulin_pred = tr_pred
            prev_bg_pred = bg_pred

        return self._build_output(insulin_preds, bg_preds, bg_history, tr_history, real_lengths, actual_t, batch_size, device)

    def _forward_inference(self, batch, map_bg, map_tr, reference_memories, mode, max_len, predicted_length):
        device = batch["bg2"].device
        batch_size = batch["bg2"].shape[0]

        bg_true, bg_mask, insulin_true, insulin_true_mask, drug_true = self._prepare_sequence_targets(batch)
        if predicted_length is not None:
            real_lengths = torch.clamp(torch.round(predicted_length).long(), min=1, max=self.cfg.max_seq_len)
        elif mode == "infer":
            infer_len = max_len if max_len is not None else self.cfg.max_seq_len
            real_lengths = torch.full((batch_size,), infer_len, dtype=torch.long, device=device)
        else:
            real_lengths = batch.get("real_lengths", self._calculate_real_lengths(bg_mask))

        actual_t = min(real_lengths.max().item(), max_len if mode == "infer" and max_len is not None else bg_true.shape[1])
        positions = torch.arange(actual_t, device=device).unsqueeze(0).expand(batch_size, -1)

        bg_memory = reference_memories["bg_states"]
        tr_memory = reference_memories["tr_states"]
        topk_scores = reference_memories["topk_scores"].to(device)
        memory_lengths = reference_memories.get("real_lengths", None)
        if memory_lengths is not None:
            memory_lengths = memory_lengths.to(device)
        topk = bg_memory.shape[1]

        initial_bg, initial_bg_mask, initial_bg_value, initial_bg_index = self._build_admission_start_initial_bg(
            batch, bg_true, bg_mask
        )
        bg_initial_h = self._build_initial_bg_embedding(
            initial_bg,
            initial_bg_mask,
            initial_bg_value,
            initial_bg_index,
            torch.zeros(batch_size, self.cfg.d_model, device=device),
        )
        bg_initial_h_k = self._expand_topk(bg_initial_h, topk)
        i_h_0 = torch.zeros(batch_size, self.cfg.d_model, device=device)
        i_h_0_k = self._expand_topk(i_h_0, topk)

        prev_bg_pred = torch.clamp(initial_bg, min=self.bg_min_value, max=self.bg_max_value)
        prev_insulin_pred = torch.zeros(batch_size, self.cfg.d_insulin, device=device, dtype=bg_true.dtype)
        prev_insulin_mask = torch.ones_like(prev_insulin_pred)

        bg_history = []
        tr_history = []
        bg_preds = []
        insulin_preds = []
        bg_h_k = None
        insulin_h_k = None

        for t in range(actual_t):
            idx = positions[:, t].clamp(max=self.time_embedding.num_embeddings - 1)
            time_emb = self.time_embedding(idx)
            drug_current = drug_true[:, t] if t < drug_true.shape[1] else drug_true[:, -1]
            drug_emb = self.drug_encoder(drug_current) + time_emb
            drug_emb_k = self._expand_topk(drug_emb, topk)
            regimen_context = self._regimen_context_from_flag_scalar(prev_insulin_pred[..., :1])
            insulin_emb = self.insulin_encoder(torch.cat([prev_insulin_pred, prev_insulin_mask], dim=-1)) + time_emb + regimen_context
            insulin_emb = torch.nan_to_num(
                insulin_emb,
                nan=self.insulin_min_value,
                posinf=self.insulin_max_value,
                neginf=self.insulin_min_value,
            )
            insulin_emb = torch.clamp(insulin_emb, min=-50.0, max=50.0)
            insulin_emb_k = self._expand_topk(insulin_emb, topk)

            if t == 0:
                tr_query_k = torch.cat([bg_initial_h_k, insulin_emb_k, drug_emb_k, i_h_0_k], dim=-1)
            else:
                tr_query_k = torch.cat([bg_h_k, insulin_emb_k, drug_emb_k, insulin_h_k], dim=-1)

            tr_query_k = self.tr_transform(tr_query_k)
            tr_query_k = torch.clamp(tr_query_k, min=-50.0, max=50.0)
            insulin_h_k = self._decode_topk(
                tr_query_k,
                tr_memory,
                self.tr_decoder,
                self.layernorm_tr,
                batch_size,
                topk,
                memory_lengths=memory_lengths,
            )
            insulin_base_k = self._expand_topk(prev_insulin_pred, topk)
            insulin_prior_k = self._reference_step_values(
                reference_memories,
                "insulin_values",
                t,
                self.cfg.d_insulin,
                device,
                insulin_h_k.dtype,
            )
            tr_pred_k, tr_pred_mask_k = self._project_insulin_prediction(
                insulin_h_k,
                insulin_base_k,
                insulin_prior_k,
            )
            tr_pred = self._aggregate_topk(tr_pred_k, topk_scores)
            tr_pred_mask = self._aggregate_topk(tr_pred_mask_k, topk_scores)
            tr_pred = torch.nan_to_num(
                tr_pred,
                nan=self.insulin_min_value,
                posinf=self.insulin_max_value,
                neginf=self.insulin_min_value,
            )
            tr_pred = self._clamp_insulin_prediction(tr_pred)
            prev_insulin_mask = self._build_insulin_mask_from_prediction(tr_pred_mask)

            if t == 0:
                bg_emb = self._build_initial_bg_embedding(
                    initial_bg,
                    initial_bg_mask,
                    initial_bg_value,
                    initial_bg_index,
                    time_emb,
                )
            else:
                bg_emb = self.bg_value_encoder(
                    torch.cat([prev_bg_pred, torch.ones_like(prev_bg_pred, device=device)], dim=-1)
                ) + time_emb
                bg_emb = torch.nan_to_num(
                    bg_emb,
                    nan=self.bg_min_value,
                    posinf=self.bg_max_value,
                    neginf=self.bg_min_value,
                )
            bg_emb = torch.clamp(bg_emb, min=-50.0, max=50.0)
            bg_emb_k = self._expand_topk(bg_emb, topk)

            bg_insulin_condition_k = self._build_bg_insulin_condition(insulin_h_k)
            if t == 0:
                bg_query_k = torch.cat([bg_initial_h_k, bg_emb_k, bg_insulin_condition_k], dim=-1)
            else:
                bg_query_k = torch.cat([bg_h_k, bg_emb_k, bg_insulin_condition_k], dim=-1)

            bg_query_k = self.bg_transform(bg_query_k)
            bg_query_k = torch.clamp(bg_query_k, min=-50.0, max=50.0)
            bg_h_k = self._decode_topk(
                bg_query_k,
                bg_memory,
                self.bg_decoder,
                self.layernorm_bg,
                batch_size,
                topk,
                memory_lengths=memory_lengths,
            )
            bg_base_k = self._expand_topk(prev_bg_pred, topk)
            bg_prior_k = self._reference_step_values(
                reference_memories,
                "bg_values",
                t,
                self.cfg.d_bg,
                device,
                bg_h_k.dtype,
            )
            bg_pred_k = self._project_bg_prediction(bg_h_k, bg_base_k, bg_prior_k)
            bg_pred = self._aggregate_topk(bg_pred_k, topk_scores)
            bg_pred = torch.nan_to_num(
                bg_pred,
                nan=self.bg_min_value,
                posinf=self.bg_max_value,
                neginf=self.bg_min_value,
            )

            insulin_preds.append(tr_pred)
            bg_preds.append(bg_pred)
            tr_history.append(self._aggregate_topk(insulin_h_k, topk_scores))
            bg_history.append(self._aggregate_topk(bg_h_k, topk_scores))
            prev_insulin_pred = tr_pred
            prev_bg_pred = bg_pred

        return self._build_output(insulin_preds, bg_preds, bg_history, tr_history, real_lengths, actual_t, batch_size, device)

    def _build_output(self, insulin_preds, bg_preds, bg_history, tr_history, real_lengths, max_t, batch_size, device):
        if insulin_preds:
            insulin_preds_tensor = torch.stack(insulin_preds, dim=1)
            bg_preds_tensor = torch.stack(bg_preds, dim=1)
            tr_hiddens_tensor = torch.stack(tr_history, dim=1)
            bg_hiddens_tensor = torch.stack(bg_history, dim=1)
        else:
            insulin_preds_tensor = torch.zeros(batch_size, 0, self.cfg.d_insulin, device=device)
            bg_preds_tensor = torch.zeros(batch_size, 0, self.cfg.d_bg, device=device)
            tr_hiddens_tensor = torch.zeros(batch_size, 0, self.cfg.d_model, device=device)
            bg_hiddens_tensor = torch.zeros(batch_size, 0, self.cfg.d_model, device=device)

        if len(tr_history) > 0:
            final_tr_h = torch.stack(
                [tr_hiddens_tensor[b, min(real_lengths[b].item() - 1, max_t - 1)] for b in range(batch_size)]
            )
            final_bg_h = torch.stack(
                [bg_hiddens_tensor[b, min(real_lengths[b].item() - 1, max_t - 1)] for b in range(batch_size)]
            )
        else:
            final_tr_h = torch.zeros(batch_size, self.cfg.d_model, device=device)
            final_bg_h = torch.zeros(batch_size, self.cfg.d_model, device=device)

        return {
            "insulin_preds": insulin_preds_tensor,
            "bg_preds": bg_preds_tensor,
            "tr_h": final_tr_h,
            "bg_h": final_bg_h,
            "tr_hiddens": tr_hiddens_tensor,
            "bg_hiddens": bg_hiddens_tensor,
            "real_lengths": real_lengths,
        }


class DischargeHeadImproved(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dropout = cfg_get(cfg, "dropout", 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg_get(cfg, "nhead", 8),
            dim_feedforward=cfg_get(cfg, "d_ff", cfg.d_model * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_summarizer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.global_pool = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model // 2, cfg.d_insulin * cfg.d_insulin_route_stage + 1),
            nn.Softplus(),
        )

    def forward(self, tr_hiddens, bg_hiddens, real_lengths, batch):
        batch_size, seq_len, _ = tr_hiddens.shape
        device = tr_hiddens.device

        tr_summary = self.sequence_summarizer(tr_hiddens)
        bg_summary = self.sequence_summarizer(bg_hiddens)

        mask = torch.arange(seq_len, device=device).unsqueeze(0) < real_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        tr_mean = (tr_summary * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        bg_mean = (bg_summary * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        tr_max = (tr_summary + (1 - mask) * (-1e9)).max(dim=1)[0]
        bg_max = (bg_summary + (1 - mask) * (-1e9)).max(dim=1)[0]

        tr_global = self.global_pool(torch.cat([tr_mean, tr_max], dim=1))
        bg_global = self.global_pool(torch.cat([bg_mean, bg_max], dim=1))
        combined = torch.cat([tr_global, bg_global], dim=1)
        discharge_pred = self.output(combined)
        return torch.clamp(discharge_pred, min=0.0, max=200.0)


DischargeHead = DischargeHeadImproved
