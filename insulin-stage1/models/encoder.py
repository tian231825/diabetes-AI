# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from configs.Config import opt_config

def cfg_get(cfg, key, default=None):
    """兼容argparse.Namespace和dict的属性获取"""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

# =========================
# 改进组件：残差连接
# =========================
class ResidualConnection(nn.Module):
    """带有LayerNorm的残差连接"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))

# =========================
# 改进组件：个性化感知嵌入
# =========================
class PersonalityAwareEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dropout = cfg_get(cfg, 'dropout', 0.1)
        self.use_person_self_attention = bool(cfg_get(cfg, 'use_person_self_attention', 0))
        self.d_person = int(cfg.d_person)

        self.person_encoder = nn.Sequential(
            nn.Linear(cfg.d_person * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

        self.person_attention = nn.MultiheadAttention(
            cfg.d_model,
            num_heads=cfg_get(cfg, 'person_nhead', 4),
            dropout=dropout,
            batch_first=True
        )
        if self.use_person_self_attention:
            self.person_feature_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2, cfg.d_model),
                    nn.LayerNorm(cfg.d_model),
                )
                for _ in range(self.d_person)
            ])

        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, person_value, person_mask, context=None):
        person_input = torch.cat([person_value, person_mask], dim=1)
        person_emb = self.person_encoder(person_input)

        if self.use_person_self_attention:
            if person_value.shape[-1] != self.d_person or person_mask.shape[-1] != self.d_person:
                raise ValueError(
                    f"Expected {self.d_person} personal features, got "
                    f"{person_value.shape[-1]} values and {person_mask.shape[-1]} masks."
                )
            feature_pairs = torch.stack([person_value, person_mask], dim=-1)
            feature_context = torch.stack([
                encoder(feature_pairs[:, feature_idx, :])
                for feature_idx, encoder in enumerate(self.person_feature_encoders)
            ], dim=1)
            person_query = person_emb.unsqueeze(1)
            self_context, _ = self.person_attention(
                person_query,
                feature_context,
                feature_context,
            )
            person_emb = person_emb + self.dropout(self_context.squeeze(1))

        if context is not None:
            person_query = person_emb.unsqueeze(1)
            person_context, _ = self.person_attention(person_query, context, context)
            person_emb = person_emb + self.dropout(person_context.squeeze(1))

        person_emb = self.output_proj(person_emb)
        return person_emb

# =========================
# SimpleTransformer（通用版）
# =========================
class SimpleTransformer(nn.Module):
    def __init__(self, cfg, in_dim, is_decoder=False):
        super().__init__()
        dropout = cfg_get(cfg, 'dropout', 0.1)
        d_ff = cfg_get(cfg, 'd_ff', cfg.d_model * 4)

        self.proj = nn.Linear(in_dim, cfg.d_model)
        self.layer_norm_proj = nn.LayerNorm(cfg.d_model)

        if is_decoder:
            layer = nn.TransformerDecoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.core = nn.TransformerDecoder(layer, cfg.n_layer)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.core = nn.TransformerEncoder(layer, cfg.n_layer)

        self.is_decoder = is_decoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, memory=None, mem_mask=None):
        device = x.device
        x = self.layer_norm_proj(self.proj(x))
        x = self.dropout(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, min=-50.0, max=50.0)

        # 执行 Transformer 核心逻辑
        if not self.is_decoder:
            out = self.core(x, src_key_padding_mask=mask)
        else:
            if memory is not None:
                memory = torch.nan_to_num(memory, nan=0.0, posinf=1e4, neginf=-1e4)
                memory = torch.clamp(memory, min=-50.0, max=50.0)
            out = self.core(x, memory, tgt_mask=mask, memory_key_padding_mask=mem_mask)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.clamp(out, min=-50.0, max=50.0)

# =========================
# Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dropout = cfg_get(cfg, 'dropout', 0.1)

        self.person_embedding = PersonalityAwareEmbedding(cfg)

        self.bg_enc = SimpleTransformer(cfg, cfg.d_model * 3)
        self.tr_enc = SimpleTransformer(cfg, cfg.d_model * 3)

        self.bg_value2embedding = nn.Sequential(
            nn.Linear(cfg.d_bg * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

        self.treatment_value2embedding = nn.Sequential(
            nn.Linear(cfg.d_insulin * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

        self.drug_value2embedding = nn.Sequential(
            nn.Linear(cfg.d_drug, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

        self.treatment_context_fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

        # self.bg_mask_embedding = nn.Linear(cfg.d_bg, cfg.d_model)

        self.residual_bg = ResidualConnection(cfg.d_model, dropout)
        self.residual_tr = ResidualConnection(cfg.d_model, dropout)

        self.time_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    def encode_person(self, person_value, person_mask, context=None):
        return self.person_embedding(person_value, person_mask, context=context)

    @staticmethod
    def build_initial_bg(bg, bg_mask):
        B, T, d_bg = bg.shape
        device = bg.device
        initial_bg = torch.zeros(B, d_bg, device=device)
        initial_bg_mask = torch.zeros(B, d_bg, device=device)

        for b in range(B):
            for t in range(T):
                mask_t = bg_mask[b, t]
                if mask_t.any():
                    k = mask_t.nonzero(as_tuple=False)[0].item()
                    initial_bg[b, k] = bg[b, t, k]
                    initial_bg_mask[b, k] = 1.0
                    break
        return initial_bg, initial_bg_mask

    def forward(self, batch):
        device = next(self.parameters()).device

        for key in ["bg1", "bg1_mask", "insulin1", "insulin1_mask", "drug1", "person_value", "person_mask"]:
            if key in batch and torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)

        bg = batch["bg1"]
        bg_mask = batch["bg1_mask"]
        insulin = batch["insulin1"]
        insulin_mask = batch["insulin1_mask"]
        drug = batch.get("drug1")
        
        initial_bg, initial_bg_mask = self.build_initial_bg(bg, bg_mask)
        
        if insulin.dim() == 4:
            insulin = insulin.flatten(start_dim=2)
        if insulin_mask.dim() == 4:
            insulin_mask = insulin_mask.flatten(start_dim=2)
        B, T1, _ = bg.shape
        if drug is None:
            drug = torch.zeros(B, T1, self.cfg.d_drug, device=device, dtype=bg.dtype)
        elif drug.dim() == 2:
            drug = drug.unsqueeze(0)

        person_value = batch["person_value"]
        person_mask = batch["person_mask"]
        real_lengths = batch.get("real_lengths", None)
        if real_lengths is None:
            valid_timesteps = bg_mask.sum(dim=-1) > 0
            real_lengths = valid_timesteps.long().sum(dim=1).clamp(min=1)
        elif not torch.is_tensor(real_lengths):
            real_lengths = torch.as_tensor(real_lengths, dtype=torch.long, device=device)
        real_lengths = real_lengths.to(device).long()

        person_emb = self.encode_person(person_value, person_mask)

        # bg_h = person_emb.clone()
        # tr_h = person_emb.clone()
        i_h_0 = torch.zeros(B, self.cfg.d_model, device=device)

        bg_states = []
        tr_states = []
        
        bg_initial_h = self.bg_value2embedding(torch.cat([initial_bg, initial_bg_mask], dim=-1))
 

        # bg_initial_h_ = self.bg_enc(
        #     torch.cat([bg_h, bg_initial_now_h, i_h_0], dim=1).unsqueeze(1)
        # ).squeeze(1)

        max_pos = self.time_embedding.num_embeddings - 1
        positions = torch.arange(T1, device=device).clamp(max=max_pos).unsqueeze(0).expand(B, -1)

        for t in range(T1):
            if real_lengths is not None:
                valid_mask = (t < real_lengths).float().unsqueeze(1)
            else:
                valid_mask = torch.ones(B, 1, device=device)

            time_emb = self.time_embedding(positions[:, t])
            
            insulin_now_h = self.treatment_value2embedding(
                torch.cat([insulin[:, t], insulin_mask[:, t]], dim=-1)
            )
            drug_now_h = self.drug_value2embedding(drug[:, t])
            treatment_now_h = self.treatment_context_fusion(
                torch.cat([insulin_now_h, drug_now_h], dim=-1)
            ) + time_emb


            if t == 0:
                tr_input = torch.cat([bg_initial_h, treatment_now_h, i_h_0], dim=1).unsqueeze(1)
            else:
                tr_input = torch.cat([bg_states[-1], treatment_now_h, tr_h], dim=1).unsqueeze(1)

            tr_h_new = self.tr_enc(tr_input).squeeze(1)

            # TODO 此处与decoder不对称暂时没融入
            # tr_h = self.residual_tr(tr_h + person_emb * 0.1, tr_h_new)
            # tr_h = tr_h * valid_mask
            prev_tr_h = tr_states[-1] if t > 0 else i_h_0
            tr_h = tr_h_new * valid_mask + prev_tr_h * (1.0 - valid_mask)
            
            bg_now_h = self.bg_value2embedding(torch.cat([bg[:, t], bg_mask[:, t]], dim=-1) )
            bg_input_h = bg_now_h + time_emb

            if t == 0:
                bg_input = torch.cat([bg_initial_h, bg_input_h, tr_h], dim=1).unsqueeze(1)
            else:
                bg_input = torch.cat([bg_states[-1], bg_input_h, tr_h], dim=1).unsqueeze(1)

            bg_h_new = self.bg_enc(bg_input).squeeze(1)
            
            # TODO 此处与decoder不对称暂时没融入
            # bg_h = self.residual_bg(bg_h + person_emb * 0.1, bg_h_new)
            # bg_h = bg_h * valid_mask
            prev_bg_h = bg_states[-1] if t > 0 else bg_initial_h
            bg_h = bg_h_new * valid_mask + prev_bg_h * (1.0 - valid_mask)

            bg_states.append(bg_h)
            tr_states.append(tr_h)

        bg_states = torch.stack(bg_states, dim=1)  # [B, T, d_model]
        tr_states = torch.stack(tr_states, dim=1)  # [B, T, d_model]

        # 返回完整序列而不是只返回最后一个向量
        # 这样Stage2可以attend到参考患者的完整时序信息
        return {
            'bg_states': bg_states,  # [B, T, d_model]
            'tr_states': tr_states,  # [B, T, d_model]
            'person_emb': person_emb,  # [B, d_model]
            'real_lengths': real_lengths if real_lengths is not None else torch.full((B,), T1, device=bg_states.device)
        }

# # =========================
# # Mapper
# # =========================
# class Mapper(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(cfg.d_model, cfg.d_model),
#             nn.ReLU(),
#             nn.Linear(cfg.d_model, cfg.d_model // 2)
#         )

#     def forward(self, x):
#         return self.net(x)
