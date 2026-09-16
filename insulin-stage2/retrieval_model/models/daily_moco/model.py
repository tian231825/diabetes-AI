# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.loss import moco_contrastive_loss


class StaticMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.d_person * 2
        self.use_person_self_attention = bool(getattr(cfg, 'use_person_self_attention', 0))
        self.d_person = int(cfg.d_person)
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        if self.use_person_self_attention:
            person_nhead = int(getattr(cfg, 'person_nhead', 4))
            if cfg.d_model % person_nhead != 0:
                raise ValueError(
                    f"d_model={cfg.d_model} must be divisible by person_nhead={person_nhead}."
                )
            self.person_feature_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2, cfg.d_model),
                    nn.LayerNorm(cfg.d_model),
                )
                for _ in range(self.d_person)
            ])
            self.person_self_attention = nn.MultiheadAttention(
                cfg.d_model,
                num_heads=person_nhead,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.person_attention_dropout = nn.Dropout(cfg.dropout)
            self.person_attention_output = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
            )

    def forward(self, person_value, person_mask):
        x = torch.cat([person_value, person_mask], dim=-1)
        emb = self.net(x)
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
            self_context, _ = self.person_self_attention(
                emb.unsqueeze(1),
                feature_context,
                feature_context,
            )
            emb = self.person_attention_output(
                emb + self.person_attention_dropout(self_context.squeeze(1))
            )
        return emb


class DailySequenceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.regimen_flag_dim = int(getattr(cfg, 'regimen_flag_dim', 2))
        self.insulin_dose_dim = max(int(cfg.d_insulin) - self.regimen_flag_dim, 0)
        self.drug_dim = int(getattr(cfg, 'd_drug', 0))
        token_dim = cfg.d_bg + self.regimen_flag_dim + self.insulin_dose_dim + self.insulin_dose_dim + cfg.d_bg + self.drug_dim
        self.input_proj = nn.Sequential(
            nn.Linear(token_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.day_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.n_layer)
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, bg, bg_mask, insulin, insulin_mask, drug, lengths, bg_mean, bg_std, insulin_dose_mean, insulin_dose_std, drug_mean, drug_std):
        bg_norm = (bg - bg_mean.view(1, 1, -1)) / bg_std.view(1, 1, -1)
        if insulin_mask is None:
            insulin_mask = torch.ones_like(insulin)
        insulin_mask = insulin_mask.float()
        regimen_flags = insulin[..., :self.regimen_flag_dim]
        insulin_dose = insulin[..., self.regimen_flag_dim:]
        insulin_dose_mask = insulin_mask[..., self.regimen_flag_dim:]
        if drug is None:
            drug = torch.zeros(bg.shape[0], bg.shape[1], self.drug_dim, device=bg.device, dtype=bg.dtype)
        drug_norm = (drug - drug_mean.view(1, 1, -1)) / drug_std.view(1, 1, -1) if self.drug_dim > 0 else drug
        if self.insulin_dose_dim > 0:
            insulin_dose_norm = (insulin_dose - insulin_dose_mean.view(1, 1, -1)) / insulin_dose_std.view(1, 1, -1)
            insulin_dose_norm = insulin_dose_norm * insulin_dose_mask
            tokens = torch.cat([bg_norm, regimen_flags, insulin_dose_norm, insulin_dose_mask, bg_mask, drug_norm], dim=-1)
        else:
            tokens = torch.cat([bg_norm, regimen_flags, bg_mask, drug_norm], dim=-1)

        hidden = self.input_proj(tokens)
        day_idx = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        hidden = hidden + self.day_embedding(day_idx)

        key_padding_mask = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled, encoded


class FuseMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.repr_dim),
            nn.LayerNorm(cfg.repr_dim),
        )

    def forward(self, static_emb, seq_emb):
        return self.net(torch.cat([static_emb, seq_emb], dim=-1))


class ProjectionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.repr_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.proj_dim),
        )

    def forward(self, r):
        return F.normalize(self.net(r), dim=-1)


class DailyEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.static_encoder = StaticMLP(cfg)
        self.seq_encoder = DailySequenceEncoder(cfg)
        self.fuse = FuseMLP(cfg)
        self.projector = ProjectionHead(cfg)

    def forward(self, batch, stats):
        static_emb = self.static_encoder(batch["person_value"], batch["person_mask"])
        seq_emb, seq_states = self.seq_encoder(
            batch["bg"],
            batch["bg_mask"],
            batch["insulin"],
            batch.get("insulin_mask"),
            batch.get("drug"),
            batch["len"].long(),
            stats.bg_mean.to(batch["bg"].device),
            stats.bg_std.to(batch["bg"].device),
            stats.insulin_dose_mean.to(batch["bg"].device),
            stats.insulin_dose_std.to(batch["bg"].device),
            stats.drug_mean.to(batch["bg"].device),
            stats.drug_std.to(batch["bg"].device),
        )
        static_emb = F.normalize(static_emb, dim=-1)
        seq_emb = F.normalize(seq_emb, dim=-1)
        static_weight = float(getattr(self.seq_encoder.cfg, "static_weight", 0.5))
        seq_weight = float(getattr(self.seq_encoder.cfg, "seq_weight", 1.5))
        r = self.fuse(static_emb * static_weight, seq_emb * seq_weight)
        h = F.normalize(r, dim=-1)
        z = self.projector(r)
        return {
            "r": r,
            "h": h,
            "z": z,
            "static_emb": static_emb,
            "seq_emb": seq_emb,
            "seq_states": seq_states,
        }


class DailyMoCoSim(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_q = DailyEncoder(cfg)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(cfg.queue_size, cfg.proj_dim), dim=-1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        m = self.cfg.moco_momentum
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[0]
        ptr = int(self.queue_ptr.item())

        if batch_size >= queue_size:
            self.queue.copy_(keys[-queue_size:])
            self.queue_ptr[0] = 0
            return

        end = ptr + batch_size
        if end <= queue_size:
            self.queue[ptr:end] = keys
        else:
            first = queue_size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[:end - queue_size] = keys[first:]
        self.queue_ptr[0] = end % queue_size

    def forward_train(self, batch_q, batch_k, stats, update_momentum=True):
        q_out = self.encoder_q(batch_q, stats)
        with torch.no_grad():
            if update_momentum:
                self.momentum_update()
            k_out = self.encoder_k(batch_k, stats)

        z_q = q_out["z"]
        z_k = k_out["z"]

        pos = torch.sum(z_q * z_k, dim=-1, keepdim=True) / self.cfg.contrast_temp
        neg = z_q @ self.queue.t() / self.cfg.contrast_temp
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = moco_contrastive_loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "q": q_out,
            "k": k_out,
            "keys_to_enqueue": z_k.detach(),
        }

    def encode(self, batch, stats, use_momentum=False):
        encoder = self.encoder_k if use_momentum else self.encoder_q
        return encoder(batch, stats)






