# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = int(cfg.d_person) * 2
        hidden_dim = int(cfg.person_hidden_dim)
        emb_dim = int(cfg.person_emb_dim)
        dropout = float(cfg.dropout)
        self.use_person_self_attention = bool(getattr(cfg, 'use_person_self_attention', 0))
        self.d_person = int(cfg.d_person)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        if self.use_person_self_attention:
            person_nhead = int(getattr(cfg, 'person_nhead', 4))
            if emb_dim % person_nhead != 0:
                raise ValueError(
                    f"person_emb_dim={emb_dim} must be divisible by person_nhead={person_nhead}."
                )
            self.person_feature_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2, emb_dim),
                    nn.LayerNorm(emb_dim),
                )
                for _ in range(self.d_person)
            ])
            self.person_self_attention = nn.MultiheadAttention(
                emb_dim,
                num_heads=person_nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.person_attention_dropout = nn.Dropout(dropout)
            self.person_attention_output = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
            )

    def forward(self, person_value, person_mask):
        person_value = person_value * person_mask
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
        return F.normalize(emb, dim=-1)


class PersonPairSimilarityModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PersonEncoder(cfg)
        emb_dim = int(cfg.person_emb_dim)
        hidden_dim = int(cfg.person_hidden_dim)
        dropout = float(cfg.dropout)
        self.sim_head = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_person(self, person_value, person_mask):
        return self.encoder(person_value, person_mask)

    def forward(self, person_value_a, person_mask_a, person_value_b, person_mask_b):
        emb_a = self.encode_person(person_value_a, person_mask_a)
        emb_b = self.encode_person(person_value_b, person_mask_b)
        cosine_sim = torch.sum(emb_a * emb_b, dim=-1)
        pair_feat = torch.cat([torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1)
        sim_residual = self.sim_head(pair_feat).squeeze(-1)
        sim = cosine_sim + 0.1 * sim_residual
        return {
            'person_emb_a': emb_a,
            'person_emb_b': emb_b,
            'cosine_sim': cosine_sim,
            'pred_sim': sim,
        }
