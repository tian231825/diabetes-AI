# -*- coding: utf-8 -*-
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.dataloader2 import DiabetesDataset, collate_fn


@dataclass
class FeatureStats:
    bg_mean: torch.Tensor
    bg_std: torch.Tensor
    insulin_dose_mean: torch.Tensor
    insulin_dose_std: torch.Tensor
    drug_mean: torch.Tensor
    drug_std: torch.Tensor

    def to_dict(self):
        return {
            'bg_mean': self.bg_mean.cpu(),
            'bg_std': self.bg_std.cpu(),
            'insulin_dose_mean': self.insulin_dose_mean.cpu(),
            'insulin_dose_std': self.insulin_dose_std.cpu(),
            'drug_mean': self.drug_mean.cpu(),
            'drug_std': self.drug_std.cpu(),
        }

    @classmethod
    def from_dict(cls, payload):
        insulin_mean = payload.get('insulin_dose_mean', payload.get('insulin_mean'))
        insulin_std = payload.get('insulin_dose_std', payload.get('insulin_std'))
        return cls(
            bg_mean=torch.as_tensor(payload['bg_mean'], dtype=torch.float32),
            bg_std=torch.as_tensor(payload['bg_std'], dtype=torch.float32),
            insulin_dose_mean=torch.as_tensor(insulin_mean, dtype=torch.float32),
            insulin_dose_std=torch.as_tensor(insulin_std, dtype=torch.float32),
            drug_mean=torch.as_tensor(payload.get('drug_mean', []), dtype=torch.float32),
            drug_std=torch.as_tensor(payload.get('drug_std', []), dtype=torch.float32),
        )


class DailySimilarityDataset(DiabetesDataset):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg, mode=mode)
        kept = [
            (idx, item)
            for idx, item in zip(self.filtered_indices, self.filtered_data)
            if self._has_s1_history(item)
        ]
        self.filtered_indices = [idx for idx, _ in kept]
        self.filtered_data = [item for _, item in kept]
        self.n = len(self.filtered_data)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        insulin = item['insulin1']
        insulin_mask = item.get('insulin1_mask')
        if insulin_mask is None:
            insulin_mask = torch.ones_like(insulin)
        return {
            'check_id': item['check_id'],
            'person_value': item['person_value'],
            'person_mask': item['person_mask'],
            'bg': item['bg1'],
            'bg_mask': item['bg1_mask'],
            'insulin': insulin,
            'insulin_mask': insulin_mask,
            'drug': item['drug1'],
            'len': item['len1'],
        }


def compute_feature_stats(dataset):
    bg_sum = None
    bg_sq_sum = None
    bg_count = None
    insulin_dose_sum = None
    insulin_dose_sq_sum = None
    insulin_dose_count = None
    drug_sum = None
    drug_sq_sum = None
    drug_count = None
    flag_dim = int(getattr(dataset.cfg, 'regimen_flag_dim', 3))

    for idx in range(len(dataset)):
        item = dataset[idx]
        bg = item['bg'].float()
        bg_mask = item['bg_mask'].float()
        insulin = item['insulin'].float()
        insulin_mask = item.get('insulin_mask')
        drug = item['drug'].float()
        if insulin_mask is None:
            insulin_mask = torch.ones_like(insulin)
        insulin_mask = insulin_mask.float()
        insulin_dose = insulin[:, flag_dim:]
        insulin_dose_mask = insulin_mask[:, flag_dim:]

        masked_bg = bg * bg_mask
        masked_insulin_dose = insulin_dose * insulin_dose_mask
        if bg_sum is None:
            bg_sum = masked_bg.sum(dim=0)
            bg_sq_sum = (bg.pow(2) * bg_mask).sum(dim=0)
            bg_count = bg_mask.sum(dim=0)
            insulin_dose_sum = masked_insulin_dose.sum(dim=0)
            insulin_dose_sq_sum = (insulin_dose.pow(2) * insulin_dose_mask).sum(dim=0)
            insulin_dose_count = insulin_dose_mask.sum(dim=0)
            drug_sum = drug.sum(dim=0)
            drug_sq_sum = drug.pow(2).sum(dim=0)
            drug_count = torch.full_like(drug_sum, float(drug.shape[0]))
        else:
            bg_sum += masked_bg.sum(dim=0)
            bg_sq_sum += (bg.pow(2) * bg_mask).sum(dim=0)
            bg_count += bg_mask.sum(dim=0)
            insulin_dose_sum += masked_insulin_dose.sum(dim=0)
            insulin_dose_sq_sum += (insulin_dose.pow(2) * insulin_dose_mask).sum(dim=0)
            insulin_dose_count += insulin_dose_mask.sum(dim=0)
            drug_sum += drug.sum(dim=0)
            drug_sq_sum += drug.pow(2).sum(dim=0)
            drug_count += torch.full_like(drug_sum, float(drug.shape[0]))

    bg_mean = bg_sum / bg_count.clamp_min(1.0)
    bg_var = bg_sq_sum / bg_count.clamp_min(1.0) - bg_mean.pow(2)
    bg_std = torch.sqrt(bg_var.clamp_min(1e-6))

    insulin_dose_count = insulin_dose_count.clamp_min(1.0)
    insulin_dose_mean = insulin_dose_sum / insulin_dose_count
    insulin_dose_var = insulin_dose_sq_sum / insulin_dose_count - insulin_dose_mean.pow(2)
    insulin_dose_std = torch.sqrt(insulin_dose_var.clamp_min(1e-6))
    drug_count = drug_count.clamp_min(1.0)
    drug_mean = drug_sum / drug_count
    drug_var = drug_sq_sum / drug_count - drug_mean.pow(2)
    drug_std = torch.sqrt(drug_var.clamp_min(1e-6))

    return FeatureStats(
        bg_mean=bg_mean.float(),
        bg_std=bg_std.float(),
        insulin_dose_mean=insulin_dose_mean.float(),
        insulin_dose_std=insulin_dose_std.float(),
        drug_mean=drug_mean.float(),
        drug_std=drug_std.float(),
    )
