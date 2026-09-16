# -*- encoding: utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class DiversityAwareLoss(nn.Module):
    def __init__(self, base_loss="l1", alpha=0.1, beta=0.05, eps=1e-8):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target, person_features=None, mask=None):
        if self.base_loss == "mse":
            base_loss = ((pred - target) ** 2 * mask.float()).sum() / (mask.sum() + self.eps) if mask is not None else F.mse_loss(pred, target)
        else:
            base_loss = (torch.abs(pred - target) * mask.float()).sum() / (mask.sum() + self.eps) if mask is not None else F.l1_loss(pred, target)

        loss_dict = {"base": base_loss.item()}
        total_loss = base_loss

        if self.alpha > 0 and pred.shape[0] > 1:
            diversity_loss = self._diversity_regularization(pred, person_features)
            total_loss = total_loss + self.alpha * diversity_loss
            loss_dict["diversity"] = diversity_loss.item()

        if self.beta > 0:
            sparsity_loss = self._sparsity_penalty(pred)
            total_loss = total_loss + self.beta * sparsity_loss
            loss_dict["sparsity"] = sparsity_loss.item()

        return total_loss, loss_dict

    def _diversity_regularization(self, pred, person_features):
        batch_size = pred.shape[0]
        pred_flat = pred.flatten(1)
        pred_norm = F.normalize(pred_flat, dim=1, eps=self.eps)
        pred_sim = torch.matmul(pred_norm, pred_norm.t())
        if person_features is not None:
            person_norm = F.normalize(person_features, dim=1, eps=self.eps)
            person_sim = torch.matmul(person_norm, person_norm.t())
            diversity_loss = F.relu(pred_sim - person_sim)
            mask = torch.ones_like(diversity_loss) - torch.eye(batch_size, device=diversity_loss.device)
            return (diversity_loss * mask).sum() / (mask.sum() + self.eps)

        mask = torch.ones_like(pred_sim) - torch.eye(batch_size, device=pred_sim.device)
        return (pred_sim * mask).sum() / (mask.sum() + self.eps)

    def _sparsity_penalty(self, pred):
        return torch.exp(-torch.abs(pred).mean())


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, pred, person_features):
        pred_norm = F.normalize(pred, dim=1)
        person_norm = F.normalize(person_features, dim=1)
        pred_sim = torch.matmul(pred_norm, pred_norm.t()) / self.temperature
        person_sim = torch.matmul(person_norm, person_norm.t())
        pos_mask = person_sim > 0.7
        neg_mask = person_sim < 0.3
        pos_loss = F.relu(self.margin - pred_sim[pos_mask]).mean() if pos_mask.any() else 0
        neg_loss = F.relu(pred_sim[neg_mask] + self.margin).mean() if neg_mask.any() else 0
        return pos_loss + neg_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        mse = (pred - target) ** 2
        if mask is not None:
            mse = mse * mask.float()
        pt = torch.exp(-mse)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * mse
        if mask is not None:
            return loss.sum() / (mask.sum() + 1e-8)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.insulin_type_loss_weight = cfg_get(cfg, "insulin_type_loss_weight", 0.0)
        self.insulin_type_weight_basic = cfg_get(cfg, "insulin_type_weight_basic", 1.0)
        self.insulin_type_weight_premix = cfg_get(cfg, "insulin_type_weight_premix", 1.0)
        self.insulin_type_weight_none = cfg_get(cfg, "insulin_type_weight_none", 1.0)
        self.premix_recall_focus_loss_weight = cfg_get(cfg, "premix_recall_focus_loss_weight", 0.0)
        self.insulin_regression_loss_weight = cfg_get(cfg, "insulin_regression_loss_weight", 1.0)
        self.basic_dose_loss_weight = cfg_get(cfg, "basic_dose_loss_weight", 1.0)
        self.premix_dose_loss_weight = cfg_get(cfg, "premix_dose_loss_weight", 1.0)
        self.basic_sc_focus_loss_weight = cfg_get(cfg, "basic_sc_focus_loss_weight", 0.0)
        self.premix_sc_focus_loss_weight = cfg_get(cfg, "premix_sc_focus_loss_weight", 0.0)
        self.effective_dose_threshold = cfg_get(cfg, "effective_dose_threshold", 2.0)
        self.effective_dose_weight = cfg_get(cfg, "effective_dose_weight", 1.0)
        self.effective_presence_loss_weight = cfg_get(cfg, "effective_presence_loss_weight", 0.0)
        self.inactive_dose_suppress_weight = cfg_get(cfg, "inactive_dose_suppress_weight", 0.0)
        self.insulin_delta_loss_weight = cfg_get(cfg, "insulin_delta_loss_weight", 0.0)
        self.insulin_delta_change_threshold = cfg_get(cfg, "insulin_delta_change_threshold", 1.0)
        self.insulin_delta_change_weight = cfg_get(cfg, "insulin_delta_change_weight", 3.0)
        self.insulin_daily_total_delta_loss_weight = cfg_get(cfg, "insulin_daily_total_delta_loss_weight", 0.0)

        self.diversity_loss = DiversityAwareLoss(
            base_loss="l1",
            alpha=cfg_get(cfg, "diversity_alpha", 0.1),
            beta=cfg_get(cfg, "sparsity_beta", 0.05),
        )
        self.contrastive_loss = ContrastiveLoss(temperature=cfg_get(cfg, "contrast_temp", 0.5)) if cfg_get(cfg, "use_contrastive", True) else None
        self.focal_loss = FocalLoss(alpha=cfg_get(cfg, "focal_alpha", 0.25), gamma=cfg_get(cfg, "focal_gamma", 2.0)) if cfg_get(cfg, "use_focal", True) else None

        self.w_base = cfg_get(cfg, "w_base", 1.0)
        self.w_contrast = cfg_get(cfg, "w_contrast", 0.2)
        self.w_focal = cfg_get(cfg, "w_focal", 0.5)

    def _get_sc_slot_slices(self, dose_dim):
        if dose_dim == 5:
            return slice(1, 5), slice(1, 5)
        if dose_dim >= 8:
            return slice(0, 4), slice(4, 8)
        return None, None

    def _masked_slot_l1(self, pred, target, mask):
        valid = mask.float()
        denom = valid.sum()
        if denom.item() <= 1e-8:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return (torch.abs(pred - target) * valid).sum() / denom

    def _binary_flag_loss(self, dose_pred, regimen, regimen_mask):
        zero = torch.zeros((), device=dose_pred.device, dtype=dose_pred.dtype)
        if dose_pred.shape[-1] != 5 or regimen is None or regimen_mask is None or regimen.shape[-1] < 2:
            return zero

        valid_mask = (regimen_mask[..., :2].sum(dim=-1) > 0)
        basic_mask = valid_mask & (regimen[..., 0] > 0.5)
        premix_mask = valid_mask & (regimen[..., 1] > 0.5)
        supervised_mask = basic_mask | premix_mask
        if not supervised_mask.any():
            return zero

        target_flag = premix_mask.float()
        pred_flag = dose_pred[..., 0].clamp(min=1e-5, max=1.0 - 1e-5)
        day_weight = torch.ones_like(pred_flag)
        if self.insulin_type_weight_basic != 1.0:
            day_weight = torch.where(
                basic_mask,
                torch.full_like(day_weight, float(self.insulin_type_weight_basic)),
                day_weight,
            )
        if self.insulin_type_weight_premix != 1.0:
            day_weight = torch.where(
                premix_mask,
                torch.full_like(day_weight, float(self.insulin_type_weight_premix)),
                day_weight,
            )

        bce = F.binary_cross_entropy(pred_flag, target_flag, reduction="none")
        supervised_weight = supervised_mask.float() * day_weight
        return (bce * supervised_weight).sum() / (supervised_weight.sum() + 1e-8)

    def _insulin_delta_losses(self, dose_pred, dose_target, dose_mask):
        zero = torch.zeros((), device=dose_pred.device, dtype=dose_pred.dtype)
        if dose_pred.dim() < 3 or dose_pred.shape[1] < 2:
            return zero, zero

        pred_delta = dose_pred[:, 1:, :] - dose_pred[:, :-1, :]
        target_delta = dose_target[:, 1:, :] - dose_target[:, :-1, :]
        pair_mask = (dose_mask[:, 1:, :] * dose_mask[:, :-1, :]).float()

        change_threshold = float(self.insulin_delta_change_threshold)
        change_weight = max(float(self.insulin_delta_change_weight), 1.0)
        if change_threshold > 0 and change_weight > 1.0:
            changed = (target_delta.abs() >= change_threshold).float()
            pair_mask = pair_mask * (1.0 + (change_weight - 1.0) * changed)

        delta_loss = self._masked_slot_l1(pred_delta, target_delta, pair_mask)

        if self.insulin_daily_total_delta_loss_weight <= 0:
            return delta_loss, zero

        day_mask = dose_mask.float()
        valid_pair = ((day_mask[:, 1:, :].sum(dim=-1) > 0) & (day_mask[:, :-1, :].sum(dim=-1) > 0)).float()
        pred_total_delta = (dose_pred[:, 1:, :] * day_mask[:, 1:, :]).sum(dim=-1) - (
            dose_pred[:, :-1, :] * day_mask[:, :-1, :]
        ).sum(dim=-1)
        target_total_delta = (dose_target[:, 1:, :] * day_mask[:, 1:, :]).sum(dim=-1) - (
            dose_target[:, :-1, :] * day_mask[:, :-1, :]
        ).sum(dim=-1)
        if change_threshold > 0 and change_weight > 1.0:
            total_changed = (target_total_delta.abs() >= change_threshold).float()
            valid_pair = valid_pair * (1.0 + (change_weight - 1.0) * total_changed)
        total_delta_loss = (torch.abs(pred_total_delta - target_total_delta) * valid_pair).sum() / (
            valid_pair.sum() + 1e-8
        )
        return delta_loss, total_delta_loss

    def forward(self, pred, target, person_features=None, hidden_states=None, mask=None, task_name=None, regimen=None, regimen_mask=None):
        loss_dict = {}

        if task_name == "insulin":
            dose_pred = pred
            dose_target = target
            dose_mask = mask if mask is not None else torch.ones_like(dose_target)

            if regimen is not None and regimen.dim() >= 2 and regimen.size(1) != dose_pred.size(1):
                aligned_len = min(dose_pred.size(1), regimen.size(1))
                dose_pred = dose_pred[:, :aligned_len]
                dose_target = dose_target[:, :aligned_len]
                dose_mask = dose_mask[:, :aligned_len]
                regimen = regimen[:, :aligned_len]
                if regimen_mask is not None:
                    regimen_mask = regimen_mask[:, :aligned_len]

            dose_weight = torch.ones_like(dose_pred)
            basic_sc_slice, premix_sc_slice = self._get_sc_slot_slices(dose_pred.shape[-1])

            if regimen is not None and regimen_mask is not None and regimen.shape[-1] >= 2:
                basic_days = ((regimen[..., 0] > 0.5) & (regimen_mask[..., 0] > 0)).unsqueeze(-1)
                premix_days = ((regimen[..., 1] > 0.5) & (regimen_mask[..., 1] > 0)).unsqueeze(-1)
                if basic_sc_slice is not None:
                    dose_weight[..., basic_sc_slice] = torch.where(
                        basic_days.expand_as(dose_weight[..., basic_sc_slice]),
                        torch.full_like(dose_weight[..., basic_sc_slice], max(self.basic_dose_loss_weight, 1.0)),
                        dose_weight[..., basic_sc_slice],
                    )
                if premix_sc_slice is not None:
                    dose_weight[..., premix_sc_slice] = torch.where(
                        premix_days.expand_as(dose_weight[..., premix_sc_slice]),
                        torch.full_like(dose_weight[..., premix_sc_slice], max(self.premix_dose_loss_weight, 1.0)),
                        dose_weight[..., premix_sc_slice],
                    )

            effective_target = (dose_target >= self.effective_dose_threshold).float() * dose_mask
            inactive_target = (dose_target < self.effective_dose_threshold).float() * dose_mask
            if self.effective_dose_weight != 1.0:
                dose_weight = dose_weight * (1.0 + (self.effective_dose_weight - 1.0) * effective_target)

            weighted_dose_mask = dose_mask * dose_weight
            diversity_loss, diversity_dict = self.diversity_loss(dose_pred, dose_target, person_features, weighted_dose_mask)
            loss_dict.update({f"diversity_{k}": v for k, v in diversity_dict.items()})
            total_loss = self.insulin_regression_loss_weight * self.w_base * diversity_loss

            if self.effective_presence_loss_weight > 0:
                effective_gap = F.relu(self.effective_dose_threshold - dose_pred)
                effective_gap_loss = (effective_gap * effective_target).sum() / (effective_target.sum() + 1e-8)
                total_loss = total_loss + self.insulin_regression_loss_weight * self.effective_presence_loss_weight * effective_gap_loss
                loss_dict["effective_presence"] = effective_gap_loss.item()

            if self.inactive_dose_suppress_weight > 0:
                inactive_over = F.relu(dose_pred - self.effective_dose_threshold)
                inactive_suppress_loss = (inactive_over * inactive_target).sum() / (inactive_target.sum() + 1e-8)
                total_loss = total_loss + self.insulin_regression_loss_weight * self.inactive_dose_suppress_weight * inactive_suppress_loss
                loss_dict["inactive_suppress"] = inactive_suppress_loss.item()

            if self.insulin_delta_loss_weight > 0 or self.insulin_daily_total_delta_loss_weight > 0:
                insulin_delta_loss, insulin_total_delta_loss = self._insulin_delta_losses(
                    dose_pred,
                    dose_target,
                    dose_mask,
                )
                if self.insulin_delta_loss_weight > 0:
                    total_loss = total_loss + self.insulin_regression_loss_weight * self.insulin_delta_loss_weight * insulin_delta_loss
                    loss_dict["insulin_delta"] = insulin_delta_loss.item()
                if self.insulin_daily_total_delta_loss_weight > 0:
                    total_loss = total_loss + self.insulin_regression_loss_weight * self.insulin_daily_total_delta_loss_weight * insulin_total_delta_loss
                    loss_dict["insulin_daily_total_delta"] = insulin_total_delta_loss.item()

            if self.focal_loss is not None:
                focal_loss = self.focal_loss(dose_pred, dose_target, weighted_dose_mask)
                total_loss = total_loss + self.insulin_regression_loss_weight * self.w_focal * focal_loss
                loss_dict["focal"] = focal_loss.item()

            if self.insulin_type_loss_weight > 0:
                insulin_type_loss = self._binary_flag_loss(dose_pred, regimen, regimen_mask)
                total_loss = total_loss + self.insulin_type_loss_weight * insulin_type_loss
                loss_dict["insulin_type"] = insulin_type_loss.item()

            if regimen is not None and regimen_mask is not None and regimen.shape[-1] >= 2 and basic_sc_slice is not None:
                basic_day_mask = ((regimen[..., 0] > 0.5) & (regimen_mask[..., 0] > 0)).unsqueeze(-1).float() * dose_mask[..., basic_sc_slice]
                premix_day_mask = ((regimen[..., 1] > 0.5) & (regimen_mask[..., 1] > 0)).unsqueeze(-1).float() * dose_mask[..., premix_sc_slice]

                if self.basic_sc_focus_loss_weight > 0:
                    basic_sc_focus_loss = self._masked_slot_l1(
                        dose_pred[..., basic_sc_slice],
                        dose_target[..., basic_sc_slice],
                        basic_day_mask,
                    )
                    total_loss = total_loss + self.insulin_regression_loss_weight * self.basic_sc_focus_loss_weight * basic_sc_focus_loss
                    loss_dict["basic_sc_focus"] = basic_sc_focus_loss.item()

                if self.premix_sc_focus_loss_weight > 0:
                    premix_sc_focus_loss = self._masked_slot_l1(
                        dose_pred[..., premix_sc_slice],
                        dose_target[..., premix_sc_slice],
                        premix_day_mask,
                    )
                    total_loss = total_loss + self.insulin_regression_loss_weight * self.premix_sc_focus_loss_weight * premix_sc_focus_loss
                    loss_dict["premix_sc_focus"] = premix_sc_focus_loss.item()
        else:
            diversity_loss, diversity_dict = self.diversity_loss(pred, target, person_features, mask)
            loss_dict.update({f"diversity_{k}": v for k, v in diversity_dict.items()})
            total_loss = self.w_base * diversity_loss

            if self.focal_loss is not None:
                focal_loss = self.focal_loss(pred, target, mask)
                total_loss = total_loss + self.w_focal * focal_loss
                loss_dict["focal"] = focal_loss.item()

        if self.contrastive_loss is not None and hidden_states is not None and person_features is not None:
            contrast_loss = self.contrastive_loss(hidden_states, person_features)
            total_loss = total_loss + self.w_contrast * contrast_loss
            loss_dict["contrastive"] = contrast_loss.item() if torch.is_tensor(contrast_loss) else contrast_loss

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


def get_loss_fn(cfg, delta: float = 1.0):
    if cfg.loss_type == "mse":
        def masked_mse(pred, target, mask: Optional[torch.Tensor] = None):
            if mask is None:
                return nn.MSELoss(reduction="mean")(pred, target)
            loss = nn.MSELoss(reduction="none")(pred, target)
            return (loss * mask.float()).sum() / (mask.sum() + 1e-8)

        return masked_mse

    if cfg.loss_type == "huber":
        def masked_huber(pred, target, mask: Optional[torch.Tensor] = None):
            if mask is None:
                return nn.HuberLoss(delta=delta, reduction="mean")(pred, target)
            loss = nn.HuberLoss(delta=delta, reduction="none")(pred, target)
            return (loss * mask.float()).sum() / (mask.sum() + 1e-8)

        return masked_huber

    if cfg.loss_type == "smooth_l1":
        def masked_smooth_l1(pred, target, mask: Optional[torch.Tensor] = None):
            if mask is None:
                return nn.SmoothL1Loss(beta=cfg.smooth_l1_beta, reduction="mean")(pred, target)
            loss = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta, reduction="none")(pred, target)
            return (loss * mask.float()).sum() / (mask.sum() + 1e-8)

        return masked_smooth_l1

    raise ValueError(f"Unknown loss type: {cfg.loss_type}")
