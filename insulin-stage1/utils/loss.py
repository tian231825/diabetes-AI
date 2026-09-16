# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.Config import opt_config

def cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

# Loss Classes: DiversityAwareLoss, ContrastiveLoss, FocalLoss, CombinedLoss

class DiversityAwareLoss(nn.Module):
    def __init__(self, base_loss='l1', alpha=0.1, beta=0.05, eps=1e-8):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target, person_features=None, mask=None):
        batch_size = pred.shape[0]
        if self.base_loss == 'mse':
            if mask is not None:
                mask = mask.float()
                # base_loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() + self.eps)
                mask_sum = mask.sum()
                if mask_sum < 1e-6:
                    mask_sum = 1.0  # 避免除零
                base_loss = ((pred - target) ** 2 * mask).sum() / mask_sum
            else:
                base_loss = F.mse_loss(pred, target)
        else:
            if mask is not None:
                mask = mask.float()
                mask_sum = mask.sum()
                if mask_sum < 1e-6:
                    mask_sum = 1.0  # 避免除零
                base_loss = (torch.abs(pred - target) * mask).sum() / mask_sum
            else:
                base_loss = F.l1_loss(pred, target)
        loss_dict = {'base': base_loss.item()}
        total_loss = base_loss
        if self.alpha > 0 and batch_size > 1:
            diversity_loss = self._diversity_regularization(pred, person_features)
            loss_dict['diversity'] = diversity_loss.item()
            total_loss = total_loss + self.alpha * diversity_loss
        if self.beta > 0:
            sparsity_loss = self._sparsity_penalty(pred)
            loss_dict['sparsity'] = sparsity_loss.item()
            total_loss = total_loss + self.beta * sparsity_loss
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
            diversity_loss = (diversity_loss * mask).sum() / (mask.sum() + self.eps)
        else:
            mask = torch.ones_like(pred_sim) - torch.eye(batch_size, device=pred_sim.device)
            diversity_loss = (pred_sim * mask).sum() / (mask.sum() + self.eps)
        return diversity_loss

    def _sparsity_penalty(self, pred):
        pred_abs_mean = torch.abs(pred).mean()
        return torch.exp(-pred_abs_mean)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, pred, person_features):
        batch_size = pred.shape[0]
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
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        mse = (pred - target) ** 2
        if mask is not None: mse = mse * mask.float()
        pt = torch.exp(-mse)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * mse
        if mask is not None:
            loss = loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'mean': loss = loss.mean()
        elif self.reduction == 'sum': loss = loss.sum()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 这一层把“分类正确”“剂量接近”“临床阈值可用”三种目标揉到一起，
        # 是当前项目里最核心的训练目标定义。
        self.insulin_flag_dims = cfg_get(cfg, 'insulin_flag_dims', 3)
        self.insulin_type_loss_weight = cfg_get(cfg, 'insulin_type_loss_weight', 0.5)
        self.type_weight_basic = cfg_get(cfg, 'insulin_type_weight_basic', 1.0)
        self.type_weight_premix = cfg_get(cfg, 'insulin_type_weight_premix', 1.0)
        self.type_weight_none = cfg_get(cfg, 'insulin_type_weight_none', 1.0)
        self.premix_dose_loss_weight = cfg_get(cfg, 'premix_dose_loss_weight', 1.0)
        self.basic_sc_focus_loss_weight = cfg_get(cfg, 'basic_sc_focus_loss_weight', 0.0)
        self.basic_merged_focus_loss_weight = cfg_get(cfg, 'basic_merged_focus_loss_weight', 0.0)
        self.premix_sc_focus_loss_weight = cfg_get(cfg, 'premix_sc_focus_loss_weight', 0.0)
        self.premix_recall_focus_loss_weight = cfg_get(cfg, 'premix_recall_focus_loss_weight', 0.0)
        self.pump_dose_weight = cfg_get(cfg, 'pump_dose_weight', 1.0)
        self.iv_dose_weight = cfg_get(cfg, 'iv_dose_weight', 1.0)
        self.micro_dose_weight = cfg_get(cfg, 'micro_dose_weight', 1.0)
        self.effective_dose_threshold = cfg_get(cfg, 'effective_dose_threshold', 2.0)
        self.effective_dose_weight = cfg_get(cfg, 'effective_dose_weight', 1.0)
        self.effective_presence_loss_weight = cfg_get(cfg, 'effective_presence_loss_weight', 0.0)
        self.inactive_dose_suppress_weight = cfg_get(cfg, 'inactive_dose_suppress_weight', 0.0)
        self.insulin_delta_loss_weight = cfg_get(cfg, 'insulin_delta_loss_weight', 0.0)
        self.insulin_delta_change_threshold = cfg_get(cfg, 'insulin_delta_change_threshold', 1.0)
        self.insulin_delta_change_weight = cfg_get(cfg, 'insulin_delta_change_weight', 3.0)
        self.insulin_daily_total_delta_loss_weight = cfg_get(cfg, 'insulin_daily_total_delta_loss_weight', 0.0)
        self.high_bg_threshold = cfg_get(cfg, 'high_bg_threshold', 10.0)
        self.high_bg_focus_loss_weight = cfg_get(cfg, 'high_bg_focus_loss_weight', 0.0)
        self.bg_delta_loss_weight = cfg_get(cfg, 'bg_delta_loss_weight', 0.0)
        self.bg_pattern_loss_weight = cfg_get(cfg, 'bg_pattern_loss_weight', 0.0)
        self.bg_underpredict_loss_weight = cfg_get(cfg, 'bg_underpredict_loss_weight', 0.0)
        self.high_bg_underpredict_extra_weight = cfg_get(cfg, 'high_bg_underpredict_extra_weight', 1.0)
        self.diversity_loss = DiversityAwareLoss(
            base_loss='l1',
            alpha=cfg_get(cfg, 'diversity_alpha', 0.1),
            beta=cfg_get(cfg, 'sparsity_beta', 0.05)
        )
        if cfg_get(cfg, 'use_contrastive', True):
            self.contrastive_loss = ContrastiveLoss(temperature=cfg_get(cfg, 'contrast_temp', 0.5))
        else: self.contrastive_loss = None
        
        if cfg_get(cfg, 'use_focal', True):
            self.focal_loss = FocalLoss(alpha=cfg_get(cfg, 'focal_alpha', 0.25), gamma=cfg_get(cfg, 'focal_gamma', 2.0))
        else: self.focal_loss = None
        
        self.w_base = cfg_get(cfg, 'w_base', 1.0)
        self.w_contrast = cfg_get(cfg, 'w_contrast', 0.2)
        self.w_focal = cfg_get(cfg, 'w_focal', 0.5)

    def _get_sc_slot_slices(self, dose_dim):
        if dose_dim >= 14:
            return slice(0, 4), slice(4, 8)
        if dose_dim >= 8:
            return slice(0, 4), slice(4, 8)
        return None, None

    def _masked_slot_l1(self, pred, target, mask):
        if mask is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        valid = mask.float()
        denom = valid.sum()
        if denom.item() <= 1e-8:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return (torch.abs(pred - target) * valid).sum() / denom

    def _bg_delta_loss(self, pred, target, mask):
        if pred.dim() < 3 or pred.shape[1] < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
        target_delta = target[:, 1:, :] - target[:, :-1, :]
        if mask is None:
            delta_mask = torch.ones_like(pred_delta)
        else:
            delta_mask = (mask[:, 1:, :] * mask[:, :-1, :]).float()
        return self._masked_slot_l1(pred_delta, target_delta, delta_mask)

    def _bg_pattern_loss(self, pred, target, mask):
        if pred.dim() < 3:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        if mask is None:
            valid = torch.ones_like(pred)
        else:
            valid = mask.float()
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pred_center = pred - (pred * valid).sum(dim=-1, keepdim=True) / denom
        target_center = target - (target * valid).sum(dim=-1, keepdim=True) / denom
        return self._masked_slot_l1(pred_center, target_center, valid)

    def _bg_underpredict_loss(self, pred, target, mask):
        under_gap = F.relu(target - pred)
        if mask is None:
            weight = torch.ones_like(under_gap)
        else:
            weight = mask.float()
        if self.high_bg_underpredict_extra_weight != 1.0:
            high_bg_boost = 1.0 + (self.high_bg_underpredict_extra_weight - 1.0) * (target >= self.high_bg_threshold).float()
            weight = weight * high_bg_boost
        denom = weight.sum().clamp_min(1e-8)
        return (under_gap * weight).sum() / denom

    def _insulin_delta_losses(self, dose_pred, dose_target, dose_mask):
        zero = torch.zeros((), device=dose_pred.device, dtype=dose_pred.dtype)
        if dose_pred.dim() < 3 or dose_pred.shape[1] < 2:
            return zero, zero

        pred_delta = dose_pred[:, 1:, :] - dose_pred[:, :-1, :]
        target_delta = dose_target[:, 1:, :] - dose_target[:, :-1, :]
        if dose_mask is None:
            pair_mask = torch.ones_like(pred_delta)
        else:
            pair_mask = (dose_mask[:, 1:, :] * dose_mask[:, :-1, :]).float()

        change_threshold = float(self.insulin_delta_change_threshold)
        change_weight = max(float(self.insulin_delta_change_weight), 1.0)
        if change_threshold > 0 and change_weight > 1.0:
            changed = (target_delta.abs() >= change_threshold).float()
            pair_mask = pair_mask * (1.0 + (change_weight - 1.0) * changed)

        delta_loss = self._masked_slot_l1(pred_delta, target_delta, pair_mask)

        if self.insulin_daily_total_delta_loss_weight <= 0:
            return delta_loss, zero

        if dose_mask is None:
            day_mask = torch.ones_like(dose_target)
        else:
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

    def forward(self, pred, target, person_features=None, hidden_states=None, mask=None, task_name=None):
        loss_dict = {}
        if task_name == "insulin":
            if self.insulin_flag_dims > 0 and pred.shape[-1] >= self.insulin_flag_dims:
                type_pred = pred[..., :self.insulin_flag_dims].clamp(min=1e-6, max=1.0)
                type_pred = type_pred / type_pred.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                type_target = target[..., :self.insulin_flag_dims]
                dose_pred = pred[..., self.insulin_flag_dims:]
                dose_target = target[..., self.insulin_flag_dims:]

                if mask is not None:
                    type_mask = mask[..., :self.insulin_flag_dims]
                    dose_mask = mask[..., self.insulin_flag_dims:]
                    valid_type = (type_mask.sum(dim=-1) > 0).float()
                else:
                    type_mask = None
                    dose_mask = None
                    valid_type = torch.ones_like(type_target[..., 0])
            else:
                type_pred = None
                type_target = None
                dose_pred = pred
                dose_target = target
                type_mask = None
                dose_mask = mask
                valid_type = None

            if self.insulin_flag_dims >= 2:
                premix_type = type_target[..., 1].unsqueeze(-1)
                dose_weight = 1.0 + (self.premix_dose_loss_weight - 1.0) * premix_type
            else:
                dose_weight = 1.0

            # 不同胰岛素编码下，泵 / IV / 微泵槽位的位置不同，
            # 这里按输出维度自动切换索引。
            if dose_pred.shape[-1] >= 14:
                slot_weight = torch.ones_like(dose_pred)
                slot_weight[..., 8] *= self.pump_dose_weight
                slot_weight[..., 9] *= self.pump_dose_weight
                slot_weight[..., 10] *= self.pump_dose_weight
                slot_weight[..., 11] *= self.pump_dose_weight
                slot_weight[..., 12] *= self.iv_dose_weight
                slot_weight[..., 13] *= self.micro_dose_weight
                dose_weight = dose_weight * slot_weight
            elif dose_pred.shape[-1] >= 10:
                slot_weight = torch.ones_like(dose_pred)
                slot_weight[..., 1] *= self.pump_dose_weight
                slot_weight[..., 3] *= self.pump_dose_weight
                slot_weight[..., 5] *= self.pump_dose_weight
                slot_weight[..., 7] *= self.pump_dose_weight
                slot_weight[..., 8] *= self.iv_dose_weight
                slot_weight[..., 9] *= self.micro_dose_weight
                dose_weight = dose_weight * slot_weight
            elif dose_pred.shape[-1] >= 6:
                slot_weight = torch.ones_like(dose_pred)
                slot_weight[..., 4] *= self.micro_dose_weight
                slot_weight[..., 5] *= self.iv_dose_weight
                dose_weight = dose_weight * slot_weight

            if dose_mask is not None:
                effective_target = (dose_target >= self.effective_dose_threshold).float() * dose_mask
                inactive_target = (dose_target < self.effective_dose_threshold).float() * dose_mask
            else:
                effective_target = (dose_target >= self.effective_dose_threshold).float()
                inactive_target = (dose_target < self.effective_dose_threshold).float()

            # 这部分不是硬截断输出，而是用额外 loss 鼓励“该有值的位”跨过临床阈值。
            if self.effective_dose_weight != 1.0:
                dose_weight = dose_weight * (
                    1.0 + (self.effective_dose_weight - 1.0) * effective_target
                )

            weighted_dose_mask = dose_mask * dose_weight if dose_mask is not None else None
            diversity_loss, diversity_dict = self.diversity_loss(dose_pred, dose_target, person_features, weighted_dose_mask)
            loss_dict.update({f'diversity_{k}': v for k, v in diversity_dict.items()})
            total_loss = self.w_base * diversity_loss

            if self.insulin_flag_dims > 0 and type_pred is not None:
                class_weights = [self.type_weight_basic, self.type_weight_premix, self.type_weight_none]
                if self.insulin_flag_dims > len(class_weights):
                    class_weights.extend([1.0] * (self.insulin_flag_dims - len(class_weights)))
                type_weight_tensor = torch.tensor(
                    class_weights[:self.insulin_flag_dims],
                    dtype=type_pred.dtype,
                    device=type_pred.device,
                )
                target_idx = torch.argmax(type_target, dim=-1)
                type_loss = F.nll_loss(
                    torch.log(type_pred.reshape(-1, self.insulin_flag_dims)),
                    target_idx.reshape(-1),
                    weight=type_weight_tensor,
                    reduction='none'
                ).reshape_as(valid_type)
                type_loss = (type_loss * valid_type).sum() / (valid_type.sum() + 1e-8)
                loss_dict['type'] = type_loss.item()
                total_loss = total_loss + self.insulin_type_loss_weight * type_loss

            if self.premix_recall_focus_loss_weight > 0 and self.insulin_flag_dims >= 2:
                premix_prob = type_pred[..., 1].clamp(min=1e-6, max=1.0 - 1e-6)
                premix_target = type_target[..., 1]
                premix_valid = valid_type
                positive_boost = 1.0 + 2.0 * premix_target
                premix_bce = F.binary_cross_entropy(
                    premix_prob,
                    premix_target,
                    reduction='none'
                )
                premix_recall_focus_loss = (
                    premix_bce * positive_boost * premix_valid
                ).sum() / (premix_valid.sum() + 1e-8)
                loss_dict['premix_recall_focus'] = premix_recall_focus_loss.item()
                total_loss = total_loss + self.premix_recall_focus_loss_weight * premix_recall_focus_loss

            if self.effective_presence_loss_weight > 0:
                effective_gap = F.relu(self.effective_dose_threshold - dose_pred)
                effective_gap_loss = (effective_gap * effective_target).sum() / (effective_target.sum() + 1e-8)
                loss_dict['effective_presence'] = effective_gap_loss.item()
                total_loss = total_loss + self.effective_presence_loss_weight * effective_gap_loss

            if self.inactive_dose_suppress_weight > 0:
                inactive_over = F.relu(dose_pred - self.effective_dose_threshold)
                inactive_suppress_loss = (inactive_over * inactive_target).sum() / (inactive_target.sum() + 1e-8)
                loss_dict['inactive_suppress'] = inactive_suppress_loss.item()
                total_loss = total_loss + self.inactive_dose_suppress_weight * inactive_suppress_loss

            if self.insulin_delta_loss_weight > 0 or self.insulin_daily_total_delta_loss_weight > 0:
                insulin_delta_loss, insulin_total_delta_loss = self._insulin_delta_losses(
                    dose_pred,
                    dose_target,
                    dose_mask,
                )
                if self.insulin_delta_loss_weight > 0:
                    loss_dict['insulin_delta'] = insulin_delta_loss.item()
                    total_loss = total_loss + self.insulin_delta_loss_weight * insulin_delta_loss
                if self.insulin_daily_total_delta_loss_weight > 0:
                    loss_dict['insulin_daily_total_delta'] = insulin_total_delta_loss.item()
                    total_loss = total_loss + self.insulin_daily_total_delta_loss_weight * insulin_total_delta_loss

            if self.focal_loss is not None:
                focal_loss = self.focal_loss(dose_pred, dose_target, dose_mask)
                loss_dict['focal'] = focal_loss.item()
                total_loss = total_loss + self.w_focal * focal_loss

            basic_sc_slice, premix_sc_slice = self._get_sc_slot_slices(dose_pred.shape[-1])
            if self.insulin_flag_dims > 0 and basic_sc_slice is not None and dose_mask is not None:
                basic_day_mask = type_target[..., 0].unsqueeze(-1) * dose_mask[..., basic_sc_slice]
                premix_day_mask = type_target[..., 1].unsqueeze(-1) * dose_mask[..., premix_sc_slice]

                if self.basic_sc_focus_loss_weight > 0:
                    basic_sc_focus_loss = self._masked_slot_l1(
                        dose_pred[..., basic_sc_slice],
                        dose_target[..., basic_sc_slice],
                        basic_day_mask,
                    )
                    loss_dict['basic_sc_focus'] = basic_sc_focus_loss.item()
                    total_loss = total_loss + self.basic_sc_focus_loss_weight * basic_sc_focus_loss

                if self.basic_merged_focus_loss_weight > 0:
                    non_premix_day_mask = (1.0 - type_target[..., 1]).unsqueeze(-1) * dose_mask[..., basic_sc_slice]
                    basic_merged_focus_loss = self._masked_slot_l1(
                        dose_pred[..., basic_sc_slice],
                        dose_target[..., basic_sc_slice],
                        non_premix_day_mask,
                    )
                    loss_dict['basic_merged_focus'] = basic_merged_focus_loss.item()
                    total_loss = total_loss + self.basic_merged_focus_loss_weight * basic_merged_focus_loss

                if self.premix_sc_focus_loss_weight > 0:
                    premix_sc_focus_loss = self._masked_slot_l1(
                        dose_pred[..., premix_sc_slice],
                        dose_target[..., premix_sc_slice],
                        premix_day_mask,
                    )
                    loss_dict['premix_sc_focus'] = premix_sc_focus_loss.item()
                    total_loss = total_loss + self.premix_sc_focus_loss_weight * premix_sc_focus_loss
        else:
            diversity_loss, diversity_dict = self.diversity_loss(pred, target, person_features, mask)
            loss_dict.update({f'diversity_{k}': v for k, v in diversity_dict.items()})
            total_loss = self.w_base * diversity_loss

            if task_name == "bg" and self.high_bg_focus_loss_weight > 0:
                if mask is not None:
                    high_bg_mask = ((target >= self.high_bg_threshold).float() * mask.float())
                else:
                    high_bg_mask = (target >= self.high_bg_threshold).float()
                high_bg_focus_loss = self._masked_slot_l1(pred, target, high_bg_mask)
                loss_dict['high_bg_focus'] = high_bg_focus_loss.item()
                total_loss = total_loss + self.high_bg_focus_loss_weight * high_bg_focus_loss

            if task_name == "bg" and self.bg_delta_loss_weight > 0:
                bg_delta_loss = self._bg_delta_loss(pred, target, mask)
                loss_dict['bg_delta'] = bg_delta_loss.item()
                total_loss = total_loss + self.bg_delta_loss_weight * bg_delta_loss

            if task_name == "bg" and self.bg_pattern_loss_weight > 0:
                bg_pattern_loss = self._bg_pattern_loss(pred, target, mask)
                loss_dict['bg_pattern'] = bg_pattern_loss.item()
                total_loss = total_loss + self.bg_pattern_loss_weight * bg_pattern_loss

            if task_name == "bg" and self.bg_underpredict_loss_weight > 0:
                bg_underpredict_loss = self._bg_underpredict_loss(pred, target, mask)
                loss_dict['bg_underpredict'] = bg_underpredict_loss.item()
                total_loss = total_loss + self.bg_underpredict_loss_weight * bg_underpredict_loss

            if self.focal_loss is not None:
                focal_loss = self.focal_loss(pred, target, mask)
                loss_dict['focal'] = focal_loss.item()
                total_loss = total_loss + self.w_focal * focal_loss

        if self.contrastive_loss is not None and hidden_states is not None and person_features is not None:
            contrast_loss = self.contrastive_loss(hidden_states, person_features)
            loss_dict['contrastive'] = contrast_loss.item() if torch.is_tensor(contrast_loss) else contrast_loss
            total_loss = total_loss + self.w_contrast * contrast_loss

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

def get_loss_fn(cfg, delta: float = 1.0):
    # Standard loss functions (mse, huber, smooth_l1)
    # Used for simple masking if not using CombinedLoss
    if cfg.loss_type == "mse":
        def masked_mse(pred, target, mask: Optional[torch.Tensor] = None):
            if mask is None: return nn.MSELoss(reduction="mean")(pred, target)
            mask = mask.float()
            loss = nn.MSELoss(reduction='none')(pred, target)
            return (loss * mask).sum() / mask.sum()
        return masked_mse
    elif cfg.loss_type == "huber":
        def masked_huber(pred, target, mask: Optional[torch.Tensor] = None):
            if mask is None: return nn.HuberLoss(delta=delta, reduction="mean")(pred, target)
            mask = mask.float()
            loss = nn.HuberLoss(delta=delta, reduction='none')(pred, target)
            return (loss * mask).sum() / mask.sum()
        return masked_huber
    elif cfg.loss_type == "smooth_l1":
        def masked_smooth_l1(pred, target, mask: Optional[torch.Tensor] = None):
            if mask is None: return nn.SmoothL1Loss(beta=cfg.smooth_l1_beta, reduction="mean")(pred, target)
            mask = mask.float()
            loss = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta, reduction='none')(pred, target)
            return (loss * mask).sum() / mask.sum()
        return masked_smooth_l1
    else: raise ValueError(f"Unknown loss type: {cfg.loss_type}")
