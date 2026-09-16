# -*- encoding: utf-8 -*-
"""Training and evaluation entry point for stage 1."""
import os
import time
import logging
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from configs.Config import opt_config
from data.dataloader2 import DiabetesDataset, collate_fn
from data.MedicalDBPreprocessor import MedicalDBPreprocessor
from models.model import TwoStageModel
from utils.logger import setup_logging, set_seed
from utils.loss import CombinedLoss


def log_message(message=""):
    print(message)
    logging.getLogger(__name__).info(message)


def bootstrap_mean_ci(values, n_boot=2000, ci=95, seed=42):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v

    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2.0
    return (
        float(np.percentile(boot_means, alpha)),
        float(np.percentile(boot_means, 100 - alpha)),
    )


def format_ci(low, high):
    if not np.isfinite(low) or not np.isfinite(high):
        return "[nan, nan]"
    return f"[{low:.4f}, {high:.4f}]"


def collect_metric_histories(model, loader, criterion, cfg, device):
    metric_keys = [
        "insulin_dose_mae",
        "insulin_dose_mape",
        "insulin_dose_rmse",
        "insulin_dose_r2",
        "bg_insulin_mae_sum",
        "bg_mae", "bg_rmse", "bg_r2",
        "daily_insulin_acc", "daily_insulin_precision", "daily_insulin_mae", "daily_insulin_mape", "daily_insulin_rmse", "daily_insulin_r2",
        "micro_acc", "micro_precision", "micro_mae", "micro_mape", "micro_rmse", "micro_r2",
        "iv_acc", "iv_precision", "iv_mae", "iv_mape", "iv_rmse", "iv_r2",
    ]
    metric_histories = {key: [] for key in metric_keys}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch, tf_ratio=0.0, mode="val")
            _, loss_dict = compute_batch_loss(batch, outputs, criterion, cfg, mode="val")
            for key in metric_keys:
                metric_histories[key].append(loss_dict.get(key, 0.0))

    return {
        key: bootstrap_mean_ci(metric_histories[key], seed=getattr(cfg, "seed", 42))
        for key in metric_keys
    }


def build_optimizer(model, cfg):
    optimizer_name = str(getattr(cfg, "optimizer", "adamw")).lower()
    lr = getattr(cfg, "lr", 1e-4)
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, cfg):
    if not getattr(cfg, "use_scheduler", False):
        return None, None

    scheduler_type = str(getattr(cfg, "scheduler_type", "plateau")).lower()
    min_lr = getattr(cfg, "scheduler_min_lr", 1e-6)
    if scheduler_type == "cosine":
        warmup_epochs = max(0, int(getattr(cfg, "warmup_epochs", 0)))
        total_cosine_epochs = max(1, int(getattr(cfg, "epochs", 1)) - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_cosine_epochs,
            eta_min=min_lr
        )
        return cosine_scheduler, scheduler_type

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=getattr(cfg, "scheduler_factor", 0.85),
        patience=getattr(cfg, "scheduler_patience", 10),
        min_lr=min_lr,
        threshold=getattr(cfg, "scheduler_threshold", 1e-3),
        threshold_mode="abs",
        cooldown=getattr(cfg, "scheduler_cooldown", 0),
    )
    return plateau_scheduler, scheduler_type


def get_tf_ratio(epoch, max_epoch, start=1.0, end=0.0, decay_ratio=1.0):
    """计算teacher forcing比例，按设定阶段线性衰减后保持不变"""
    decay_epochs = max(1, int(max_epoch * decay_ratio))
    progress = min(epoch / decay_epochs, 1.0)
    ratio = start + (end - start) * progress
    return max(min(start, end), min(max(start, end), ratio))


def apply_two_stage_curriculum(criterion, cfg, epoch):
    if not getattr(cfg, "use_two_stage_curriculum", 0):
        return "single"

    total_epochs = max(1, int(getattr(cfg, "epochs", 1)))
    phase1_epochs = max(1, int(total_epochs * float(getattr(cfg, "two_stage_phase1_ratio", 0.2))))

    if epoch < phase1_epochs:
        criterion.insulin_type_loss_weight = float(getattr(cfg, "phase1_insulin_type_loss_weight", criterion.insulin_type_loss_weight))
        criterion.premix_recall_focus_loss_weight = float(getattr(cfg, "phase1_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0)))
        criterion.basic_sc_focus_loss_weight = float(getattr(cfg, "phase1_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0)))
        criterion.premix_sc_focus_loss_weight = float(getattr(cfg, "phase1_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0)))
        return "phase1"

    criterion.insulin_type_loss_weight = float(getattr(cfg, "phase2_insulin_type_loss_weight", criterion.insulin_type_loss_weight))
    criterion.premix_recall_focus_loss_weight = float(getattr(cfg, "phase2_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0)))
    criterion.basic_sc_focus_loss_weight = float(getattr(cfg, "phase2_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0)))
    criterion.premix_sc_focus_loss_weight = float(getattr(cfg, "phase2_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0)))
    return "phase2"


def create_sequence_mask(real_lengths, max_len, feature_dim, device):
    """
    根据real_lengths创建序列mask
    
    Args:
        real_lengths: [B] 每个样本的实际长度
        max_len: int 最大序列长度
        feature_dim: int 特征维度
        device: torch.device
        
    Returns:
        mask: [B, max_len, feature_dim] 序列mask
    """
    B = real_lengths.shape[0]
    mask = torch.zeros(B, max_len, feature_dim, device=device)
    
    for b in range(B):
        length = min(real_lengths[b].item(), max_len)
        mask[b, :length, :] = 1.0
    
    return mask


def ensure_mask_dimensions(mask, target_shape):
    """
    确保mask维度与target一致
    
    Args:
        mask: 原始mask，可能是[B, T]、[B, T, 1]或[B, T, D]
        target_shape: 目标形状 [B, T, D]
        
    Returns:
        mask: [B, T, D] 扩展后的mask
    """
    if mask.dim() == 2:
        # [B, T] -> [B, T, 1] -> [B, T, D]
        mask = mask.unsqueeze(-1).expand(target_shape)
    elif mask.dim() == 3 and mask.shape[-1] == 1:
        # [B, T, 1] -> [B, T, D]
        mask = mask.expand(target_shape)
    elif mask.dim() == 3 and mask.shape[-1] != target_shape[-1]:
        # 维度不匹配，报错
        raise ValueError(f"Mask shape {mask.shape} incompatible with target shape {target_shape}")
    
    return mask


def calculate_real_lengths_from_mask(bg_mask):
    """
    从bg_mask计算每个样本的实际长度
    
    Args:
        bg_mask: [B, T, d_bg] 血糖mask
        
    Returns:
        real_lengths: [B] 每个样本的实际长度
    """
    B = bg_mask.shape[0]
    device = bg_mask.device
    real_lengths = torch.zeros(B, dtype=torch.long, device=device)
    
    for b in range(B):
        valid_timesteps = (bg_mask[b].sum(dim=1) > 0).nonzero(as_tuple=False)
        if len(valid_timesteps) > 0:
            real_lengths[b] = valid_timesteps[-1].item() + 1
        else:
            real_lengths[b] = 1
    
    return real_lengths


def resolve_none_types_following_future(gt_type, none_idx=2):
    """
    仅用于统计：
    将 none 天优先并入后续第一个非 none 标签；若后面没有，则并入前一个非 none 标签。
    """
    resolved = gt_type.clone()
    if gt_type.dim() != 2:
        return resolved

    B, T = gt_type.shape
    for b in range(B):
        for t in range(T):
            if int(resolved[b, t].item()) != none_idx:
                continue

            replacement = none_idx
            for j in range(t + 1, T):
                candidate = int(gt_type[b, j].item())
                if candidate != none_idx:
                    replacement = candidate
                    break

            if replacement == none_idx:
                for j in range(t - 1, -1, -1):
                    candidate = int(gt_type[b, j].item())
                    if candidate != none_idx:
                        replacement = candidate
                        break

            resolved[b, t] = replacement

    return resolved


def build_merged_dose_mask(merged_type, dose_mask):
    """
    仅用于统计：
    none 天一旦被归并到 basic/premix，就按归并后的整个剂量范围参与 MAE 统计。

    这里不会改训练标签，只是给 merged MAE 构造一个“论文统计用”的有效位掩码。
    """
    merged_mask = dose_mask.clone()
    if merged_type.dim() != 2:
        return merged_mask

    none_positions = (dose_mask.sum(dim=-1) <= 2)
    if dose_mask.shape[-1] >= 14:
        basic_template = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=dose_mask.dtype,
            device=dose_mask.device,
        )
        premix_template = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=dose_mask.dtype,
            device=dose_mask.device,
        )
        none_template = torch.tensor(
            [0.0] * 12 + [1.0, 1.0],
            dtype=dose_mask.dtype,
            device=dose_mask.device,
        )
        templates = torch.stack([basic_template, premix_template, none_template], dim=0)
        replacement = templates[merged_type.clamp(min=0, max=2)]
        merged_mask = torch.where(none_positions.unsqueeze(-1), replacement, merged_mask)
    else:
        full_mask = torch.ones_like(dose_mask)
        merged_mask = torch.where(none_positions.unsqueeze(-1), full_mask, merged_mask)
    return merged_mask


def compute_masked_regression_metrics(pred, target, mask):
    """
    在给定 mask 上计算 MAE / RMSE / R2。
    """
    valid = mask > 0
    if not valid.any():
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0, "count": 0}

    pred_valid = pred[valid].float()
    target_valid = target[valid].float()
    err = pred_valid - target_valid
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    denom = torch.where(
        target_valid.abs() <= 1e-6,
        target_valid.abs() + 1.0,
        target_valid.abs(),
    )
    mape = (err.abs() / denom).mean().item() * 100.0

    if target_valid.numel() < 2:
        r2 = 0.0
    else:
        target_mean = target_valid.mean()
        ss_tot = ((target_valid - target_mean) ** 2).sum()
        ss_res = (err ** 2).sum()
        if ss_tot.item() <= 1e-12:
            r2 = 0.0
        else:
            r2 = (1.0 - ss_res / ss_tot).item()

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "count": int(valid.sum().item())}


def compute_daily_total_regression_metrics(pred, target, mask):
    valid = mask > 0
    if pred.dim() != 3 or target.dim() != 3 or mask.dim() != 3:
        return compute_masked_regression_metrics(pred, target, mask)

    valid_days = valid.any(dim=-1)
    if not valid_days.any():
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0, "count": 0}

    pred_daily = (pred.float() * valid.float()).sum(dim=-1)[valid_days]
    target_daily = (target.float() * valid.float()).sum(dim=-1)[valid_days]
    err = pred_daily - target_daily
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    denom = torch.where(
        target_daily.abs() <= 1e-6,
        target_daily.abs() + 1.0,
        target_daily.abs(),
    )
    mape = (err.abs() / denom).mean().item() * 100.0

    if target_daily.numel() < 2:
        r2 = 0.0
    else:
        target_mean = target_daily.mean()
        ss_tot = ((target_daily - target_mean) ** 2).sum()
        ss_res = (err ** 2).sum()
        r2 = 0.0 if ss_tot.item() <= 1e-12 else (1.0 - ss_res / ss_tot).item()

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "count": int(valid_days.sum().item())}


def compute_positive_event_metrics(pred, target, mask, positive_threshold=1e-3):
    """
    事件有效性统计：
    - Recall: 预测有值 / ground-truth 有值（只看 gt>threshold 的位置）
    - Precision: 在预测有值的位置中，有多少同时 gt 也有值
    - MAE / RMSE / R2: 只在 pred>threshold 且 gt>threshold 的交集位置统计
    """
    valid = mask > 0
    if not valid.any():
        return {"acc": 0.0, "precision": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "count": 0}

    pred_pos = (pred > positive_threshold) & valid
    gt_pos = (target > positive_threshold) & valid
    gt_pos_count = gt_pos.sum().item()
    pred_pos_count = pred_pos.sum().item()
    tp_mask = pred_pos & gt_pos
    acc = tp_mask.sum().item() / (gt_pos_count + 1e-8) if gt_pos_count > 0 else 0.0
    precision = tp_mask.sum().item() / (pred_pos_count + 1e-8) if pred_pos_count > 0 else 0.0

    reg_metrics = compute_masked_regression_metrics(pred, target, tp_mask.float())
    return {
        "acc": acc,
        "precision": precision,
        "mae": reg_metrics["mae"],
        "mape": reg_metrics["mape"],
        "rmse": reg_metrics["rmse"],
        "r2": reg_metrics["r2"],
        "count": reg_metrics["count"],
    }


def compute_threshold_event_metrics(pred, target, mask, pred_threshold, gt_threshold=0.0):
    """
    Event-style metrics with decoupled thresholds:
    - Recall / Precision use pred > pred_threshold vs gt > gt_threshold
    - Regression metrics only use the intersection mask
    """
    valid = mask > 0
    if not valid.any():
        return {"acc": 0.0, "precision": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "count": 0}

    pred_pos = (pred > pred_threshold) & valid
    gt_pos = (target > gt_threshold) & valid
    gt_pos_count = gt_pos.sum().item()
    pred_pos_count = pred_pos.sum().item()
    tp_mask = pred_pos & gt_pos

    acc = tp_mask.sum().item() / (gt_pos_count + 1e-8) if gt_pos_count > 0 else 0.0
    precision = tp_mask.sum().item() / (pred_pos_count + 1e-8) if pred_pos_count > 0 else 0.0
    reg_metrics = compute_masked_regression_metrics(pred, target, tp_mask.float())
    return {
        "acc": acc,
        "precision": precision,
        "mae": reg_metrics["mae"],
        "mape": reg_metrics["mape"],
        "rmse": reg_metrics["rmse"],
        "r2": reg_metrics["r2"],
        "count": reg_metrics["count"],
    }


def compute_mode_metrics(pred_group, gt_group, mask_group, pred_type, gt_type, class_idx):
    """
    Mode-style metrics:
    - Recall / Precision depend on whether the regimen mode is predicted correctly
    - Regression metrics only use days where the mode is matched
    - Once the mode is matched, all valid slots are included regardless of zero values
    """
    valid_days = (mask_group.sum(dim=-1) > 0)
    gt_days = valid_days & (gt_type == class_idx)
    pred_days = valid_days & (pred_type == class_idx)
    matched_days = valid_days & (gt_type == class_idx) & (pred_type == class_idx)

    gt_count = gt_days.sum().item()
    pred_count = pred_days.sum().item()
    match_count = matched_days.sum().item()

    recall = match_count / (gt_count + 1e-8) if gt_count > 0 else 0.0
    precision = match_count / (pred_count + 1e-8) if pred_count > 0 else 0.0

    matched_mask = mask_group * matched_days.unsqueeze(-1).float()
    reg_metrics = compute_masked_regression_metrics(pred_group, gt_group, matched_mask)
    return {
        "acc": recall,
        "precision": precision,
        "mae": reg_metrics["mae"],
        "mape": reg_metrics["mape"],
        "rmse": reg_metrics["rmse"],
        "r2": reg_metrics["r2"],
        "count": reg_metrics["count"],
    }


def compute_component_metrics(dose_pred, dose_gt, dose_mask, pred_type, gt_type, insulin_flag_dims, positive_threshold=1e-3):
    """
    当前 13 维胰岛素输出下的论文统计口径：
    - micro: 全天微泵
    - iv: 全天静脉
    - pump: 胰岛素泵相关位
    - basic_sc: 四针皮下相关位（仅 basic 天）
    - premix_sc: 预混皮下相关位（仅 premix 天）
    """
    metrics = {}
    if dose_pred.numel() == 0:
        return metrics

    if insulin_flag_dims == 0 and dose_pred.shape[-1] >= 6:
        group_defs = {
            "micro": {"dose_indices": [4]},
            "iv": {"dose_indices": [5]},
            "daily_insulin": {"dose_indices": [0, 1, 2, 3]},
        }
        for name, group_def in group_defs.items():
            pred_group = dose_pred[..., group_def["dose_indices"]]
            gt_group = dose_gt[..., group_def["dose_indices"]]
            mask_group = dose_mask[..., group_def["dose_indices"]]
            if name == "iv":
                group_metrics = compute_threshold_event_metrics(
                    pred_group, gt_group, mask_group, pred_threshold=1.0, gt_threshold=0.0
                )
            elif name == "micro":
                group_metrics = compute_threshold_event_metrics(
                    pred_group, gt_group, mask_group, pred_threshold=5.0, gt_threshold=0.0
                )
            else:
                group_metrics = compute_masked_regression_metrics(
                    pred_group, gt_group, mask_group.float()
                )
            if "acc" in group_metrics:
                metrics[f"{name}_acc"] = group_metrics["acc"]
            if "precision" in group_metrics:
                metrics[f"{name}_precision"] = group_metrics["precision"]
            metrics[f"{name}_mae"] = group_metrics["mae"]
            metrics[f"{name}_mape"] = group_metrics["mape"]
            metrics[f"{name}_rmse"] = group_metrics["rmse"]
            metrics[f"{name}_r2"] = group_metrics["r2"]
        return metrics

    # 17 维版本和 13 维版本的剂量槽位语义不同，这里按输出维度自动切换索引表，
    # 保证论文统计指标和当前胰岛素编码方式一致。
    if dose_pred.shape[-1] >= 14:
        group_defs = {
            "micro": {"dose_indices": [13], "type_filter": None},
            "iv": {"dose_indices": [12], "type_filter": None},
            "pump": {"dose_indices": [8, 9, 10, 11], "type_filter": None},
            "basic_sc": {"dose_indices": [0, 1, 2, 3], "type_filter": 0},
            "premix_sc": {"dose_indices": [4, 5, 6, 7], "type_filter": 1 if insulin_flag_dims >= 2 else None},
        }
    else:
        group_defs = {
            "micro": {"dose_indices": [9], "type_filter": None},
            "iv": {"dose_indices": [8], "type_filter": None},
            "pump": {"dose_indices": [1, 3, 5, 7], "type_filter": None},
            "basic_sc": {"dose_indices": [0, 2, 4, 6], "type_filter": 0},
            "premix_sc": {"dose_indices": [0, 2, 4, 6], "type_filter": 1 if insulin_flag_dims >= 2 else None},
        }

    for name, group_def in group_defs.items():
        dose_indices = group_def["dose_indices"]
        if max(dose_indices) >= dose_pred.shape[-1]:
            continue

        pred_group = dose_pred[..., dose_indices]
        gt_group = dose_gt[..., dose_indices]
        mask_group = dose_mask[..., dose_indices]

        if name == "iv":
            group_metrics = compute_threshold_event_metrics(
                pred_group, gt_group, mask_group, pred_threshold=1.0, gt_threshold=0.0
            )
        elif name == "micro":
            group_metrics = compute_threshold_event_metrics(
                pred_group, gt_group, mask_group, pred_threshold=5.0, gt_threshold=0.0
            )
        elif name == "pump":
            group_metrics = compute_mode_metrics(
                pred_group, gt_group, mask_group, pred_type=pred_type, gt_type=gt_type, class_idx=0
            )
        elif name == "basic_sc":
            group_metrics = compute_mode_metrics(
                pred_group, gt_group, mask_group, pred_type=pred_type, gt_type=gt_type, class_idx=0
            )
        elif name == "premix_sc":
            if insulin_flag_dims >= 2:
                group_metrics = compute_mode_metrics(
                    pred_group, gt_group, mask_group, pred_type=pred_type, gt_type=gt_type, class_idx=1
                )
            else:
                group_metrics = {"acc": 0.0, "precision": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "count": 0}
        else:
            type_filter = group_def["type_filter"]
            if type_filter is not None:
                group_days = (gt_type == type_filter).unsqueeze(-1).float()
                mask_group = mask_group * group_days
            group_metrics = compute_positive_event_metrics(
                pred_group,
                gt_group,
                mask_group,
                positive_threshold=positive_threshold,
            )

        metrics[f"{name}_acc"] = group_metrics["acc"]
        metrics[f"{name}_precision"] = group_metrics["precision"]
        metrics[f"{name}_mae"] = group_metrics["mae"]
        metrics[f"{name}_mape"] = group_metrics["mape"]
        metrics[f"{name}_rmse"] = group_metrics["rmse"]
        metrics[f"{name}_r2"] = group_metrics["r2"]

    return metrics


def compute_merged_group_metrics(dose_pred, dose_gt, dose_mask, pred_type, gt_type, insulin_flag_dims):
    """
    Merged basic/premix metrics:
    - resolve none days into nearby non-none regimen for both gt and prediction
    - use merged dose mask over the full merged dose slots
    - recall/precision are mode-style over merged regimens
    - regression metrics only use days where merged mode matches
    """
    metrics = {}
    if dose_pred.numel() == 0 or insulin_flag_dims < 3:
        return metrics

    merged_gt_type = resolve_none_types_following_future(gt_type, none_idx=2)
    merged_pred_type = resolve_none_types_following_future(pred_type, none_idx=2)
    merged_dose_mask = build_merged_dose_mask(merged_gt_type, dose_mask)

    for class_idx, class_name in enumerate(["basic", "premix"]):
        group_metrics = compute_mode_metrics(
            dose_pred,
            dose_gt,
            merged_dose_mask,
            pred_type=merged_pred_type,
            gt_type=merged_gt_type,
            class_idx=class_idx,
        )
        metrics[f"{class_name}_merged_acc"] = group_metrics["acc"]
        metrics[f"{class_name}_merged_precision"] = group_metrics["precision"]
        metrics[f"{class_name}_merged_mae"] = group_metrics["mae"]
        metrics[f"{class_name}_merged_rmse"] = group_metrics["rmse"]
        metrics[f"{class_name}_merged_r2"] = group_metrics["r2"]

    return metrics


def build_train_sampler(dataset, cfg):
    """
    仅改变训练采样分布：
    对包含 premix 天的序列提高采样权重，不改变标签和 loss 定义。
    """
    if not bool(getattr(cfg, "use_premix_oversample", 0)):
        return None

    insulin_flag_dims = int(getattr(cfg, "insulin_flag_dims", 3))
    if insulin_flag_dims < 2:
        return None

    premix_weight = float(getattr(cfg, "premix_oversample_weight", 1.0))
    if premix_weight <= 1.0:
        return None

    sample_weights = []
    premix_sequence_count = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        insulin2 = sample.get("insulin1", None)
        insulin2_mask = sample.get("insulin1_mask", None)
        has_premix = False
        if insulin2 is not None and insulin2_mask is not None:
            premix_flag = insulin2[:, 1]
            premix_mask = insulin2_mask[:, 1]
            has_premix = bool(((premix_flag > 0.5) & (premix_mask > 0)).any().item())

        if has_premix:
            sample_weights.append(premix_weight)
            premix_sequence_count += 1
        else:
            sample_weights.append(1.0)

    if premix_sequence_count == 0:
        log_message("Premix oversampling skipped: no premix sequence found in training split.")
        return None

    log_message(
        f"Premix oversampling enabled: {premix_sequence_count}/{len(dataset)} sequences contain premix, "
        f"oversample_weight={premix_weight:.2f}"
    )
    weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


def build_train_sampler_v2(dataset, cfg):
    """
    Adjust only the training sampling distribution.
    Oversample sequences with premix days and/or high-BG targets without changing labels or loss definitions.
    """
    use_premix_oversample = bool(getattr(cfg, "use_premix_oversample", 0))
    use_high_bg_oversample = bool(getattr(cfg, "use_high_bg_oversample", 0))
    if not use_premix_oversample and not use_high_bg_oversample:
        return None

    insulin_flag_dims = int(getattr(cfg, "insulin_flag_dims", 3))
    if use_premix_oversample and insulin_flag_dims < 2:
        return None

    premix_weight = float(getattr(cfg, "premix_oversample_weight", 1.0))
    high_bg_weight = float(getattr(cfg, "high_bg_oversample_weight", 1.0))
    high_bg_threshold = float(getattr(cfg, "high_bg_threshold", 10.0))

    sample_weights = []
    premix_sequence_count = 0
    high_bg_sequence_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        insulin = sample.get("insulin1", None)
        insulin_mask = sample.get("insulin1_mask", None)
        bg = sample.get("bg1", None)
        bg_mask = sample.get("bg1_mask", None)

        has_premix = False
        if use_premix_oversample and premix_weight > 1.0 and insulin is not None and insulin_mask is not None:
            premix_flag = insulin[:, 1]
            premix_mask = insulin_mask[:, 1]
            has_premix = bool(((premix_flag > 0.5) & (premix_mask > 0)).any().item())

        has_high_bg = False
        if use_high_bg_oversample and high_bg_weight > 1.0 and bg is not None and bg_mask is not None:
            has_high_bg = bool(((bg >= high_bg_threshold) & (bg_mask > 0)).any().item())

        weight = 1.0
        if has_premix:
            weight *= premix_weight
            premix_sequence_count += 1
        if has_high_bg:
            weight *= high_bg_weight
            high_bg_sequence_count += 1
        sample_weights.append(weight)

    if max(sample_weights) <= 1.0:
        log_message("Sequence oversampling skipped: no eligible premix/high-bg sequence found in training split.")
        return None

    summary_parts = []
    if use_premix_oversample and premix_weight > 1.0:
        summary_parts.append(f"premix={premix_sequence_count}/{len(dataset)} weight={premix_weight:.2f}")
    if use_high_bg_oversample and high_bg_weight > 1.0:
        summary_parts.append(
            f"high_bg={high_bg_sequence_count}/{len(dataset)} weight={high_bg_weight:.2f} threshold={high_bg_threshold:.1f}"
        )
    log_message("Sequence oversampling enabled: " + ", ".join(summary_parts))
    weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


def maybe_augment_premix_batch(batch, cfg):
    """
    仅在训练阶段对 premix 天做轻量增强：
    - 只改 insulin1 的剂量位
    - 只改 premix 天
    - 只改非零且有效的剂量位
    - 不改标签类型、不改 mask、不改 bg 真值
    """
    if not bool(getattr(cfg, "use_premix_augmentation", 0)):
        return batch

    insulin = batch.get("insulin1", None)
    insulin_mask = batch.get("insulin1_mask", None)
    if insulin is None or insulin_mask is None:
        return batch

    insulin_flag_dims = int(getattr(cfg, "insulin_flag_dims", 3))
    if insulin.shape[-1] <= insulin_flag_dims or insulin_flag_dims < 2:
        return batch

    aug_prob = float(getattr(cfg, "premix_aug_prob", 0.3))
    aug_scale = float(getattr(cfg, "premix_aug_scale", 0.05))
    if aug_prob <= 0.0 or aug_scale <= 0.0:
        return batch

    # 这里只改训练输入，不改 ground-truth 标签和 mask。
    insulin_aug = insulin.clone()
    dose = insulin_aug[..., insulin_flag_dims:]
    dose_mask = insulin_mask[..., insulin_flag_dims:]

    premix_days = (insulin[..., 1] > 0.5) & (insulin_mask[..., 1] > 0)
    if not premix_days.any():
        batch["insulin1"] = insulin_aug
        return batch

    valid_nonzero = premix_days.unsqueeze(-1) & (dose_mask > 0) & (dose.abs() > 1e-8)
    if not valid_nonzero.any():
        batch["insulin1"] = insulin_aug
        return batch

    day_apply = (torch.rand_like(premix_days.float()) < aug_prob) & premix_days
    apply_mask = day_apply.unsqueeze(-1) & valid_nonzero
    if not apply_mask.any():
        batch["insulin1"] = insulin_aug
        return batch

    noise = (torch.rand_like(dose) * 2.0 - 1.0) * aug_scale
    scale = 1.0 + noise
    augmented_dose = torch.clamp(dose * scale, min=0.0)
    dose = torch.where(apply_mask, augmented_dose, dose)
    insulin_aug[..., insulin_flag_dims:] = dose
    batch["insulin1"] = insulin_aug
    return batch


def compute_batch_loss(batch, outputs, criterion, cfg, mode="train"):
    """
    统一的loss计算函数（训练和验证共用）
    
    Args:
        batch: 数据batch
        outputs: 模型输出
        criterion: loss函数
        cfg: 配置
        mode: "train" or "val"
        
    Returns:
        total_loss: 总loss
        loss_dict: loss详细信息字典
    """
    device = batch["bg1"].device
    
    # 提取预测和真实值
    insulin_pred = outputs[0]
    bg_pred = outputs[1]
    # bread_insulin = outputs[2]
    
    insulin_gt = batch["insulin1"]
    insulin_gt_mask = batch["insulin1_mask"]
    bg_gt = batch["bg1"]
    bg_mask = batch["bg1_mask"]
    
    # 统一截断到最小长度
    min_T = min(insulin_pred.shape[1], insulin_gt.shape[1], bg_pred.shape[1], bg_gt.shape[1])
    
    insulin_pred = insulin_pred[:, :min_T, :]
    insulin_gt = insulin_gt[:, :min_T, :]
    insulin_gt_mask = insulin_gt_mask[:, :min_T, :] if insulin_gt_mask.shape[1] >= min_T else insulin_gt_mask
    bg_pred = bg_pred[:, :min_T, :]
    bg_gt = bg_gt[:, :min_T, :]
    bg_mask = bg_mask[:, :min_T, :] if bg_mask.shape[1] >= min_T else bg_mask
    
    # 确保bg_mask维度正确
    bg_mask = ensure_mask_dimensions(bg_mask, bg_gt.shape)
    
    # 计算real_lengths
    real_lengths = calculate_real_lengths_from_mask(bg_mask)
    
    # 创建正确的insulin_mask（基于real_lengths和特征有效位）
    B, T, D_insulin = insulin_pred.shape
    insulin_seq_mask = create_sequence_mask(real_lengths, T, D_insulin, device)
    insulin_gt_mask = ensure_mask_dimensions(insulin_gt_mask, insulin_gt.shape)
    insulin_mask = insulin_seq_mask * insulin_gt_mask

    # 计算胰岛素损失
    insulin_loss, insulin_loss_dict = criterion(
        pred=insulin_pred,
        target=insulin_gt,
        person_features=batch["person_value"],
        hidden_states=None,
        mask=insulin_mask,
        task_name="insulin"
    )
    
    # 计算血糖损失
    bg_loss, bg_loss_dict = criterion(
        pred=bg_pred,
        target=bg_gt,
        person_features=batch["person_value"],
        hidden_states=None,
        mask=bg_mask,
        task_name="bg"
    )
    
    # 组合loss字典
    loss_dict = {
        'insulin_' + k: v for k, v in insulin_loss_dict.items()
    }
    loss_dict.update({
        'bg_' + k: v for k, v in bg_loss_dict.items()
    })

    bg_metrics = compute_masked_regression_metrics(bg_pred, bg_gt, bg_mask.float())
    loss_dict['bg_mae'] = bg_metrics['mae']
    loss_dict['bg_rmse'] = bg_metrics['rmse']
    loss_dict['bg_r2'] = bg_metrics['r2']

    insulin_flag_dims = int(getattr(cfg, "insulin_flag_dims", 3))
    if insulin_flag_dims > 0 and insulin_pred.shape[-1] >= insulin_flag_dims and insulin_gt.shape[-1] >= insulin_flag_dims:
        pred_type = torch.argmax(insulin_pred[..., :insulin_flag_dims], dim=-1)
        gt_type = torch.argmax(insulin_gt[..., :insulin_flag_dims], dim=-1)
        valid_type = (insulin_mask[..., :insulin_flag_dims].sum(dim=-1) > 0)
        dose_pred = insulin_pred[..., insulin_flag_dims:]
        dose_gt = insulin_gt[..., insulin_flag_dims:]
        dose_mask = insulin_mask[..., insulin_flag_dims:]

        if valid_type.any():
            correct = (pred_type == gt_type) & valid_type
            loss_dict['insulin_type_acc'] = correct.float().sum().item() / (valid_type.float().sum().item() + 1e-8)
            class_names = ["basic", "premix", "none"]
            for class_idx, class_name in enumerate(class_names[:insulin_flag_dims]):
                class_valid = valid_type & (gt_type == class_idx)
                if class_valid.any():
                    class_correct = (((pred_type == gt_type) & class_valid).float().sum().item())
                    class_total = class_valid.float().sum().item()
                    loss_dict[f'insulin_type_acc_{class_name}'] = class_correct / (class_total + 1e-8)
                else:
                    loss_dict[f'insulin_type_acc_{class_name}'] = 0.0
    else:
        pred_type = None
        gt_type = None
        dose_pred = insulin_pred
        dose_gt = insulin_gt
        dose_mask = insulin_mask

    dose_total_trend_penalty = torch.zeros((), device=device)
    if dose_pred.numel() > 0:
        dose_abs_err = (dose_pred - dose_gt).abs()
        valid_dose = dose_mask > 0
        if valid_dose.any():
            loss_dict['insulin_dose_mae'] = (dose_abs_err * dose_mask).sum().item() / (dose_mask.sum().item() + 1e-8)
        else:
            loss_dict['insulin_dose_mae'] = 0.0
        overall_dose_metrics = compute_daily_total_regression_metrics(dose_pred, dose_gt, dose_mask.float())
        loss_dict['insulin_dose_mape'] = overall_dose_metrics['mape']
        loss_dict['insulin_dose_rmse'] = overall_dose_metrics['rmse']
        loss_dict['insulin_dose_r2'] = overall_dose_metrics['r2']
        loss_dict['bg_insulin_mae_sum'] = loss_dict['insulin_dose_mae'] + loss_dict.get('bg_mae', 0.0)

        if insulin_flag_dims > 0:
            class_names = ["basic", "premix", "none"]
            for class_idx, class_name in enumerate(class_names[:insulin_flag_dims]):
                class_valid = (gt_type == class_idx).unsqueeze(-1) & valid_dose
                class_mask = class_valid.float() * dose_mask
                class_mask_sum = class_mask.sum().item()
                if class_mask_sum > 0:
                    class_mae = (dose_abs_err * class_mask).sum().item() / (class_mask_sum + 1e-8)
                else:
                    class_mae = 0.0
                loss_dict[f'insulin_dose_mae_{class_name}'] = class_mae

        if insulin_flag_dims >= 3:
            merged_type = resolve_none_types_following_future(gt_type, none_idx=2)
            merged_dose_mask = build_merged_dose_mask(merged_type, dose_mask)
            merged_valid_dose = merged_dose_mask > 0
            original_none = (gt_type == 2)
            loss_dict['none_to_basic_count'] = (original_none & (merged_type == 0)).float().sum().item()
            loss_dict['none_to_premix_count'] = (original_none & (merged_type == 1)).float().sum().item()
            merged_class_names = ["basic", "premix"]
            for class_idx, class_name in enumerate(merged_class_names):
                class_valid = (merged_type == class_idx).unsqueeze(-1) & merged_valid_dose
                class_mask = class_valid.float() * merged_dose_mask
                class_mask_sum = class_mask.sum().item()
                if class_mask_sum > 0:
                    class_mae = (dose_abs_err * class_mask).sum().item() / (class_mask_sum + 1e-8)
                else:
                    class_mae = 0.0
                loss_dict[f'insulin_dose_mae_{class_name}_merged'] = class_mae
            loss_dict['merged_bg_mae_sum'] = (
                loss_dict.get('insulin_dose_mae_basic_merged', 0.0)
                + loss_dict.get('insulin_dose_mae_premix_merged', 0.0)
                + loss_dict.get('bg_mae', 0.0)
            )

        component_metrics = compute_component_metrics(
            dose_pred=dose_pred,
            dose_gt=dose_gt,
            dose_mask=dose_mask,
            pred_type=pred_type,
            gt_type=gt_type,
            insulin_flag_dims=insulin_flag_dims,
            positive_threshold=float(getattr(cfg, "metric_positive_threshold", 1e-3)),
        )
        loss_dict.update(component_metrics)
        loss_dict['daily_insulin_mape'] = overall_dose_metrics['mape']
        merged_group_metrics = compute_merged_group_metrics(
            dose_pred=dose_pred,
            dose_gt=dose_gt,
            dose_mask=dose_mask,
            pred_type=pred_type,
            gt_type=gt_type,
            insulin_flag_dims=insulin_flag_dims,
        )
        loss_dict.update(merged_group_metrics)

        bg_down_insulin_rise_penalty_weight = float(getattr(cfg, "bg_down_insulin_rise_penalty_weight", 0.0))
        if bg_down_insulin_rise_penalty_weight > 0 and min_T >= 2:
            bg_mask_float = bg_mask.float()
            dose_mask_float = dose_mask.float()
            bg_day_valid = (bg_mask_float.sum(dim=-1) > 0).float()
            dose_day_valid = (dose_mask_float.sum(dim=-1) > 0).float()
            valid_pair = (
                (bg_day_valid[:, 1:] > 0)
                & (bg_day_valid[:, :-1] > 0)
                & (dose_day_valid[:, 1:] > 0)
                & (dose_day_valid[:, :-1] > 0)
            ).float()

            if valid_pair.sum().item() > 0:
                bg_day_gt = (bg_gt * bg_mask_float).sum(dim=-1) / bg_mask_float.sum(dim=-1).clamp_min(1.0)
                dose_total_pred = (dose_pred * dose_mask_float).sum(dim=-1)
                dose_total_gt = (dose_gt * dose_mask_float).sum(dim=-1)

                bg_down_threshold = float(getattr(cfg, "bg_down_threshold", 0.2))
                target_insulin_nonincrease_margin = float(getattr(cfg, "target_insulin_nonincrease_margin", 0.5))
                predicted_insulin_rise_margin = float(getattr(cfg, "predicted_insulin_rise_margin", 0.5))

                target_bg_delta = bg_day_gt[:, 1:] - bg_day_gt[:, :-1]
                target_insulin_total_delta = dose_total_gt[:, 1:] - dose_total_gt[:, :-1]
                pred_insulin_total_delta = dose_total_pred[:, 1:] - dose_total_pred[:, :-1]

                bg_down_mask = (target_bg_delta <= -bg_down_threshold).float()
                insulin_should_not_rise_mask = (target_insulin_total_delta <= target_insulin_nonincrease_margin).float()
                trigger_mask = valid_pair * bg_down_mask * insulin_should_not_rise_mask

                if trigger_mask.sum().item() > 0:
                    pred_extra_rise = F.relu(pred_insulin_total_delta - predicted_insulin_rise_margin)
                    dose_total_trend_penalty = (pred_extra_rise * trigger_mask).sum() / (trigger_mask.sum() + 1e-8)
                    loss_dict['insulin_bg_down_rise_penalty'] = dose_total_trend_penalty.item()
    
    # 计算总loss
    total_loss = cfg.lambda_insulin * insulin_loss + cfg.lambda_bg * bg_loss
    if float(getattr(cfg, "bg_down_insulin_rise_penalty_weight", 0.0)) > 0:
        total_loss = total_loss + float(getattr(cfg, "bg_down_insulin_rise_penalty_weight", 0.0)) * dose_total_trend_penalty
    
    # # 计算出院带药损失
    # bread_insulin_gt = batch.get("bread_insulin_gt", None)
    # if bread_insulin_gt is not None:
    #     bread_loss, bread_loss_dict = criterion(
    #         pred=bread_insulin,
    #         target=bread_insulin_gt,
    #         person_features=batch["person_value"],
    #         hidden_states=None,
    #         mask=None
    #     )
    #     total_loss = total_loss + cfg.lambda_discharge * bread_loss
    #     loss_dict.update({
    #         'bread_' + k: v for k, v in bread_loss_dict.items()
    #     })
    
    # # 计算长度预测损失（如果启用）
    # if len(outputs) > 3 and getattr(cfg, 'use_length_predictor', False):
    #     predicted_length = outputs[3]  # [B]
    #     length_loss = torch.nn.functional.l1_loss(predicted_length, real_lengths.float())
    #     total_loss = total_loss + getattr(cfg, 'lambda_length', 0.1) * length_loss
    #     loss_dict['length_loss'] = length_loss.item()
    
    loss_dict['total'] = total_loss.item()
    loss_dict['insulin_pred_min'] = insulin_pred.detach().min().item()
    loss_dict['insulin_pred_max'] = insulin_pred.detach().max().item()
    loss_dict['insulin_gt_min'] = insulin_gt.detach().min().item()
    loss_dict['insulin_gt_max'] = insulin_gt.detach().max().item()
    loss_dict['bg_pred_min'] = bg_pred.detach().min().item()
    loss_dict['bg_pred_max'] = bg_pred.detach().max().item()
    loss_dict['bg_gt_min'] = bg_gt.detach().min().item()
    loss_dict['bg_gt_max'] = bg_gt.detach().max().item()
    loss_dict['insulin_mask_sum'] = insulin_mask.detach().sum().item()
    loss_dict['bg_mask_sum'] = bg_mask.detach().sum().item()
    
    return total_loss, loss_dict
    
def train_val_test(cfg):
    run_tag = f"topk{int(getattr(cfg, 'top_k', 1))}_{time.strftime('%Y%m%d-%H%M%S')}"
    save_best_path = os.path.join(cfg.save_path, run_tag)
    
    if not os.path.exists(save_best_path): 
        os.makedirs(save_best_path)
    
    setup_logging(os.path.join(save_best_path, 'output.log'), console_output=True)
    set_seed(cfg.seed)
    device = cfg.device
    writer = None
    global_step = 0
    if getattr(cfg, 'use_tensorboard', 1):
        train_dir = os.path.dirname(os.path.abspath(__file__))
        tensorboard_path = os.path.normpath(
            os.path.join(train_dir, "..", "tf-logs", run_tag)
        )
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_path)
        writer.add_text("run/config", f"top_k={int(getattr(cfg, 'top_k', 1))}", 0)
        writer.add_scalar("config/top_k", float(getattr(cfg, 'top_k', 1)), 0)
        log_message(f"TensorBoard logging enabled: {tensorboard_path}")
    log_message(f"Run tag: {run_tag}")
    log_message(f"Effective retrieval top_k: {int(getattr(cfg, 'top_k', 1))}")
    
    # Datasets
    train_dataset = DiabetesDataset(cfg, mode="train")
    val_dataset = DiabetesDataset(cfg, mode="val")
    
    num_workers = max(0, int(getattr(cfg, "num_workers", 0)))
    pin_memory = bool(getattr(cfg, "pin_memory", 0)) and str(device).startswith("cuda")
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_sampler = build_train_sampler_v2(train_dataset, cfg)
    if train_sampler is not None:
        train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    cfg.d_insulin = train_dataset[0]["d_insulin"]
    # cfg.d_insulin_route_stage = train_dataset[0]["d_insulin_route_stage"]
    cfg.d_person = train_dataset[0]["d_per1"]
    cfg.d_drug = train_dataset[0]["d_drug"]
    cfg.save_best_path = save_best_path
    if cfg.cut_time != 0: 
        cfg.max_T1 = cfg.cut_time
    
    # Model
    model = TwoStageModel(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    
    # Learning rate scheduler
    scheduler, scheduler_type = build_scheduler(optimizer, cfg)
    if scheduler is not None:
        if scheduler_type == "plateau":
            log_message(
                "Learning rate scheduler enabled: "
                f"type=plateau, factor={getattr(cfg, 'scheduler_factor', 0.85)}, "
                f"patience={getattr(cfg, 'scheduler_patience', 10)}, "
                f"threshold={getattr(cfg, 'scheduler_threshold', 1e-3)}, "
                f"min_lr={getattr(cfg, 'scheduler_min_lr', 1e-6)}"
            )
        else:
            log_message(
                "Learning rate scheduler enabled: "
                f"type=cosine, warmup_epochs={getattr(cfg, 'warmup_epochs', 0)}, "
                f"min_lr={getattr(cfg, 'scheduler_min_lr', 1e-6)}"
            )
    
    # Loss Function
    criterion = CombinedLoss(cfg)
    
    best_val_loss = float('inf')
    best_monitor_value = float('inf')
    best_es_val_loss = float('inf')
    best_es_monitor_value = float('inf')
    best_epoch = -1
    best_metrics = {}
    early_stop_counter = 0
    # This experiment series is intended to run full-length; hard-disable early stopping.
    use_early_stop = False
    early_stop_patience = getattr(cfg, 'early_stop_patience', 30)
    early_stop_min_delta = getattr(cfg, 'early_stop_min_delta', 1e-3)
    early_stop_start_epoch = int(getattr(cfg, 'early_stop_start_epoch', 0))
    min_lr_counter = 0
    min_lr = float(getattr(cfg, "scheduler_min_lr", 0.0))

    log_message("Early stopping disabled: training will continue to max epochs unless interrupted.")
    monitor_metric_name = "bg_daily_insulin_mae_sum"
    log_message(f"Best checkpoint / plateau monitor metric: {monitor_metric_name}")
    
    for epoch in range(cfg.epochs):
        curriculum_phase = apply_two_stage_curriculum(criterion, cfg, epoch)
        if scheduler is not None and scheduler_type == "cosine":
            warmup_epochs = max(0, int(getattr(cfg, "warmup_epochs", 0)))
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_scale = float(epoch + 1) / float(warmup_epochs)
                target_lr = max(min_lr, getattr(cfg, "lr", 1e-4) * warmup_scale)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = target_lr

        # ==================== Training ====================
        model.train()
        total_loss = 0.0
        tf_ratio = get_tf_ratio(
            epoch,
            cfg.epochs,
            start=getattr(cfg, 'tf_start', getattr(cfg, 'tf_ratio', 0.9)),
            end=getattr(cfg, 'tf_end', 0.3),
            decay_ratio=getattr(cfg, 'tf_decay_ratio', 0.7)
        )
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch = maybe_augment_premix_batch(batch, cfg)
            
            # Forward
            outputs = model(batch, tf_ratio=tf_ratio, mode="train")
            
            # Calculate Loss (using unified function)
            loss, loss_dict = compute_batch_loss(batch, outputs, criterion, cfg, mode="train")
            
            # Backward
            loss.backward()
            
            # # Gradient clipping
            # if hasattr(cfg, 'max_grad_norm') and cfg.max_grad_norm > 0:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            #     if batch_idx % 10 == 0:
            #         loss_dict['grad_norm'] = grad_norm.item()
            # 将梯度中 NaN / Inf 替换为 0
            pre_clip_sq_norm = 0.0
            nonfinite_grad_params = 0
            zero_grad_params = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad = p.grad.detach()
                    if not torch.isfinite(grad).all():
                        nonfinite_grad_params += 1
                    param_grad_norm = grad.float().norm(2).item()
                    pre_clip_sq_norm += param_grad_norm * param_grad_norm
                    if grad.abs().max().item() == 0.0:
                        zero_grad_params += 1
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)

            grad_norm = None
            if cfg.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if batch_idx % 10 == 0:
                    if grad_norm is not None:
                        loss_dict['grad_norm'] = grad_norm.item()
                    loss_dict['grad_norm_preclip'] = pre_clip_sq_norm ** 0.5
                    loss_dict['nonfinite_grad_params'] = nonfinite_grad_params
                    loss_dict['zero_grad_params'] = zero_grad_params

            if nonfinite_grad_params > 0:
                if batch_idx % 10 == 0:
                    log_message(f"WARNING: Skipping optimizer step due to non-finite gradients in {nonfinite_grad_params} parameters")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            if writer is not None:
                writer.add_scalar("train/batch_total_loss", loss.item(), global_step)
                if 'insulin_diversity_base' in loss_dict:
                    writer.add_scalar("train/batch_insulin_loss", loss_dict['insulin_diversity_base'], global_step)
                if 'insulin_type' in loss_dict:
                    writer.add_scalar("train/batch_insulin_type_loss", loss_dict['insulin_type'], global_step)
                if 'insulin_type_acc' in loss_dict:
                    writer.add_scalar("train/batch_insulin_type_acc", loss_dict['insulin_type_acc'], global_step)
                if 'insulin_type_acc_basic' in loss_dict:
                    writer.add_scalar("train/batch_insulin_type_acc_basic", loss_dict['insulin_type_acc_basic'], global_step)
                if 'insulin_type_acc_premix' in loss_dict:
                    writer.add_scalar("train/batch_insulin_type_acc_premix", loss_dict['insulin_type_acc_premix'], global_step)
                if 'bg_diversity_base' in loss_dict:
                    writer.add_scalar("train/batch_bg_loss", loss_dict['bg_diversity_base'], global_step)
                if 'grad_norm' in loss_dict:
                    writer.add_scalar("train/batch_grad_norm", loss_dict['grad_norm'], global_step)
                if 'grad_norm_preclip' in loss_dict:
                    writer.add_scalar("train/batch_preclip_grad_norm", loss_dict['grad_norm_preclip'], global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.add_scalar("train/tf_ratio", tf_ratio, global_step)
                writer.add_scalar("train/nonfinite_grad_params", nonfinite_grad_params, global_step)
                writer.add_scalar("train/zero_grad_params", zero_grad_params, global_step)
            global_step += 1
            
            # Logging
            if batch_idx % 10 == 0:
                loss_str = f"Epoch {epoch}, Batch {batch_idx}, Total Loss: {loss.item():.4f}"
                if 'insulin_diversity_base' in loss_dict:
                    loss_str += f", Insulin: {loss_dict['insulin_diversity_base']:.4f}"
                if 'insulin_type' in loss_dict:
                    loss_str += f", InsType: {loss_dict['insulin_type']:.4f}"
                if 'insulin_type_acc' in loss_dict:
                    loss_str += f", InsTypeAcc: {loss_dict['insulin_type_acc']:.4f}"
                if 'bg_diversity_base' in loss_dict:
                    loss_str += f", BG: {loss_dict['bg_diversity_base']:.4f}"
                if 'bread_diversity_base' in loss_dict:
                    loss_str += f", Discharge: {loss_dict['bread_diversity_base']:.4f}"
                if 'grad_norm' in loss_dict:
                    loss_str += f", GradNorm: {loss_dict['grad_norm']:.4f}"
                if 'grad_norm_preclip' in loss_dict:
                    loss_str += f", PreClipGradNorm: {loss_dict['grad_norm_preclip']:.4f}"
                if 'nonfinite_grad_params' in loss_dict:
                    loss_str += f", NonFiniteGradParams: {loss_dict['nonfinite_grad_params']}"
                if 'zero_grad_params' in loss_dict:
                    loss_str += f", ZeroGradParams: {loss_dict['zero_grad_params']}"
                log_message(loss_str)
                log_message(
                    f"  Range BG pred[{loss_dict['bg_pred_min']:.3f}, {loss_dict['bg_pred_max']:.3f}] "
                    f"gt[{loss_dict['bg_gt_min']:.3f}, {loss_dict['bg_gt_max']:.3f}] "
                    f"mask_sum={loss_dict['bg_mask_sum']:.1f}"
                )
                log_message(
                    f"  Range Insulin pred[{loss_dict['insulin_pred_min']:.3f}, {loss_dict['insulin_pred_max']:.3f}] "
                    f"gt[{loss_dict['insulin_gt_min']:.3f}, {loss_dict['insulin_gt_max']:.3f}] "
                    f"mask_sum={loss_dict['insulin_mask_sum']:.1f}"
                )
                
                # Warning for abnormal loss
                if loss.item() > 100:
                    log_message(f"WARNING: Loss too large ({loss.item():.2f}), possible numerical instability")
        
        avg_train_loss = total_loss / len(train_loader)
        if writer is not None:
            writer.add_scalar("train/epoch_total_loss", avg_train_loss, epoch)
            writer.add_scalar("train/epoch_tf_ratio", tf_ratio, epoch)
            writer.add_scalar("train/epoch_insulin_type_loss_weight", criterion.insulin_type_loss_weight, epoch)
        log_message(f"\n{'='*60}")
        log_message(f"Epoch {epoch} Training Completed")
        log_message(f"Average Training Loss: {avg_train_loss:.4f}")
        log_message(f"Curriculum Phase: {curriculum_phase}")
        log_message(f"Current Insulin Type Loss Weight: {criterion.insulin_type_loss_weight:.4f}")
        log_message(f"{'='*60}\n")
        
        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        val_loss_components = {
            'insulin': 0.0,
            'insulin_dose_mae': 0.0,
            'insulin_dose_mape': 0.0,
            'insulin_dose_rmse': 0.0,
            'insulin_dose_r2': 0.0,
            'bg_insulin_mae_sum': 0.0,
            'bg_mae': 0.0,
            'bg_rmse': 0.0,
            'bg_r2': 0.0,
            'daily_insulin_acc': 0.0,
            'daily_insulin_precision': 0.0,
            'daily_insulin_mae': 0.0,
            'daily_insulin_mape': 0.0,
            'daily_insulin_rmse': 0.0,
            'daily_insulin_r2': 0.0,
            'micro_acc': 0.0,
            'micro_precision': 0.0,
            'micro_mae': 0.0,
            'micro_mape': 0.0,
            'micro_rmse': 0.0,
            'micro_r2': 0.0,
            'iv_acc': 0.0,
            'iv_precision': 0.0,
            'iv_mae': 0.0,
            'iv_mape': 0.0,
            'iv_rmse': 0.0,
            'iv_r2': 0.0,
            'bg': 0.0,
            'bread': 0.0,
            'length': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(batch, tf_ratio=0.0, mode="val")
                
                # Calculate validation loss (using unified function)
                batch_loss, loss_dict = compute_batch_loss(batch, outputs, criterion, cfg, mode="val")
                
                val_loss += batch_loss.item()
                
                # Accumulate loss components for reporting
                if 'insulin_diversity_base' in loss_dict:
                    val_loss_components['insulin'] += loss_dict['insulin_diversity_base']
                if 'insulin_dose_mae' in loss_dict:
                    val_loss_components['insulin_dose_mae'] += loss_dict['insulin_dose_mae']
                for metric_name in [
                    'insulin_dose_mape', 'insulin_dose_rmse', 'insulin_dose_r2',
                    'bg_insulin_mae_sum',
                    'bg_mae', 'bg_rmse', 'bg_r2',
                    'daily_insulin_acc', 'daily_insulin_precision', 'daily_insulin_mae', 'daily_insulin_mape', 'daily_insulin_rmse', 'daily_insulin_r2',
                    'micro_acc', 'micro_precision', 'micro_mae', 'micro_mape', 'micro_rmse', 'micro_r2',
                    'iv_acc', 'iv_precision', 'iv_mae', 'iv_mape', 'iv_rmse', 'iv_r2',
                ]:
                    if metric_name in loss_dict:
                        val_loss_components[metric_name] += loss_dict[metric_name]
                if 'bg_diversity_base' in loss_dict:
                    val_loss_components['bg'] += loss_dict['bg_diversity_base']
                if 'bread_diversity_base' in loss_dict:
                    val_loss_components['bread'] += loss_dict['bread_diversity_base']
                if 'length_loss' in loss_dict:
                    val_loss_components['length'] += loss_dict['length_loss']
        
        avg_val_loss = val_loss / len(val_loader)
        # Print validation results
        log_message(f"\n{'='*60}")
        log_message(f"Epoch {epoch} Validation Completed")
        log_message(f"Average Validation Loss: {avg_val_loss:.4f}")
        log_message(f"  - Insulin Loss: {val_loss_components['insulin'] / len(val_loader):.4f}")
        log_message(f"  - Insulin Dose MAE/MAPE/RMSE/R2: {val_loss_components['insulin_dose_mae'] / len(val_loader):.4f} / {val_loss_components['insulin_dose_mape'] / len(val_loader):.2f}% / {val_loss_components['insulin_dose_rmse'] / len(val_loader):.4f} / {val_loss_components['insulin_dose_r2'] / len(val_loader):.4f}")
        log_message(f"  - BG+Insulin MAE Sum: {val_loss_components['bg_insulin_mae_sum'] / len(val_loader):.4f}")
        log_message(f"  - BG Loss: {val_loss_components['bg'] / len(val_loader):.4f}")
        log_message(f"  - BG MAE/RMSE/R2: {val_loss_components['bg_mae'] / len(val_loader):.4f} / {val_loss_components['bg_rmse'] / len(val_loader):.4f} / {val_loss_components['bg_r2'] / len(val_loader):.4f}")
        log_message(f"  - Daily-Insulin MAE/MAPE/RMSE/R2/Recall/Precision: {val_loss_components['daily_insulin_mae'] / len(val_loader):.4f} / {val_loss_components['daily_insulin_mape'] / len(val_loader):.2f}% / {val_loss_components['daily_insulin_rmse'] / len(val_loader):.4f} / {val_loss_components['daily_insulin_r2'] / len(val_loader):.4f} / {val_loss_components['daily_insulin_acc'] / len(val_loader):.4f} / {val_loss_components['daily_insulin_precision'] / len(val_loader):.4f}")
        log_message(f"  - Micro MAE/MAPE/RMSE/R2/Recall/Precision: {val_loss_components['micro_mae'] / len(val_loader):.4f} / {val_loss_components['micro_mape'] / len(val_loader):.2f}% / {val_loss_components['micro_rmse'] / len(val_loader):.4f} / {val_loss_components['micro_r2'] / len(val_loader):.4f} / {val_loss_components['micro_acc'] / len(val_loader):.4f} / {val_loss_components['micro_precision'] / len(val_loader):.4f}")
        log_message(f"  - IV MAE/MAPE/RMSE/R2/Recall/Precision: {val_loss_components['iv_mae'] / len(val_loader):.4f} / {val_loss_components['iv_mape'] / len(val_loader):.2f}% / {val_loss_components['iv_rmse'] / len(val_loader):.4f} / {val_loss_components['iv_r2'] / len(val_loader):.4f} / {val_loss_components['iv_acc'] / len(val_loader):.4f} / {val_loss_components['iv_precision'] / len(val_loader):.4f}")
        if val_loss_components['bread'] > 0:
            log_message(f"  - Discharge Loss: {val_loss_components['bread'] / len(val_loader):.4f}")
        if val_loss_components['length'] > 0:
            log_message(f"  - Length Loss: {val_loss_components['length'] / len(val_loader):.4f}")
        
        avg_insulin_loss = val_loss_components['insulin'] / len(val_loader)
        avg_insulin_dose_mae = val_loss_components['insulin_dose_mae'] / len(val_loader)
        avg_insulin_dose_rmse = val_loss_components['insulin_dose_rmse'] / len(val_loader)
        avg_insulin_dose_r2 = val_loss_components['insulin_dose_r2'] / len(val_loader)
        avg_bg_insulin_mae_sum = val_loss_components['bg_insulin_mae_sum'] / len(val_loader)
        avg_daily_insulin_mae = val_loss_components['daily_insulin_mae'] / len(val_loader)
        avg_bg_loss = val_loss_components['bg'] / len(val_loader)
        avg_bg_mae = val_loss_components['bg_mae'] / len(val_loader)
        avg_bg_rmse = val_loss_components['bg_rmse'] / len(val_loader)
        avg_bg_r2 = val_loss_components['bg_r2'] / len(val_loader)
        avg_bg_daily_insulin_mae_sum = avg_bg_mae + avg_daily_insulin_mae
        monitor_value = avg_bg_daily_insulin_mae_sum
        flag_penalty = 0.0
        legacy_selection_metric = avg_bg_insulin_mae_sum
        legacy_flag_penalty = 0.0

        # Update learning rate scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(monitor_value)
            else:
                warmup_epochs = max(0, int(getattr(cfg, "warmup_epochs", 0)))
                if epoch >= warmup_epochs:
                    scheduler.step()

        # Learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar("val/cards/bg_mae", avg_bg_mae, epoch)
            writer.add_scalar("val/cards/daily_insulin_mae", avg_daily_insulin_mae, epoch)
            writer.add_scalar("val/cards/bg_daily_insulin_mae_sum", avg_bg_daily_insulin_mae_sum, epoch)
            writer.add_scalar("val/current_val_loss", avg_val_loss, epoch)
            writer.add_scalar("val/learning_rate", current_lr, epoch)
            writer.add_scalar("val/selection/bg_daily_insulin_mae_sum", monitor_value, epoch)
        log_message(f"Current Selection Metric ({monitor_metric_name}): {monitor_value:.4f}")
        log_message(f"Current BG+Daily-Insulin MAE Sum: {avg_bg_daily_insulin_mae_sum:.4f}")
        log_message(f"Current BG+Insulin MAE Sum: {avg_bg_insulin_mae_sum:.4f}")
        log_message(f"Current Learning Rate: {current_lr:.8f}")
        log_message(f"{'='*60}\n")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        improved = monitor_value < best_monitor_value
        meaningful_improved = monitor_value < (best_es_monitor_value - early_stop_min_delta)
        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_metrics = {
                "selection_metric_name": monitor_metric_name,
                "selection_metric": monitor_value,
                "flag_penalty": flag_penalty,
                "legacy_selection_metric": legacy_selection_metric,
                "legacy_flag_penalty": legacy_flag_penalty,
                "bg_daily_insulin_mae_sum": avg_bg_daily_insulin_mae_sum,
                "val_loss": avg_val_loss,
                "insulin_loss": avg_insulin_loss,
                "insulin_dose_mae": avg_insulin_dose_mae,
                "insulin_dose_mape": val_loss_components['insulin_dose_mape'] / len(val_loader),
                "insulin_dose_rmse": avg_insulin_dose_rmse,
                "insulin_dose_r2": avg_insulin_dose_r2,
                "bg_insulin_mae_sum": avg_bg_insulin_mae_sum,
                "bg_loss": avg_bg_loss,
                "bg_mae": avg_bg_mae,
                "bg_rmse": avg_bg_rmse,
                "bg_r2": avg_bg_r2,
                "daily_insulin_acc": val_loss_components['daily_insulin_acc'] / len(val_loader),
                "daily_insulin_precision": val_loss_components['daily_insulin_precision'] / len(val_loader),
                "daily_insulin_mae": val_loss_components['daily_insulin_mae'] / len(val_loader),
                "daily_insulin_mape": val_loss_components['daily_insulin_mape'] / len(val_loader),
                "daily_insulin_rmse": val_loss_components['daily_insulin_rmse'] / len(val_loader),
                "daily_insulin_r2": val_loss_components['daily_insulin_r2'] / len(val_loader),
                "micro_acc": val_loss_components['micro_acc'] / len(val_loader),
                "micro_precision": val_loss_components['micro_precision'] / len(val_loader),
                "micro_mae": val_loss_components['micro_mae'] / len(val_loader),
                "micro_mape": val_loss_components['micro_mape'] / len(val_loader),
                "micro_rmse": val_loss_components['micro_rmse'] / len(val_loader),
                "micro_r2": val_loss_components['micro_r2'] / len(val_loader),
                "iv_acc": val_loss_components['iv_acc'] / len(val_loader),
                "iv_precision": val_loss_components['iv_precision'] / len(val_loader),
                "iv_mae": val_loss_components['iv_mae'] / len(val_loader),
                "iv_mape": val_loss_components['iv_mape'] / len(val_loader),
                "iv_rmse": val_loss_components['iv_rmse'] / len(val_loader),
                "iv_r2": val_loss_components['iv_r2'] / len(val_loader),
                "learning_rate": current_lr,
            }
            torch.save(model.state_dict(), os.path.join(save_best_path, "model.pth"))
            log_message(
                f"[SAVED] Best model (Epoch {epoch}, {monitor_metric_name}: {monitor_value:.4f}, Val Loss: {avg_val_loss:.4f})\n"
            )

        if meaningful_improved:
            best_es_val_loss = avg_val_loss
            best_es_monitor_value = monitor_value
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if current_lr <= (min_lr * 1.01 if min_lr > 0 else 0.0):
            min_lr_counter += 1
        else:
            min_lr_counter = 0

        if writer is not None:
            writer.add_scalar("val/best_val_loss", best_val_loss, epoch)
            writer.add_scalar("train/early_stop_counter", early_stop_counter, epoch)
            writer.add_scalar("train/min_lr_counter", min_lr_counter, epoch)

        if use_early_stop and epoch >= early_stop_start_epoch and early_stop_counter >= early_stop_patience:
            log_message(f"[EARLY STOP] No validation improvement greater than {early_stop_min_delta:.4f} for {early_stop_patience} epochs.")
            break
 
        if use_early_stop and epoch >= early_stop_start_epoch and getattr(cfg, "stop_on_min_lr", 0) and min_lr > 0 and min_lr_counter >= getattr(cfg, "min_lr_patience", 8):
            log_message(
                f"[EARLY STOP] Learning rate stayed near min_lr={min_lr:.8f} "
                f"for {min_lr_counter} epochs without meaningful validation improvement."
            )
            break

    if writer is not None:
        writer.close()

    if best_epoch >= 0:
        best_ci = {}
        best_model_path = os.path.join(save_best_path, "model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            best_ci = collect_metric_histories(model, val_loader, criterion, cfg, device)

        best_sc_bg_sum = best_metrics.get("bg_insulin_mae_sum", float("nan"))

        log_message("\n" + "=" * 60)
        log_message(f"Best Validation Summary (Epoch {best_epoch})")
        if "selection_metric" in best_metrics:
            log_message(f"  - Selection Metric ({best_metrics.get('selection_metric_name', 'selection_metric')}): {best_metrics.get('selection_metric', float('nan')):.4f}")
        log_message(f"  - BG+Daily-Insulin MAE Sum: {best_metrics.get('bg_daily_insulin_mae_sum', float('nan')):.4f}")
        log_message(f"  - BG+Insulin MAE Sum: {best_sc_bg_sum:.4f}")
        log_message(f"  - Best Val Loss: {best_metrics.get('val_loss', float('nan')):.4f}")
        log_message(f"  - Insulin Loss: {best_metrics.get('insulin_loss', float('nan')):.4f}")
        log_message(f"  - Insulin Dose MAE/MAPE/RMSE/R2: {best_metrics.get('insulin_dose_mae', float('nan')):.4f} / {best_metrics.get('insulin_dose_mape', float('nan')):.2f}% / {best_metrics.get('insulin_dose_rmse', float('nan')):.4f} / {best_metrics.get('insulin_dose_r2', float('nan')):.4f}")
        log_message(f"    Insulin Dose 95%CI: MAE {format_ci(*best_ci.get('insulin_dose_mae', (float('nan'), float('nan'))))} | MAPE {format_ci(*best_ci.get('insulin_dose_mape', (float('nan'), float('nan'))))} | RMSE {format_ci(*best_ci.get('insulin_dose_rmse', (float('nan'), float('nan'))))} | R2 {format_ci(*best_ci.get('insulin_dose_r2', (float('nan'), float('nan'))))}")
        log_message(f"  - BG+Insulin MAE Sum: {best_metrics.get('bg_insulin_mae_sum', float('nan')):.4f} 95%CI {format_ci(*best_ci.get('bg_insulin_mae_sum', (float('nan'), float('nan'))))}")
        log_message(f"  - BG Loss: {best_metrics.get('bg_loss', float('nan')):.4f}")
        log_message(f"  - BG MAE/RMSE/R2: {best_metrics.get('bg_mae', float('nan')):.4f} / {best_metrics.get('bg_rmse', float('nan')):.4f} / {best_metrics.get('bg_r2', float('nan')):.4f}")
        log_message(f"    BG 95%CI: MAE {format_ci(*best_ci.get('bg_mae', (float('nan'), float('nan'))))} | RMSE {format_ci(*best_ci.get('bg_rmse', (float('nan'), float('nan'))))} | R2 {format_ci(*best_ci.get('bg_r2', (float('nan'), float('nan'))))}")
        log_message(f"  - Daily-Insulin MAE/MAPE/RMSE/R2/Recall/Precision: {best_metrics.get('daily_insulin_mae', float('nan')):.4f} / {best_metrics.get('daily_insulin_mape', float('nan')):.2f}% / {best_metrics.get('daily_insulin_rmse', float('nan')):.4f} / {best_metrics.get('daily_insulin_r2', float('nan')):.4f} / {best_metrics.get('daily_insulin_acc', float('nan')):.4f} / {best_metrics.get('daily_insulin_precision', float('nan')):.4f}")
        log_message(f"    Daily-Insulin 95%CI: MAE {format_ci(*best_ci.get('daily_insulin_mae', (float('nan'), float('nan'))))} | MAPE {format_ci(*best_ci.get('daily_insulin_mape', (float('nan'), float('nan'))))} | RMSE {format_ci(*best_ci.get('daily_insulin_rmse', (float('nan'), float('nan'))))} | R2 {format_ci(*best_ci.get('daily_insulin_r2', (float('nan'), float('nan'))))} | Recall {format_ci(*best_ci.get('daily_insulin_acc', (float('nan'), float('nan'))))} | Precision {format_ci(*best_ci.get('daily_insulin_precision', (float('nan'), float('nan'))))}")
        log_message(f"  - Micro MAE/MAPE/RMSE/R2/Recall/Precision: {best_metrics.get('micro_mae', float('nan')):.4f} / {best_metrics.get('micro_mape', float('nan')):.2f}% / {best_metrics.get('micro_rmse', float('nan')):.4f} / {best_metrics.get('micro_r2', float('nan')):.4f} / {best_metrics.get('micro_acc', float('nan')):.4f} / {best_metrics.get('micro_precision', float('nan')):.4f}")
        log_message(f"    Micro 95%CI: MAE {format_ci(*best_ci.get('micro_mae', (float('nan'), float('nan'))))} | MAPE {format_ci(*best_ci.get('micro_mape', (float('nan'), float('nan'))))} | RMSE {format_ci(*best_ci.get('micro_rmse', (float('nan'), float('nan'))))} | R2 {format_ci(*best_ci.get('micro_r2', (float('nan'), float('nan'))))} | Recall {format_ci(*best_ci.get('micro_acc', (float('nan'), float('nan'))))} | Precision {format_ci(*best_ci.get('micro_precision', (float('nan'), float('nan'))))}")
        log_message(f"  - IV MAE/MAPE/RMSE/R2/Recall/Precision: {best_metrics.get('iv_mae', float('nan')):.4f} / {best_metrics.get('iv_mape', float('nan')):.2f}% / {best_metrics.get('iv_rmse', float('nan')):.4f} / {best_metrics.get('iv_r2', float('nan')):.4f} / {best_metrics.get('iv_acc', float('nan')):.4f} / {best_metrics.get('iv_precision', float('nan')):.4f}")
        log_message(f"    IV 95%CI: MAE {format_ci(*best_ci.get('iv_mae', (float('nan'), float('nan'))))} | MAPE {format_ci(*best_ci.get('iv_mape', (float('nan'), float('nan'))))} | RMSE {format_ci(*best_ci.get('iv_rmse', (float('nan'), float('nan'))))} | R2 {format_ci(*best_ci.get('iv_r2', (float('nan'), float('nan'))))} | Recall {format_ci(*best_ci.get('iv_acc', (float('nan'), float('nan'))))} | Precision {format_ci(*best_ci.get('iv_precision', (float('nan'), float('nan'))))}")
        log_message(f"  - Learning Rate: {best_metrics.get('learning_rate', float('nan')):.8f}")
        log_message("=" * 60 + "\n")


if __name__ == "__main__":
    cfg = opt_config()
    train_val_test(cfg)
