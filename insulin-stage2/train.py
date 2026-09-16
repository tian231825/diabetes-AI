# -*- encoding: utf-8 -*-
"""Training and evaluation entry point for stage 2."""
import os
import json
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
from utils.logger import make_torch_generator, seed_worker, set_seed, setup_logging
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


def _compute_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    mu = float(values.mean())
    sigma = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mu, sigma


def compute_rule_stats_from_dataset(dataset, effective_dose_threshold=2.0):
    bb_ratios = []
    pm_am_ratios = []
    pm_pm_ratios = []

    for sample in dataset:
        insulin2 = sample["insulin2"].detach().cpu().numpy().astype(np.float64)
        regimen2 = sample["regimen2"].detach().cpu().numpy().astype(np.float64)
        regimen2_mask = sample["regimen2_mask"].detach().cpu().numpy().astype(np.float64)

        labeled_bb_day_mask = (regimen2_mask[:, 0] > 0) & (regimen2[:, 0] > 0.5)
        labeled_pm_day_mask = (regimen2_mask[:, 1] > 0) & (regimen2[:, 1] > 0.5)
        dose_slots = insulin2[:, -4:]
        bb_trigger_mask = (dose_slots > effective_dose_threshold).any(axis=1)
        pm_trigger_mask = (dose_slots > effective_dose_threshold).any(axis=1)

        bb_day_mask = labeled_bb_day_mask & bb_trigger_mask
        pm_day_mask = labeled_pm_day_mask & pm_trigger_mask

        if bb_day_mask.any():
            bb_slots = dose_slots[bb_day_mask].sum(axis=0)
            bb_total = float(bb_slots.sum())
            if bb_total > 1e-8:
                bb_ratios.append(float(bb_slots[3] / bb_total))

        if pm_day_mask.any():
            pm_slots = dose_slots[pm_day_mask].sum(axis=0)
            p_am = float(pm_slots[0] + pm_slots[1])
            p_pm = float(pm_slots[2])
            pm_total = float(pm_slots.sum())
            if pm_total > 1e-8:
                pm_am_ratios.append(float(p_am / pm_total))
                pm_pm_ratios.append(float(p_pm / pm_total))

    mu_bb, sigma_bb = _compute_mean_std(bb_ratios)
    mu_pm_am, sigma_pm_am = _compute_mean_std(pm_am_ratios)
    mu_pm_pm, sigma_pm_pm = _compute_mean_std(pm_pm_ratios)
    return {
        "mu_bb": 0.45 if mu_bb is None else mu_bb,
        "sigma_bb": 0.10 if sigma_bb is None else sigma_bb,
        "mu_pm_am": 0.50 if mu_pm_am is None else mu_pm_am,
        "sigma_pm_am": 0.10 if sigma_pm_am is None else sigma_pm_am,
        "mu_pm_pm": 0.50 if mu_pm_pm is None else mu_pm_pm,
        "sigma_pm_pm": 0.10 if sigma_pm_pm is None else sigma_pm_pm,
        "n_bb_patients": len(bb_ratios),
        "n_pm_patients": len(pm_am_ratios),
        "source": "computed_from_train_split",
    }


def load_rule_stats_for_training(cfg, train_dataset):
    default_stats = {
        "mu_bb": 0.45,
        "sigma_bb": 0.10,
        "mu_pm_am": 0.50,
        "sigma_pm_am": 0.10,
        "mu_pm_pm": 0.50,
        "sigma_pm_pm": 0.10,
        "n_bb_patients": 0,
        "n_pm_patients": 0,
        "source": "default_fallback",
    }
    stats_path = getattr(cfg, "rule_stats_path", "") or ""
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        bb_stats = payload.get("bb_stats", {})
        pm_am_stats = payload.get("pm_am_stats", {})
        pm_pm_stats = payload.get("pm_pm_stats", {})
        return {
            "mu_bb": float(bb_stats.get("mu", default_stats["mu_bb"])),
            "sigma_bb": float(bb_stats.get("sigma", default_stats["sigma_bb"])),
            "mu_pm_am": float(pm_am_stats.get("mu", default_stats["mu_pm_am"])),
            "sigma_pm_am": float(pm_am_stats.get("sigma", default_stats["sigma_pm_am"])),
            "mu_pm_pm": float(pm_pm_stats.get("mu", default_stats["mu_pm_pm"])),
            "sigma_pm_pm": float(pm_pm_stats.get("sigma", default_stats["sigma_pm_pm"])),
            "n_bb_patients": int(payload.get("bb_patient_count_with_valid_ratio", 0)),
            "n_pm_patients": int(payload.get("pm_patient_count_with_valid_ratio", 0)),
            "source": stats_path,
        }
    try:
        return compute_rule_stats_from_dataset(
            train_dataset,
            effective_dose_threshold=float(getattr(cfg, "effective_dose_threshold", 2.0)),
        )
    except Exception:
        return default_stats


def collect_metric_histories(model, loader, criterion, cfg, device):
    metric_keys = [
        "insulin_dose_mae",
        "insulin_dose_mape",
        "insulin_dose_mae_basic",
        "insulin_dose_mape_basic",
        "insulin_dose_mae_premix",
        "insulin_dose_mape_premix",
        "insulin_dose_mae_basic_matched",
        "insulin_dose_mape_basic_matched",
        "insulin_dose_mae_premix_matched",
        "insulin_dose_mape_premix_matched",
        "insulin_dose_rmse_basic",
        "insulin_dose_rmse_premix",
        "insulin_dose_rmse_basic_matched",
        "insulin_dose_rmse_premix_matched",
        "insulin_dose_r2_basic",
        "insulin_dose_r2_premix",
        "insulin_dose_r2_basic_matched",
        "insulin_dose_r2_premix_matched",
        "bg_mae", "bg_rmse", "bg_r2",
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


def linear_schedule_value(epoch, total_epochs, start_value, end_value):
    total_epochs = max(1, int(total_epochs))
    if total_epochs <= 1:
        return float(end_value)
    progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
    return float(start_value) + (float(end_value) - float(start_value)) * progress


def apply_two_stage_curriculum(criterion, cfg, epoch):
    if not getattr(cfg, "use_two_stage_curriculum", 0):
        cfg.current_lambda_bg = float(getattr(cfg, "lambda_bg", 1.0))
        cfg.current_lambda_insulin = float(getattr(cfg, "lambda_insulin", 1.0))
        return "single"

    total_epochs = max(1, int(getattr(cfg, "epochs", 1)))
    phase1_epochs = max(1, int(total_epochs * float(getattr(cfg, "two_stage_phase1_ratio", 0.2))))

    if bool(getattr(cfg, "dynamic_multi_task_loss", 0)):
        criterion.insulin_type_loss_weight = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_insulin_type_loss_weight", criterion.insulin_type_loss_weight),
            getattr(cfg, "phase2_insulin_type_loss_weight", criterion.insulin_type_loss_weight),
        )
        criterion.premix_recall_focus_loss_weight = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0)),
            getattr(cfg, "phase2_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0)),
        )
        criterion.insulin_regression_loss_weight = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_insulin_regression_loss_weight", getattr(criterion, "insulin_regression_loss_weight", 1.0)),
            getattr(cfg, "phase2_insulin_regression_loss_weight", getattr(criterion, "insulin_regression_loss_weight", 1.0)),
        )
        criterion.basic_sc_focus_loss_weight = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0)),
            getattr(cfg, "phase2_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0)),
        )
        criterion.premix_sc_focus_loss_weight = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0)),
            getattr(cfg, "phase2_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0)),
        )
        cfg.current_lambda_bg = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_lambda_bg", getattr(cfg, "lambda_bg", 1.0)),
            getattr(cfg, "phase2_lambda_bg", getattr(cfg, "lambda_bg", 1.0)),
        )
        cfg.current_lambda_insulin = linear_schedule_value(
            epoch, total_epochs,
            getattr(cfg, "phase1_lambda_insulin", getattr(cfg, "lambda_insulin", 1.0)),
            getattr(cfg, "phase2_lambda_insulin", getattr(cfg, "lambda_insulin", 1.0)),
        )
        return "phase1" if epoch < phase1_epochs else "phase2"

    if epoch < phase1_epochs:
        criterion.insulin_type_loss_weight = float(getattr(cfg, "phase1_insulin_type_loss_weight", criterion.insulin_type_loss_weight))
        criterion.premix_recall_focus_loss_weight = float(getattr(cfg, "phase1_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0)))
        criterion.insulin_regression_loss_weight = float(getattr(cfg, "phase1_insulin_regression_loss_weight", getattr(criterion, "insulin_regression_loss_weight", 1.0)))
        criterion.basic_sc_focus_loss_weight = float(getattr(cfg, "phase1_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0)))
        criterion.premix_sc_focus_loss_weight = float(getattr(cfg, "phase1_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0)))
        cfg.current_lambda_bg = float(getattr(cfg, "phase1_lambda_bg", getattr(cfg, "lambda_bg", 1.0)))
        cfg.current_lambda_insulin = float(getattr(cfg, "phase1_lambda_insulin", getattr(cfg, "lambda_insulin", 1.0)))
        return "phase1"

    criterion.insulin_type_loss_weight = float(getattr(cfg, "phase2_insulin_type_loss_weight", criterion.insulin_type_loss_weight))
    criterion.premix_recall_focus_loss_weight = float(getattr(cfg, "phase2_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0)))
    criterion.insulin_regression_loss_weight = float(getattr(cfg, "phase2_insulin_regression_loss_weight", getattr(criterion, "insulin_regression_loss_weight", 1.0)))
    criterion.basic_sc_focus_loss_weight = float(getattr(cfg, "phase2_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0)))
    criterion.premix_sc_focus_loss_weight = float(getattr(cfg, "phase2_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0)))
    cfg.current_lambda_bg = float(getattr(cfg, "phase2_lambda_bg", getattr(cfg, "lambda_bg", 1.0)))
    cfg.current_lambda_insulin = float(getattr(cfg, "phase2_lambda_insulin", getattr(cfg, "lambda_insulin", 1.0)))
    return "phase2"


def compute_selection_metric_from_insulin_bg(
    insulin_bg_sum,
    cfg,
):
    insulin_bg_sum = float(insulin_bg_sum)
    raw_sc_weight = float(getattr(cfg, "selection_raw_sc_weight", 1.0))
    return raw_sc_weight * insulin_bg_sum, 0.0


def compute_balanced_regimen_dose_metric(
    insulin_dose_mae_basic,
    insulin_dose_mae_premix,
):
    basic_mae = float(insulin_dose_mae_basic)
    premix_mae = float(insulin_dose_mae_premix)
    return 0.5 * basic_mae + 0.5 * premix_mae


def compute_selection_metric_with_regression_objective(
    insulin_bg_sum,
    regimen_recall_premix,
    cfg,
):
    base_value, aux_penalty = compute_selection_metric_from_insulin_bg(
        insulin_bg_sum,
        cfg,
    )
    premix_recall = float(regimen_recall_premix)
    premix_floor = float(getattr(cfg, "selection_premix_recall_floor", 0.0))
    premix_penalty_weight = float(getattr(cfg, "selection_premix_recall_penalty_weight", 0.0))
    premix_recall_penalty = 0.0
    if premix_floor > 0.0 and premix_penalty_weight > 0.0:
        premix_recall_penalty = premix_penalty_weight * max(0.0, premix_floor - premix_recall)
    return base_value + premix_recall_penalty, aux_penalty, premix_recall_penalty

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
    if dose_mask.shape[-1] >= 12:
        basic_template = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            dtype=dose_mask.dtype,
            device=dose_mask.device,
        )
        premix_template = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=dose_mask.dtype,
            device=dose_mask.device,
        )
        none_template = torch.tensor(
            [0.0] * 12,
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
        "rmse": reg_metrics["rmse"],
        "r2": reg_metrics["r2"],
        "count": reg_metrics["count"],
    }


def infer_regimen_labels_from_dose(dose_tensor, positive_threshold=2.0):
    """
    Infer day-level basic/premix label from 8-d regression output:
    - 0: basic
    - 1: premix
    - -1: all zero / no regimen
    """
    if dose_tensor.shape[-1] == 5:
        dose_part = torch.where(
            dose_tensor[..., 1:5] >= positive_threshold,
            dose_tensor[..., 1:5],
            torch.zeros_like(dose_tensor[..., 1:5]),
        )
        dose_active = dose_part.sum(dim=-1) > 0
        flag_part = dose_tensor[..., 0]
        labels = torch.full(dose_active.shape, -1, dtype=torch.long, device=dose_tensor.device)
        labels = torch.where(dose_active & (flag_part >= 0.5), torch.ones_like(labels), labels)
        labels = torch.where(dose_active & (flag_part < 0.5), torch.zeros_like(labels), labels)
        return labels

    if dose_tensor.shape[-1] < 8:
        return torch.full(dose_tensor.shape[:2], -1, dtype=torch.long, device=dose_tensor.device)

    basic_part = torch.where(dose_tensor[..., :4] >= positive_threshold, dose_tensor[..., :4], torch.zeros_like(dose_tensor[..., :4]))
    premix_part = torch.where(dose_tensor[..., 4:8] >= positive_threshold, dose_tensor[..., 4:8], torch.zeros_like(dose_tensor[..., 4:8]))

    basic_score = basic_part.sum(dim=-1)
    premix_score = premix_part.sum(dim=-1)
    basic_active = basic_score > 0
    premix_active = premix_score > 0

    labels = torch.full_like(basic_score, -1, dtype=torch.long)
    labels = torch.where(basic_active & ~premix_active, torch.zeros_like(labels), labels)
    labels = torch.where(~basic_active & premix_active, torch.ones_like(labels), labels)
    both_active = basic_active & premix_active
    labels = torch.where(both_active & (basic_score >= premix_score), torch.zeros_like(labels), labels)
    labels = torch.where(both_active & (premix_score > basic_score), torch.ones_like(labels), labels)
    return labels


def gaussian_consistency_score(ratio_tensor, mu, sigma, min_sigma=1e-3):
    sigma = max(float(abs(sigma)), min_sigma)
    return torch.exp(-0.5 * ((ratio_tensor - float(mu)) / sigma) ** 2)


def build_eval_dose_tensors(batch, outputs, device):
    insulin_pred, bg_pred = outputs[0], outputs[1]
    bg_gt = batch["bg2"]
    bg_mask = batch["bg2_mask"]
    insulin_gt = batch["insulin2"]
    insulin_gt_mask = batch["insulin2_mask"]

    min_T = min(
        insulin_pred.shape[1],
        bg_pred.shape[1],
        insulin_gt.shape[1],
        bg_gt.shape[1],
        bg_mask.shape[1],
        insulin_gt_mask.shape[1],
    )

    dose_pred = insulin_pred[:, :min_T, :]
    dose_gt = insulin_gt[:, :min_T, :]
    bg_gt = bg_gt[:, :min_T, :]
    bg_mask = bg_mask[:, :min_T, :]
    insulin_gt_mask = insulin_gt_mask[:, :min_T, :]

    bg_mask = ensure_mask_dimensions(bg_mask, bg_gt.shape)
    real_lengths = calculate_real_lengths_from_mask(bg_mask)
    insulin_seq_mask = create_sequence_mask(real_lengths, min_T, dose_pred.shape[-1], device)
    insulin_gt_mask = ensure_mask_dimensions(insulin_gt_mask, dose_gt.shape)
    dose_mask = insulin_seq_mask * insulin_gt_mask
    return dose_pred, dose_gt, dose_mask, min_T


def build_gt_regimen_labels(batch, dose_gt, dose_mask, min_T, cfg, device):
    gt_regimen = batch.get("regimen2")
    gt_regimen_mask = batch.get("regimen2_mask")
    if gt_regimen is not None and gt_regimen.shape[1] >= min_T:
        gt_regimen = gt_regimen[:, :min_T, :]
    if gt_regimen_mask is not None and gt_regimen_mask.shape[1] >= min_T:
        gt_regimen_mask = gt_regimen_mask[:, :min_T, :]

    if gt_regimen is not None and gt_regimen_mask is not None:
        gt_valid_label = gt_regimen_mask.sum(dim=-1) > 0
        gt_label = torch.full(gt_valid_label.shape, -1, dtype=torch.long, device=device)
        gt_label = torch.where(gt_valid_label & (gt_regimen[..., 1] > 0.5), torch.ones_like(gt_label), gt_label)
        gt_label = torch.where(gt_valid_label & (gt_regimen[..., 0] > 0.5), torch.zeros_like(gt_label), gt_label)
    else:
        gt_label = infer_regimen_labels_from_dose(
            dose_gt,
            positive_threshold=float(getattr(cfg, "effective_dose_threshold", 2.0)),
        )

    stats_valid_label = dose_mask.sum(dim=-1) > 0
    gt_label_for_stats = torch.where(
        gt_label >= 0,
        gt_label,
        torch.full_like(gt_label, 2),
    )
    gt_label_for_stats = resolve_none_types_following_future(gt_label_for_stats, none_idx=2)
    return gt_label, gt_label_for_stats, stats_valid_label


def compute_topk_regimen_prior(model, batch, horizon):
    topk_indices, topk_scores = model._retrieve_topk_indices(batch)
    weights = torch.softmax(topk_scores.float(), dim=1).cpu().numpy()

    batch_size = topk_indices.shape[0]
    prior_basic = np.zeros((batch_size, horizon), dtype=np.float64)
    prior_premix = np.zeros((batch_size, horizon), dtype=np.float64)

    for b in range(batch_size):
        for k, kb_index in enumerate(topk_indices[b].tolist()):
            ref = model._knowledge_base[kb_index]
            ref_regimen = ref.get("regimen2")
            ref_regimen_mask = ref.get("regimen2_mask")
            if ref_regimen is None or ref_regimen_mask is None:
                continue

            ref_regimen = ref_regimen.detach().cpu().numpy() if hasattr(ref_regimen, "detach") else np.asarray(ref_regimen)
            ref_regimen_mask = ref_regimen_mask.detach().cpu().numpy() if hasattr(ref_regimen_mask, "detach") else np.asarray(ref_regimen_mask)
            ref_horizon = min(horizon, ref_regimen.shape[0], ref_regimen_mask.shape[0])
            if ref_horizon <= 0:
                continue

            valid = ref_regimen_mask[:ref_horizon].sum(axis=-1) > 0
            basic = valid & (ref_regimen[:ref_horizon, 0] > 0.5)
            premix = valid & (ref_regimen[:ref_horizon, 1] > 0.5)

            w = float(weights[b, k])
            prior_basic[b, :ref_horizon] += w * basic.astype(np.float64)
            prior_premix[b, :ref_horizon] += w * premix.astype(np.float64)

    total = prior_basic + prior_premix
    zero_mask = total <= 1e-8
    prior_basic = np.where(zero_mask, 0.5, prior_basic / np.clip(total, 1e-8, None))
    prior_premix = np.where(zero_mask, 0.5, prior_premix / np.clip(total, 1e-8, None))

    device = batch["person_value"].device
    return (
        torch.as_tensor(prior_basic, dtype=torch.float32, device=device),
        torch.as_tensor(prior_premix, dtype=torch.float32, device=device),
    )


def infer_rule_probabilistic_regimen_labels(dose_tensor, prior_basic, prior_premix, cfg, rule_stats):
    if dose_tensor.shape[-1] == 5:
        threshold = float(getattr(cfg, "effective_dose_threshold", 2.0))
        dose_part = torch.clamp(dose_tensor[..., 1:5], min=0.0)
        dose_active = (dose_part > threshold).any(dim=-1)
        flag_prob = torch.sigmoid((dose_tensor[..., 0] - 0.5) * 4.0)
        probs = torch.stack([1.0 - flag_prob, flag_prob], dim=-1)
        pred_label = torch.argmax(probs, dim=-1).long()
        pred_label = torch.where(dose_active, pred_label, torch.full_like(pred_label, -1))
        return pred_label, probs

    threshold = float(getattr(cfg, "effective_dose_threshold", 2.0))
    temperature = max(float(getattr(cfg, "rule_softmax_temperature", 0.2)), 1e-6)
    prior_weight = float(getattr(cfg, "rule_prior_weight", 0.35))
    eps = 1e-8

    dose_tensor = torch.clamp(dose_tensor, min=0.0)
    bb_slots = dose_tensor[..., :4]
    pm_slots = dose_tensor[..., 4:8]

    bb_trigger = (bb_slots > threshold).any(dim=-1)
    pm_am = pm_slots[..., 0] + pm_slots[..., 1]
    pm_pm = pm_slots[..., 2] + pm_slots[..., 3]
    pm_trigger = (pm_am > threshold) | (pm_pm > threshold)

    bb_total = bb_slots.sum(dim=-1)
    pm_total = pm_am + pm_pm

    r_bb = bb_slots[..., 3] / (bb_total + eps)
    r_pm_am = pm_am / (pm_total + eps)
    r_pm_pm = pm_pm / (pm_total + eps)

    bb_score = gaussian_consistency_score(r_bb, rule_stats["mu_bb"], rule_stats["sigma_bb"])
    pm_am_score = gaussian_consistency_score(r_pm_am, rule_stats["mu_pm_am"], rule_stats["sigma_pm_am"])
    pm_pm_score = gaussian_consistency_score(r_pm_pm, rule_stats["mu_pm_pm"], rule_stats["sigma_pm_pm"])
    pm_score = torch.sqrt(torch.clamp(pm_am_score * pm_pm_score, min=0.0))

    huge_neg = torch.full_like(bb_score, -1e4)
    bb_logit = torch.where(
        bb_trigger,
        bb_score / temperature + prior_weight * torch.log(prior_basic + eps),
        huge_neg,
    )
    pm_logit = torch.where(
        pm_trigger,
        pm_score / temperature + prior_weight * torch.log(prior_premix + eps),
        huge_neg,
    )

    logits = torch.stack([bb_logit, pm_logit], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).long()
    no_trigger = ~(bb_trigger | pm_trigger)
    pred_label = torch.where(no_trigger, torch.full_like(pred_label, -1), pred_label)
    return pred_label, probs


def compute_regimen_classification_metrics(pred_label, gt_label_for_stats, stats_valid_label):
    overall = compute_binary_regimen_metrics(
        pred_label=pred_label,
        gt_label=gt_label_for_stats,
        valid_mask=stats_valid_label.float(),
        class_idx=1,
    )
    metrics = {"acc": overall["acc"]}
    for class_name, class_idx in [("basic", 0), ("premix", 1)]:
        cls = compute_binary_regimen_metrics(
            pred_label=pred_label,
            gt_label=gt_label_for_stats,
            valid_mask=stats_valid_label.float(),
            class_idx=class_idx,
        )
        metrics[f"recall_{class_name}"] = cls["recall"]
        metrics[f"precision_{class_name}"] = cls["precision"]
        metrics[f"f1_{class_name}"] = cls["f1"]
    return metrics


def compute_binary_regimen_metrics(pred_label, gt_label, valid_mask, class_idx):
    eval_mask = valid_mask > 0
    gt_pos = eval_mask & (gt_label == class_idx)
    pred_pos = eval_mask & (pred_label == class_idx)
    tp = (gt_pos & pred_pos).sum().item()
    gt_count = gt_pos.sum().item()
    pred_count = pred_pos.sum().item()
    acc = ((pred_label == gt_label) & eval_mask).float().sum().item() / (eval_mask.float().sum().item() + 1e-8) if eval_mask.any() else 0.0
    recall = tp / (gt_count + 1e-8) if gt_count > 0 else 0.0
    precision = tp / (pred_count + 1e-8) if pred_count > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


def compute_mode_metrics(pred_group, gt_group, mask_group, pred_label, gt_label, class_idx):
    """
    Mode-style metrics:
    - Recall / Precision depend on whether the regimen mode is predicted correctly
    - Regression metrics only use days where the mode is matched
    - Once the mode is matched, all valid slots are included regardless of zero values
    """
    valid_days = (mask_group.sum(dim=-1) > 0)
    gt_days = valid_days & (gt_label == class_idx)
    pred_days = valid_days & (pred_label == class_idx)
    matched_days = valid_days & (gt_label == class_idx) & (pred_label == class_idx)

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
        "rmse": reg_metrics["rmse"],
        "r2": reg_metrics["r2"],
        "count": reg_metrics["count"],
    }


def compute_gt_regimen_regression_metrics(pred_group, gt_group, mask_group, gt_type, class_idx):
    """
    Expanded regression metrics used only for selection:
    - only count days whose ground-truth regimen matches class_idx
    - do not require predicted regimen to match
    """
    gt_days = (gt_type == class_idx).unsqueeze(-1).float()
    reg_mask = mask_group * gt_days
    return compute_masked_regression_metrics(pred_group, gt_group, reg_mask)


def aggregate_premix_pair_tensors(pred_group, gt_group, mask_group):
    """
    将 premix 四个槽位聚合成两个总量：
    - 早总 = 早长 + 早短
    - 晚总 = 晚长 + 晚短
    聚合后按 2 取平均，保持和原始两个槽位一致的误差尺度。
    """
    if pred_group.shape[-1] != 4:
        return pred_group, gt_group, mask_group

    pred_pair = torch.stack(
        [
            (pred_group[..., 0] + pred_group[..., 1]) / 2.0,
            (pred_group[..., 2] + pred_group[..., 3]) / 2.0,
        ],
        dim=-1,
    )
    gt_pair = torch.stack(
        [
            (gt_group[..., 0] + gt_group[..., 1]) / 2.0,
            (gt_group[..., 2] + gt_group[..., 3]) / 2.0,
        ],
        dim=-1,
    )
    mask_pair = torch.stack(
        [
            torch.maximum(mask_group[..., 0], mask_group[..., 1]),
            torch.maximum(mask_group[..., 2], mask_group[..., 3]),
        ],
        dim=-1,
    )
    return pred_pair, gt_pair, mask_pair


def compute_component_metrics(dose_pred, dose_gt, dose_mask, pred_label, gt_label, positive_threshold=2.0):
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

    # 17 维版本和 13 维版本的剂量槽位语义不同，这里按输出维度自动切换索引表，
    # 保证论文统计指标和当前胰岛素编码方式一致。
    if dose_pred.shape[-1] == 5:
        group_defs = {
            "basic_sc": {"dose_indices": [1, 2, 3, 4], "type_filter": 0},
            "premix_sc": {"dose_indices": [1, 2, 3, 4], "type_filter": 1},
        }
    elif dose_pred.shape[-1] >= 8:
        group_defs = {
            "basic_sc": {"dose_indices": [0, 1, 2, 3], "type_filter": 0},
            "premix_sc": {"dose_indices": [4, 5, 6, 7], "type_filter": 1},
        }
    else:
        return metrics

    for name, group_def in group_defs.items():
        dose_indices = group_def["dose_indices"]
        if max(dose_indices) >= dose_pred.shape[-1]:
            continue

        pred_group = dose_pred[..., dose_indices]
        gt_group = dose_gt[..., dose_indices]
        mask_group = dose_mask[..., dose_indices]

        if name == "basic_sc":
            group_metrics = compute_mode_metrics(
                pred_group,
                gt_group,
                mask_group,
                pred_label=pred_label,
                gt_label=gt_label,
                class_idx=0,
            )
        elif name == "premix_sc":
            if dose_pred.shape[-1] >= 8 and pred_group.shape[-1] >= 4:
                pred_group, gt_group, mask_group = aggregate_premix_pair_tensors(
                    pred_group,
                    gt_group,
                    mask_group,
                )
            group_metrics = compute_mode_metrics(
                pred_group,
                gt_group,
                mask_group,
                pred_label=pred_label,
                gt_label=gt_label,
                class_idx=1,
            )

        metrics[f"{name}_acc"] = group_metrics["acc"]
        metrics[f"{name}_precision"] = group_metrics["precision"]
        metrics[f"{name}_mae"] = group_metrics["mae"]
        metrics[f"{name}_rmse"] = group_metrics["rmse"]
        metrics[f"{name}_r2"] = group_metrics["r2"]

    return metrics


def compute_expanded_sc_metrics(dose_pred, dose_gt, dose_mask, gt_label):
    """
    Selection-only metrics:
    - basic: compute regression on gt basic days over basic_sc slots
    - premix: compute regression on gt premix days over aggregated [morning total, evening total]
    """
    metrics = {}
    if dose_pred.numel() == 0:
        return metrics

    if dose_pred.shape[-1] == 5:
        basic_slice = slice(1, 5)
        premix_slice = slice(1, 5)
    elif dose_pred.shape[-1] >= 8:
        basic_slice = slice(0, 4)
        premix_slice = slice(4, 8)
    else:
        return metrics

    basic_metrics = compute_gt_regimen_regression_metrics(
        dose_pred[..., basic_slice],
        dose_gt[..., basic_slice],
        dose_mask[..., basic_slice],
        gt_type=gt_label,
        class_idx=0,
    )
    metrics["expanded_basic_sc_mae"] = basic_metrics["mae"]
    metrics["expanded_basic_sc_rmse"] = basic_metrics["rmse"]
    metrics["expanded_basic_sc_r2"] = basic_metrics["r2"]

    premix_pred = dose_pred[..., premix_slice]
    premix_gt = dose_gt[..., premix_slice]
    premix_mask = dose_mask[..., premix_slice]
    if dose_pred.shape[-1] >= 8:
        premix_pred, premix_gt, premix_mask = aggregate_premix_pair_tensors(
            premix_pred,
            premix_gt,
            premix_mask,
        )
    premix_metrics = compute_gt_regimen_regression_metrics(
        premix_pred,
        premix_gt,
        premix_mask,
        gt_type=gt_label,
        class_idx=1,
    )
    metrics["expanded_premix_sc_mae"] = premix_metrics["mae"]
    metrics["expanded_premix_sc_rmse"] = premix_metrics["rmse"]
    metrics["expanded_premix_sc_r2"] = premix_metrics["r2"]

    return metrics


def compute_merged_group_metrics(dose_pred, dose_gt, dose_mask, pred_type, gt_type, insulin_flag_dims):
    """
    Merged basic/premix metrics:
    - resolve none days into nearby non-none regimen for both gt and prediction
    - use merged dose mask over the full merged dose slots
    - recall/precision are event-style metrics on merged regimen days
    - regression metrics use merged regimen day masks
    """
    metrics = {}
    if dose_pred.numel() == 0 or insulin_flag_dims < 3:
        return metrics

    merged_gt_type = resolve_none_types_following_future(gt_type, none_idx=2)
    merged_dose_mask = build_merged_dose_mask(merged_gt_type, dose_mask)

    for class_idx, class_name in enumerate(["basic", "premix"]):
        pred_group = dose_pred
        gt_group = dose_gt
        mask_group = merged_dose_mask
        if class_name == "premix" and dose_pred.shape[-1] >= 8:
            pred_group = dose_pred[..., 4:8]
            gt_group = dose_gt[..., 4:8]
            mask_group = merged_dose_mask[..., 4:8]
            pred_group, gt_group, mask_group = aggregate_premix_pair_tensors(
                pred_group, gt_group, mask_group
            )
        group_day_mask = (merged_gt_type == class_idx).unsqueeze(-1).float()
        metric_mask = mask_group * group_day_mask
        reg_metrics = compute_masked_regression_metrics(
            pred_group,
            gt_group,
            metric_mask,
        )
        event_metrics = compute_positive_event_metrics(
            pred_group,
            gt_group,
            metric_mask,
        )
        metrics[f"{class_name}_merged_acc"] = event_metrics["acc"]
        metrics[f"{class_name}_merged_precision"] = event_metrics["precision"]
        metrics[f"{class_name}_merged_mae"] = reg_metrics["mae"]
        metrics[f"{class_name}_merged_rmse"] = reg_metrics["rmse"]
        metrics[f"{class_name}_merged_r2"] = reg_metrics["r2"]

    return metrics


def build_train_sampler(dataset, cfg):
    """
    仅改变训练采样分布：
    对包含 premix 天的序列提高采样权重，不改变标签和 loss 定义。
    """
    if not bool(getattr(cfg, "use_premix_oversample", 0)):
        return None

    premix_weight = float(getattr(cfg, "premix_oversample_weight", 1.0))
    if premix_weight <= 1.0:
        return None

    sample_weights = []
    premix_sequence_count = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        regimen2 = sample.get("regimen2", None)
        regimen2_mask = sample.get("regimen2_mask", None)
        has_premix = False
        if regimen2 is not None and regimen2_mask is not None:
            premix_flag = regimen2[:, 1]
            premix_mask = regimen2_mask[:, 1]
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


def maybe_augment_premix_batch(batch, cfg):
    """
    仅在训练阶段对 premix 天做轻量增强：
    - 只改 insulin2 的剂量位
    - 只改 premix 天
    - 只改非零且有效的剂量位
    - 不改标签类型、不改 mask、不改 bg 真值
    """
    if not bool(getattr(cfg, "use_premix_augmentation", 0)):
        return batch

    insulin = batch.get("insulin2", None)
    insulin_mask = batch.get("insulin2_mask", None)
    if insulin is None or insulin_mask is None:
        return batch

    aug_prob = float(getattr(cfg, "premix_aug_prob", 0.3))
    aug_scale = float(getattr(cfg, "premix_aug_scale", 0.05))
    if aug_prob <= 0.0 or aug_scale <= 0.0:
        return batch

    # 这里只改训练输入，不改 ground-truth 标签和 mask。
    insulin_aug = insulin.clone()
    dose = insulin_aug
    dose_mask = insulin_mask
    regimen = batch.get("regimen2", None)
    regimen_mask = batch.get("regimen2_mask", None)
    if regimen is None or regimen_mask is None:
        return batch
    premix_days = (regimen[..., 1] > 0.5) & (regimen_mask[..., 1] > 0)
    if not premix_days.any():
        batch["insulin2"] = insulin_aug
        return batch

    valid_nonzero = premix_days.unsqueeze(-1) & (dose_mask > 0) & (dose.abs() > 1e-8)
    if not valid_nonzero.any():
        batch["insulin2"] = insulin_aug
        return batch

    day_apply = (torch.rand_like(premix_days.float()) < aug_prob) & premix_days
    apply_mask = day_apply.unsqueeze(-1) & valid_nonzero
    if not apply_mask.any():
        batch["insulin2"] = insulin_aug
        return batch

    noise = (torch.rand_like(dose) * 2.0 - 1.0) * aug_scale
    scale = 1.0 + noise
    augmented_dose = torch.clamp(dose * scale, min=0.0)
    dose = torch.where(apply_mask, augmented_dose, dose)
    insulin_aug[...] = dose
    batch["insulin2"] = insulin_aug
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
    
    insulin_gt = batch["insulin2"]
    insulin_gt_mask = batch["insulin2_mask"]
    bg_gt = batch["bg2"]
    bg_mask = batch["bg2_mask"]
    
    # 统一截断到最小长度
    min_T = min(insulin_pred.shape[1], insulin_gt.shape[1], bg_pred.shape[1], bg_gt.shape[1])
    
    insulin_pred = insulin_pred[:, :min_T, :]
    insulin_gt = insulin_gt[:, :min_T, :]
    insulin_gt_mask = insulin_gt_mask[:, :min_T, :] if insulin_gt_mask.shape[1] >= min_T else insulin_gt_mask
    bg_pred = bg_pred[:, :min_T, :]
    bg_gt = bg_gt[:, :min_T, :]
    bg_mask = bg_mask[:, :min_T, :] if bg_mask.shape[1] >= min_T else bg_mask
    gt_regimen = batch.get("regimen2")
    gt_regimen_mask = batch.get("regimen2_mask")
    if gt_regimen is not None and gt_regimen.shape[1] >= min_T:
        gt_regimen = gt_regimen[:, :min_T, :]
    if gt_regimen_mask is not None and gt_regimen_mask.shape[1] >= min_T:
        gt_regimen_mask = gt_regimen_mask[:, :min_T, :]
    
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
        task_name="insulin",
        regimen=gt_regimen,
        regimen_mask=gt_regimen_mask,
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

    dose_total_trend_penalty = torch.zeros((), device=device)
    dose_pred = insulin_pred
    dose_gt = insulin_gt
    dose_mask = insulin_mask

    if gt_regimen is not None and gt_regimen_mask is not None:
        gt_valid_label = (gt_regimen_mask.sum(dim=-1) > 0)
        gt_label = torch.full(gt_valid_label.shape, -1, dtype=torch.long, device=device)
        gt_label = torch.where(gt_valid_label & (gt_regimen[..., 1] > 0.5), torch.ones_like(gt_label), gt_label)
        gt_label = torch.where(gt_valid_label & (gt_regimen[..., 0] > 0.5), torch.zeros_like(gt_label), gt_label)
    else:
        gt_valid_label = (dose_mask.sum(dim=-1) > 0)
        gt_label = infer_regimen_labels_from_dose(dose_gt, positive_threshold=float(getattr(cfg, "effective_dose_threshold", 2.0)))

    pred_label = infer_regimen_labels_from_dose(
        dose_pred,
        positive_threshold=float(getattr(cfg, "effective_dose_threshold", 2.0)),
    )

    stats_valid_label = (dose_mask.sum(dim=-1) > 0)
    gt_label_for_stats = torch.where(
        gt_label >= 0,
        gt_label,
        torch.full_like(gt_label, 2),
    )
    gt_label_for_stats = resolve_none_types_following_future(gt_label_for_stats, none_idx=2)

    overall_type_metrics = compute_binary_regimen_metrics(
        pred_label=pred_label,
        gt_label=gt_label_for_stats,
        valid_mask=stats_valid_label.float(),
        class_idx=1,
    )
    loss_dict["insulin_type_acc"] = overall_type_metrics["acc"]

    for class_name, class_idx in [("basic", 0), ("premix", 1)]:
        cls_metrics = compute_binary_regimen_metrics(
            pred_label=pred_label,
            gt_label=gt_label_for_stats,
            valid_mask=stats_valid_label.float(),
            class_idx=class_idx,
        )
        loss_dict[f"insulin_type_acc_{class_name}"] = cls_metrics["recall"]
        loss_dict[f"insulin_type_precision_{class_name}"] = cls_metrics["precision"]
        loss_dict[f"insulin_type_f1_{class_name}"] = cls_metrics["f1"]

    dose_abs_err = (dose_pred - dose_gt).abs()
    valid_dose = dose_mask > 0
    loss_dict["insulin_dose_mae"] = (dose_abs_err * dose_mask).sum().item() / (dose_mask.sum().item() + 1e-8) if valid_dose.any() else 0.0
    overall_dose_metrics = compute_masked_regression_metrics(
        dose_pred,
        dose_gt,
        dose_mask,
    )
    loss_dict["insulin_dose_mape"] = overall_dose_metrics["mape"]
    overall_matched_mask = ((gt_label == pred_label) & (gt_label >= 0)).unsqueeze(-1).float() * dose_mask
    overall_matched_metrics = compute_masked_regression_metrics(
        dose_pred,
        dose_gt,
        overall_matched_mask,
    )
    loss_dict["insulin_dose_mae_matched"] = overall_matched_metrics["mae"]
    loss_dict["insulin_dose_mape_matched"] = overall_matched_metrics["mape"]

    basic_mask = ((gt_label == 0).unsqueeze(-1).float()) * dose_mask
    basic_metrics = compute_masked_regression_metrics(
        dose_pred,
        dose_gt,
        basic_mask,
    )
    loss_dict["insulin_dose_mae_basic"] = basic_metrics["mae"]
    loss_dict["insulin_dose_mape_basic"] = basic_metrics["mape"]
    loss_dict["insulin_dose_rmse_basic"] = basic_metrics["rmse"]
    loss_dict["insulin_dose_r2_basic"] = basic_metrics["r2"]
    basic_matched_mask = ((gt_label == 0) & (pred_label == 0)).unsqueeze(-1).float() * dose_mask
    basic_matched_metrics = compute_masked_regression_metrics(
        dose_pred,
        dose_gt,
        basic_matched_mask,
    )
    loss_dict["insulin_dose_mae_basic_matched"] = basic_matched_metrics["mae"]
    loss_dict["insulin_dose_mape_basic_matched"] = basic_matched_metrics["mape"]
    loss_dict["insulin_dose_rmse_basic_matched"] = basic_matched_metrics["rmse"]
    loss_dict["insulin_dose_r2_basic_matched"] = basic_matched_metrics["r2"]

    premix_mask = ((gt_label == 1).unsqueeze(-1).float()) * dose_mask
    premix_metrics = compute_masked_regression_metrics(
        dose_pred,
        dose_gt,
        premix_mask,
    )
    loss_dict["insulin_dose_mae_premix"] = premix_metrics["mae"]
    loss_dict["insulin_dose_mape_premix"] = premix_metrics["mape"]
    loss_dict["insulin_dose_rmse_premix"] = premix_metrics["rmse"]
    loss_dict["insulin_dose_r2_premix"] = premix_metrics["r2"]
    premix_matched_mask = ((gt_label == 1) & (pred_label == 1)).unsqueeze(-1).float() * dose_mask
    premix_matched_metrics = compute_masked_regression_metrics(
        dose_pred,
        dose_gt,
        premix_matched_mask,
    )
    loss_dict["insulin_dose_mae_premix_matched"] = premix_matched_metrics["mae"]
    loss_dict["insulin_dose_mape_premix_matched"] = premix_matched_metrics["mape"]
    loss_dict["insulin_dose_rmse_premix_matched"] = premix_matched_metrics["rmse"]
    loss_dict["insulin_dose_r2_premix_matched"] = premix_matched_metrics["r2"]

    loss_dict["bg_sc_sum"] = (
        loss_dict.get("insulin_dose_mae_basic", 0.0)
        + loss_dict.get("insulin_dose_mae_premix", 0.0)
        + loss_dict.get("bg_mae", 0.0)
    )

    component_metrics = compute_component_metrics(
        dose_pred=dose_pred,
        dose_gt=dose_gt,
        dose_mask=dose_mask,
        pred_label=pred_label,
        gt_label=gt_label,
        positive_threshold=float(getattr(cfg, "effective_dose_threshold", 2.0)),
    )
    loss_dict.update(component_metrics)

    expanded_metrics = compute_expanded_sc_metrics(
        dose_pred=dose_pred,
        dose_gt=dose_gt,
        dose_mask=dose_mask,
        gt_label=gt_label,
    )
    loss_dict.update(expanded_metrics)

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
                loss_dict["insulin_bg_down_rise_penalty"] = dose_total_trend_penalty.item()
    
    # 计算总loss
    lambda_insulin = float(getattr(cfg, "current_lambda_insulin", getattr(cfg, "lambda_insulin", 1.0)))
    lambda_bg = float(getattr(cfg, "current_lambda_bg", getattr(cfg, "lambda_bg", 1.0)))
    total_loss = lambda_insulin * insulin_loss + lambda_bg * bg_loss
    if bg_down_insulin_rise_penalty_weight > 0:
        total_loss = total_loss + bg_down_insulin_rise_penalty_weight * dose_total_trend_penalty
    
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
    loss_dict['lambda_insulin'] = lambda_insulin
    loss_dict['lambda_bg'] = lambda_bg
    
    return total_loss, loss_dict
    
def train_val_test(cfg):
    run_prefix = f"top{int(getattr(cfg, 'top_k', 0))}_"
    save_best_path = os.path.join(cfg.save_path, f"{run_prefix}{time.strftime('%Y%m%d-%H%M%S')}")
    
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
            os.path.join(train_dir, "..", "tf-logs", os.path.basename(save_best_path))
        )
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_path)
        log_message(f"TensorBoard logging enabled: {tensorboard_path}")
    
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
        loader_kwargs["worker_init_fn"] = seed_worker

    train_sampler = build_train_sampler(train_dataset, cfg)
    if train_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=False,
            generator=make_torch_generator(int(getattr(cfg, "seed", 42))),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=make_torch_generator(int(getattr(cfg, "seed", 42))),
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=make_torch_generator(int(getattr(cfg, "seed", 42)) + 1),
        **loader_kwargs,
    )
    
    cfg.d_insulin = train_dataset[0]["d_insulin"]
    cfg.d_regimen = train_dataset[0]["regimen2"].shape[-1]
    # cfg.d_insulin_route_stage = train_dataset[0]["d_insulin_route_stage"]
    cfg.d_person = train_dataset[0]["d_per1"]
    cfg.d_drug = train_dataset[0]["d_drug"]
    cfg.save_best_path = save_best_path
    if cfg.cut_time != 0:
        cfg.max_T1 = cfg.cut_time
    
    # Model
    model = TwoStageModel(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    rule_stats = load_rule_stats_for_training(cfg, train_dataset)
    log_message(
        "Rule-based regimen stats for validation | "
        f"mu_BB={rule_stats['mu_bb']:.4f}, sigma_BB={rule_stats['sigma_bb']:.4f}, "
        f"mu_PM_AM={rule_stats['mu_pm_am']:.4f}, sigma_PM_AM={rule_stats['sigma_pm_am']:.4f}, "
        f"mu_PM_PM={rule_stats['mu_pm_pm']:.4f}, sigma_PM_PM={rule_stats['sigma_pm_pm']:.4f}, "
        f"n_BB={int(rule_stats.get('n_bb_patients', 0))}, n_PM={int(rule_stats.get('n_pm_patients', 0))}, "
        f"source={rule_stats.get('source', 'unknown')}"
    )
    
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
    use_early_stop = bool(getattr(cfg, 'use_early_stop', 0))
    early_stop_patience = getattr(cfg, 'early_stop_patience', 30)
    early_stop_min_delta = getattr(cfg, 'early_stop_min_delta', 1e-3)
    min_lr_counter = 0
    min_lr = float(getattr(cfg, "scheduler_min_lr", 0.0))
    selection_warmup_epochs = max(0, int(getattr(cfg, "selection_warmup_epochs", 0)))

    if use_early_stop:
        log_message(
            f"Early stopping enabled: patience={early_stop_patience}, "
            f"min_delta={early_stop_min_delta:.6f}"
        )
    else:
        log_message("Early stopping disabled: training will continue to max epochs unless interrupted.")
    scheduler_metric_name = "insulin_bg_sum" if bool(getattr(cfg, "use_two_stage_curriculum", 0)) else "val_loss"
    checkpoint_metric_name = "insulin_bg_sum" if bool(getattr(cfg, "use_two_stage_curriculum", 0)) else "val_loss"
    log_message(f"Scheduler / plateau monitor metric: {scheduler_metric_name}")
    log_message(f"Best checkpoint monitor metric: {checkpoint_metric_name}")
    
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
            writer.add_scalar("train/epoch_lambda_insulin", float(getattr(cfg, "current_lambda_insulin", getattr(cfg, "lambda_insulin", 1.0))), epoch)
            writer.add_scalar("train/epoch_lambda_bg", float(getattr(cfg, "current_lambda_bg", getattr(cfg, "lambda_bg", 1.0))), epoch)
            writer.add_scalar("train/epoch_premix_recall_focus_loss_weight", getattr(criterion, "premix_recall_focus_loss_weight", 0.0), epoch)
            writer.add_scalar("train/epoch_basic_sc_focus_loss_weight", getattr(criterion, "basic_sc_focus_loss_weight", 0.0), epoch)
            writer.add_scalar("train/epoch_premix_sc_focus_loss_weight", getattr(criterion, "premix_sc_focus_loss_weight", 0.0), epoch)
        log_message(f"\n{'='*60}")
        log_message(f"Epoch {epoch} Training Completed")
        log_message(f"Average Training Loss: {avg_train_loss:.4f}")
        log_message(f"Curriculum Phase: {curriculum_phase}")
        log_message(f"Current Lambda Insulin: {float(getattr(cfg, 'current_lambda_insulin', getattr(cfg, 'lambda_insulin', 1.0))):.4f}")
        log_message(f"Current Lambda BG: {float(getattr(cfg, 'current_lambda_bg', getattr(cfg, 'lambda_bg', 1.0))):.4f}")
        log_message(f"Current Insulin Regression Loss Weight: {getattr(criterion, 'insulin_regression_loss_weight', 1.0):.4f}")
        log_message(f"Current Premix Recall Focus Loss Weight: {getattr(criterion, 'premix_recall_focus_loss_weight', 0.0):.4f}")
        log_message(f"Current Basic-SC Focus Loss Weight: {getattr(criterion, 'basic_sc_focus_loss_weight', 0.0):.4f}")
        log_message(f"Current Premix-SC Focus Loss Weight: {getattr(criterion, 'premix_sc_focus_loss_weight', 0.0):.4f}")
        log_message(f"{'='*60}\n")
        
        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        val_loss_components = {
            'insulin': 0.0,
            'rule_prob_type_acc': 0.0,
            'rule_prob_type_acc_basic': 0.0,
            'rule_prob_type_acc_premix': 0.0,
            'rule_prob_type_precision_basic': 0.0,
            'rule_prob_type_precision_premix': 0.0,
            'rule_prob_type_f1_basic': 0.0,
            'rule_prob_type_f1_premix': 0.0,
            'insulin_dose_mae': 0.0,
            'insulin_dose_mape': 0.0,
            'insulin_dose_mae_basic': 0.0,
            'insulin_dose_mape_basic': 0.0,
            'insulin_dose_mae_premix': 0.0,
            'insulin_dose_mape_premix': 0.0,
            'insulin_dose_mae_basic_matched': 0.0,
            'insulin_dose_mape_basic_matched': 0.0,
            'insulin_dose_mae_premix_matched': 0.0,
            'insulin_dose_mape_premix_matched': 0.0,
            'insulin_dose_rmse_basic': 0.0,
            'insulin_dose_rmse_premix': 0.0,
            'insulin_dose_rmse_basic_matched': 0.0,
            'insulin_dose_rmse_premix_matched': 0.0,
            'insulin_dose_r2_basic': 0.0,
            'insulin_dose_r2_premix': 0.0,
            'insulin_dose_r2_basic_matched': 0.0,
            'insulin_dose_r2_premix_matched': 0.0,
            'bg_mae': 0.0,
            'bg_rmse': 0.0,
            'bg_r2': 0.0,
            'bg': 0.0,
            'bread': 0.0,
            'length': 0.0,
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(batch, tf_ratio=0.0, mode="val")
                
                # Calculate validation loss (using unified function)
                batch_loss, loss_dict = compute_batch_loss(batch, outputs, criterion, cfg, mode="val")
                dose_pred, dose_gt, dose_mask, min_T = build_eval_dose_tensors(batch, outputs, device)
                _, gt_label_for_stats, stats_valid_label = build_gt_regimen_labels(
                    batch=batch,
                    dose_gt=dose_gt,
                    dose_mask=dose_mask,
                    min_T=min_T,
                    cfg=cfg,
                    device=device,
                )
                prior_basic, prior_premix = compute_topk_regimen_prior(model, batch, min_T)
                rule_pred_label, _ = infer_rule_probabilistic_regimen_labels(
                    dose_pred,
                    prior_basic,
                    prior_premix,
                    cfg,
                    rule_stats,
                )
                rule_metrics = compute_regimen_classification_metrics(
                    rule_pred_label,
                    gt_label_for_stats,
                    stats_valid_label,
                )
                loss_dict["rule_prob_type_acc"] = rule_metrics["acc"]
                loss_dict["rule_prob_type_acc_basic"] = rule_metrics["recall_basic"]
                loss_dict["rule_prob_type_acc_premix"] = rule_metrics["recall_premix"]
                loss_dict["rule_prob_type_precision_basic"] = rule_metrics["precision_basic"]
                loss_dict["rule_prob_type_precision_premix"] = rule_metrics["precision_premix"]
                loss_dict["rule_prob_type_f1_basic"] = rule_metrics["f1_basic"]
                loss_dict["rule_prob_type_f1_premix"] = rule_metrics["f1_premix"]
                
                val_loss += batch_loss.item()
                
                # Accumulate loss components for reporting
                if 'insulin_diversity_base' in loss_dict:
                    val_loss_components['insulin'] += loss_dict['insulin_diversity_base']
                if 'rule_prob_type_acc' in loss_dict:
                    val_loss_components['rule_prob_type_acc'] += loss_dict['rule_prob_type_acc']
                if 'rule_prob_type_acc_basic' in loss_dict:
                    val_loss_components['rule_prob_type_acc_basic'] += loss_dict['rule_prob_type_acc_basic']
                if 'rule_prob_type_acc_premix' in loss_dict:
                    val_loss_components['rule_prob_type_acc_premix'] += loss_dict['rule_prob_type_acc_premix']
                if 'rule_prob_type_precision_basic' in loss_dict:
                    val_loss_components['rule_prob_type_precision_basic'] += loss_dict['rule_prob_type_precision_basic']
                if 'rule_prob_type_precision_premix' in loss_dict:
                    val_loss_components['rule_prob_type_precision_premix'] += loss_dict['rule_prob_type_precision_premix']
                if 'rule_prob_type_f1_basic' in loss_dict:
                    val_loss_components['rule_prob_type_f1_basic'] += loss_dict['rule_prob_type_f1_basic']
                if 'rule_prob_type_f1_premix' in loss_dict:
                    val_loss_components['rule_prob_type_f1_premix'] += loss_dict['rule_prob_type_f1_premix']
                if 'insulin_dose_mae' in loss_dict:
                    val_loss_components['insulin_dose_mae'] += loss_dict['insulin_dose_mae']
                if 'insulin_dose_mape' in loss_dict:
                    val_loss_components['insulin_dose_mape'] += loss_dict['insulin_dose_mape']
                if 'insulin_dose_mae_basic' in loss_dict:
                    val_loss_components['insulin_dose_mae_basic'] += loss_dict['insulin_dose_mae_basic']
                if 'insulin_dose_mape_basic' in loss_dict:
                    val_loss_components['insulin_dose_mape_basic'] += loss_dict['insulin_dose_mape_basic']
                if 'insulin_dose_mae_premix' in loss_dict:
                    val_loss_components['insulin_dose_mae_premix'] += loss_dict['insulin_dose_mae_premix']
                if 'insulin_dose_mape_premix' in loss_dict:
                    val_loss_components['insulin_dose_mape_premix'] += loss_dict['insulin_dose_mape_premix']
                if 'insulin_dose_mae_basic_matched' in loss_dict:
                    val_loss_components['insulin_dose_mae_basic_matched'] += loss_dict['insulin_dose_mae_basic_matched']
                if 'insulin_dose_mape_basic_matched' in loss_dict:
                    val_loss_components['insulin_dose_mape_basic_matched'] += loss_dict['insulin_dose_mape_basic_matched']
                if 'insulin_dose_mae_premix_matched' in loss_dict:
                    val_loss_components['insulin_dose_mae_premix_matched'] += loss_dict['insulin_dose_mae_premix_matched']
                if 'insulin_dose_mape_premix_matched' in loss_dict:
                    val_loss_components['insulin_dose_mape_premix_matched'] += loss_dict['insulin_dose_mape_premix_matched']
                if 'insulin_dose_rmse_basic' in loss_dict:
                    val_loss_components['insulin_dose_rmse_basic'] += loss_dict['insulin_dose_rmse_basic']
                if 'insulin_dose_rmse_premix' in loss_dict:
                    val_loss_components['insulin_dose_rmse_premix'] += loss_dict['insulin_dose_rmse_premix']
                if 'insulin_dose_rmse_basic_matched' in loss_dict:
                    val_loss_components['insulin_dose_rmse_basic_matched'] += loss_dict['insulin_dose_rmse_basic_matched']
                if 'insulin_dose_rmse_premix_matched' in loss_dict:
                    val_loss_components['insulin_dose_rmse_premix_matched'] += loss_dict['insulin_dose_rmse_premix_matched']
                if 'insulin_dose_r2_basic' in loss_dict:
                    val_loss_components['insulin_dose_r2_basic'] += loss_dict['insulin_dose_r2_basic']
                if 'insulin_dose_r2_premix' in loss_dict:
                    val_loss_components['insulin_dose_r2_premix'] += loss_dict['insulin_dose_r2_premix']
                if 'insulin_dose_r2_basic_matched' in loss_dict:
                    val_loss_components['insulin_dose_r2_basic_matched'] += loss_dict['insulin_dose_r2_basic_matched']
                if 'insulin_dose_r2_premix_matched' in loss_dict:
                    val_loss_components['insulin_dose_r2_premix_matched'] += loss_dict['insulin_dose_r2_premix_matched']
                for metric_name in [
                    'bg_mae', 'bg_rmse', 'bg_r2',
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
        avg_insulin_loss = val_loss_components['insulin'] / len(val_loader)
        avg_rule_prob_type_acc = val_loss_components['rule_prob_type_acc'] / len(val_loader)
        avg_rule_prob_type_acc_basic = val_loss_components['rule_prob_type_acc_basic'] / len(val_loader)
        avg_rule_prob_type_acc_premix = val_loss_components['rule_prob_type_acc_premix'] / len(val_loader)
        avg_rule_prob_type_precision_basic = val_loss_components['rule_prob_type_precision_basic'] / len(val_loader)
        avg_rule_prob_type_precision_premix = val_loss_components['rule_prob_type_precision_premix'] / len(val_loader)
        avg_rule_prob_type_f1_basic = val_loss_components['rule_prob_type_f1_basic'] / len(val_loader)
        avg_rule_prob_type_f1_premix = val_loss_components['rule_prob_type_f1_premix'] / len(val_loader)
        avg_insulin_dose_mae = val_loss_components['insulin_dose_mae'] / len(val_loader)
        avg_insulin_dose_mape = val_loss_components['insulin_dose_mape'] / len(val_loader)
        avg_insulin_dose_mae_basic = val_loss_components['insulin_dose_mae_basic'] / len(val_loader)
        avg_insulin_dose_mape_basic = val_loss_components['insulin_dose_mape_basic'] / len(val_loader)
        avg_insulin_dose_mae_premix = val_loss_components['insulin_dose_mae_premix'] / len(val_loader)
        avg_insulin_dose_mape_premix = val_loss_components['insulin_dose_mape_premix'] / len(val_loader)
        avg_insulin_dose_mae_basic_matched = val_loss_components['insulin_dose_mae_basic_matched'] / len(val_loader)
        avg_insulin_dose_mape_basic_matched = val_loss_components['insulin_dose_mape_basic_matched'] / len(val_loader)
        avg_insulin_dose_mae_premix_matched = val_loss_components['insulin_dose_mae_premix_matched'] / len(val_loader)
        avg_insulin_dose_mape_premix_matched = val_loss_components['insulin_dose_mape_premix_matched'] / len(val_loader)
        avg_insulin_dose_rmse_basic = val_loss_components['insulin_dose_rmse_basic'] / len(val_loader)
        avg_insulin_dose_rmse_premix = val_loss_components['insulin_dose_rmse_premix'] / len(val_loader)
        avg_insulin_dose_rmse_basic_matched = val_loss_components['insulin_dose_rmse_basic_matched'] / len(val_loader)
        avg_insulin_dose_rmse_premix_matched = val_loss_components['insulin_dose_rmse_premix_matched'] / len(val_loader)
        avg_insulin_dose_r2_basic = val_loss_components['insulin_dose_r2_basic'] / len(val_loader)
        avg_insulin_dose_r2_premix = val_loss_components['insulin_dose_r2_premix'] / len(val_loader)
        avg_insulin_dose_r2_basic_matched = val_loss_components['insulin_dose_r2_basic_matched'] / len(val_loader)
        avg_insulin_dose_r2_premix_matched = val_loss_components['insulin_dose_r2_premix_matched'] / len(val_loader)
        avg_bg_loss = val_loss_components['bg'] / len(val_loader)
        avg_bg_mae = val_loss_components['bg_mae'] / len(val_loader)
        avg_bg_rmse = val_loss_components['bg_rmse'] / len(val_loader)
        avg_bg_r2 = val_loss_components['bg_r2'] / len(val_loader)
        insulin_bg_sum = avg_bg_mae + avg_insulin_dose_mae
        dose_balanced_mae = compute_balanced_regimen_dose_metric(
            avg_insulin_dose_mae_basic,
            avg_insulin_dose_mae_premix,
        )
        premix_recall_penalty = 0.0
        if scheduler_metric_name == "insulin_bg_sum":
            scheduler_monitor_value = insulin_bg_sum
        else:
            scheduler_monitor_value = avg_val_loss
        if checkpoint_metric_name == "insulin_bg_premix_guard":
            monitor_value, _, premix_recall_penalty = compute_selection_metric_with_regression_objective(
                insulin_bg_sum,
                avg_rule_prob_type_acc_premix,
                cfg,
            )
        elif checkpoint_metric_name == "insulin_bg_sum":
            monitor_value = insulin_bg_sum
        else:
            monitor_value = avg_val_loss

        # Print validation results
        log_message(f"\n{'='*60}")
        log_message(f"Epoch {epoch} Validation Completed")
        log_message(f"Average Validation Loss: {avg_val_loss:.4f}")
        log_message(f"  - Rule-Prob Type Recall Basic/Premix: {avg_rule_prob_type_acc_basic:.4f} / {avg_rule_prob_type_acc_premix:.4f}")
        log_message(f"  - Rule-Prob Type Accuracy (overall): {avg_rule_prob_type_acc:.4f}")
        log_message(f"  - Rule-Prob Type Precision Basic/Premix: {avg_rule_prob_type_precision_basic:.4f} / {avg_rule_prob_type_precision_premix:.4f}")
        log_message(f"  - Rule-Prob Type F1 Basic/Premix: {avg_rule_prob_type_f1_basic:.4f} / {avg_rule_prob_type_f1_premix:.4f}")
        log_message(f"  - Insulin Dose MAE/MAPE (overall): {avg_insulin_dose_mae:.4f} / {avg_insulin_dose_mape:.2f}%")
        log_message(f"  - Basic Dose MAE/MAPE/RMSE/R2 (gt days, 8d): {avg_insulin_dose_mae_basic:.4f} / {avg_insulin_dose_mape_basic:.2f}% / {avg_insulin_dose_rmse_basic:.4f} / {avg_insulin_dose_r2_basic:.4f}")
        log_message(f"  - Premix Dose MAE/MAPE/RMSE/R2 (gt days, 8d): {avg_insulin_dose_mae_premix:.4f} / {avg_insulin_dose_mape_premix:.2f}% / {avg_insulin_dose_rmse_premix:.4f} / {avg_insulin_dose_r2_premix:.4f}")
        log_message(f"  - Basic Dose MAE/MAPE/RMSE/R2 (matched days, 8d): {avg_insulin_dose_mae_basic_matched:.4f} / {avg_insulin_dose_mape_basic_matched:.2f}% / {avg_insulin_dose_rmse_basic_matched:.4f} / {avg_insulin_dose_r2_basic_matched:.4f}")
        log_message(f"  - Premix Dose MAE/MAPE/RMSE/R2 (matched days, 8d): {avg_insulin_dose_mae_premix_matched:.4f} / {avg_insulin_dose_mape_premix_matched:.2f}% / {avg_insulin_dose_rmse_premix_matched:.4f} / {avg_insulin_dose_r2_premix_matched:.4f}")
        log_message(f"  - BG MAE/RMSE/R2: {avg_bg_mae:.4f} / {avg_bg_rmse:.4f} / {avg_bg_r2:.4f}")
        if premix_recall_penalty > 0:
            log_message(f"  - Premix Recall Penalty: {premix_recall_penalty:.4f}")
        if val_loss_components['bread'] > 0:
            log_message(f"  - Discharge Loss: {val_loss_components['bread'] / len(val_loader):.4f}")
        if val_loss_components['length'] > 0:
            log_message(f"  - Length Loss: {val_loss_components['length'] / len(val_loader):.4f}")

        # Update learning rate scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(scheduler_monitor_value)
            else:
                warmup_epochs = max(0, int(getattr(cfg, "warmup_epochs", 0)))
                if epoch >= warmup_epochs:
                    scheduler.step()

        # Learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar("val/current_val_loss", avg_val_loss, epoch)
            writer.add_scalar("val/learning_rate", current_lr, epoch)
            writer.add_scalar("val/selection_metric", monitor_value, epoch)
            writer.add_scalar("val/bg_mae", avg_bg_mae, epoch)
            writer.add_scalar("val/insulin_dose_mae", avg_insulin_dose_mae, epoch)
            writer.add_scalar("val/basic_dose_mae", avg_insulin_dose_mae_basic, epoch)
            writer.add_scalar("val/premix_dose_mae", avg_insulin_dose_mae_premix, epoch)
            writer.add_scalar("val/basic_dose_mae_matched", avg_insulin_dose_mae_basic_matched, epoch)
            writer.add_scalar("val/premix_dose_mae_matched", avg_insulin_dose_mae_premix_matched, epoch)
            writer.add_scalar("val/basic_dose_rmse", avg_insulin_dose_rmse_basic, epoch)
            writer.add_scalar("val/premix_dose_rmse", avg_insulin_dose_rmse_premix, epoch)
            writer.add_scalar("val/basic_dose_r2", avg_insulin_dose_r2_basic, epoch)
            writer.add_scalar("val/premix_dose_r2", avg_insulin_dose_r2_premix, epoch)
            writer.add_scalar("val/rule_prob_type_acc", avg_rule_prob_type_acc, epoch)
            writer.add_scalar("val/rule_prob_type_acc_basic", avg_rule_prob_type_acc_basic, epoch)
            writer.add_scalar("val/rule_prob_type_acc_premix", avg_rule_prob_type_acc_premix, epoch)
            writer.add_scalar("val/rule_prob_type_precision_basic", avg_rule_prob_type_precision_basic, epoch)
            writer.add_scalar("val/rule_prob_type_precision_premix", avg_rule_prob_type_precision_premix, epoch)
            writer.add_scalar("val/rule_prob_type_f1_basic", avg_rule_prob_type_f1_basic, epoch)
            writer.add_scalar("val/rule_prob_type_f1_premix", avg_rule_prob_type_f1_premix, epoch)
            writer.add_scalar("val/premix_recall_penalty", premix_recall_penalty, epoch)
        log_message(f"Current Selection Metric ({checkpoint_metric_name}): {monitor_value:.4f}")
        log_message(f"Current Learning Rate: {current_lr:.8f}")
        log_message(f"{'='*60}\n")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        eligible_for_best = epoch >= selection_warmup_epochs
        improved = eligible_for_best and (monitor_value < best_monitor_value)
        meaningful_improved = monitor_value < (best_es_monitor_value - early_stop_min_delta)
        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_metrics = {
                "selection_metric_name": checkpoint_metric_name,
                "selection_metric": monitor_value,
                "selection_insulin_bg_sum": insulin_bg_sum,
                "selection_dose_balanced_mae": dose_balanced_mae,
                "premix_recall_penalty": premix_recall_penalty,
                "val_loss": avg_val_loss,
                "insulin_loss": avg_insulin_loss,
                "rule_prob_type_acc": avg_rule_prob_type_acc,
                "rule_prob_type_acc_basic": avg_rule_prob_type_acc_basic,
                "rule_prob_type_acc_premix": avg_rule_prob_type_acc_premix,
                "rule_prob_type_precision_basic": avg_rule_prob_type_precision_basic,
                "rule_prob_type_precision_premix": avg_rule_prob_type_precision_premix,
                "rule_prob_type_f1_basic": avg_rule_prob_type_f1_basic,
                "rule_prob_type_f1_premix": avg_rule_prob_type_f1_premix,
                "insulin_dose_mae": avg_insulin_dose_mae,
                "insulin_dose_mape": avg_insulin_dose_mape,
                "insulin_dose_mae_basic": avg_insulin_dose_mae_basic,
                "insulin_dose_mape_basic": avg_insulin_dose_mape_basic,
                "insulin_dose_mae_premix": avg_insulin_dose_mae_premix,
                "insulin_dose_mape_premix": avg_insulin_dose_mape_premix,
                "insulin_dose_mae_basic_matched": avg_insulin_dose_mae_basic_matched,
                "insulin_dose_mape_basic_matched": avg_insulin_dose_mape_basic_matched,
                "insulin_dose_mae_premix_matched": avg_insulin_dose_mae_premix_matched,
                "insulin_dose_mape_premix_matched": avg_insulin_dose_mape_premix_matched,
                "insulin_dose_rmse_basic": avg_insulin_dose_rmse_basic,
                "insulin_dose_rmse_premix": avg_insulin_dose_rmse_premix,
                "insulin_dose_rmse_basic_matched": avg_insulin_dose_rmse_basic_matched,
                "insulin_dose_rmse_premix_matched": avg_insulin_dose_rmse_premix_matched,
                "insulin_dose_r2_basic": avg_insulin_dose_r2_basic,
                "insulin_dose_r2_premix": avg_insulin_dose_r2_premix,
                "insulin_dose_r2_basic_matched": avg_insulin_dose_r2_basic_matched,
                "insulin_dose_r2_premix_matched": avg_insulin_dose_r2_premix_matched,
                "bg_loss": avg_bg_loss,
                "bg_mae": avg_bg_mae,
                "bg_rmse": avg_bg_rmse,
                "bg_r2": avg_bg_r2,
                "learning_rate": current_lr,
            }
            torch.save(model.state_dict(), os.path.join(save_best_path, "model.pth"))
            log_message(
                f"[SAVED] Best model (Epoch {epoch}, {checkpoint_metric_name}: {monitor_value:.4f}, Val Loss: {avg_val_loss:.4f})\n"
            )
        elif epoch == 0 and selection_warmup_epochs > 0:
            log_message(
                f"Selection warmup enabled: best checkpoint tracking starts at epoch {selection_warmup_epochs}."
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

        if use_early_stop and early_stop_counter >= early_stop_patience:
            log_message(f"[EARLY STOP] No validation improvement greater than {early_stop_min_delta:.4f} for {early_stop_patience} epochs.")
            break
 
        if use_early_stop and getattr(cfg, "stop_on_min_lr", 0) and min_lr > 0 and min_lr_counter >= getattr(cfg, "min_lr_patience", 8):
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

        log_message("\n" + "=" * 60)
        log_message(f"Best Validation Summary (Epoch {best_epoch})")
        if "selection_metric" in best_metrics:
            log_message(f"  - Selection Metric ({best_metrics.get('selection_metric_name', 'selection_metric')}): {best_metrics.get('selection_metric', float('nan')):.4f}")
        log_message(f"  - Best Val Loss: {best_metrics.get('val_loss', float('nan')):.4f}")
        log_message(f"  - Rule-Prob Type Recall Basic/Premix: {best_metrics.get('rule_prob_type_acc_basic', float('nan')):.4f} / {best_metrics.get('rule_prob_type_acc_premix', float('nan')):.4f}")
        log_message(f"  - Rule-Prob Type Accuracy (overall): {best_metrics.get('rule_prob_type_acc', float('nan')):.4f}")
        log_message(f"  - Rule-Prob Type Precision Basic/Premix: {best_metrics.get('rule_prob_type_precision_basic', float('nan')):.4f} / {best_metrics.get('rule_prob_type_precision_premix', float('nan')):.4f}")
        log_message(f"  - Rule-Prob Type F1 Basic/Premix: {best_metrics.get('rule_prob_type_f1_basic', float('nan')):.4f} / {best_metrics.get('rule_prob_type_f1_premix', float('nan')):.4f}")
        log_message(f"  - Insulin Dose MAE/MAPE (overall): {best_metrics.get('insulin_dose_mae', float('nan')):.4f} / {best_metrics.get('insulin_dose_mape', float('nan')):.2f}%")
        log_message(f"    Insulin Dose 95%CI: MAE {format_ci(*best_ci.get('insulin_dose_mae', (float('nan'), float('nan'))))} | MAPE {format_ci(*best_ci.get('insulin_dose_mape', (float('nan'), float('nan'))))}")
        log_message(
            f"  - Basic Dose MAE/MAPE/RMSE/R2 (gt days, 8d): "
            f"{best_metrics.get('insulin_dose_mae_basic', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_mape_basic', float('nan')):.2f}% / "
            f"{best_metrics.get('insulin_dose_rmse_basic', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_r2_basic', float('nan')):.4f}"
        )
        log_message(
            f"    Basic Dose 95%CI: MAE {format_ci(*best_ci.get('insulin_dose_mae_basic', (float('nan'), float('nan'))))} | "
            f"MAPE {format_ci(*best_ci.get('insulin_dose_mape_basic', (float('nan'), float('nan'))))} | "
            f"RMSE {format_ci(*best_ci.get('insulin_dose_rmse_basic', (float('nan'), float('nan'))))} | "
            f"R2 {format_ci(*best_ci.get('insulin_dose_r2_basic', (float('nan'), float('nan'))))}"
        )
        log_message(
            f"  - Premix Dose MAE/MAPE/RMSE/R2 (gt days, 8d): "
            f"{best_metrics.get('insulin_dose_mae_premix', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_mape_premix', float('nan')):.2f}% / "
            f"{best_metrics.get('insulin_dose_rmse_premix', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_r2_premix', float('nan')):.4f}"
        )
        log_message(
            f"    Premix Dose 95%CI: MAE {format_ci(*best_ci.get('insulin_dose_mae_premix', (float('nan'), float('nan'))))} | "
            f"MAPE {format_ci(*best_ci.get('insulin_dose_mape_premix', (float('nan'), float('nan'))))} | "
            f"RMSE {format_ci(*best_ci.get('insulin_dose_rmse_premix', (float('nan'), float('nan'))))} | "
            f"R2 {format_ci(*best_ci.get('insulin_dose_r2_premix', (float('nan'), float('nan'))))}"
        )
        log_message(
            f"  - Basic Dose MAE/MAPE/RMSE/R2 (matched days, 8d): "
            f"{best_metrics.get('insulin_dose_mae_basic_matched', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_mape_basic_matched', float('nan')):.2f}% / "
            f"{best_metrics.get('insulin_dose_rmse_basic_matched', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_r2_basic_matched', float('nan')):.4f}"
        )
        log_message(
            f"    Basic Dose Matched 95%CI: MAE {format_ci(*best_ci.get('insulin_dose_mae_basic_matched', (float('nan'), float('nan'))))} | "
            f"MAPE {format_ci(*best_ci.get('insulin_dose_mape_basic_matched', (float('nan'), float('nan'))))} | "
            f"RMSE {format_ci(*best_ci.get('insulin_dose_rmse_basic_matched', (float('nan'), float('nan'))))} | "
            f"R2 {format_ci(*best_ci.get('insulin_dose_r2_basic_matched', (float('nan'), float('nan'))))}"
        )
        log_message(
            f"  - Premix Dose MAE/MAPE/RMSE/R2 (matched days, 8d): "
            f"{best_metrics.get('insulin_dose_mae_premix_matched', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_mape_premix_matched', float('nan')):.2f}% / "
            f"{best_metrics.get('insulin_dose_rmse_premix_matched', float('nan')):.4f} / "
            f"{best_metrics.get('insulin_dose_r2_premix_matched', float('nan')):.4f}"
        )
        log_message(
            f"    Premix Dose Matched 95%CI: MAE {format_ci(*best_ci.get('insulin_dose_mae_premix_matched', (float('nan'), float('nan'))))} | "
            f"MAPE {format_ci(*best_ci.get('insulin_dose_mape_premix_matched', (float('nan'), float('nan'))))} | "
            f"RMSE {format_ci(*best_ci.get('insulin_dose_rmse_premix_matched', (float('nan'), float('nan'))))} | "
            f"R2 {format_ci(*best_ci.get('insulin_dose_r2_premix_matched', (float('nan'), float('nan'))))}"
        )
        log_message(f"  - BG MAE/RMSE/R2: {best_metrics.get('bg_mae', float('nan')):.4f} / {best_metrics.get('bg_rmse', float('nan')):.4f} / {best_metrics.get('bg_r2', float('nan')):.4f}")
        log_message(f"    BG 95%CI: MAE {format_ci(*best_ci.get('bg_mae', (float('nan'), float('nan'))))} | RMSE {format_ci(*best_ci.get('bg_rmse', (float('nan'), float('nan'))))} | R2 {format_ci(*best_ci.get('bg_r2', (float('nan'), float('nan'))))}")
        if best_metrics.get('premix_recall_penalty', 0.0) > 0:
            log_message(f"  - Premix Recall Penalty: {best_metrics.get('premix_recall_penalty', 0.0):.4f}")
        log_message(f"  - Learning Rate: {best_metrics.get('learning_rate', float('nan')):.8f}")
        log_message("=" * 60 + "\n")


if __name__ == "__main__":
    cfg = opt_config()
    train_val_test(cfg)
