# -*- encoding: utf-8 -*-
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from configs.Config import opt_config
from data.dataloader2 import DiabetesDataset, collate_fn
from data.MedicalDBPreprocessor import MedicalDBPreprocessor
from models.model import TwoStageModel
from utils.logger import set_seed
from utils.loss import CombinedLoss
from train import (
    calculate_real_lengths_from_mask,
    compute_batch_loss,
    compute_binary_regimen_metrics,
    compute_masked_regression_metrics,
    create_sequence_mask,
    ensure_mask_dimensions,
    infer_regimen_labels_from_dose,
    resolve_none_types_following_future,
)
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


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


CARD_PAGE_BG = (250, 248, 243)
CARD_HEADER_BG = (222, 233, 244)
CARD_SECTION_BG = (234, 240, 230)
CARD_GRID = (170, 178, 184)
CARD_TEXT = (32, 39, 45)
CARD_MUTED = (98, 109, 118)
CARD_ALT_BG = (245, 248, 251)
CARD_WHITE = (255, 255, 255)


def _load_card_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _measure_multiline(draw, text, font, spacing):
    if not text:
        return font.size
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def _draw_card_table(draw, left, top, title, headers, rows, cell_width, header_height, row_height, title_font, header_font, body_font, note=""):
    table_width = cell_width * len(headers)
    draw.rounded_rectangle((left, top, left + table_width, top + 42), radius=10, fill=CARD_SECTION_BG)
    draw.text((left + 14, top + 9), title, font=title_font, fill=CARD_TEXT)
    y = top + 54
    if note:
        draw.text((left + 2, y), note, font=body_font, fill=CARD_MUTED)
        y += 28

    for col_idx, header in enumerate(headers):
        x0 = left + col_idx * cell_width
        x1 = x0 + cell_width
        draw.rectangle((x0, y, x1, y + header_height), fill=CARD_HEADER_BG, outline=CARD_GRID, width=1)
        bbox = draw.textbbox((0, 0), header, font=header_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x0 + (cell_width - text_w) / 2, y + (header_height - text_h) / 2 - 1), header, font=header_font, fill=CARD_TEXT)
    y += header_height

    spacing = 4
    for row_idx, row in enumerate(rows):
        bg = CARD_WHITE if row_idx % 2 == 0 else CARD_ALT_BG
        actual_row_height = row_height
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue
            actual_row_height = max(actual_row_height, _measure_multiline(draw, cell, body_font, spacing) + 20)
        for col_idx, cell in enumerate(row):
            x0 = left + col_idx * cell_width
            x1 = x0 + cell_width
            draw.rectangle((x0, y, x1, y + actual_row_height), fill=bg, outline=CARD_GRID, width=1)
            font = header_font if col_idx == 0 else body_font
            fill = CARD_TEXT if col_idx == 0 else CARD_MUTED
            if col_idx == 0:
                bbox = draw.textbbox((0, 0), cell, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text((x0 + (cell_width - text_w) / 2, y + (actual_row_height - text_h) / 2 - 1), cell, font=font, fill=fill)
            else:
                draw.multiline_text((x0 + 10, y + 10), cell, font=font, fill=fill, spacing=spacing, align="left")
        y += actual_row_height
    return y


def _format_pair_cell(gt_value, pred_value):
    return f"{float(gt_value):.1f}\n{float(pred_value):.1f}"


def _build_pair_rows(target_array, pred_array, row_labels):
    day_count = min(target_array.shape[0], pred_array.shape[0])
    headers = ["Day"] + row_labels[:target_array.shape[1]]
    rows = []
    for day_idx in range(day_count):
        row = [f"D{day_idx + 1}"]
        for col_idx in range(target_array.shape[1]):
            row.append(_format_pair_cell(target_array[day_idx, col_idx], pred_array[day_idx, col_idx]))
        rows.append(row)
    return headers, rows


def save_prediction_case_card(sample, predictions, save_path):
    target_bg = sample["bg2"].numpy()
    target_insulin = sample["insulin2"].numpy()
    pred_bg = predictions["bg_prediction"][0]
    pred_insulin = predictions["insulin_prediction"][0]

    total_days = min(target_bg.shape[0], target_insulin.shape[0], pred_bg.shape[0], pred_insulin.shape[0])
    target_bg = target_bg[:total_days]
    pred_bg = pred_bg[:total_days]
    target_insulin = target_insulin[:total_days]
    pred_insulin = pred_insulin[:total_days]

    bg_labels = ["BG_0", "BG_1", "BG_2", "BG_3", "BG_4", "BG_5", "BG_6"]
    if target_insulin.shape[1] == 5:
        insulin_labels = ["Premix_Flag", "Dose_B", "Dose_L", "Dose_D", "Dose_Long"]
    elif target_insulin.shape[1] == 8:
        insulin_labels = ["Basic_B", "Basic_L", "Basic_D", "Basic_N", "Premix_B_Long", "Premix_B_Short", "Premix_D_Long", "Premix_D_Short"]
    else:
        insulin_labels = [f"Ins_{i}" for i in range(target_insulin.shape[1])]

    bg_headers, bg_rows = _build_pair_rows(target_bg, pred_bg, bg_labels)
    insulin_headers, insulin_rows = _build_pair_rows(target_insulin, pred_insulin, insulin_labels)

    title_font = _load_card_font(22, bold=True)
    header_font = _load_card_font(18, bold=True)
    body_font = _load_card_font(16, bold=False)
    note_font = _load_card_font(15, bold=False)

    cell_width = 126
    header_height = 38
    base_row_height = 52
    left = 28
    top = 28
    width = max(len(bg_headers), len(insulin_headers)) * cell_width + left * 2
    temp_img = Image.new("RGB", (width, 2200), CARD_PAGE_BG)
    temp_draw = ImageDraw.Draw(temp_img)
    y = top
    temp_draw.rounded_rectangle((left, y, width - left, y + 56), radius=14, fill=CARD_HEADER_BG)
    temp_draw.text((left + 18, y + 12), f"Case {sample['check_id']}", font=title_font, fill=CARD_TEXT)
    y += 76
    y = _draw_card_table(temp_draw, left, y, "Blood Glucose", bg_headers, bg_rows, cell_width, header_height, base_row_height, title_font, header_font, body_font, note="Each cell: ground truth on top, prediction below")
    y += 26
    y = _draw_card_table(temp_draw, left, y, "Insulin", insulin_headers, insulin_rows, cell_width, header_height, base_row_height, title_font, header_font, body_font)
    y += 28
    temp_draw.text((left, y), "Cell format: GT / Pred (stacked vertically)", font=note_font, fill=CARD_MUTED)
    final_height = y + 40
    card = temp_img.crop((0, 0, width, final_height))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    card.save(save_path)


def extract_retrieval_topk_details(model, batch):
    with torch.no_grad():
        topk_indices, topk_scores = model._retrieve_topk_indices(batch)
        batch_size = topk_indices.shape[0]
        topk = topk_indices.shape[1]
        device = batch["person_value"].device
        reference_batch = model._build_reference_batch(topk_indices)
        reference_batch = model._move_batch_to_device(reference_batch, device)
        reference_outputs = model._encode_reference_batch(reference_batch, batch_size, topk)
        reference_encoder = model.stage1 if hasattr(model, "stage1") else model.reference_encoder
        current_person_emb = reference_encoder.encode_person(
            batch["person_value"], batch["person_mask"]
        )
        refer_person_emb = reference_outputs["person_emb"]
        mapped_bg_states = model._map_reference_state_sequence(
            reference_outputs["bg_states"],
            refer_person_emb,
            current_person_emb,
            model.bg_Mapper,
            model.bg_memory_map_gate_logit,
        )
        mapped_tr_states = model._map_reference_state_sequence(
            reference_outputs["tr_states"],
            refer_person_emb,
            current_person_emb,
            model.tr_Mapper,
            model.tr_memory_map_gate_logit,
        )

    topk_scores_np = topk_scores.detach().cpu().numpy()
    temperature = max(float(getattr(model.cfg, "memory_fusion_temperature", 1.0)), 1e-6)
    topk_weights_np = torch.softmax(topk_scores.float() / temperature, dim=1).detach().cpu().numpy()
    map_bg_norm_np = torch.norm(mapped_bg_states, dim=-1).mean(dim=-1).detach().cpu().numpy()
    map_tr_norm_np = torch.norm(mapped_tr_states, dim=-1).mean(dim=-1).detach().cpu().numpy()

    details = []
    for b in range(topk_indices.shape[0]):
        raw_contrib = topk_weights_np[b] * ((map_bg_norm_np[b] + map_tr_norm_np[b]) / 2.0)
        if raw_contrib.sum() > 0:
            contribution_strength = raw_contrib / raw_contrib.sum()
        else:
            contribution_strength = np.full_like(raw_contrib, 1.0 / max(len(raw_contrib), 1))

        sample_details = []
        for rank, kb_index in enumerate(topk_indices[b].detach().cpu().tolist(), start=1):
            retrieved_sample = model._knowledge_base[kb_index]
            sample_details.append(
                {
                    "rank": rank,
                    "check_id": retrieved_sample["check_id"],
                    "similarity_score": float(topk_scores_np[b, rank - 1]),
                    "similarity_weight": float(topk_weights_np[b, rank - 1]),
                    "contribution_strength": float(contribution_strength[rank - 1]),
                }
            )
        details.append(sample_details)
    return details


def print_retrieval_topk_details(sample_ids, retrieval_details):
    for sample_idx, items in enumerate(retrieval_details):
        sample_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else f"sample_{sample_idx}"
        print(f"[Retrieval TopK] query_id={sample_id}")
        for item in items:
            print(
                f"  top{item['rank']:>2}: id={item['check_id']} | "
                f"similarity={item['similarity_score']:.6f} | "
                f"weight={item['similarity_weight']:.6f} | "
                f"contribution_strength={item['contribution_strength']:.6f}"
            )


def format_ci(low, high):
    if not np.isfinite(low) or not np.isfinite(high):
        return "[nan, nan]"
    return f"[{low:.4f}, {high:.4f}]"


def hydrate_cfg_from_checkpoint(cfg, state_dict):
    person_proj_weight = state_dict.get("stage1.person_value2embedding.0.weight")
    if person_proj_weight is not None and person_proj_weight.ndim == 2:
        cfg.d_person = int(person_proj_weight.shape[1] // 2)

    insulin_proj_weight = state_dict.get("stage1.treatment_value2embedding.0.weight")
    if insulin_proj_weight is not None and insulin_proj_weight.ndim == 2:
        cfg.d_insulin = int(insulin_proj_weight.shape[1] // 2)

    drug_proj_weight = state_dict.get("stage1.drug_value2embedding.0.weight")
    if drug_proj_weight is not None and drug_proj_weight.ndim == 2:
        cfg.d_drug = int(drug_proj_weight.shape[1])

    topk_weight = state_dict.get("bg_map_aggregator.output_proj.0.weight")
    if topk_weight is not None and topk_weight.ndim == 2:
        d_model = int(getattr(cfg, "d_model", topk_weight.shape[0]))
        in_features = int(topk_weight.shape[1])
        if d_model > 0 and in_features % d_model == 0:
            cfg.top_k = max(1, in_features // d_model)

    return cfg


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


def load_rule_stats(cfg):
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
    if not stats_path or not os.path.exists(stats_path):
        try:
            train_dataset = DiabetesDataset(cfg, mode="train")
            computed = compute_rule_stats_from_dataset(
                train_dataset,
                effective_dose_threshold=float(getattr(cfg, "effective_dose_threshold", 2.0)),
            )
            return computed
        except Exception:
            return default_stats

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
    metrics = {
        "acc": overall["acc"],
    }
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


def _to_float_tensor(value, name, expected_dim=None):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if expected_dim is not None and tensor.dim() != expected_dim:
        raise ValueError(f"{name} must have {expected_dim} dims, got shape {tuple(tensor.shape)}")
    return tensor


def _infer_model_drug_dim(model):
    try:
        return int(model.stage2.drug_encoder[0].in_features)
    except Exception:
        return None


def _build_repeated_drug_sequence(drug_value, horizon, drug_dim):
    if drug_value is None:
        return torch.zeros(horizon, drug_dim, dtype=torch.float32)

    drug_tensor = torch.as_tensor(drug_value, dtype=torch.float32)
    if drug_tensor.dim() == 1:
        if drug_tensor.shape[0] != drug_dim:
            raise ValueError(f"drug2 vector dim mismatch: expected {drug_dim}, got {drug_tensor.shape[0]}")
        return drug_tensor.unsqueeze(0).repeat(horizon, 1)

    if drug_tensor.dim() == 2:
        if drug_tensor.shape[1] != drug_dim:
            raise ValueError(f"drug2 feature dim mismatch: expected {drug_dim}, got {drug_tensor.shape[1]}")
        if drug_tensor.shape[0] == 1:
            return drug_tensor.repeat(horizon, 1)
        if drug_tensor.shape[0] == horizon:
            return drug_tensor
        raise ValueError(f"drug2 sequence length must be 1 or {horizon}, got {drug_tensor.shape[0]}")

    raise ValueError(f"drug2 must be 1D or 2D, got shape {tuple(drug_tensor.shape)}")


def _setup_inference_service_logger(log_dir, logger_name="inference_service"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    log_file = os.path.join(log_dir, "inference_service.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_file


def build_service_sample(cfg, payload, model=None):
    """
    Build a single-sample dict compatible with the model's current inference path.

    Required payload fields:
    - person_value
    - c_pep_value
    - bg1
    - bg1_mask
    - insulin1
    - insulin1_mask

    Optional:
    - person_mask
    - c_pep_mask
    - drug2: one daily drug vector; it will be repeated across predicted days
    - prediction_horizon: placeholder horizon for autoregressive rollout
    - check_id
    """
    bg1 = _to_float_tensor(payload["bg1"], "bg1", expected_dim=2)
    bg1_mask = _to_float_tensor(payload["bg1_mask"], "bg1_mask", expected_dim=2)
    insulin1 = _to_float_tensor(payload["insulin1"], "insulin1", expected_dim=2)
    insulin1_mask = _to_float_tensor(payload["insulin1_mask"], "insulin1_mask", expected_dim=2)
    person_value = _to_float_tensor(payload["person_value"], "person_value", expected_dim=1)
    c_pep_value = _to_float_tensor(payload["c_pep_value"], "c_pep_value", expected_dim=1)

    person_mask = _to_float_tensor(
        payload.get("person_mask", torch.ones_like(person_value)),
        "person_mask",
        expected_dim=1,
    )
    c_pep_mask = _to_float_tensor(
        payload.get("c_pep_mask", torch.ones_like(c_pep_value)),
        "c_pep_mask",
        expected_dim=1,
    )

    if bg1.shape != bg1_mask.shape:
        raise ValueError(f"bg1 and bg1_mask shape mismatch: {tuple(bg1.shape)} vs {tuple(bg1_mask.shape)}")
    if insulin1.shape != insulin1_mask.shape:
        raise ValueError(f"insulin1 and insulin1_mask shape mismatch: {tuple(insulin1.shape)} vs {tuple(insulin1_mask.shape)}")

    horizon = int(payload.get("prediction_horizon", getattr(cfg, "max_seq_len", 7)))
    if horizon <= 0:
        raise ValueError("prediction_horizon must be positive")

    insulin_dim = int(insulin1.shape[-1])
    bg_dim = int(bg1.shape[-1])
    if getattr(cfg, "d_insulin", insulin_dim) != insulin_dim:
        cfg.d_insulin = insulin_dim
    if getattr(cfg, "d_bg", bg_dim) != bg_dim:
        cfg.d_bg = bg_dim

    drug_dim = None
    if "drug2" in payload and payload["drug2"] is not None:
        drug_tensor_raw = torch.as_tensor(payload["drug2"], dtype=torch.float32)
        drug_dim = int(drug_tensor_raw.shape[-1])
    elif model is not None:
        drug_dim = _infer_model_drug_dim(model)
    elif hasattr(cfg, "d_drug"):
        drug_dim = int(cfg.d_drug)

    if drug_dim is None:
        raise ValueError("drug2 or d_drug is required to build the inference service sample")

    cfg.d_drug = drug_dim
    drug2 = _build_repeated_drug_sequence(payload.get("drug2"), horizon, drug_dim)

    bg2 = torch.zeros(horizon, bg_dim, dtype=torch.float32)
    bg2_mask = torch.zeros_like(bg2)
    insulin2 = torch.zeros(horizon, insulin_dim, dtype=torch.float32)
    insulin2_mask = torch.zeros_like(insulin2)

    sample = {
        "check_id": payload.get("check_id", "service_request"),
        "person_value": person_value,
        "person_mask": person_mask,
        "c_pep_value": c_pep_value,
        "c_pep_mask": c_pep_mask,
        "bg1": bg1,
        "bg1_mask": bg1_mask,
        "insulin1": insulin1,
        "insulin1_mask": insulin1_mask,
        "len1": int(bg1.shape[0]),
        "bg2": bg2,
        "bg2_mask": bg2_mask,
        "insulin2": insulin2,
        "insulin2_mask": insulin2_mask,
        "drug2": drug2,
        "len2": int(horizon),
        "d_insulin": insulin_dim,
        "d_per1": int(person_value.shape[0]),
        "d_drug": drug_dim,
        "data_mode": payload.get("data_mode", "service"),
        # Optional compatibility field for length predictor path.
        "bg": torch.cat([bg1, bg2], dim=0),
    }
    return sample


class InferenceService:
    """
    Reusable inference service.

    Load model once, then call predict(payload) repeatedly.
    """
    def __init__(self, cfg, log_dir=None, logger_name="inference_service"):
        self.cfg = cfg
        self.device = cfg.device
        self.log_dir = log_dir or os.path.join(
            cfg.load_model_path if cfg.load_model_path else cfg.save_path,
            "service_logs",
        )
        self.logger, self.log_file = _setup_inference_service_logger(self.log_dir, logger_name=logger_name)
        self.model = None
        self._load_model()

    def _load_model(self):
        self.logger.info("Initializing inference service")
        self.logger.info("Device: %s", self.device)
        self.logger.info("Checkpoint root: %s", self.cfg.load_model_path if self.cfg.load_model_path else self.cfg.save_path)

        self.model = None
        self.logger.info("Waiting for first request to infer d_drug and build model")

    def _ensure_model_loaded(self, payload):
        if self.model is not None:
            return

        sample_for_dim = build_service_sample(self.cfg, payload, model=None)
        self.cfg.d_drug = sample_for_dim["d_drug"]
        self.cfg.d_insulin = sample_for_dim["d_insulin"]
        self.cfg.d_person = sample_for_dim["d_per1"]

        checkpoint_path = None
        if self.cfg.load_model_path:
            checkpoint_path = os.path.join(self.cfg.load_model_path, "model.pth")

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            prev_top_k = getattr(self.cfg, "top_k", None)
            self.cfg = hydrate_cfg_from_checkpoint(self.cfg, state_dict)
            if getattr(self.cfg, "top_k", None) != prev_top_k:
                self.logger.info("Adjust top_k from %s to %s based on checkpoint", prev_top_k, getattr(self.cfg, "top_k", None))
            self.model = TwoStageModel(self.cfg).to(self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info("Loaded checkpoint: %s", checkpoint_path)
        else:
            self.model = TwoStageModel(self.cfg).to(self.device)
            self.logger.warning("Checkpoint not found, using current model initialization")
        self.model.eval()

    def predict(self, payload):
        request_id = payload.get("request_id", datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        check_id = payload.get("check_id", "service_request")
        try:
            self._ensure_model_loaded(payload)
            sample = build_service_sample(self.cfg, payload, model=self.model)
            batch = collate_fn([sample])
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            self.logger.info(
                "Request start | request_id=%s | check_id=%s | len1=%s | horizon=%s | bg1_shape=%s | insulin1_shape=%s | drug2_shape=%s",
                request_id,
                check_id,
                sample["len1"],
                sample["len2"],
                tuple(sample["bg1"].shape),
                tuple(sample["insulin1"].shape),
                tuple(sample["drug2"].shape),
            )

            with torch.no_grad():
                outputs = self.model(batch, tf_ratio=0.0, mode="test")

            insulin_pred = outputs[0][0].detach().cpu().numpy()
            bg_pred = outputs[1][0].detach().cpu().numpy()
            predicted_length = int(bg_pred.shape[0])
            if len(outputs) >= 3 and torch.is_tensor(outputs[2]):
                predicted_length = int(torch.round(outputs[2][0]).item())
                predicted_length = max(1, min(predicted_length, int(bg_pred.shape[0])))
                insulin_pred = insulin_pred[:predicted_length]
                bg_pred = bg_pred[:predicted_length]

            self.logger.info(
                "Request success | request_id=%s | check_id=%s | predicted_length=%s | bg_pred_shape=%s | insulin_pred_shape=%s",
                request_id,
                check_id,
                predicted_length,
                tuple(bg_pred.shape),
                tuple(insulin_pred.shape),
            )

            return {
                "request_id": request_id,
                "check_id": check_id,
                "predicted_length": predicted_length,
                "insulin_prediction": insulin_pred,
                "bg_prediction": bg_pred,
            }
        except Exception as exc:
            self.logger.exception(
                "Request failed | request_id=%s | check_id=%s | error=%s",
                request_id,
                check_id,
                str(exc),
            )
            raise

    def get_log_file(self):
        return self.log_file


def inference_service_request(cfg, payload, model=None):
    """
    Service-style inference interface.

    Input:
    - stage1 history (`bg1`, `insulin1`, masks)
    - patient features
    - c-peptide features
    - one daily drug vector or a fixed drug sequence

    Output:
    - predicted insulin sequence
    - predicted bg sequence
    - predicted length if the current checkpoint supports length prediction
    """
    if isinstance(model, InferenceService):
        return model.predict(payload)

    service = InferenceService(cfg) if model is None else None
    if service is not None:
        return service.predict(payload)

    local_model = model
    sample = build_service_sample(cfg, payload, model=local_model)
    batch = collate_fn([sample])
    batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = local_model(batch, tf_ratio=0.0, mode="test")

    insulin_pred = outputs[0][0].detach().cpu().numpy()
    bg_pred = outputs[1][0].detach().cpu().numpy()
    predicted_length = int(bg_pred.shape[0])
    if len(outputs) >= 3 and torch.is_tensor(outputs[2]):
        predicted_length = int(torch.round(outputs[2][0]).item())
        predicted_length = max(1, min(predicted_length, int(bg_pred.shape[0])))
        insulin_pred = insulin_pred[:predicted_length]
        bg_pred = bg_pred[:predicted_length]

    return {
        "request_id": payload.get("request_id"),
        "check_id": sample["check_id"],
        "predicted_length": predicted_length,
        "insulin_prediction": insulin_pred,
        "bg_prediction": bg_pred,
    }


def inference_single_sample(cfg, sample_data):
    """
    Run inference on a single preprocessed sample dict.
    """
    device = cfg.device

    checkpoint_path = None
    if cfg.load_model_path:
        candidate = os.path.join(cfg.load_model_path, "model.pth")
        if os.path.exists(candidate):
            checkpoint_path = candidate
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        cfg = hydrate_cfg_from_checkpoint(cfg, state_dict)
        model = TwoStageModel(cfg).to(device)
        model.load_state_dict(state_dict)
    else:
        model = TwoStageModel(cfg).to(device)
    model.eval()

    # Wrap sample in list and collate
    batch = collate_fn([sample_data])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(batch, tf_ratio=0.0, mode="test")

    retrieval_details = extract_retrieval_topk_details(model, batch)
    print_retrieval_topk_details(batch.get("check_id", []), retrieval_details)

    insulin_pred = outputs[0]
    bg_pred = outputs[1]
    # discharge_pred = outputs[2]
    # actual_lengths = outputs[3] if len(outputs) > 3 else None
    print(insulin_pred.cpu().numpy(),bg_pred.cpu().numpy())
    return {
        "insulin_prediction": insulin_pred.cpu().numpy(),
        "bg_prediction": bg_pred.cpu().numpy(),
        "retrieval_topk": retrieval_details[0] if retrieval_details else [],
        # "discharge_prediction": discharge_pred.cpu().numpy(),
        # "predicted_lengths": actual_lengths.cpu().numpy() if actual_lengths is not None else None
    }

def evaluate_test_loss(cfg, test_dataset):
    device = cfg.device
    rule_stats = load_rule_stats(cfg)
    rule_stats_path = getattr(cfg, "rule_stats_path", "") or ""

    if cfg.load_model_path and os.path.exists(cfg.load_model_path):
        checkpoint_file = os.path.join(cfg.load_model_path, "model.pth")
        state_dict = torch.load(checkpoint_file, map_location=device)
        prev_top_k = getattr(cfg, "top_k", None)
        cfg = hydrate_cfg_from_checkpoint(cfg, state_dict)
        if getattr(cfg, "top_k", None) != prev_top_k:
            print(f"[inference] Adjust top_k from {prev_top_k} to {getattr(cfg, 'top_k', None)} based on checkpoint.")
        model = TwoStageModel(cfg).to(device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {cfg.load_model_path}")
    else:
        model = TwoStageModel(cfg).to(device)
        print("No valid load_model_path found, using random weights.")
    if rule_stats_path and os.path.exists(rule_stats_path):
        print(f"Loaded rule-based regimen stats from {rule_stats_path}")
    elif rule_stats.get("source") == "computed_from_train_split":
        print("Rule-based regimen stats file not provided; computed mu/sigma from the train split on the fly.")
    else:
        print("Rule-based regimen stats file not provided or not found; using default mu/sigma priors.")
    print(
        "Rule stats | "
        f"mu_BB={rule_stats['mu_bb']:.4f}, sigma_BB={rule_stats['sigma_bb']:.4f}, "
        f"mu_PM_AM={rule_stats['mu_pm_am']:.4f}, sigma_PM_AM={rule_stats['sigma_pm_am']:.4f}, "
        f"mu_PM_PM={rule_stats['mu_pm_pm']:.4f}, sigma_PM_PM={rule_stats['sigma_pm_pm']:.4f}, "
        f"n_BB={int(rule_stats.get('n_bb_patients', 0))}, n_PM={int(rule_stats.get('n_pm_patients', 0))}"
    )

    model.eval()
    criterion = CombinedLoss(cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    insulin_loss_total = 0.0
    bg_loss_total = 0.0
    metric_sums = {
        "insulin_dose_mae": 0.0,
        "insulin_dose_mape": 0.0,
        "insulin_dose_mae_matched": 0.0,
        "insulin_dose_mape_matched": 0.0,
        "insulin_dose_mae_basic": 0.0,
        "insulin_dose_mape_basic": 0.0,
        "insulin_dose_mae_premix": 0.0,
        "insulin_dose_mape_premix": 0.0,
        "insulin_dose_mae_basic_matched": 0.0,
        "insulin_dose_mape_basic_matched": 0.0,
        "insulin_dose_mae_premix_matched": 0.0,
        "insulin_dose_mape_premix_matched": 0.0,
        "insulin_dose_rmse_basic": 0.0,
        "insulin_dose_rmse_premix": 0.0,
        "insulin_dose_rmse_basic_matched": 0.0,
        "insulin_dose_rmse_premix_matched": 0.0,
        "insulin_dose_r2_basic": 0.0,
        "insulin_dose_r2_premix": 0.0,
        "insulin_dose_r2_basic_matched": 0.0,
        "insulin_dose_r2_premix_matched": 0.0,
        "bg_mae": 0.0,
        "bg_rmse": 0.0,
        "bg_r2": 0.0,
        "rule_prob_type_acc": 0.0,
        "rule_prob_type_acc_basic": 0.0,
        "rule_prob_type_acc_premix": 0.0,
        "rule_prob_type_precision_basic": 0.0,
        "rule_prob_type_precision_premix": 0.0,
        "rule_prob_type_f1_basic": 0.0,
        "rule_prob_type_f1_premix": 0.0,
    }
    metric_histories = {key: [] for key in metric_sums.keys()}

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch, tf_ratio=0.0, mode="test")
            batch_loss, loss_dict = compute_batch_loss(batch, outputs, criterion, cfg, mode="test")
            total_loss += batch_loss.item()
            insulin_loss_total += loss_dict.get("insulin_diversity_base", 0.0)
            bg_loss_total += loss_dict.get("bg_diversity_base", 0.0)

            dose_pred, dose_gt, dose_mask, min_T = build_eval_dose_tensors(batch, outputs, device)
            _, gt_label_for_stats, stats_valid_label = build_gt_regimen_labels(
                batch, dose_gt, dose_mask, min_T, cfg, device
            )
            prior_basic, prior_premix = compute_topk_regimen_prior(model, batch, min_T)
            rule_pred_label, _ = infer_rule_probabilistic_regimen_labels(
                dose_pred, prior_basic, prior_premix, cfg, rule_stats
            )
            rule_metrics = compute_regimen_classification_metrics(
                rule_pred_label, gt_label_for_stats, stats_valid_label
            )
            loss_dict["rule_prob_type_acc"] = rule_metrics["acc"]
            loss_dict["rule_prob_type_acc_basic"] = rule_metrics["recall_basic"]
            loss_dict["rule_prob_type_acc_premix"] = rule_metrics["recall_premix"]
            loss_dict["rule_prob_type_precision_basic"] = rule_metrics["precision_basic"]
            loss_dict["rule_prob_type_precision_premix"] = rule_metrics["precision_premix"]
            loss_dict["rule_prob_type_f1_basic"] = rule_metrics["f1_basic"]
            loss_dict["rule_prob_type_f1_premix"] = rule_metrics["f1_premix"]

            for key in metric_sums.keys():
                value = loss_dict.get(key, 0.0)
                metric_sums[key] += value
                metric_histories[key].append(value)

    avg_total_loss = total_loss / len(test_loader)
    avg_insulin_loss = insulin_loss_total / len(test_loader)
    avg_bg_loss = bg_loss_total / len(test_loader)
    metric_avgs = {key: metric_sums[key] / len(test_loader) for key in metric_sums.keys()}
    metric_cis = {
        key: bootstrap_mean_ci(metric_histories[key], seed=cfg.seed)
        for key in metric_histories.keys()
    }
    regimen_metric_label = "Flag-Based Type" if int(getattr(cfg, "d_insulin", 0)) == 5 else "Rule-Prob Type"

    print("\n" + "=" * 60)
    print("Test Dataset Evaluation Completed")
    print(f"Average Test Loss: {avg_total_loss:.4f}")
    print(f"  - Insulin Loss: {avg_insulin_loss:.4f}")
    print(f"  - BG Loss: {avg_bg_loss:.4f}")
    print(f"  - Insulin Dose MAE/MAPE (overall): {metric_avgs['insulin_dose_mae']:.4f} / {metric_avgs['insulin_dose_mape']:.2f}%")
    print(f"    Insulin Dose 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape'])}")
    print(f"  - Insulin Dose MAE/MAPE (matched days, 8d overall): {metric_avgs['insulin_dose_mae_matched']:.4f} / {metric_avgs['insulin_dose_mape_matched']:.2f}%")
    print(f"    Insulin Dose Matched 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae_matched'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape_matched'])}")
    print(f"  - Basic Dose MAE/MAPE/RMSE/R2 (gt days, 8d): {metric_avgs['insulin_dose_mae_basic']:.4f} / {metric_avgs['insulin_dose_mape_basic']:.2f}% / {metric_avgs['insulin_dose_rmse_basic']:.4f} / {metric_avgs['insulin_dose_r2_basic']:.4f}")
    print(f"    Basic Dose 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae_basic'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape_basic'])} | RMSE {format_ci(*metric_cis['insulin_dose_rmse_basic'])} | R2 {format_ci(*metric_cis['insulin_dose_r2_basic'])}")
    print(f"  - Premix Dose MAE/MAPE/RMSE/R2 (gt days, 8d): {metric_avgs['insulin_dose_mae_premix']:.4f} / {metric_avgs['insulin_dose_mape_premix']:.2f}% / {metric_avgs['insulin_dose_rmse_premix']:.4f} / {metric_avgs['insulin_dose_r2_premix']:.4f}")
    print(f"    Premix Dose 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae_premix'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape_premix'])} | RMSE {format_ci(*metric_cis['insulin_dose_rmse_premix'])} | R2 {format_ci(*metric_cis['insulin_dose_r2_premix'])}")
    print(f"  - Basic Dose MAE/MAPE/RMSE/R2 (matched days, 8d): {metric_avgs['insulin_dose_mae_basic_matched']:.4f} / {metric_avgs['insulin_dose_mape_basic_matched']:.2f}% / {metric_avgs['insulin_dose_rmse_basic_matched']:.4f} / {metric_avgs['insulin_dose_r2_basic_matched']:.4f}")
    print(f"    Basic Dose Matched 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae_basic_matched'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape_basic_matched'])} | RMSE {format_ci(*metric_cis['insulin_dose_rmse_basic_matched'])} | R2 {format_ci(*metric_cis['insulin_dose_r2_basic_matched'])}")
    print(f"  - Premix Dose MAE/MAPE/RMSE/R2 (matched days, 8d): {metric_avgs['insulin_dose_mae_premix_matched']:.4f} / {metric_avgs['insulin_dose_mape_premix_matched']:.2f}% / {metric_avgs['insulin_dose_rmse_premix_matched']:.4f} / {metric_avgs['insulin_dose_r2_premix_matched']:.4f}")
    print(f"    Premix Dose Matched 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae_premix_matched'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape_premix_matched'])} | RMSE {format_ci(*metric_cis['insulin_dose_rmse_premix_matched'])} | R2 {format_ci(*metric_cis['insulin_dose_r2_premix_matched'])}")
    print(f"  - {regimen_metric_label} Recall Basic/Premix: "
          f"{metric_avgs['rule_prob_type_acc_basic']:.4f} / {metric_avgs['rule_prob_type_acc_premix']:.4f}")
    print(f"    {regimen_metric_label} Recall 95%CI: "
          f"Basic {format_ci(*metric_cis['rule_prob_type_acc_basic'])} | Premix {format_ci(*metric_cis['rule_prob_type_acc_premix'])}")
    print(f"  - {regimen_metric_label} Accuracy (overall): {metric_avgs['rule_prob_type_acc']:.4f}")
    print(f"    {regimen_metric_label} Accuracy 95%CI: {format_ci(*metric_cis['rule_prob_type_acc'])}")
    print(f"  - {regimen_metric_label} Precision Basic/Premix: "
          f"{metric_avgs['rule_prob_type_precision_basic']:.4f} / {metric_avgs['rule_prob_type_precision_premix']:.4f}")
    print(f"    {regimen_metric_label} Precision 95%CI: "
          f"Basic {format_ci(*metric_cis['rule_prob_type_precision_basic'])} | Premix {format_ci(*metric_cis['rule_prob_type_precision_premix'])}")
    print(f"  - {regimen_metric_label} F1 Basic/Premix: "
          f"{metric_avgs['rule_prob_type_f1_basic']:.4f} / {metric_avgs['rule_prob_type_f1_premix']:.4f}")
    print(f"    {regimen_metric_label} F1 95%CI: "
          f"Basic {format_ci(*metric_cis['rule_prob_type_f1_basic'])} | Premix {format_ci(*metric_cis['rule_prob_type_f1_premix'])}")
    print(f"  - BG MAE/RMSE/R2: {metric_avgs['bg_mae']:.4f} / {metric_avgs['bg_rmse']:.4f} / {metric_avgs['bg_r2']:.4f}")
    print(f"    BG 95%CI: MAE {format_ci(*metric_cis['bg_mae'])} | RMSE {format_ci(*metric_cis['bg_rmse'])} | R2 {format_ci(*metric_cis['bg_r2'])}")
    print("=" * 60 + "\n")

    result = {
        "avg_test_loss": avg_total_loss,
        "avg_insulin_loss": avg_insulin_loss,
        "avg_bg_loss": avg_bg_loss,
        "insulin_dose_mae": metric_avgs["insulin_dose_mae"],
        "insulin_dose_mape": metric_avgs["insulin_dose_mape"],
        "insulin_dose_mae_matched": metric_avgs["insulin_dose_mae_matched"],
        "insulin_dose_mape_matched": metric_avgs["insulin_dose_mape_matched"],
        "insulin_dose_mae_basic": metric_avgs["insulin_dose_mae_basic"],
        "insulin_dose_mape_basic": metric_avgs["insulin_dose_mape_basic"],
        "insulin_dose_mae_premix": metric_avgs["insulin_dose_mae_premix"],
        "insulin_dose_mape_premix": metric_avgs["insulin_dose_mape_premix"],
        "insulin_dose_mae_basic_matched": metric_avgs["insulin_dose_mae_basic_matched"],
        "insulin_dose_mape_basic_matched": metric_avgs["insulin_dose_mape_basic_matched"],
        "insulin_dose_mae_premix_matched": metric_avgs["insulin_dose_mae_premix_matched"],
        "insulin_dose_mape_premix_matched": metric_avgs["insulin_dose_mape_premix_matched"],
        "insulin_dose_rmse_basic": metric_avgs["insulin_dose_rmse_basic"],
        "insulin_dose_rmse_premix": metric_avgs["insulin_dose_rmse_premix"],
        "insulin_dose_rmse_basic_matched": metric_avgs["insulin_dose_rmse_basic_matched"],
        "insulin_dose_rmse_premix_matched": metric_avgs["insulin_dose_rmse_premix_matched"],
        "insulin_dose_r2_basic": metric_avgs["insulin_dose_r2_basic"],
        "insulin_dose_r2_premix": metric_avgs["insulin_dose_r2_premix"],
        "insulin_dose_r2_basic_matched": metric_avgs["insulin_dose_r2_basic_matched"],
        "insulin_dose_r2_premix_matched": metric_avgs["insulin_dose_r2_premix_matched"],
        "rule_prob_type_acc": metric_avgs["rule_prob_type_acc"],
        "rule_prob_type_acc_basic": metric_avgs["rule_prob_type_acc_basic"],
        "rule_prob_type_acc_premix": metric_avgs["rule_prob_type_acc_premix"],
        "rule_prob_type_precision_basic": metric_avgs["rule_prob_type_precision_basic"],
        "rule_prob_type_precision_premix": metric_avgs["rule_prob_type_precision_premix"],
        "rule_prob_type_f1_basic": metric_avgs["rule_prob_type_f1_basic"],
        "rule_prob_type_f1_premix": metric_avgs["rule_prob_type_f1_premix"],
        "bg_mae": metric_avgs["bg_mae"],
        "bg_rmse": metric_avgs["bg_rmse"],
        "bg_r2": metric_avgs["bg_r2"],
    }
    for key, (ci_low, ci_high) in metric_cis.items():
        result[f"{key}_ci95_low"] = ci_low
        result[f"{key}_ci95_high"] = ci_high
    return result

def visualize_predictions(sample, predictions, save_path):
    import matplotlib.font_manager as fm
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Complete observed sequence.
    target_bg = sample["bg2"].numpy()           # (days, 7)
    target_insulin = sample["insulin2"].numpy()
    total_days = min(
        sample["bg2"].shape[0],
        sample["insulin2"].shape[0],
        predictions["bg_prediction"].shape[1],
        predictions["insulin_prediction"].shape[1],
    )
    
    # Complete generated sequence.
    gen_bg = predictions['bg_prediction'][0][:total_days]
    gen_insulin = predictions['insulin_prediction'][0][:total_days]
    
    time_points_bg = ['BG_0', 'BG_1', 'BG_2', 'BG_3', 'BG_4', 'BG_5', 'BG_6']
    if target_insulin.shape[1] == 5:
        time_points_ins = [
            'Premix_Flag', 'Dose_B', 'Dose_L', 'Dose_D', 'Dose_Long',
        ]
    elif target_insulin.shape[1] == 8:
        time_points_ins = [
            'Basic_SC_B', 'Basic_SC_L', 'Basic_SC_D', 'Basic_SC_N',
            'Premix_SC_B_Long', 'Premix_SC_B_Short', 'Premix_SC_D_Long', 'Premix_SC_D_Short',
        ]
    else:
        time_points_ins = [f'Insulin_{i}' for i in range(target_insulin.shape[1])]
    days = np.arange(1, total_days + 1)
    
    # Spread points within each day so different time slots are visible.
    bg_offsets = np.linspace(-0.3, 0.3, 7)
    ins_offsets = np.linspace(-0.3, 0.3, len(time_points_ins))
    
    # Create blood-glucose and insulin subplots.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # ========== Blood glucose subplot ==========
    for tp_idx, tp_name in enumerate(time_points_bg):
        x = days + bg_offsets[tp_idx]
        ax1.scatter(x, target_bg[:total_days, tp_idx],
                    marker='o', facecolors='none',
                    edgecolors=f'C{tp_idx}', linewidths=1.5,
                    alpha=0.4, label=f'{tp_name}_Observed')
        ax1.scatter(x, gen_bg[:total_days, tp_idx],
                    marker='o', c=f'C{tp_idx}', alpha=0.8,
                    label=f'{tp_name}_Generated')
    
    ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Autoregressive start')
    ax1.set_title(f"Patient {sample['check_id']} - blood-glucose autoregressive prediction")
    ax1.set_ylabel('Blood glucose (mmol/L)')
    ax1.set_xticks(days)
    ax1.set_xticklabels([])
    ax1.legend(ncol=4, fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ========== Insulin subplot ==========
    for ins_idx, ins_name in enumerate(time_points_ins):
        x = days + ins_offsets[ins_idx]
        ax2.scatter(x, target_insulin[:total_days, ins_idx],
                    marker='*', facecolors='none',
                    edgecolors=f'C{ins_idx}', linewidths=1.5,
                    alpha=0.4, label=f'{ins_name}_Observed')
        ax2.scatter(x, gen_insulin[:total_days, ins_idx],
                    marker='o', c=f'C{ins_idx}', alpha=0.8,
                    label=f'{ins_name}_Generated')
    
    ax2.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Autoregressive start')
    ax2.set_title(f"Patient {sample['check_id']} - insulin autoregressive prediction")
    ax2.set_xlabel('Hospital day')
    ax2.set_ylabel('Insulin dose (IU)')
    ax2.set_xticks(days)
    ax2.legend(ncol=4, fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    
if __name__ == "__main__":
    # Example usage setup
    cfg = opt_config()
    set_seed(cfg.seed)

    # Load a sample (e.g., from test set)
    test_dataset = DiabetesDataset(cfg, mode="test")
    
    
    cfg.d_insulin = test_dataset[0]["d_insulin"]
    # cfg.d_insulin_route_stage = train_dataset[0]["d_insulin_route_stage"]
    cfg.d_person = test_dataset[0]["d_per1"]
    cfg.d_drug = test_dataset[0]["d_drug"]
    if cfg.cut_time != 0:
        cfg.max_T1 = cfg.cut_time
    evaluate_test_loss(cfg, test_dataset)

    output_dir = os.path.join(cfg.load_model_path if cfg.load_model_path else cfg.save_path, "inference_outputs")
    os.makedirs(output_dir, exist_ok=True)

    num_visualizations = min(10, len(test_dataset))
    for i in range(num_visualizations):
        sample = test_dataset[i]
        # Run inference
        result = inference_single_sample(cfg, sample)
        save_path = os.path.join(output_dir, f"{sample['check_id']}_autoregressive.png")
        visualize_predictions(sample, result, save_path)
        print(f"Saved visualization to: {save_path}")
        card_save_path = os.path.join(output_dir, f"{sample['check_id']}_prediction_card.png")
        save_prediction_case_card(sample, result, card_save_path)
        print(f"Saved case card to: {card_save_path}")

