# -*- encoding: utf-8 -*-
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.Config import opt_config
from data.dataloader2 import DiabetesDataset, collate_fn
from models.model import TwoStageModel
from utils.logger import set_seed


TOPK_DIR_PATTERN = re.compile(r"^topk?(?P<k>\d+)_(?P<ts>\d{8}-\d{6})$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the latest run directory for each top-k under autodl-tmp and report insulin dose MAE."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="./autodl-tmp",
        help="Directory containing topK_YYYYMMDD-HHMMSS run folders.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of top-k values to evaluate. Default: evaluate every detected top-k.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save the summary JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional override for DataLoader workers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional directory for generated table/chart outputs. Defaults to runs_root.",
    )
    return parser.parse_args()


def _parse_dir_meta(entry):
    match = TOPK_DIR_PATTERN.match(entry.name)
    if not match:
        return None
    topk = int(match.group("k"))
    try:
        ts = datetime.strptime(match.group("ts"), "%Y%m%d-%H%M%S")
    except ValueError:
        ts = datetime.fromtimestamp(entry.stat().st_mtime)
    return topk, ts


def find_latest_runs(runs_root, topk_filter=None):
    latest = {}
    for entry in os.scandir(runs_root):
        if not entry.is_dir():
            continue
        parsed = _parse_dir_meta(entry)
        if parsed is None:
            continue
        topk, ts = parsed
        if topk_filter is not None and topk not in topk_filter:
            continue
        prev = latest.get(topk)
        if prev is None or ts > prev["timestamp"]:
            latest[topk] = {
                "top_k": topk,
                "timestamp": ts,
                "path": entry.path,
                "name": entry.name,
            }
    return dict(sorted(latest.items(), key=lambda kv: kv[0]))


def build_cfg(overrides):
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        cfg = opt_config()
    finally:
        sys.argv = original_argv

    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


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


def ensure_mask_dimensions(mask, target_shape):
    if mask.dim() == 2:
        return mask.unsqueeze(-1).expand(target_shape)
    if mask.dim() == 3 and mask.shape[-1] == 1:
        return mask.expand(target_shape)
    if mask.dim() == 3 and mask.shape[-1] != target_shape[-1]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} incompatible with target shape {tuple(target_shape)}")
    return mask


def calculate_real_lengths_from_mask(bg_mask):
    real_lengths = torch.ones(bg_mask.shape[0], dtype=torch.long, device=bg_mask.device)
    for b in range(bg_mask.shape[0]):
        valid_timesteps = (bg_mask[b].sum(dim=1) > 0).nonzero(as_tuple=False)
        if len(valid_timesteps) > 0:
            real_lengths[b] = valid_timesteps[-1].item() + 1
    return real_lengths


def create_sequence_mask(real_lengths, max_len, feature_dim, device):
    mask = torch.zeros(real_lengths.shape[0], max_len, feature_dim, device=device)
    for b in range(real_lengths.shape[0]):
        length = min(int(real_lengths[b].item()), max_len)
        mask[b, :length, :] = 1.0
    return mask


def build_stage2_dose_tensors(batch, outputs, device):
    insulin_pred, bg_pred = outputs[0], outputs[1]
    bg_gt = batch["bg2"]
    bg_mask = batch["bg2_mask"]
    insulin_gt = batch["insulin2"]
    insulin_gt_mask = batch["insulin2_mask"]

    min_t = min(
        insulin_pred.shape[1],
        bg_pred.shape[1],
        insulin_gt.shape[1],
        bg_gt.shape[1],
        bg_mask.shape[1],
        insulin_gt_mask.shape[1],
    )
    dose_pred = insulin_pred[:, :min_t, :]
    dose_gt = insulin_gt[:, :min_t, :]
    bg_gt = bg_gt[:, :min_t, :]
    bg_mask = bg_mask[:, :min_t, :]
    insulin_gt_mask = insulin_gt_mask[:, :min_t, :]

    bg_mask = ensure_mask_dimensions(bg_mask, bg_gt.shape)
    real_lengths = calculate_real_lengths_from_mask(bg_mask)
    seq_mask = create_sequence_mask(real_lengths, min_t, dose_pred.shape[-1], device)
    dose_mask = seq_mask * ensure_mask_dimensions(insulin_gt_mask, dose_gt.shape)
    if dose_pred.shape[-1] != 5:
        raise ValueError(f"Stage2 top-k inference expects 5D insulin output, got {dose_pred.shape[-1]}D")
    return dose_pred[..., 1:5], dose_gt[..., 1:5], dose_mask[..., 1:5]


def evaluate_stage2_dose_mae(cfg, test_dataset):
    device = cfg.device
    checkpoint_file = os.path.join(cfg.load_model_path, "model.pth")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")

    state_dict = torch.load(checkpoint_file, map_location=device)
    prev_top_k = getattr(cfg, "top_k", None)
    cfg = hydrate_cfg_from_checkpoint(cfg, state_dict)
    if getattr(cfg, "top_k", None) != prev_top_k:
        print(f"[topk inference] Adjust top_k from {prev_top_k} to {getattr(cfg, 'top_k', None)} based on checkpoint.")

    model = TwoStageModel(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(getattr(cfg, "num_workers", 0) or 0),
    )
    abs_err_sum = 0.0
    abs_pct_sum = 0.0
    sq_err_sum = 0.0
    count = 0
    pred_values = []
    target_values = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch, tf_ratio=0.0, mode="test")
            pred, target, mask = build_stage2_dose_tensors(batch, outputs, device)
            valid = mask > 0
            if not valid.any():
                continue
            pred_valid = pred[valid].float()
            target_valid = target[valid].float()
            err = pred_valid - target_valid
            denom = torch.where(target_valid.abs() <= 1e-6, target_valid.abs() + 1.0, target_valid.abs())
            abs_err_sum += err.abs().sum().item()
            abs_pct_sum += (err.abs() / denom).sum().item()
            sq_err_sum += (err ** 2).sum().item()
            count += int(valid.sum().item())
            pred_values.append(pred_valid.detach().cpu())
            target_values.append(target_valid.detach().cpu())

    if count == 0:
        return {"insulin_dose_mae": 0.0, "insulin_dose_mape": 0.0, "insulin_dose_rmse": 0.0, "insulin_dose_r2": 0.0, "dose_count": 0}

    target_all = torch.cat(target_values)
    pred_all = torch.cat(pred_values)
    ss_tot = ((target_all - target_all.mean()) ** 2).sum().item()
    ss_res = ((pred_all - target_all) ** 2).sum().item()
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "insulin_dose_mae": abs_err_sum / count,
        "insulin_dose_mape": abs_pct_sum / count * 100.0,
        "insulin_dose_rmse": float(np.sqrt(sq_err_sum / count)),
        "insulin_dose_r2": r2,
        "dose_count": count,
    }


def evaluate_run(run_info, cli_args):
    cfg = build_cfg(
        {
            "load_model_path": run_info["path"],
            "save_path": cli_args.runs_root,
            "top_k": run_info["top_k"],
            "batch_size": cli_args.batch_size,
            "device": cli_args.device,
            "num_workers": cli_args.num_workers,
        }
    )
    set_seed(cfg.seed)

    test_dataset = DiabetesDataset(cfg, mode="test")
    sample0 = test_dataset[0]
    cfg.d_insulin = sample0["d_insulin"]
    cfg.d_person = sample0["d_per1"]
    cfg.d_drug = sample0["d_drug"]
    if cfg.cut_time != 0:
        cfg.max_T1 = cfg.cut_time

    metrics = evaluate_stage2_dose_mae(cfg, test_dataset)
    summary = {
        "top_k": run_info["top_k"],
        "run_name": run_info["name"],
        "load_model_path": run_info["path"],
        "insulin_dose_mae": metrics["insulin_dose_mae"],
        "insulin_dose_mape": metrics["insulin_dose_mape"],
        "insulin_dose_rmse": metrics["insulin_dose_rmse"],
        "insulin_dose_r2": metrics["insulin_dose_r2"],
        "dose_count": metrics["dose_count"],
    }
    return summary


def save_k_mae_table(results, output_dir):
    csv_path = os.path.join(output_dir, "latest_topk_insulin_dose_mae_table.csv")
    fieldnames = [
        "top_k",
        "run_name",
        "load_model_path",
        "insulin_dose_mae",
        "insulin_dose_mape",
        "insulin_dose_rmse",
        "insulin_dose_r2",
        "dose_count",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)
    return csv_path
def main():
    args = parse_args()
    runs_root = os.path.abspath(args.runs_root)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else runs_root
    os.makedirs(output_dir, exist_ok=True)

    latest_runs = find_latest_runs(runs_root, set(args.topk) if args.topk else None)
    if not latest_runs:
        raise RuntimeError(f"No matching topk run directories found under: {runs_root}")

    results = []
    print(f"Evaluating latest runs under: {runs_root}")
    for topk, run_info in latest_runs.items():
        print("\n" + "=" * 80)
        print(f"[topk={topk}] latest run: {run_info['name']}")
        result = evaluate_run(run_info, args)
        results.append(result)
        print(
            f"[topk={topk}] Insulin Dose MAE={result['insulin_dose_mae']:.4f} | "
            f"MAPE={result['insulin_dose_mape']:.2f}% | "
            f"RMSE={result['insulin_dose_rmse']:.4f} | "
            f"R2={result['insulin_dose_r2']:.4f} | "
            f"Dose count={result['dose_count']}"
        )

    print("\n" + "=" * 80)
    print("Summary")
    for item in results:
        print(
            f"topk={item['top_k']:>2} | {item['run_name']} | "
            f"Insulin Dose MAE={item['insulin_dose_mae']:.4f} | "
            f"MAPE={item['insulin_dose_mape']:.2f}% | "
            f"RMSE={item['insulin_dose_rmse']:.4f} | "
            f"R2={item['insulin_dose_r2']:.4f} | "
            f"Dose count={item['dose_count']}"
        )

    csv_path = save_k_mae_table(results, output_dir)
    print(f"\nSaved K-MAE table to: {csv_path}")

    if args.output_json:
        output_json = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary JSON to: {output_json}")


if __name__ == "__main__":
    main()
