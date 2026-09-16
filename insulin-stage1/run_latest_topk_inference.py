# -*- encoding: utf-8 -*-
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from configs.Config import opt_config
from data.dataloader2 import DiabetesDataset
from inference import evaluate_test_loss
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

    metrics = evaluate_test_loss(cfg, test_dataset)
    summary = {
        "top_k": run_info["top_k"],
        "run_name": run_info["name"],
        "load_model_path": run_info["path"],
        "insulin_dose_mae": metrics["insulin_dose_mae"],
        "insulin_dose_mape": metrics["insulin_dose_mape"],
        "insulin_dose_rmse": metrics["insulin_dose_rmse"],
        "insulin_dose_r2": metrics["insulin_dose_r2"],
        "daily_insulin_mae": metrics["daily_insulin_mae"],
        "daily_insulin_rmse": metrics["daily_insulin_rmse"],
        "daily_insulin_r2": metrics["daily_insulin_r2"],
        "bg_mae": metrics["bg_mae"],
        "bg_insulin_mae_sum": metrics["bg_insulin_mae_sum"],
        "avg_test_loss": metrics["avg_test_loss"],
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
        "daily_insulin_mae",
        "daily_insulin_rmse",
        "daily_insulin_r2",
        "bg_mae",
        "bg_insulin_mae_sum",
        "avg_test_loss",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)
    return csv_path


def save_k_mae_chart(results, output_dir):
    ks = np.array([item["top_k"] for item in results], dtype=np.int32)
    maes = np.array([item["insulin_dose_mae"] for item in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ks.astype(str), maes, color="#4C78A8", alpha=0.85, label="Insulin Dose MAE")

    if len(results) >= 2:
        degree = 1 if len(results) == 2 else 2
        coeffs = np.polyfit(ks, maes, deg=degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(ks.min(), ks.max(), 200)
        ax.plot(
            x_dense,
            poly(x_dense),
            color="#E45756",
            linewidth=2.2,
            label="Trend Line",
        )

    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mae:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Latest Top-K vs Insulin Dose MAE")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Insulin Dose MAE")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = os.path.join(output_dir, "latest_topk_insulin_dose_mae_chart.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png_path


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
            f"Daily-Insulin MAE={result['daily_insulin_mae']:.4f} | "
            f"BG MAE={result['bg_mae']:.4f} | BG+Insulin MAE Sum={result['bg_insulin_mae_sum']:.4f}"
        )

    print("\n" + "=" * 80)
    print("Summary")
    for item in results:
        print(
            f"topk={item['top_k']:>2} | {item['run_name']} | "
            f"Insulin Dose MAE={item['insulin_dose_mae']:.4f} | "
            f"Daily-Insulin MAE={item['daily_insulin_mae']:.4f} | "
            f"BG MAE={item['bg_mae']:.4f} | BG+Insulin MAE Sum={item['bg_insulin_mae_sum']:.4f}"
        )

    csv_path = save_k_mae_table(results, output_dir)
    chart_path = save_k_mae_chart(results, output_dir)
    print(f"\nSaved K-MAE table to: {csv_path}")
    print(f"Saved K-MAE chart to: {chart_path}")

    if args.output_json:
        output_json = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary JSON to: {output_json}")


if __name__ == "__main__":
    main()
