# -*- coding: utf-8 -*-
import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.daily_moco.dataset import DailySimilarityDataset, FeatureStats
from models.daily_moco.inference import HVectorEncoder


def _load_checkpoint_cfg(checkpoint_path):
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    cfg = SimpleNamespace(**payload["cfg"])
    return cfg, payload


def _get_raw_treatment_info(dataset, check_id):
    raw_person = dataset.data[check_id]
    stage1 = raw_person["c_pep_before"]
    regimen_sequence = stage1.get("胰岛素类型", [])
    insulin_orders = stage1.get("胰岛素医嘱执行", [])
    return {
        "regimen_sequence": regimen_sequence,
        "insulin_orders": insulin_orders,
    }


def _summarize_processed_treatment(item):
    insulin = item["insulin"].float()
    insulin_mask = item["insulin_mask"].float()
    flag_dim = 3 if insulin.shape[-1] >= 3 else 0
    regimen_flags = insulin[:, :flag_dim] if flag_dim > 0 else torch.zeros(insulin.shape[0], 0)
    dose_values = insulin[:, flag_dim:]
    dose_mask = insulin_mask[:, flag_dim:]

    regimen_names = []
    for day_flag in regimen_flags:
        if day_flag.numel() < 3:
            regimen_names.append("unknown")
        elif day_flag[0] > 0.5:
            regimen_names.append("basic")
        elif day_flag[1] > 0.5:
            regimen_names.append("premix")
        elif day_flag[2] > 0.5:
            regimen_names.append("none")
        else:
            regimen_names.append("unknown")

    regimen_counts = {
        "basic_days": regimen_names.count("basic"),
        "premix_days": regimen_names.count("premix"),
        "none_days": regimen_names.count("none"),
        "unknown_days": regimen_names.count("unknown"),
    }
    return {
        "seq_len": int(item["len"]),
        "regimen_from_processed_flags": regimen_names,
        "regimen_day_counts": regimen_counts,
        "processed_insulin": insulin.tolist(),
        "processed_insulin_mask": insulin_mask.tolist(),
        "processed_dose_values": dose_values.tolist(),
        "processed_dose_mask": dose_mask.tolist(),
    }


def _sample_people(dataset, sample_size, seed):
    n = len(dataset)
    if n == 0:
        return []
    sample_size = min(sample_size, n)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices[:sample_size]


def _cosine(left, right):
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return float(torch.sum(left * right).item())


def _build_similarity_matrix(sampled_people, field_name):
    matrix = []
    for left in sampled_people:
        row = []
        for right in sampled_people:
            row.append(_cosine(left[field_name], right[field_name]))
        matrix.append(row)
    return matrix


def generate_pair_report(checkpoint_path, output_dir=None, mode="val", sample_size=10, seed=None, device=None):
    checkpoint_path = Path(checkpoint_path)
    cfg, payload = _load_checkpoint_cfg(checkpoint_path)
    stats = FeatureStats.from_dict(payload["feature_stats"])
    seed = int(cfg.seed if seed is None else seed)

    dataset = DailySimilarityDataset(cfg, mode=mode)
    if len(dataset) == 0 and mode != "train":
        dataset = DailySimilarityDataset(cfg, mode="train")
        mode = "train"

    encoder = HVectorEncoder(str(checkpoint_path), device=device)
    picked_indices = _sample_people(dataset, sample_size=sample_size, seed=seed)

    sampled_people = []
    for idx in picked_indices:
        item = dataset[idx]
        outputs = encoder.encode_outputs(
            item["person_value"],
            item["person_mask"],
            item["bg"],
            item["bg_mask"],
            item["insulin"],
            item["drug"],
            insulin_mask=item["insulin_mask"],
            lengths=item["len"],
        )
        h = outputs["h"][0]
        static_emb = outputs["static_emb"][0]
        seq_emb = outputs["seq_emb"][0]
        check_id = item["check_id"]
        sampled_people.append(
            {
                "dataset_index": int(idx),
                "check_id": check_id,
                "h": h,
                "static_emb": static_emb,
                "seq_emb": seq_emb,
                "h_norm": float(torch.norm(h, p=2).item()),
                "static_emb_norm": float(torch.norm(static_emb, p=2).item()),
                "seq_emb_norm": float(torch.norm(seq_emb, p=2).item()),
                "raw_treatment": _get_raw_treatment_info(dataset, check_id),
                "processed_treatment": _summarize_processed_treatment(item),
            }
        )

    pair_rows = []
    for left, right in combinations(sampled_people, 2):
        pair_rows.append(
            {
                "left_check_id": left["check_id"],
                "right_check_id": right["check_id"],
                "cos_static_emb": _cosine(left["static_emb"], right["static_emb"]),
                "cos_seq_emb": _cosine(left["seq_emb"], right["seq_emb"]),
                "cos_h": _cosine(left["h"], right["h"]),
            }
        )
    pair_rows.sort(key=lambda x: x["cos_h"], reverse=True)

    matrix_ids = [person["check_id"] for person in sampled_people]
    h_matrix = _build_similarity_matrix(sampled_people, "h")
    static_matrix = _build_similarity_matrix(sampled_people, "static_emb")
    seq_matrix = _build_similarity_matrix(sampled_people, "seq_emb")

    report = {
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "sample_size": len(sampled_people),
        "seed": seed,
        "matrix_check_ids": matrix_ids,
        "cos_h_matrix": h_matrix,
        "cos_static_emb_matrix": static_matrix,
        "cos_seq_emb_matrix": seq_matrix,
        "people": [
            {
                "dataset_index": person["dataset_index"],
                "check_id": person["check_id"],
                "h_norm": person["h_norm"],
                "static_emb_norm": person["static_emb_norm"],
                "seq_emb_norm": person["seq_emb_norm"],
                "raw_treatment": person["raw_treatment"],
                "processed_treatment": person["processed_treatment"],
            }
            for person in sampled_people
        ],
        "pairwise_cos_h": pair_rows,
    }

    output_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sample_pair_report.json"
    txt_path = output_dir / "sample_pair_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Sample size: {len(sampled_people)}")
    lines.append(f"Seed: {seed}")
    lines.append("")
    lines.append("Sampled People")
    lines.append("=" * 80)
    for i, person in enumerate(report["people"], start=1):
        lines.append(
            f"[{i}] check_id={person['check_id']} | dataset_index={person['dataset_index']} | "
            f"h_norm={person['h_norm']:.4f} | static_norm={person['static_emb_norm']:.4f} | seq_norm={person['seq_emb_norm']:.4f}"
        )
        lines.append(f"raw regimen sequence: {person['raw_treatment']['regimen_sequence']}")
        lines.append(f"processed regimen counts: {person['processed_treatment']['regimen_day_counts']}")
        lines.append(f"processed regimen flags: {person['processed_treatment']['regimen_from_processed_flags']}")
        lines.append(f"processed insulin: {person['processed_treatment']['processed_insulin']}")
        lines.append(f"processed insulin mask: {person['processed_treatment']['processed_insulin_mask']}")
        lines.append(f"raw insulin orders: {person['raw_treatment']['insulin_orders']}")
        lines.append("-" * 80)

    def append_matrix(title, matrix):
        lines.append("")
        lines.append(title)
        lines.append("=" * 80)
        if matrix_ids:
            header = ["check_id"] + matrix_ids
            lines.append("\t".join(header))
            for check_id, row in zip(matrix_ids, matrix):
                formatted = [f"{value:.4f}" for value in row]
                lines.append("\t".join([check_id] + formatted))

    append_matrix("cos(static_emb) Matrix", static_matrix)
    append_matrix("cos(seq_emb) Matrix", seq_matrix)
    append_matrix("cos(h) Matrix", h_matrix)

    lines.append("")
    lines.append("Pairwise Similarities")
    lines.append("=" * 80)
    if matrix_ids:
        for row in pair_rows:
            lines.append(
                f"{row['left_check_id']} <-> {row['right_check_id']}: "
                f"cos_static={row['cos_static_emb']:.6f} | "
                f"cos_seq={row['cos_seq_emb']:.6f} | "
                f"cos_h={row['cos_h']:.6f}"
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[PAIR REPORT] Saved JSON to: {json_path}")
    print(f"[PAIR REPORT] Saved TXT to:  {txt_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Sample people from DailyMoCo and report pairwise cos(h) with treatment data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_pair_report(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        sample_size=args.sample_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
