# -*- coding: utf-8 -*-
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.daily_moco.inference import HVectorEncoder, cosine_similarity
from models.person_pair.inference import PersonPairSimilarityInference
from shared.dataloader2 import DiabetesDataset


def _load_cfg_from_checkpoint(checkpoint_path):
    payload = torch.load(Path(checkpoint_path), map_location='cpu')
    return SimpleNamespace(**payload['cfg']), payload


def _sample_indices(n, sample_size, seed):
    if n <= 0:
        return []
    sample_size = min(int(sample_size), int(n))
    rng = random.Random(int(seed))
    indices = list(range(n))
    rng.shuffle(indices)
    return indices[:sample_size]


def _regimen_name(day_flag):
    if len(day_flag) < 3:
        return 'unknown'
    if day_flag[0] > 0.5:
        return 'basic'
    if day_flag[1] > 0.5:
        return 'premix'
    if day_flag[2] > 0.5:
        return 'none'
    return 'unknown'


def _get_treatment_info(dataset, item):
    check_id = item['check_id']
    raw_person = dataset.data[check_id]
    stage2 = raw_person['c_pep_after']
    insulin = item['insulin2'].float()
    insulin_mask = item['insulin2_mask'].float()
    flag_dim = 3 if insulin.shape[-1] >= 3 else 0
    regimen_flags = insulin[:, :flag_dim].tolist() if flag_dim > 0 else []
    regimen_names = [_regimen_name(day_flag) for day_flag in regimen_flags]
    return {
        'raw_regimen_sequence': stage2.get('胰岛素类型', []),
        'raw_insulin_orders': stage2.get('胰岛素医嘱执行', []),
        'raw_drug_orders': stage2.get('非胰岛素降糖药医嘱执行', []),
        'processed_regimen_flags': regimen_flags,
        'processed_regimen_names': regimen_names,
        'processed_insulin': insulin.tolist(),
        'processed_insulin_mask': insulin_mask.tolist(),
        'processed_drug': item['drug2'].float().tolist(),
        'seq_len': int(item['len2']),
    }


def _score_matrix(people, score_fn):
    matrix = []
    for left in people:
        row = []
        for right in people:
            row.append(float(score_fn(left, right)))
        matrix.append(row)
    return matrix


def _topk_neighbors(matrix, ids, topk):
    topk = max(1, min(int(topk), max(len(ids) - 1, 1)))
    mapping = {}
    for i, check_id in enumerate(ids):
        pairs = [(ids[j], matrix[i][j]) for j in range(len(ids)) if j != i]
        pairs.sort(key=lambda x: x[1], reverse=True)
        mapping[check_id] = pairs[:topk]
    return mapping


def _overlap_report(student_topk, teacher_topk):
    rows = []
    overlaps = []
    for check_id, student_neighbors in student_topk.items():
        teacher_neighbors = teacher_topk.get(check_id, [])
        student_ids = [item[0] for item in student_neighbors]
        teacher_ids = [item[0] for item in teacher_neighbors]
        overlap_ids = [nid for nid in student_ids if nid in teacher_ids]
        overlap = len(overlap_ids)
        denom = max(len(teacher_ids), 1)
        overlaps.append(overlap / denom)
        rows.append(
            {
                'check_id': check_id,
                'student_topk': student_neighbors,
                'teacher_topk': teacher_neighbors,
                'overlap_count': overlap,
                'overlap_ratio': overlap / denom,
                'overlap_ids': overlap_ids,
            }
        )
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return rows, avg_overlap


def generate_pair_report(
    checkpoint_path,
    teacher_checkpoint,
    output_dir=None,
    mode='val',
    sample_size=10,
    topk=3,
    seed=None,
    device=None,
):
    checkpoint_path = Path(checkpoint_path)
    cfg, _ = _load_cfg_from_checkpoint(checkpoint_path)
    seed = int(cfg.seed if seed is None else seed)

    dataset = DiabetesDataset(cfg, mode=mode)
    if len(dataset) == 0 and mode != 'train':
        dataset = DiabetesDataset(cfg, mode='train')
        mode = 'train'

    student_infer = PersonPairSimilarityInference(str(checkpoint_path), device=device)
    teacher_infer = HVectorEncoder(str(teacher_checkpoint), device=device or getattr(cfg, 'teacher_device', None))
    picked_indices = _sample_indices(len(dataset), sample_size=sample_size, seed=seed)

    people = []
    for idx in picked_indices:
        item = dataset[idx]
        teacher_h = teacher_infer.encode(
            item['person_value'],
            item['person_mask'],
            item['bg2'],
            item['bg2_mask'],
            item['insulin2'],
            item['drug2'],
            insulin_mask=item['insulin2_mask'],
            lengths=item['len2'],
        )[0]
        person_emb = student_infer.encode_person(
            {
                'person_value': item['person_value'],
                'person_mask': item['person_mask'],
            }
        )[0]
        people.append(
            {
                'dataset_index': int(idx),
                'check_id': item['check_id'],
                'person_value': item['person_value'].float(),
                'person_mask': item['person_mask'].float(),
                'student_emb': person_emb.float(),
                'teacher_h': teacher_h.float(),
                'treatment': _get_treatment_info(dataset, item),
            }
        )

    def _student_score(left, right):
        return student_infer.predict_similarity(
            {'person_value': left['person_value'], 'person_mask': left['person_mask']},
            {'person_value': right['person_value'], 'person_mask': right['person_mask']},
        )

    def _teacher_score(left, right):
        return cosine_similarity(left['teacher_h'], right['teacher_h']).item()

    student_matrix = _score_matrix(people, _student_score)
    teacher_matrix = _score_matrix(people, _teacher_score)
    ids = [person['check_id'] for person in people]

    pair_rows = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            pair_rows.append(
                {
                    'left_check_id': ids[i],
                    'right_check_id': ids[j],
                    'pred_sim': student_matrix[i][j],
                    'teacher_sim': teacher_matrix[i][j],
                    'sim_gap': student_matrix[i][j] - teacher_matrix[i][j],
                }
            )
    pair_rows.sort(key=lambda x: x['pred_sim'], reverse=True)

    student_topk = _topk_neighbors(student_matrix, ids, topk)
    teacher_topk = _topk_neighbors(teacher_matrix, ids, topk)
    retrieval_rows, avg_overlap = _overlap_report(student_topk, teacher_topk)

    report = {
        'checkpoint': str(checkpoint_path),
        'teacher_checkpoint': str(teacher_checkpoint),
        'mode': mode,
        'sample_size': len(people),
        'topk': int(topk),
        'seed': seed,
        'matrix_check_ids': ids,
        'pred_sim_matrix': student_matrix,
        'teacher_sim_matrix': teacher_matrix,
        'avg_topk_overlap': avg_overlap,
        'people': [
            {
                'dataset_index': person['dataset_index'],
                'check_id': person['check_id'],
                'student_emb_norm': float(torch.norm(person['student_emb'], p=2).item()),
                'teacher_h_norm': float(torch.norm(person['teacher_h'], p=2).item()),
                'treatment': person['treatment'],
            }
            for person in people
        ],
        'pairwise_similarity': pair_rows,
        'retrieval_topk': retrieval_rows,
    }

    output_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'sample_pair_report.json'
    txt_path = output_dir / 'sample_pair_report.txt'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f'Checkpoint: {checkpoint_path}')
    lines.append(f'Teacher checkpoint: {teacher_checkpoint}')
    lines.append(f'Mode: {mode}')
    lines.append(f'Sample size: {len(people)}')
    lines.append(f'Top-k: {topk}')
    lines.append(f'Seed: {seed}')
    lines.append(f'Average top-k overlap with teacher: {avg_overlap:.4f}')
    lines.append('')
    lines.append('Sampled People')
    lines.append('=' * 80)
    for idx, person in enumerate(report['people'], start=1):
        lines.append(
            f"[{idx}] check_id={person['check_id']} | dataset_index={person['dataset_index']} | "
            f"student_emb_norm={person['student_emb_norm']:.4f} | teacher_h_norm={person['teacher_h_norm']:.4f}"
        )
        lines.append(f"raw regimen sequence: {person['treatment']['raw_regimen_sequence']}")
        lines.append(f"processed regimen names: {person['treatment']['processed_regimen_names']}")
        lines.append(f"processed regimen flags: {person['treatment']['processed_regimen_flags']}")
        lines.append(f"processed insulin: {person['treatment']['processed_insulin']}")
        lines.append(f"processed insulin mask: {person['treatment']['processed_insulin_mask']}")
        lines.append(f"processed drug: {person['treatment']['processed_drug']}")
        lines.append(f"raw insulin orders: {person['treatment']['raw_insulin_orders']}")
        lines.append(f"raw drug orders: {person['treatment']['raw_drug_orders']}")
        lines.append('-' * 80)

    def append_matrix(title, matrix):
        lines.append('')
        lines.append(title)
        lines.append('=' * 80)
        if ids:
            lines.append('\t'.join(['check_id'] + ids))
            for check_id, row in zip(ids, matrix):
                lines.append('\t'.join([check_id] + [f'{value:.4f}' for value in row]))

    append_matrix('pred_sim Matrix', student_matrix)
    append_matrix('teacher_sim Matrix', teacher_matrix)

    lines.append('')
    lines.append('Pairwise Similarities')
    lines.append('=' * 80)
    for row in pair_rows:
        lines.append(
            f"{row['left_check_id']} <-> {row['right_check_id']}: "
            f"pred_sim={row['pred_sim']:.6f} | "
            f"teacher_sim={row['teacher_sim']:.6f} | "
            f"gap={row['sim_gap']:.6f}"
        )

    lines.append('')
    lines.append('Top-k Retrieval')
    lines.append('=' * 80)
    for row in retrieval_rows:
        student_fmt = ', '.join([f'{cid}({score:.4f})' for cid, score in row['student_topk']])
        teacher_fmt = ', '.join([f'{cid}({score:.4f})' for cid, score in row['teacher_topk']])
        lines.append(
            f"{row['check_id']}: overlap={row['overlap_count']}/{topk} ({row['overlap_ratio']:.4f}) | "
            f"overlap_ids={row['overlap_ids']}"
        )
        lines.append(f"  student_topk: {student_fmt}")
        lines.append(f"  teacher_topk: {teacher_fmt}")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[PAIR REPORT] Saved JSON to: {json_path}')
    print(f'[PAIR REPORT] Saved TXT to:  {txt_path}')
    return report


def main():
    parser = argparse.ArgumentParser(description='Sample people from PersonPair and report retrieval quality against teacher similarity')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    generate_pair_report(
        checkpoint_path=args.checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        sample_size=args.sample_size,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
    )


if __name__ == '__main__':
    main()
