# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
RESULTS_ROOT = ROOT / 'results'


def build_base_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=0)

    parser.add_argument('--file_path_d1', type=str, default=str(DATA_ROOT / 'latest_data' / 'data_simplified_basic.json'))
    parser.add_argument('--file_path_d2', type=str, default=str(DATA_ROOT / 'latest_data' / 'data_simplified_premix.json'))
    parser.add_argument('--file_path_d1_extra', type=str, default=str(DATA_ROOT / 'latest_data' / 'data_simplified_basic_extra.json'))
    parser.add_argument('--file_path_d2_extra', type=str, default=str(DATA_ROOT / 'latest_data' / 'data_simplified_premix_extra.json'))
    parser.add_argument('--personality_path', type=str, default=str(DATA_ROOT / 'merged_v3.json'))
    parser.add_argument('--personality_path_extra', type=str, default=str(DATA_ROOT / 'merged_v4_L.json'))

    parser.add_argument('--cut_time', type=int, default=7)
    parser.add_argument('--stage2_max_days', type=int, default=10)
    parser.add_argument('--drug_mode', type=int, default=2)
    parser.add_argument('--d_person', type=int, default=40)
    parser.add_argument('--personality', type=int, default=1)
    parser.add_argument('--c_pep_scheme', type=str, default='binary_abnormal')

    parser.add_argument('--require_c_pep_for_pool', type=int, default=0)
    parser.add_argument('--enable_zero_day_case_filter', type=int, default=0)
    parser.add_argument('--restrict_to_s1_history', type=int, default=1)
    parser.add_argument('--data_partition', type=str, default='all')
    parser.add_argument('--teacher_data_ratio', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--scheduler_patience', type=int, default=8)
    parser.add_argument('--scheduler_factor', type=float, default=0.2)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-8)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=50)
    return parser


def finalize_args(args):
    args.require_c_pep_for_pool = bool(args.require_c_pep_for_pool)
    return args


