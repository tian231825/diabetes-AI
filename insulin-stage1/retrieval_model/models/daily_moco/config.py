# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from shared.config import RESULTS_ROOT, build_base_parser, finalize_args


def build_config():
    parser = build_base_parser('DailyMoCoSim self-supervised similarity model')
    parser.set_defaults(batch_size=16, epochs=300, lr=5e-4, weight_decay=1e-4, data_partition='teacher')

    parser.add_argument('--save_dir', type=str, default=str(RESULTS_ROOT / 'daily_moco'))
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--person_nhead', type=int, default=4)
    parser.add_argument('--use_person_self_attention', type=int, default=0)
    parser.add_argument('--static_weight', type=float, default=0.25)
    parser.add_argument('--seq_weight', type=float, default=1.75)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--repr_dim', type=int, default=128)
    parser.add_argument('--queue_size', type=int, default=512)
    parser.add_argument('--moco_momentum', type=float, default=0.99)
    parser.add_argument('--contrast_temp', type=float, default=0.07)
    parser.add_argument('--max_seq_len', type=int, default=7)
    parser.add_argument('--d_bg', type=int, default=7)
    parser.add_argument('--d_insulin', type=int, default=6)
    parser.add_argument('--d_drug', type=int, default=17)
    parser.add_argument('--regimen_flag_dim', type=int, default=0)
    parser.add_argument('--use_regimen', type=int, default=1)
    parser.add_argument('--use_amp', type=int, default=0)
    parser.add_argument('--min_prefix_ratio', type=float, default=0.5)
    parser.add_argument('--weak_person_mask_prob', type=float, default=0.03)
    parser.add_argument('--strong_person_mask_prob', type=float, default=0.06)
    parser.add_argument('--weak_bg_mask_prob', type=float, default=0.05)
    parser.add_argument('--strong_bg_mask_prob', type=float, default=0.08)
    parser.add_argument('--weak_channel_dropout', type=float, default=0.05)
    parser.add_argument('--strong_channel_dropout', type=float, default=0.05)
    parser.add_argument('--weak_bg_noise_std', type=float, default=0.02)
    parser.add_argument('--strong_bg_noise_std', type=float, default=0.03)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)

    args = parser.parse_args()
    args = finalize_args(args)
    args.use_amp = bool(args.use_amp)
    return args

