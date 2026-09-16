# -*- coding: utf-8 -*-
from pathlib import Path

from shared.config import RESULTS_ROOT, build_base_parser, finalize_args


DEFAULT_TEACHER_CHECKPOINT = RESULTS_ROOT / 'daily_moco' / 'current' / 'daily_moco_sim.pt'
DEFAULT_PERSON_PAIR_CHECKPOINT = RESULTS_ROOT / 'person_pair' / 'current' / 'person_pair_similarity.pt'


def build_config():
    root = Path(__file__).resolve().parent
    parser = build_base_parser('Person-pair similarity regression model')
    parser.set_defaults(batch_size=256, epochs=300, lr=1e-3, weight_decay=1e-4, data_partition='similarity')

    parser.add_argument('--teacher_device', type=str, default=None)
    parser.add_argument('--teacher_checkpoint', type=str, default=str(DEFAULT_TEACHER_CHECKPOINT))
    parser.add_argument('--person_pair_checkpoint', type=str, default=str(DEFAULT_PERSON_PAIR_CHECKPOINT))
    parser.add_argument('--save_dir', type=str, default=str(RESULTS_ROOT / 'person_pair'))
    parser.add_argument('--cache_dir', type=str, default=str(root.parents[2] / 'cache' / 'person_pair'))
    parser.add_argument('--teacher_batch_size', type=int, default=64)
    parser.add_argument('--person_hidden_dim', type=int, default=256)
    parser.add_argument('--person_emb_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--person_nhead', type=int, default=4)
    parser.add_argument('--use_person_self_attention', type=int, default=0)
    parser.add_argument('--train_pair_count', type=int, default=60000)
    parser.add_argument('--val_pair_count', type=int, default=0)
    parser.add_argument('--save_after_epoch', type=int, default=50)
    parser.add_argument('--hard_pos_k', type=int, default=8)
    parser.add_argument('--hard_neg_k', type=int, default=8)
    parser.add_argument('--random_pair_k', type=int, default=8)
    parser.add_argument('--hard_weight', type=float, default=1.5)
    parser.add_argument('--random_weight', type=float, default=1.0)
    parser.add_argument('--regression_weight', type=float, default=1.0)
    parser.add_argument('--ranking_weight', type=float, default=0.1)
    parser.add_argument('--ranking_margin', type=float, default=0.1)
    parser.add_argument('--residual_scale', type=float, default=0.05)
    parser.add_argument('--report_mode', type=str, default='val')
    parser.add_argument('--report_sample_size', type=int, default=10)
    parser.add_argument('--report_topk', type=int, default=3)

    args = parser.parse_args()
    args = finalize_args(args)
    if args.teacher_device is None:
        args.teacher_device = args.device
    return args
