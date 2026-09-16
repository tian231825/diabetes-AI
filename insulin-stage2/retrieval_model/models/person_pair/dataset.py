# -*- coding: utf-8 -*-
import hashlib
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.daily_moco.inference import HVectorEncoder, cosine_similarity
from shared.dataloader2 import DiabetesDataset, collate_fn as shared_collate_fn


class PairSimilarityDataset(Dataset):
    def __init__(self, cfg, mode='train', teacher_encoder=None):
        self.cfg = cfg
        self.mode = mode
        self.source_dataset = DiabetesDataset(cfg, mode=mode)
        self.records = []
        for idx in range(len(self.source_dataset)):
            item = self.source_dataset[idx]
            self.records.append(item)
        self.teacher_encoder = teacher_encoder or HVectorEncoder(cfg.teacher_checkpoint, device=cfg.teacher_device)
        self.teacher_h = self._load_or_build_teacher_h()
        self.teacher_sim_matrix = self.teacher_h @ self.teacher_h.t() if len(self.records) > 0 else torch.empty(0, 0)
        self.pairs = self._build_pairs()

    def _cache_file(self) -> Path:
        cache_dir = Path(self.cfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_sig = f"{Path(self.cfg.teacher_checkpoint).resolve()}::{Path(self.cfg.teacher_checkpoint).stat().st_mtime_ns}"
        digest = hashlib.md5(ckpt_sig.encode('utf-8')).hexdigest()[:10]
        split_seed = getattr(self.cfg, 'split_seed', getattr(self.cfg, 'seed', 42))
        data_partition = str(getattr(self.cfg, 'data_partition', 'all'))
        teacher_ratio = float(getattr(self.cfg, 'teacher_data_ratio', 0.5))
        teacher_ratio_tag = str(teacher_ratio).replace('.', 'p')
        return cache_dir / f'teacher_h_{self.mode}_{data_partition}_ratio{teacher_ratio_tag}_seed{split_seed}_{digest}.pt'

    def _build_teacher_batch(self, records):
        teacher_records = []
        for item in records:
            teacher_records.append(
                {
                    'check_id': item['check_id'],
                    'person_value': item['person_value'].float(),
                    'person_mask': item['person_mask'].float(),
                    'bg': item['bg2'].float(),
                    'bg_mask': item['bg2_mask'].float(),
                    'insulin': item['insulin2'].float(),
                    'insulin_mask': item['insulin2_mask'].float(),
                    'drug': item['drug2'].float(),
                    'len': int(item['len2']),
                }
            )
        return shared_collate_fn(teacher_records)

    def _load_or_build_teacher_h(self) -> torch.Tensor:
        cache_file = self._cache_file()
        check_ids = [record['check_id'] for record in self.records]
        if cache_file.exists():
            payload = torch.load(cache_file, map_location='cpu')
            if payload.get('check_ids') == check_ids:
                return payload['h'].float()

        hs = []
        batch_size = int(getattr(self.cfg, 'teacher_batch_size', 64))
        for start in range(0, len(self.records), batch_size):
            batch_records = self.records[start:start + batch_size]
            batch = self._build_teacher_batch(batch_records)
            h = self.teacher_encoder.encode(
                batch['person_value'],
                batch['person_mask'],
                batch['bg'],
                batch['bg_mask'],
                batch['insulin'],
                batch['drug'],
                insulin_mask=batch['insulin_mask'],
                lengths=batch['len'],
            )
            hs.append(h.float())

        teacher_h = torch.cat(hs, dim=0)
        torch.save({'check_ids': check_ids, 'h': teacher_h}, cache_file)
        return teacher_h

    def _build_pairs(self):
        n = len(self.records)
        if n < 2:
            return []

        all_pairs_count = n * (n - 1) // 2
        target_count = int(getattr(self.cfg, 'train_pair_count', 0) if self.mode == 'train' else getattr(self.cfg, 'val_pair_count', 0) or 0)
        if self.mode != 'train':
            if target_count <= 0 or target_count >= all_pairs_count:
                return [{'a': a, 'b': b, 'pair_kind': 'random'} for a, b in combinations(range(n), 2)]

            rng = torch.Generator().manual_seed(int(getattr(self.cfg, 'seed', 42)) + 10_000)
            pairs = set()
            while len(pairs) < target_count:
                idx = torch.randint(0, n, (target_count * 2, 2), generator=rng)
                for a, b in idx.tolist():
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    pairs.add((a, b))
                    if len(pairs) >= target_count:
                        break
            return [{'a': a, 'b': b, 'pair_kind': 'random'} for a, b in sorted(pairs)]

        hard_pos_k = int(getattr(self.cfg, 'hard_pos_k', 8))
        hard_neg_k = int(getattr(self.cfg, 'hard_neg_k', 8))
        random_pair_k = int(getattr(self.cfg, 'random_pair_k', 8))
        rng = torch.Generator().manual_seed(int(getattr(self.cfg, 'seed', 42)))
        pair_map = {}
        for anchor in range(n):
            sims = self.teacher_sim_matrix[anchor]
            sorted_idx = torch.argsort(sims, descending=True).tolist()
            pos_added = 0
            neg_added = 0
            for idx in sorted_idx:
                if idx == anchor:
                    continue
                a, b = (anchor, idx) if anchor < idx else (idx, anchor)
                key = (a, b)
                if pos_added < hard_pos_k:
                    pair_map[key] = 'hard_pos'
                    pos_added += 1
                else:
                    break
            for idx in reversed(sorted_idx):
                if idx == anchor:
                    continue
                a, b = (anchor, idx) if anchor < idx else (idx, anchor)
                key = (a, b)
                if neg_added < hard_neg_k:
                    pair_map[key] = 'hard_neg'
                    neg_added += 1
                else:
                    break
            if random_pair_k > 0 and n > 1:
                candidates = [idx for idx in range(n) if idx != anchor]
                perm = torch.randperm(len(candidates), generator=rng).tolist()
                for pick in perm[:random_pair_k]:
                    idx = candidates[pick]
                    a, b = (anchor, idx) if anchor < idx else (idx, anchor)
                    pair_map.setdefault((a, b), 'random')

        pairs = [{'a': a, 'b': b, 'pair_kind': pair_kind} for (a, b), pair_kind in sorted(pair_map.items())]
        if target_count > 0 and len(pairs) > target_count:
            perm = torch.randperm(len(pairs), generator=rng).tolist()
            pairs = [pairs[idx] for idx in perm[:target_count]]
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        if isinstance(pair, dict):
            a_idx = pair['a']
            b_idx = pair['b']
            pair_kind = pair.get('pair_kind', 'random')
        else:
            a_idx, b_idx = pair
            pair_kind = 'random'
        item_a = self.records[a_idx]
        item_b = self.records[b_idx]
        h_a = self.teacher_h[a_idx]
        h_b = self.teacher_h[b_idx]
        sim = cosine_similarity(h_a, h_b).squeeze(0).float()
        sample_weight = float(getattr(self.cfg, 'random_weight', 1.0))
        if pair_kind in {'hard_pos', 'hard_neg'}:
            sample_weight = float(getattr(self.cfg, 'hard_weight', 2.0))
        return {
            'check_id_a': item_a['check_id'],
            'check_id_b': item_b['check_id'],
            'person_value_a': item_a['person_value'].float(),
            'person_mask_a': item_a['person_mask'].float(),
            'person_value_b': item_b['person_value'].float(),
            'person_mask_b': item_b['person_mask'].float(),
            'target_sim': sim,
            'sample_weight': torch.tensor(sample_weight, dtype=torch.float32),
            'pair_kind': pair_kind,
        }


def collate_fn(batch):
    if not batch:
        return {}
    collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        collated[key] = values if isinstance(values[0], str) else torch.stack(values, dim=0)
    return collated
