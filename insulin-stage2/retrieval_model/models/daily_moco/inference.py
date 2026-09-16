# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.daily_moco.dataset import FeatureStats
from models.daily_moco.model import DailyMoCoSim


class HVectorEncoder:
    def __init__(self, checkpoint_path, device=None):
        payload = torch.load(Path(checkpoint_path), map_location='cpu')
        self.cfg = SimpleNamespace(**payload['cfg'])
        self.device = torch.device(device or self.cfg.device)
        self.stats = FeatureStats.from_dict(payload['feature_stats'])
        self.model = DailyMoCoSim(self.cfg).to(self.device)
        self.model.load_state_dict(payload['model_state_dict'])
        self.model.eval()

    def _validate_runtime_dims(self, person_value, bg, insulin, drug):
        expected_person = int(getattr(self.cfg, 'd_person', 0))
        expected_bg = int(getattr(self.cfg, 'd_bg', 0))
        expected_insulin = int(getattr(self.cfg, 'd_insulin', 0))
        expected_drug = int(getattr(self.cfg, 'd_drug', 0))
        actual_person = int(person_value.shape[-1])
        actual_bg = int(bg.shape[-1])
        actual_insulin = int(insulin.shape[-1])
        actual_drug = int(drug.shape[-1]) if drug is not None else 0
        if expected_person and actual_person != expected_person:
            raise ValueError(
                f"DailyMoCo person dimension mismatch: checkpoint={expected_person}, runtime={actual_person}. "
                "Please retrain retrieval checkpoints with the current patient_feature_v2."
            )
        if expected_bg and actual_bg != expected_bg:
            raise ValueError(f"DailyMoCo bg dimension mismatch: checkpoint={expected_bg}, runtime={actual_bg}.")
        if expected_insulin and actual_insulin != expected_insulin:
            raise ValueError(
                f"DailyMoCo insulin dimension mismatch: checkpoint={expected_insulin}, runtime={actual_insulin}."
            )
        if expected_drug != actual_drug:
            raise ValueError(f"DailyMoCo drug dimension mismatch: checkpoint={expected_drug}, runtime={actual_drug}.")

    def _prepare_batch(self, person_value, person_mask, bg, bg_mask, insulin, drug=None, insulin_mask=None, lengths=None):
        person_value = torch.as_tensor(person_value, dtype=torch.float32)
        person_mask = torch.as_tensor(person_mask, dtype=torch.float32)
        bg = torch.as_tensor(bg, dtype=torch.float32)
        bg_mask = torch.as_tensor(bg_mask, dtype=torch.float32)
        insulin = torch.as_tensor(insulin, dtype=torch.float32)
        if drug is None:
            drug = torch.zeros(bg.shape[0], getattr(self.cfg, 'd_drug', 0), dtype=torch.float32)
        drug = torch.as_tensor(drug, dtype=torch.float32)
        if insulin_mask is None:
            insulin_mask = torch.ones_like(insulin)
        else:
            insulin_mask = torch.as_tensor(insulin_mask, dtype=torch.float32)

        if person_value.dim() == 1:
            person_value = person_value.unsqueeze(0)
        if person_mask.dim() == 1:
            person_mask = person_mask.unsqueeze(0)
        if bg.dim() == 2:
            bg = bg.unsqueeze(0)
        if bg_mask.dim() == 2:
            bg_mask = bg_mask.unsqueeze(0)
        if insulin.dim() == 2:
            insulin = insulin.unsqueeze(0)
        if drug.dim() == 2:
            drug = drug.unsqueeze(0)
        if insulin_mask.dim() == 2:
            insulin_mask = insulin_mask.unsqueeze(0)

        if lengths is None:
            lengths = torch.full((bg.shape[0],), bg.shape[1], dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long)
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)

        self._validate_runtime_dims(person_value, bg, insulin, drug)

        return {
            'person_value': person_value.to(self.device),
            'person_mask': person_mask.to(self.device),
            'bg': bg.to(self.device),
            'bg_mask': bg_mask.to(self.device),
            'insulin': insulin.to(self.device),
            'drug': drug.to(self.device),
            'insulin_mask': insulin_mask.to(self.device),
            'len': lengths.to(self.device),
        }

    @torch.no_grad()
    def encode_outputs(self, person_value, person_mask, bg, bg_mask, insulin, drug=None, insulin_mask=None, lengths=None):
        batch = self._prepare_batch(
            person_value,
            person_mask,
            bg,
            bg_mask,
            insulin,
            drug,
            insulin_mask=insulin_mask,
            lengths=lengths,
        )
        outputs = self.model.encode(batch, self.stats)
        return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()}

    @torch.no_grad()
    def encode(self, person_value, person_mask, bg, bg_mask, insulin, drug=None, insulin_mask=None, lengths=None):
        outputs = self.encode_outputs(
            person_value,
            person_mask,
            bg,
            bg_mask,
            insulin,
            drug,
            insulin_mask=insulin_mask,
            lengths=lengths,
        )
        return outputs['h'].detach().cpu()


def cosine_similarity(h1, h2):
    h1 = torch.as_tensor(h1, dtype=torch.float32)
    h2 = torch.as_tensor(h2, dtype=torch.float32)
    if h1.dim() == 1:
        h1 = h1.unsqueeze(0)
    if h2.dim() == 1:
        h2 = h2.unsqueeze(0)
    return torch.sum(h1 * h2, dim=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode samples into h vectors')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location='cpu')
    encoder = HVectorEncoder(args.checkpoint, device=args.device)
    bg = payload['bg'] if 'bg' in payload else payload.get('bg2', payload['bg1'])
    bg_mask = payload['bg_mask'] if 'bg_mask' in payload else payload.get('bg2_mask', payload['bg1_mask'])
    insulin = payload['insulin'] if 'insulin' in payload else payload.get('insulin2', payload['insulin1'])
    drug = payload.get('drug', payload.get('drug2'))
    insulin_mask = payload.get('insulin_mask', payload.get('insulin2_mask', payload.get('insulin1_mask')))
    lengths = payload['len'] if 'len' in payload else payload.get('len2', payload.get('len1'))
    h = encoder.encode(
        payload['person_value'],
        payload['person_mask'],
        bg,
        bg_mask,
        insulin,
        drug,
        insulin_mask=insulin_mask,
        lengths=lengths,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'h': h}, args.output)
    print(json.dumps({'output': args.output, 'shape': list(h.shape)}, ensure_ascii=False))
