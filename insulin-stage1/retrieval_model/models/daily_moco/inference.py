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


def _infer_d_person_from_state_dict(state_dict):
    weight = state_dict.get('encoder_q.static_encoder.net.0.weight')
    if weight is None or weight.dim() != 2:
        return None
    input_dim = int(weight.shape[1])
    if input_dim % 2 != 0:
        return None
    return input_dim // 2


class HVectorEncoder:
    def __init__(self, checkpoint_path, device=None):
        payload = torch.load(Path(checkpoint_path), map_location='cpu')
        self.cfg = SimpleNamespace(**payload['cfg'])
        inferred_d_person = _infer_d_person_from_state_dict(payload['model_state_dict'])
        if inferred_d_person is not None:
            self.cfg.d_person = int(inferred_d_person)
        self.device = torch.device(device or self.cfg.device)
        self.stats = FeatureStats.from_dict(payload['feature_stats'])
        self.model = DailyMoCoSim(self.cfg).to(self.device)
        self.model.load_state_dict(payload['model_state_dict'])
        self.model.eval()

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

        expected_person = int(self.cfg.d_person)
        expected_bg = int(self.cfg.d_bg)
        expected_insulin = int(self.cfg.d_insulin)
        expected_drug = int(getattr(self.cfg, 'd_drug', 0))
        if person_value.shape[-1] != expected_person or person_mask.shape[-1] != expected_person:
            raise ValueError(
                f'DailyMoCo checkpoint expects d_person={expected_person}, got '
                f'person_value={person_value.shape[-1]}, person_mask={person_mask.shape[-1]}. '
                'This checkpoint likely needs retraining after patient-feature changes.'
            )
        if bg.shape[-1] != expected_bg or bg_mask.shape[-1] != expected_bg:
            raise ValueError(
                f'DailyMoCo checkpoint expects d_bg={expected_bg}, got '
                f'bg={bg.shape[-1]}, bg_mask={bg_mask.shape[-1]}.'
            )
        if insulin.shape[-1] != expected_insulin or insulin_mask.shape[-1] != expected_insulin:
            raise ValueError(
                f'DailyMoCo checkpoint expects d_insulin={expected_insulin}, got '
                f'insulin={insulin.shape[-1]}, insulin_mask={insulin_mask.shape[-1]}.'
            )
        if drug.shape[-1] != expected_drug:
            raise ValueError(
                f'DailyMoCo checkpoint expects d_drug={expected_drug}, got drug={drug.shape[-1]}.'
            )

        if lengths is None:
            lengths = torch.full((bg.shape[0],), bg.shape[1], dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long)
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)

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
    bg = payload['bg'] if 'bg' in payload else payload['bg1']
    bg_mask = payload['bg_mask'] if 'bg_mask' in payload else payload['bg1_mask']
    insulin = payload['insulin'] if 'insulin' in payload else payload['insulin1']
    drug = payload.get('drug', payload.get('drug1'))
    insulin_mask = payload.get('insulin_mask', payload.get('insulin1_mask'))
    lengths = payload['len'] if 'len' in payload else payload.get('len1')
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
