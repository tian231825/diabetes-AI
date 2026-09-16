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

from .config import DEFAULT_PERSON_PAIR_CHECKPOINT
from .model import PersonPairSimilarityModel


def _ensure_checkpoint_path(checkpoint_path=None):
    checkpoint = Path(checkpoint_path or DEFAULT_PERSON_PAIR_CHECKPOINT)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f'Person-pair checkpoint not found: {checkpoint}. '
            'Please update models/person_pair/config.py or pass checkpoint_path explicitly.'
        )
    return checkpoint


class PersonPairSimilarityInference:
    """
    Reusable person-pair similarity predictor.

    Supported single-person inputs:
    1. {'person_value': ..., 'person_mask': ...}
    2. {'value': ..., 'mask': ...}
    3. (person_value, person_mask)
    """

    def __init__(self, checkpoint_path=None, device=None):
        checkpoint = _ensure_checkpoint_path(checkpoint_path)
        payload = torch.load(checkpoint, map_location='cpu')
        self.checkpoint_path = checkpoint
        self.cfg = SimpleNamespace(**payload['cfg'])
        self.device = torch.device(device or self.cfg.device)
        self.model = PersonPairSimilarityModel(self.cfg).to(self.device)
        self.model.load_state_dict(payload['model_state_dict'])
        self.model.eval()

    def _split_person_feature(self, person_feature, person_mask=None):
        if person_mask is not None:
            return person_feature, person_mask

        if isinstance(person_feature, dict):
            if 'person_value' in person_feature and 'person_mask' in person_feature:
                return person_feature['person_value'], person_feature['person_mask']
            if 'value' in person_feature and 'mask' in person_feature:
                return person_feature['value'], person_feature['mask']
            raise KeyError("person feature dict must contain ('person_value', 'person_mask') or ('value', 'mask').")

        if isinstance(person_feature, (list, tuple)) and len(person_feature) == 2:
            return person_feature[0], person_feature[1]

        raise TypeError(
            'person feature must be provided as (person_value, person_mask), '
            "{'person_value': ..., 'person_mask': ...}, or {'value': ..., 'mask': ...}."
        )

    def _prepare_person(self, person_value, person_mask):
        person_value = torch.as_tensor(person_value, dtype=torch.float32)
        person_mask = torch.as_tensor(person_mask, dtype=torch.float32)
        expected_person = int(getattr(self.cfg, 'd_person', 0))
        actual_person = int(person_value.shape[-1])
        if expected_person and actual_person != expected_person:
            raise ValueError(
                f"Person-pair person dimension mismatch: checkpoint={expected_person}, runtime={actual_person}. "
                "Please retrain retrieval checkpoints with the current patient_feature_v2."
            )
        if person_value.dim() == 1:
            person_value = person_value.unsqueeze(0)
        if person_mask.dim() == 1:
            person_mask = person_mask.unsqueeze(0)
        return person_value.to(self.device), person_mask.to(self.device)

    @torch.no_grad()
    def encode_person(self, person_feature, person_mask=None, return_numpy=False):
        person_value, person_mask = self._split_person_feature(person_feature, person_mask)
        person_value, person_mask = self._prepare_person(person_value, person_mask)
        person_emb = self.model.encode_person(person_value, person_mask).detach().cpu()
        if return_numpy:
            return person_emb.numpy()
        return person_emb

    @torch.no_grad()
    def predict_similarity(
        self,
        person_a,
        person_b=None,
        person_mask_a=None,
        person_mask_b=None,
        return_tensor=False,
    ):
        if person_b is None:
            raise ValueError('person_b is required.')

        person_value_a, person_mask_a = self._split_person_feature(person_a, person_mask_a)
        person_value_b, person_mask_b = self._split_person_feature(person_b, person_mask_b)
        person_value_a, person_mask_a = self._prepare_person(person_value_a, person_mask_a)
        person_value_b, person_mask_b = self._prepare_person(person_value_b, person_mask_b)

        outputs = self.model(person_value_a, person_mask_a, person_value_b, person_mask_b)
        pred_sim = outputs['pred_sim'].detach().cpu()

        if return_tensor:
            return pred_sim
        if pred_sim.numel() == 1:
            return float(pred_sim.item())
        return pred_sim.numpy()

    @torch.no_grad()
    def predict_with_embedding(self, person_a, person_b=None, person_mask_a=None, person_mask_b=None):
        if person_b is None:
            raise ValueError('person_b is required.')

        person_value_a, person_mask_a = self._split_person_feature(person_a, person_mask_a)
        person_value_b, person_mask_b = self._split_person_feature(person_b, person_mask_b)
        person_value_a, person_mask_a = self._prepare_person(person_value_a, person_mask_a)
        person_value_b, person_mask_b = self._prepare_person(person_value_b, person_mask_b)

        outputs = self.model(person_value_a, person_mask_a, person_value_b, person_mask_b)
        pred_sim = outputs['pred_sim'].detach().cpu()
        emb_a = outputs['person_emb_a'].detach().cpu()
        emb_b = outputs['person_emb_b'].detach().cpu()
        return {
            'pred_sim': float(pred_sim.item()) if pred_sim.numel() == 1 else pred_sim.numpy(),
            'person_emb_a': emb_a,
            'person_emb_b': emb_b,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict similarity from two person profiles')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_a', type=str, required=True)
    parser.add_argument('--input_b', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    payload_a = torch.load(args.input_a, map_location='cpu')
    payload_b = torch.load(args.input_b, map_location='cpu')
    infer = PersonPairSimilarityInference(args.checkpoint, device=args.device)
    result = infer.predict_with_embedding(payload_a, payload_b)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        json.dumps(
            {
                'output': args.output,
                'pred_sim': result['pred_sim'],
                'person_emb_shape': list(result['person_emb_a'].shape),
                'checkpoint': str(infer.checkpoint_path),
            },
            ensure_ascii=False,
        )
    )
