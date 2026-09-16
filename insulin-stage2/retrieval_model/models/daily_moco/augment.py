# -*- coding: utf-8 -*-
import copy

import torch


def clone_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _sample_prefix_lengths(lengths, min_ratio):
    prefix_lengths = lengths.clone()
    for i in range(lengths.numel()):
        current = int(lengths[i].item())
        min_len = max(1, int(round(current * min_ratio)))
        prefix_lengths[i] = torch.randint(min_len, current + 1, (1,), device=lengths.device)
    return prefix_lengths


def _apply_prefix_crop(batch, prefix_lengths):
    lengths = batch["len"]
    max_len = batch["bg"].shape[1]
    time_index = torch.arange(max_len, device=batch["bg"].device).unsqueeze(0)
    keep_mask = time_index < prefix_lengths.unsqueeze(1)

    batch["bg"] = batch["bg"] * keep_mask.unsqueeze(-1)
    batch["bg_mask"] = batch["bg_mask"] * keep_mask.unsqueeze(-1).float()
    batch["insulin"] = batch["insulin"] * keep_mask.unsqueeze(-1)
    if "insulin_mask" in batch:
        batch["insulin_mask"] = batch["insulin_mask"] * keep_mask.unsqueeze(-1).float()
    batch["len"] = prefix_lengths

    if "real_lengths" in batch:
        batch["real_lengths"] = prefix_lengths.clone()

    return batch


def _apply_person_feature_mask(batch, mask_prob):
    if mask_prob <= 0:
        return batch
    if "person_value" not in batch or "person_mask" not in batch:
        return batch

    valid = batch["person_mask"] > 0
    random_mask = torch.rand_like(batch["person_value"]) < mask_prob
    drop_mask = valid & random_mask

    batch["person_value"] = batch["person_value"].masked_fill(drop_mask, 0.0)
    batch["person_mask"] = batch["person_mask"].masked_fill(drop_mask, 0.0)
    return batch


def _apply_bg_feature_mask(batch, mask_prob):
    if mask_prob <= 0:
        return batch
    valid = batch["bg_mask"] > 0
    random_mask = torch.rand_like(batch["bg"]) < mask_prob
    drop_mask = valid & random_mask
    batch["bg"] = batch["bg"].masked_fill(drop_mask, 0.0)
    batch["bg_mask"] = batch["bg_mask"].masked_fill(drop_mask, 0.0)
    return batch


def _apply_channel_dropout(batch, dropout_prob):
    if dropout_prob <= 0:
        return batch
    bg_channels = batch["bg"].shape[-1]

    bg_drop = (torch.rand(batch["bg"].shape[0], bg_channels, device=batch["bg"].device) < dropout_prob).float()

    batch["bg"] = batch["bg"] * (1.0 - bg_drop.unsqueeze(1))
    batch["bg_mask"] = batch["bg_mask"] * (1.0 - bg_drop.unsqueeze(1))
    return batch


def _apply_bg_noise(batch, noise_std):
    if noise_std <= 0:
        return batch
    noise = torch.randn_like(batch["bg"]) * noise_std
    valid = batch["bg_mask"] > 0
    batch["bg"] = batch["bg"] + noise * valid.float()
    return batch


def make_view(batch, cfg, strength="weak"):
    view = clone_batch(batch)

    lengths = view["len"].long()
    prefix_lengths = _sample_prefix_lengths(lengths, cfg.min_prefix_ratio)
    view = _apply_prefix_crop(view, prefix_lengths)

    if strength == "weak":
        view = _apply_person_feature_mask(view, cfg.weak_person_mask_prob)
        view = _apply_bg_feature_mask(view, cfg.weak_bg_mask_prob)
        view = _apply_channel_dropout(view, cfg.weak_channel_dropout)
        view = _apply_bg_noise(view, cfg.weak_bg_noise_std)
    else:
        view = _apply_person_feature_mask(view, cfg.strong_person_mask_prob)
        view = _apply_bg_feature_mask(view, cfg.strong_bg_mask_prob)
        view = _apply_channel_dropout(view, cfg.strong_channel_dropout)
        view = _apply_bg_noise(view, cfg.strong_bg_noise_std)

    return view
