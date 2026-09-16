# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def moco_contrastive_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def similarity_regression_loss(pred_sim, target_sim):
    return F.mse_loss(pred_sim, target_sim)


def weighted_similarity_regression_loss(pred_sim, target_sim, sample_weight=None):
    if sample_weight is None:
        return similarity_regression_loss(pred_sim, target_sim)
    sample_weight = sample_weight.float()
    loss = (pred_sim.float() - target_sim.float()) ** 2
    loss = loss * sample_weight
    return loss.sum() / sample_weight.sum().clamp_min(1e-8)


def batch_ranking_loss(pred_sim, target_sim, margin=0.1):
    pred_sim = pred_sim.float()
    target_sim = target_sim.float()
    batch_size = pred_sim.numel()
    if batch_size < 2:
        return pred_sim.new_tensor(0.0)

    pair_target_diff = target_sim.unsqueeze(1) - target_sim.unsqueeze(0)
    valid = pair_target_diff > 0
    if not torch.any(valid):
        return pred_sim.new_tensor(0.0)

    pair_pred_diff = pred_sim.unsqueeze(1) - pred_sim.unsqueeze(0)
    losses = F.relu(float(margin) - pair_pred_diff[valid])
    if losses.numel() == 0:
        return pred_sim.new_tensor(0.0)
    return losses.mean()


@torch.no_grad()
def regression_metrics(pred, target):
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    if pred.numel() < 2:
        pearson = 0.0
    else:
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        denom = pred_c.norm() * target_c.norm()
        pearson = float((pred_c * target_c).sum().item() / denom.clamp_min(1e-8).item())
    return {'mse': mse, 'mae': mae, 'pearson': pearson}
