# -*- coding: utf-8 -*-
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.daily_moco.inference import HVectorEncoder
from models.person_pair.config import build_config
from models.person_pair.dataset import PairSimilarityDataset, collate_fn
from models.person_pair.model import PersonPairSimilarityModel
from models.person_pair.pair_report import generate_pair_report
from shared.loss import batch_ranking_loss, regression_metrics, weighted_similarity_regression_loss
from shared.runtime import ensure_dir, move_to_device, set_seed


def run_epoch(model, loader, optimizer, device, cfg, training=True, log_every=50):
    model.train(mode=training)
    loss_meter = 0.0
    preds_all = []
    targets_all = []
    rank_weight = float(getattr(cfg, 'ranking_weight', 0.2))
    reg_weight = float(getattr(cfg, 'regression_weight', 1.0))
    rank_margin = float(getattr(cfg, 'ranking_margin', 0.1))
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        outputs = model(batch['person_value_a'], batch['person_mask_a'], batch['person_value_b'], batch['person_mask_b'])
        pred_sim = outputs['pred_sim']
        target_sim = batch['target_sim']
        reg_loss = weighted_similarity_regression_loss(pred_sim, target_sim, batch.get('sample_weight'))
        rank_loss = batch_ranking_loss(pred_sim, target_sim, margin=rank_margin)
        loss = reg_weight * reg_loss + rank_weight * rank_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_meter += loss.item() * target_sim.size(0)
        preds_all.append(pred_sim.detach().cpu())
        targets_all.append(target_sim.detach().cpu())
        if step % log_every == 0:
            print(f'Step {step}, Loss: {loss.item():.4f}')
    preds_all = torch.cat(preds_all, dim=0) if preds_all else torch.empty(0)
    targets_all = torch.cat(targets_all, dim=0) if targets_all else torch.empty(0)
    metrics = regression_metrics(preds_all, targets_all) if preds_all.numel() > 0 else {'mse': 0.0, 'mae': 0.0, 'pearson': 0.0}
    metrics['loss'] = loss_meter / max(len(loader.dataset), 1)
    return metrics


def save_checkpoint(run_dir, model, cfg, epoch, metrics):
    payload = {'epoch': epoch, 'cfg': vars(cfg), 'model_state_dict': model.state_dict(), 'metrics': metrics}
    torch.save(payload, run_dir / 'person_pair_similarity.pt')
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(cfg), f, ensure_ascii=False, indent=2)


def main():
    cfg = build_config()
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    run_dir = ensure_dir(Path(cfg.save_dir) / datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=str(run_dir / 'tb'))

    teacher_encoder = HVectorEncoder(cfg.teacher_checkpoint, device=cfg.teacher_device)
    train_dataset = PairSimilarityDataset(cfg, mode='train', teacher_encoder=teacher_encoder)
    val_dataset = PairSimilarityDataset(cfg, mode='val', teacher_encoder=teacher_encoder)
    if train_dataset.records:
        cfg.d_person = int(train_dataset.records[0]['person_value'].numel())

    train_loader = DataLoader(train_dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers), collate_fn=collate_fn, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=collate_fn, pin_memory=device.type == 'cuda')

    model = PersonPairSimilarityModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(cfg.scheduler_factor), patience=int(cfg.scheduler_patience), min_lr=float(cfg.scheduler_min_lr))

    best_val = float('inf')
    stale_epochs = 0
    best_epoch = -1
    save_after_epoch = int(getattr(cfg, 'save_after_epoch', 0))

    print('Training PersonPairSimilarity')
    print(f'Teacher checkpoint: {cfg.teacher_checkpoint}')
    print(f'Train persons: {len(train_dataset.records)} | Val persons: {len(val_dataset.records)}')
    print(f'Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}')
    print(f'Save best after epoch: {save_after_epoch}')
    print(f'Run dir: {run_dir}')

    for epoch in range(int(cfg.epochs)):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, training=True, log_every=int(cfg.log_every))
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, cfg, training=False, log_every=int(cfg.log_every))
        scheduler.step(val_metrics['loss'])

        writer.add_scalar('loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('metric/val_mae', val_metrics['mae'], epoch)
        writer.add_scalar('metric/val_pearson', val_metrics['pearson'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('best/val_loss', min(best_val, val_metrics['loss']) if best_epoch >= 0 else val_metrics['loss'], epoch)

        print('=' * 60)
        print(f'Epoch {epoch} Completed')
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train MAE: {train_metrics['mae']:.4f} | Train Pearson: {train_metrics['pearson']:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | Val MAE:   {val_metrics['mae']:.4f} | Val Pearson:   {val_metrics['pearson']:.4f}")
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.8f}')
        print('=' * 60)

        if epoch < save_after_epoch:
            continue

        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(run_dir, model, cfg, epoch, val_metrics)
            print(f'[SAVED] Best person-pair model at epoch {epoch}: val_loss={best_val:.4f}')
        else:
            stale_epochs += 1

        if int(cfg.early_stop_patience) > 0 and stale_epochs >= int(cfg.early_stop_patience):
            print(f'[EARLY STOP] No improvement for {stale_epochs} epochs. Best epoch: {best_epoch}, best val_loss: {best_val:.4f}')
            break

    writer.close()
    checkpoint_path = run_dir / 'person_pair_similarity.pt'
    if checkpoint_path.exists():
        print(f'Best checkpoint saved to: {checkpoint_path}')
        try:
            generate_pair_report(
                checkpoint_path=checkpoint_path,
                teacher_checkpoint=cfg.teacher_checkpoint,
                output_dir=run_dir,
                mode=str(getattr(cfg, 'report_mode', 'val')),
                sample_size=int(getattr(cfg, 'report_sample_size', 10)),
                topk=int(getattr(cfg, 'report_topk', 3)),
                seed=int(cfg.seed),
                device=cfg.device,
            )
        except Exception as exc:
            print(f'[PAIR REPORT] Failed to generate report: {exc}')
    else:
        print(f'[WARN] No checkpoint was saved to: {checkpoint_path}')


if __name__ == '__main__':
    main()
