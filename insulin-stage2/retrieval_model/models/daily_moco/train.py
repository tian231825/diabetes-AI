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

from models.daily_moco.augment import make_view
from models.daily_moco.config import build_config
from models.daily_moco.dataset import DailySimilarityDataset, compute_feature_stats, collate_fn
from models.daily_moco.model import DailyMoCoSim
from models.daily_moco.pair_report import generate_pair_report
from shared.config import apply_feature_dims_from_dataset
from shared.runtime import ensure_dir, make_torch_generator, move_to_device, seed_worker, set_seed


def run_epoch(model, loader, optimizer, scaler, device, stats, cfg, training=True):
    total_loss = 0.0
    total_steps = 0
    model.train(training)
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        view_q = make_view(batch, cfg, strength='weak')
        view_k = make_view(batch, cfg, strength='strong')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp and device.type == 'cuda'):
            outputs = model.forward_train(view_q, view_k, stats, update_momentum=training)
            loss = outputs['loss']
        if training:
            if scaler is not None and cfg.use_amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                if float(getattr(cfg, 'grad_clip_norm', 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(getattr(cfg, 'grad_clip_norm', 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                optimizer.step()
            with torch.no_grad():
                model.dequeue_and_enqueue(outputs['keys_to_enqueue'])
        total_loss += float(loss.item())
        total_steps += 1
        if training and step % cfg.log_every == 0:
            print(f'Step {step}, Loss: {loss.item():.4f}')
    return total_loss / max(total_steps, 1)


def save_checkpoint(model, cfg, epoch, val_loss, stats, run_dir):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'cfg': vars(cfg),
        'val_loss': val_loss,
        'feature_stats': stats.to_dict(),
    }
    torch.save(payload, run_dir / 'daily_moco_sim.pt')
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(cfg), f, ensure_ascii=False, indent=2)


def main():
    cfg = build_config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_dir = ensure_dir(Path(cfg.save_dir) / datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=str(run_dir / 'tb'))

    train_dataset = DailySimilarityDataset(cfg, mode='train')
    val_dataset = DailySimilarityDataset(cfg, mode='val')
    cfg = apply_feature_dims_from_dataset(cfg, train_dataset)
    stats = compute_feature_stats(train_dataset)

    num_workers = int(cfg.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(int(cfg.seed)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(int(cfg.seed) + 1),
    )

    model = DailyMoCoSim(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.scheduler_min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.use_amp and device.type == 'cuda')

    best_val_loss = float('inf')
    stale_epochs = 0

    print('Training DailyMoCoSim')
    print(f'Train size: {len(train_dataset)} | Val size: {len(val_dataset)}')
    print(f'Queue size: {cfg.queue_size} | Temp: {cfg.contrast_temp:.3f} | Momentum: {cfg.moco_momentum:.4f}')
    print(f'Run dir: {run_dir}')

    for epoch in range(cfg.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, stats, cfg, training=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, None, device, stats, cfg, training=False)
        scheduler.step(val_loss)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('best/val_loss', min(best_val_loss, val_loss), epoch)
        print('=' * 60)
        print(f'Epoch {epoch} Completed')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss:   {val_loss:.4f}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.8f}')
        print('=' * 60)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            save_checkpoint(model, cfg, epoch, val_loss, stats, run_dir)
            print(f'[SAVED] Best self-supervised model at epoch {epoch}: val_loss={val_loss:.4f}')
        else:
            stale_epochs += 1
        if int(cfg.early_stop_patience) > 0 and stale_epochs >= int(cfg.early_stop_patience):
            print(f'[EARLY STOP] No improvement for {stale_epochs} epochs. Best val_loss={best_val_loss:.4f}')
            break
    writer.close()
    best_ckpt = run_dir / "daily_moco_sim.pt"
    print(f'Best checkpoint saved to: {best_ckpt}')
    if best_ckpt.exists():
        try:
            generate_pair_report(
                checkpoint_path=best_ckpt,
                output_dir=run_dir,
                mode='val',
                sample_size=10,
                seed=cfg.seed,
                device=str(device),
            )
        except Exception as exc:
            print(f'[PAIR REPORT] Failed to generate sample pair report: {exc}')


if __name__ == '__main__':
    main()



