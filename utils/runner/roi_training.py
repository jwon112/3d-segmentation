"""
ROI Model Training
ROI detector 학습 관련 함수
"""

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from utils.experiment_utils import is_main_process
from losses import combined_loss_nnunet_style
from metrics import calculate_dice_score


def train_roi_model(model, train_loader, val_loader, epochs, device, lr=1e-3,
                    criterion=None, ckpt_path=None, results_dir=None, model_name='roi_model',
                    train_sampler=None, rank: int = 0, include_coords: bool = True, use_4modalities: bool = True):
    """Train ROI detector on resized volumes (binary WT segmentation)."""
    if criterion is None:
        criterion = combined_loss_nnunet_style
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_dice = 0.0
    best_epoch = 0
    os.makedirs(results_dir or "experiment_result", exist_ok=True)
    disable_progress = not is_main_process(rank)

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        n_samples = 0
        for inputs, labels in tqdm(
            train_loader,
            desc=f"[ROI] Train {epoch+1}/{epochs}",
            leave=False,
            disable=disable_progress,
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            bsz = inputs.size(0)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bsz
            n_samples += bsz
        train_loss /= max(1, n_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_dices = []
        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader,
                desc=f"[ROI] Val {epoch+1}/{epochs}",
                leave=False,
                disable=disable_progress,
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                bsz = inputs.size(0)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * bsz
                val_samples += bsz
                dice_scores = calculate_dice_score(logits.detach().cpu(), labels.detach().cpu(), num_classes=2)
                if dice_scores.numel() >= 2:
                    val_dices.append(dice_scores[1].item())
        val_loss /= max(1, val_samples)
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        scheduler.step(val_loss)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            if ckpt_path and is_main_process(rank):
                # 체크포인트에 metadata 저장 (기존 체크포인트와의 호환성을 위해 state_dict도 포함)
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'metadata': {
                        'use_4modalities': use_4modalities,
                        'include_coords': include_coords,
                    }
                }
                torch.save(checkpoint, ckpt_path)
        if is_main_process(rank):
            print(f"[ROI][Epoch {epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return {
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
    }

