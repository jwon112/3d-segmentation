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
    """Train ROI detector on resized volumes (binary WT segmentation).
    
    Args:
        results_dir: 실험 결과 저장 디렉토리
                     이미 존재하는 경로면 자동으로 재개, 새로 생성되면 새로 시작
    """
    if criterion is None:
        criterion = combined_loss_nnunet_style
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_dice = 0.0
    best_epoch = 0
    start_epoch = 0
    os.makedirs(results_dir or "experiment_result", exist_ok=True)
    disable_progress = not is_main_process(rank)
    
    # Latest checkpoint 경로 (재개용)
    latest_ckpt_path = None
    if ckpt_path:
        ckpt_dir = os.path.dirname(ckpt_path)
        latest_ckpt_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
    elif results_dir:
        # ckpt_path가 없어도 results_dir가 있으면 경로 생성
        latest_ckpt_path = os.path.join(results_dir, f"{model_name}_latest.pth")
    
    # 재개 로직: latest checkpoint가 있으면 자동으로 재개
    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        if is_main_process(rank):
            print(f"[ROI][Resume] Found checkpoint: {latest_ckpt_path}")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # ReduceLROnPlateau scheduler는 state_dict가 없지만, best_val_dice를 복원하여 patience 추적
            best_val_dice = checkpoint.get('best_val_dice', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            start_epoch = checkpoint.get('epoch', 0)
            if is_main_process(rank):
                print(f"[ROI][Resume] Resumed from epoch {start_epoch}/{epochs}")
                print(f"[ROI][Resume] Best val dice: {best_val_dice:.4f} (epoch {best_epoch})")
        except Exception as e:
            if is_main_process(rank):
                print(f"[ROI][Resume] Warning: Failed to load checkpoint: {e}")
                print(f"[ROI][Resume] Starting from scratch...")
            start_epoch = 0
    
    # Epoch 범위 체크: 재개 시 epoch가 총 epoch보다 크거나 같으면 새로 시작
    if start_epoch >= epochs:
        if is_main_process(rank):
            print(f"[ROI][Resume] Warning: Checkpoint epoch {start_epoch} >= total epochs {epochs}. Starting from scratch.")
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
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
                # Best checkpoint: 모델 가중치와 메타데이터만 저장 (평가용, 기존 호환성 유지)
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'metadata': {
                        'use_4modalities': use_4modalities,
                        'include_coords': include_coords,
                        'coord_encoding_type': 'simple',  # ROI 모델은 현재 항상 simple coords 사용
                    }
                }
                torch.save(checkpoint, ckpt_path)
        
        # Latest checkpoint: 매 epoch마다 저장 (재개용)
        if latest_ckpt_path and is_main_process(rank):
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'metadata': {
                    'use_4modalities': use_4modalities,
                    'include_coords': include_coords,
                    'coord_encoding_type': 'simple',
                }
            }
            torch.save(checkpoint, latest_ckpt_path)
        
        if is_main_process(rank):
            print(f"[ROI][Epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return {
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
    }

