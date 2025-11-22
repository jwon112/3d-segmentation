"""
Result Utilities
실험 결과 저장 및 처리 관련 유틸리티
"""

import os
import pandas as pd
from typing import List, Dict, Optional


def create_result_dict(
    dataset_version: str,
    seed: int,
    fold_idx: Optional[int],
    model_name: str,
    total_params: int,
    flops: int,
    pam_train_mean: float,
    pam_inference_mean: float,
    latency_mean: float,
    metrics: Dict,
    best_val_dice: float,
    best_val_wt: float,
    best_val_tc: float,
    best_val_et: float,
    best_epoch: int
) -> Dict:
    """결과 딕셔너리 생성"""
    return {
        'dataset': dataset_version,
        'seed': seed,
        'fold': fold_idx,
        'model_name': model_name,
        'total_params': total_params,
        'flops': flops,
        'pam_train': pam_train_mean,  # bytes, batch_size=1 기준 (평균값)
        'pam_inference': pam_inference_mean,  # bytes, batch_size=1 기준 (평균값)
        'inference_latency_ms': latency_mean,  # milliseconds, batch_size=1 기준 (평균값)
        'test_dice': metrics['dice'],  # WT/TC/ET 평균
        'test_wt': metrics.get('wt', None),
        'test_tc': metrics.get('tc', None),
        'test_et': metrics.get('et', None),
        'val_dice': best_val_dice,  # WT/TC/ET 평균
        'val_wt': best_val_wt,
        'val_tc': best_val_tc,
        'val_et': best_val_et,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'best_epoch': best_epoch
    }


def create_stage_pam_result(
    dataset_version: str,
    seed: int,
    fold_idx: Optional[int],
    model_name: str,
    mode: str,
    stage_name: str,
    mem_list: List[float]
) -> Dict:
    """Stage별 PAM 결과 딕셔너리 생성"""
    if not mem_list:
        return None
    
    mem_mean = sum(mem_list) / len(mem_list)
    mem_std = (sum((x - mem_mean) ** 2 for x in mem_list) / len(mem_list)) ** 0.5 if len(mem_list) > 1 else 0.0
    
    return {
        'dataset': dataset_version,
        'seed': seed,
        'fold': fold_idx,
        'model_name': model_name,
        'mode': mode,
        'stage_name': stage_name,
        'pam_mean': mem_mean,  # bytes
        'pam_std': mem_std,  # bytes
        'num_runs': len(mem_list)
    }


def create_epoch_result_dict(
    dataset_version: str,
    seed: int,
    fold_idx: Optional[int],
    model_name: str,
    epoch_result: Dict,
    best_epoch: int,
    test_dice: Optional[float]
) -> Dict:
    """Epoch별 결과 딕셔너리 생성"""
    return {
        'dataset': dataset_version,
        'seed': seed,
        'fold': fold_idx,
        'model_name': model_name,
        'epoch': epoch_result['epoch'],
        'train_loss': epoch_result['train_loss'],
        'train_dice': epoch_result['train_dice'],
        'val_loss': epoch_result['val_loss'],
        'val_dice': epoch_result['val_dice'],
        'val_wt': epoch_result.get('val_wt', None),
        'val_tc': epoch_result.get('val_tc', None),
        'val_et': epoch_result.get('val_et', None),
        'test_dice': test_dice if epoch_result['epoch'] == best_epoch else None  # Best epoch에만 최종 test dice 기록
    }


def save_results_to_csv(
    results_dir: str,
    all_results: List[Dict],
    all_epochs_results: List[Dict],
    all_stage_pam_results: List[Dict],
    is_main_process: bool
):
    """결과를 CSV 파일로 저장"""
    if not is_main_process:
        return
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 메인 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(results_dir, "integrated_experiment_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # 모든 epoch 결과 저장
    if all_epochs_results:
        epochs_df = pd.DataFrame(all_epochs_results)
        epochs_csv_path = os.path.join(results_dir, "all_epochs_results.csv")
        epochs_df.to_csv(epochs_csv_path, index=False)
        print(f"All epochs results saved to: {epochs_csv_path}")
    
    # Stage별 PAM 결과 저장
    if all_stage_pam_results:
        stage_pam_df = pd.DataFrame(all_stage_pam_results)
        stage_pam_csv_path = os.path.join(results_dir, "stage_wise_pam_results.csv")
        stage_pam_df.to_csv(stage_pam_csv_path, index=False)
        print(f"Stage-wise PAM results saved to: {stage_pam_csv_path}")

