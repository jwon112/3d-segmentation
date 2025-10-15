#!/usr/bin/env python3
"""
3D Segmentation Visualization Module
다중 모델 지원 3D 세그멘테이션 시각화 모듈
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from scipy import ndimage
import os
from typing import List, Dict, Optional, Union

class SegmentationVisualizer:
    """3D Segmentation 결과 시각화 클래스 (다중 모델 지원)"""
    
    def __init__(self, class_names=['Background', 'NCR/NET', 'ED', 'ET'], 
                 class_colors=['black', 'red', 'green', 'blue']):
        self.class_names = class_names
        self.class_colors = class_colors
        self.n_classes = len(class_names)
        
    def visualize_prediction_slices(self, image, pred, target=None, slice_indices=None,
                                  modality_names=['FLAIR', 'T1', 'T1CE', 'T2'],
                                  model_name='Model', save_path=None):
        """예측 결과를 2D 슬라이스로 시각화 (다중 모델 지원)"""
        
        if slice_indices is None:
            d, h, w = image.shape[1:]
            slice_indices = [d//4, d//2, 3*d//4]
        
        n_slices = len(slice_indices)
        n_modalities = image.shape[0]
        
        fig, axes = plt.subplots(n_slices, n_modalities + 2, figsize=(4*(n_modalities + 2), 4*n_slices))
        if n_slices == 1:
            axes = axes.reshape(1, -1)
        
        for i, slice_idx in enumerate(slice_indices):
            # 원본 이미지들
            for j in range(n_modalities):
                ax = axes[i, j]
                slice_img = image[j, slice_idx, :, :]
                ax.imshow(slice_img, cmap='gray')
                ax.set_title(f'{modality_names[j]} - Slice {slice_idx}')
                ax.axis('off')
            
            # 예측 결과
            ax = axes[i, n_modalities]
            slice_pred = pred[slice_idx, :, :]
            im = ax.imshow(slice_pred, cmap='tab10', vmin=0, vmax=self.n_classes-1)
            ax.set_title(f'{model_name} Prediction - Slice {slice_idx}')
            ax.axis('off')
            
            # Ground Truth (있는 경우)
            ax = axes[i, n_modalities + 1]
            if target is not None:
                slice_target = target[slice_idx, :, :]
                im = ax.imshow(slice_target, cmap='tab10', vmin=0, vmax=self.n_classes-1)
                ax.set_title(f'Ground Truth - Slice {slice_idx}')
            else:
                ax.text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'No Ground Truth - Slice {slice_idx}')
            ax.axis('off')
        
        # 컬러바 추가
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, 
                    ticks=range(self.n_classes), label='Classes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'segmentation_slices_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_3d_segmentation(self, pred, target=None, threshold=0.5, 
                                 model_name='Model', save_path=None):
        """3D segmentation 결과 시각화 (다중 모델 지원)"""
        
        fig = go.Figure()
        
        # 예측 결과
        pred_coords = np.where(pred > 0)
        if len(pred_coords[0]) > 0:
            fig.add_trace(go.Scatter3d(
                x=pred_coords[2],
                y=pred_coords[1],
                z=pred_coords[0],
                mode='markers',
                marker=dict(
                    size=2,
                    color=pred[pred_coords],
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="Predicted Classes")
                ),
                name=f'{model_name} Prediction'
            ))
        
        # Ground Truth (있는 경우)
        if target is not None:
            target_coords = np.where(target > 0)
            if len(target_coords[0]) > 0:
                fig.add_trace(go.Scatter3d(
                    x=target_coords[2],
                    y=target_coords[1],
                    z=target_coords[0],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=target[target_coords],
                        colorscale='Plasma',
                        opacity=0.4
                    ),
                    name='Ground Truth'
                ))
        
        fig.update_layout(
            title=f'3D Segmentation Results - {model_name}',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Depth'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f'segmentation_3d_{model_name.lower()}.html')
    
    def create_comparison_gif(self, image, pred, target=None, output_path='segmentation_comparison.gif',
                            modality_idx=0, duration=100, model_name='Model'):
        """비교 GIF 생성 (다중 모델 지원)"""
        import imageio
        
        d, h, w = image.shape[1:]
        frames = []
        
        for slice_idx in range(0, d, 2):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 원본 이미지
            slice_img = image[modality_idx, slice_idx, :, :]
            axes[0].imshow(slice_img, cmap='gray')
            axes[0].set_title(f'Original - Slice {slice_idx}')
            axes[0].axis('off')
            
            # 예측 결과
            slice_pred = pred[slice_idx, :, :]
            axes[1].imshow(slice_pred, cmap='tab10', vmin=0, vmax=self.n_classes-1)
            axes[1].set_title(f'{model_name} Prediction - Slice {slice_idx}')
            axes[1].axis('off')
            
            # Ground Truth (있는 경우)
            if target is not None:
                slice_target = target[slice_idx, :, :]
                axes[2].imshow(slice_target, cmap='tab10', vmin=0, vmax=self.n_classes-1)
                axes[2].set_title(f'Ground Truth - Slice {slice_idx}')
            else:
                axes[2].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title(f'No Ground Truth - Slice {slice_idx}')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Figure를 이미지로 변환
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # RGBA를 RGB로 변환
            image_array = image_array[:, :, :3]
            frames.append(image_array)
            
            plt.close(fig)
        
        # GIF 생성
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"Comparison GIF saved to {output_path}")

class MetricsAnalyzer:
    """분할 성능 메트릭 분석 클래스 (다중 모델 지원)"""
    
    def __init__(self, class_names=['Background', 'NCR/NET', 'ED', 'ET']):
        self.class_names = class_names
        self.n_classes = len(class_names)
    
    def calculate_dice_scores(self, pred, target):
        """클래스별 Dice Score 계산"""
        dice_scores = []
        
        for class_idx in range(self.n_classes):
            pred_class = (pred == class_idx).astype(np.float32)
            target_class = (target == class_idx).astype(np.float32)
            
            intersection = np.sum(pred_class * target_class)
            union = np.sum(pred_class) + np.sum(target_class)
            
            if union == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = 2.0 * intersection / union
            
            dice_scores.append(dice)
        
        return np.array(dice_scores)
    
    def calculate_hausdorff_distance(self, pred, target, class_idx):
        """Hausdorff Distance 계산 (특정 클래스)"""
        from scipy.spatial.distance import directed_hausdorff
        
        pred_coords = np.where(pred == class_idx)
        target_coords = np.where(target == class_idx)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            return float('inf')
        
        pred_points = np.column_stack(pred_coords)
        target_points = np.column_stack(target_coords)
        
        d1 = directed_hausdorff(pred_points, target_points)[0]
        d2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(d1, d2)
    
    def plot_dice_scores(self, dice_scores_dict: Dict[str, List], save_path=None):
        """다중 모델 Dice Score 분포 시각화"""
        models = list(dice_scores_dict.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Model Dice Score Analysis', fontsize=16)
        
        # 모델별 색상 설정
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # 1. 박스플롯 (모델별)
        ax1 = axes[0, 0]
        box_data = []
        box_labels = []
        for model in models:
            dice_scores_array = np.array(dice_scores_dict[model])
            mean_dice = np.mean(dice_scores_array, axis=0)
            box_data.append(mean_dice[1:])  # 배경 제외
            box_labels.append(model)
        
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_title('Mean Dice Score by Model')
        ax1.set_ylabel('Dice Score')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 클래스별 비교
        ax2 = axes[0, 1]
        x = np.arange(len(self.class_names[1:]))  # 배경 제외
        width = 0.8 / n_models
        
        for i, model in enumerate(models):
            dice_scores_array = np.array(dice_scores_dict[model])
            mean_dice = np.mean(dice_scores_array, axis=0)
            ax2.bar(x + i * width, mean_dice[1:], width, 
                   label=model, color=colors[i], alpha=0.8)
        
        ax2.set_title('Mean Dice Score by Class')
        ax2.set_ylabel('Dice Score')
        ax2.set_xlabel('Classes')
        ax2.set_xticks(x + width * (n_models - 1) / 2)
        ax2.set_xticklabels(self.class_names[1:])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 전체 성능 비교
        ax3 = axes[1, 0]
        overall_scores = []
        for model in models:
            dice_scores_array = np.array(dice_scores_dict[model])
            mean_dice = np.mean(dice_scores_array, axis=0)
            overall_score = np.mean(mean_dice[1:])  # 배경 제외한 평균
            overall_scores.append(overall_score)
        
        bars = ax3.bar(models, overall_scores, color=colors, alpha=0.8)
        ax3.set_title('Overall Performance Comparison')
        ax3.set_ylabel('Mean Dice Score')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, score in zip(bars, overall_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 성능 분포 히스토그램
        ax4 = axes[1, 1]
        for i, model in enumerate(models):
            dice_scores_array = np.array(dice_scores_dict[model])
            mean_dice = np.mean(dice_scores_array, axis=0)
            overall_score = np.mean(mean_dice[1:])
            ax4.hist([overall_score], bins=20, alpha=0.7, 
                    color=colors[i], label=model)
        
        ax4.set_title('Performance Distribution')
        ax4.set_xlabel('Overall Dice Score')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('multi_model_dice_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model', save_path=None):
        """혼동 행렬 시각화 (다중 모델 지원)"""
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), 
                            labels=list(range(self.n_classes)))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm

class VolumeAnalyzer:
    """3D 볼륨 분석 클래스 (다중 모델 지원)"""
    
    def __init__(self):
        pass
    
    def calculate_volume_statistics(self, segmentation, voxel_size=(1, 1, 1)):
        """볼륨 통계 계산"""
        volumes = {}
        
        for class_idx in range(1, 4):  # 배경 제외
            class_mask = (segmentation == class_idx)
            volume_voxels = np.sum(class_mask)
            volume_mm3 = volume_voxels * np.prod(voxel_size)
            volumes[f'Class_{class_idx}'] = {
                'voxels': volume_voxels,
                'volume_mm3': volume_mm3
            }
        
        return volumes
    
    def plot_volume_distribution(self, volume_stats_dict: Dict[str, List], 
                                class_names=['NCR/NET', 'ED', 'ET'], save_path=None):
        """다중 모델 볼륨 분포 시각화"""
        models = list(volume_stats_dict.keys())
        n_models = len(models)
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for class_idx, class_name in enumerate(class_names):
            ax = axes[class_idx]
            
            for i, model in enumerate(models):
                volumes = []
                for stats in volume_stats_dict[model]:
                    volumes.append(stats[f'Class_{class_idx + 1}']['volume_mm3'])
                
                ax.hist(volumes, bins=20, alpha=0.7, color=colors[i], 
                       label=model, density=True)
            
            ax.set_title(f'{class_name} Volume Distribution')
            ax.set_xlabel('Volume (mm³)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('multi_model_volume_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def comprehensive_analysis_multi_model(models_dict: Dict[str, torch.nn.Module], 
                                      data_loader, device, num_samples=5, 
                                      results_dir=None):
    """다중 모델 종합 분석 수행"""
    
    visualizer = SegmentationVisualizer()
    metrics_analyzer = MetricsAnalyzer()
    volume_analyzer = VolumeAnalyzer()
    
    all_dice_scores = {model_name: [] for model_name in models_dict.keys()}
    all_volume_stats = {model_name: [] for model_name in models_dict.keys()}
    
    # 결과 디렉토리 생성
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {model_name}")
        print(f"{'='*50}")
        
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_samples:
                    break
                    
                images = batch['image'].to(device)
                targets = batch['segmentation'].to(device)
                patient_id = batch.get('patient_id', [f'patient_{i}'])[0] if 'patient_id' in batch else f'patient_{i}'
                
                print(f"Processing sample {i+1}: {patient_id}")
                
                # 예측
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # CPU로 이동
                image_np = images[0].cpu().numpy()
                pred_np = predictions[0].cpu().numpy()
                target_np = targets[0].cpu().numpy()
                
                # Dice Score 계산
                dice_scores = metrics_analyzer.calculate_dice_scores(pred_np, target_np)
                all_dice_scores[model_name].append(dice_scores)
                print(f"Dice Scores: {dict(zip(metrics_analyzer.class_names, dice_scores))}")
                
                # 볼륨 통계 계산
                volume_stats = volume_analyzer.calculate_volume_statistics(pred_np)
                all_volume_stats[model_name].append(volume_stats)
                
                # 시각화 (결과 디렉토리에 저장)
                if results_dir:
                    slice_path = os.path.join(results_dir, f'segmentation_slices_{model_name.lower()}_{patient_id}.png')
                    html_path = os.path.join(results_dir, f'segmentation_3d_{model_name.lower()}_{patient_id}.html')
                    gif_path = os.path.join(results_dir, f'comparison_{model_name.lower()}_{patient_id}.gif')
                else:
                    slice_path = f'segmentation_slices_{model_name.lower()}_{patient_id}.png'
                    html_path = f'segmentation_3d_{model_name.lower()}_{patient_id}.html'
                    gif_path = f'comparison_{model_name.lower()}_{patient_id}.gif'
                
                visualizer.visualize_prediction_slices(image_np, pred_np, target_np, 
                                                     model_name=model_name, save_path=slice_path)
                visualizer.visualize_3d_segmentation(pred_np, target_np, 
                                                   model_name=model_name, save_path=html_path)
                visualizer.create_comparison_gif(image_np, pred_np, target_np, 
                                               output_path=gif_path, model_name=model_name)
    
    # 전체 결과 분석
    print("\n" + "="*60)
    print("MULTI-MODEL COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)
    
    # Dice Score 분석
    dice_save_path = os.path.join(results_dir, 'multi_model_dice_analysis.png') if results_dir else None
    metrics_analyzer.plot_dice_scores(all_dice_scores, save_path=dice_save_path)
    
    # 볼륨 분포 분석
    volume_save_path = os.path.join(results_dir, 'multi_model_volume_distribution.png') if results_dir else None
    volume_analyzer.plot_volume_distribution(all_volume_stats, save_path=volume_save_path)
    
    # 평균 성능 비교
    print(f"\nModel Performance Summary:")
    print("-" * 40)
    for model_name in models_dict.keys():
        dice_scores_array = np.array(all_dice_scores[model_name])
        mean_dice = np.mean(dice_scores_array, axis=0)
        overall_score = np.mean(mean_dice[1:])  # 배경 제외
        
        print(f"{model_name}:")
        for name, score in zip(metrics_analyzer.class_names, mean_dice):
            print(f"  {name}: {score:.4f}")
        print(f"  Overall Mean Dice: {overall_score:.4f}")
        print()

if __name__ == "__main__":
    # 테스트
    from baseline import UNet3D_Simplified, UNETR_Simplified, SwinUNETR_Simplified
    from data_loader_kaggle import get_data_loaders_kaggle
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 다중 모델 로드
    models_dict = {
        'UNet3D': UNet3D_Simplified(n_channels=4, n_classes=4).to(device),
        'UNETR': UNETR_Simplified(img_size=(64, 64, 64), patch_size=(8, 8, 8),
                                 in_channels=4, out_channels=4).to(device),
        'SwinUNETR': SwinUNETR_Simplified(img_size=(64, 64, 64), patch_size=(4, 4, 4),
                                         in_channels=4, out_channels=4).to(device)
    }
    
    # 데이터 로더
    train_loader, val_loader = get_data_loaders_kaggle('.', batch_size=1, num_workers=0, max_samples=3)
    
    # 다중 모델 종합 분석
    comprehensive_analysis_multi_model(models_dict, val_loader, device, num_samples=2, results_dir='test_results')
