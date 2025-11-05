#!/usr/bin/env python3
"""
3D Segmentation Results Visualization (DataFrame-based)
다중 모델 비교를 위한 DataFrame 기반 시각화 모듈
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# FLOPs 계산을 위한 라이브러리
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop library not available. FLOPs calculation will be skipped.")
    print("   Install with: pip install thop")

def calculate_flops(model, input_size=(1, 4, 64, 64, 64)):
    """모델의 FLOPs 계산"""
    if not THOP_AVAILABLE:
        return 0
    
    try:
        # 모델의 디바이스 확인
        device = next(model.parameters()).device
        
        # 더미 입력을 모델과 같은 디바이스에 생성
        dummy_input = torch.randn(input_size).to(device)
        
        # FLOPs 계산
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops
    except Exception as e:
        print(f"⚠️  Error calculating FLOPs: {e}")
        return 0

def create_learning_curves_chart(epochs_df, results_dir):
    """학습 곡선 차트 생성
    Train/Val Loss와 Train/Val Dice Score를 각각 겹쳐서 보여주는 2개의 서브플롯
    """
    if epochs_df.empty:
        print("⚠️  No epoch data available for learning curves")
        return
    
    # 모델별로 색상 설정
    models = epochs_df['model_name'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # 2개의 서브플롯 (세로로 나열)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Learning Curves: Model Convergence and Overfitting Analysis', fontsize=14, fontweight='bold')
    
    # 1. Train Loss vs Val Loss (겹쳐서 표시)
    ax1 = axes[0]
    has_loss_data = False
    for i, model in enumerate(models):
        model_data = epochs_df[epochs_df['model_name'] == model].copy()
        model_data = model_data.sort_values('epoch')
        
        # Train Loss와 Val Loss가 모두 있는 경우에만 플롯
        if 'train_loss' in model_data.columns and 'val_loss' in model_data.columns:
            # NaN이 아닌 유효한 데이터가 있는지 확인
            train_loss_valid = model_data['train_loss'].notna().any()
            val_loss_valid = model_data['val_loss'].notna().any()
            
            if train_loss_valid:
                ax1.plot(model_data['epoch'], model_data['train_loss'], 
                        label=f'{model.upper()} (Train)', color=colors[i], linewidth=2, linestyle='-', alpha=0.8)
                has_loss_data = True
            if val_loss_valid:
                ax1.plot(model_data['epoch'], model_data['val_loss'], 
                        label=f'{model.upper()} (Val)', color=colors[i], linewidth=2, linestyle='--', alpha=0.8)
                has_loss_data = True
    
    if not has_loss_data:
        ax1.text(0.5, 0.5, 'No loss data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, fontsize=12, color='red')
        print("⚠️  Warning: No valid loss data found for learning curves")
    
    ax1.set_title('Loss: Training vs Validation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Train Dice vs Val Dice (겹쳐서 표시)
    ax2 = axes[1]
    has_dice_data = False
    for i, model in enumerate(models):
        model_data = epochs_df[epochs_df['model_name'] == model].copy()
        model_data = model_data.sort_values('epoch')
        
        # Train Dice와 Val Dice가 모두 있는 경우에만 플롯
        if 'train_dice' in model_data.columns and 'val_dice' in model_data.columns:
            # NaN이 아닌 유효한 데이터가 있는지 확인
            train_dice_valid = model_data['train_dice'].notna().any()
            val_dice_valid = model_data['val_dice'].notna().any()
            
            if train_dice_valid:
                ax2.plot(model_data['epoch'], model_data['train_dice'], 
                        label=f'{model.upper()} (Train)', color=colors[i], linewidth=2, linestyle='-', alpha=0.8)
                has_dice_data = True
            if val_dice_valid:
                ax2.plot(model_data['epoch'], model_data['val_dice'], 
                        label=f'{model.upper()} (Val)', color=colors[i], linewidth=2, linestyle='--', alpha=0.8)
                has_dice_data = True
    
    if not has_dice_data:
        ax2.text(0.5, 0.5, 'No dice data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12, color='red')
        print("⚠️  Warning: No valid dice data found for learning curves")
    
    ax2.set_title('Dice Score: Training vs Validation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Dice Score', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장 (learning_curve.png로 저장)
    chart_path = os.path.join(results_dir, "learning_curve.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves chart saved to: {chart_path}")
    plt.close()

def create_model_comparison_chart(results_df, results_dir):
    """모델 비교 차트 생성"""
    if results_df.empty:
        print("⚠️  No results data available for model comparison")
        return
    
    # 모델별 평균 성능 계산
    model_stats = results_df.groupby('model_name').agg({
        'test_dice': ['mean', 'std'],
        'val_dice': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'total_params': ['mean'],
        'flops': ['mean']
    }).round(4)
    
    # 차트 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('3D Segmentation Model Comparison', fontsize=16)
    
    models = results_df['model_name'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Test Dice Score
    ax1 = axes[0, 0]
    test_dice_means = [model_stats.loc[model, ('test_dice', 'mean')] for model in models]
    test_dice_stds = [model_stats.loc[model, ('test_dice', 'std')] for model in models]
    bars1 = ax1.bar(models, test_dice_means, yerr=test_dice_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax1.set_title('Test Dice Score')
    ax1.set_ylabel('Dice Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Validation Dice Score
    ax2 = axes[0, 1]
    val_dice_means = [model_stats.loc[model, ('val_dice', 'mean')] for model in models]
    val_dice_stds = [model_stats.loc[model, ('val_dice', 'std')] for model in models]
    bars2 = ax2.bar(models, val_dice_means, yerr=val_dice_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax2.set_title('Validation Dice Score')
    ax2.set_ylabel('Dice Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Precision
    ax3 = axes[0, 2]
    precision_means = [model_stats.loc[model, ('precision', 'mean')] for model in models]
    precision_stds = [model_stats.loc[model, ('precision', 'std')] for model in models]
    bars3 = ax3.bar(models, precision_means, yerr=precision_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax3.set_title('Precision')
    ax3.set_ylabel('Precision')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Recall
    ax4 = axes[1, 0]
    recall_means = [model_stats.loc[model, ('recall', 'mean')] for model in models]
    recall_stds = [model_stats.loc[model, ('recall', 'std')] for model in models]
    bars4 = ax4.bar(models, recall_means, yerr=recall_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax4.set_title('Recall')
    ax4.set_ylabel('Recall')
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='x', rotation=45)
    
    # Parameters
    ax5 = axes[1, 1]
    param_means = [model_stats.loc[model, ('total_params', 'mean')] for model in models]
    bars5 = ax5.bar(models, param_means, color=colors, alpha=0.7)
    ax5.set_title('Model Parameters')
    ax5.set_ylabel('Number of Parameters')
    ax5.tick_params(axis='x', rotation=45)
    ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # FLOPs
    ax6 = axes[1, 2]
    flops_means = [model_stats.loc[model, ('flops', 'mean')] for model in models]
    bars6 = ax6.bar(models, flops_means, color=colors, alpha=0.7)
    ax6.set_title('FLOPs')
    ax6.set_ylabel('FLOPs')
    ax6.tick_params(axis='x', rotation=45)
    ax6.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 저장
    chart_path = os.path.join(results_dir, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison chart saved to: {chart_path}")
    plt.close()


def create_wt_tc_et_learning_curves(epochs_df, results_dir):
    """WT/TC/ET Validation Dice 학습 곡선(모델별)을 별도 이미지로 저장.
    파일: learning_curve_wt_tc_et.png
    """
    if epochs_df.empty:
        print("⚠️  No epoch data for WT/TC/ET curves")
        return
    needed = {'model_name', 'epoch', 'val_wt', 'val_tc', 'val_et'}
    if not needed.issubset(set(epochs_df.columns)):
        print("⚠️  epochs_df lacks WT/TC/ET columns; skipping WT/TC/ET curves")
        return

    models = epochs_df['model_name'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Validation Curves: WT / TC / ET', fontsize=14, fontweight='bold')
    metrics = [('val_wt', 'WT'), ('val_tc', 'TC'), ('val_et', 'ET')]

    for ax, (col, title) in zip(axes, metrics):
        for i, m in enumerate(models):
            dfm = epochs_df[epochs_df['model_name'] == m].sort_values('epoch')
            if col in dfm.columns and dfm[col].notna().any():
                ax.plot(dfm['epoch'], dfm[col], label=m.upper(), color=colors[i], linewidth=2)
        ax.set_title(f'{title} (Validation)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Dice', fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'learning_curve_wt_tc_et.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"WT/TC/ET learning curves saved to: {out_path}")
    plt.close()


def create_wt_tc_et_summary(results_df, results_dir):
    """모델별 WT/TC/ET Test & Val Dice 요약 바차트.
    파일: wt_tc_et_summary_test.png, wt_tc_et_summary_val.png
    """
    if results_df.empty:
        print("⚠️  No results data for WT/TC/ET summary")
        return

    models = results_df['model_name'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # Test summary
    req_test = {'test_wt', 'test_tc', 'test_et'}
    if req_test.issubset(set(results_df.columns)):
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        x = np.arange(len(models))
        width = 0.22
        means_wt = [results_df[results_df['model_name'] == m]['test_wt'].mean() for m in models]
        means_tc = [results_df[results_df['model_name'] == m]['test_tc'].mean() for m in models]
        means_et = [results_df[results_df['model_name'] == m]['test_et'].mean() for m in models]
        ax.bar(x - width, means_wt, width, label='WT')
        ax.bar(x,         means_tc, width, label='TC')
        ax.bar(x + width, means_et, width, label='ET')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], rotation=30)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Dice')
        ax.set_title('WT/TC/ET Test Dice Summary')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
        outp = os.path.join(results_dir, 'wt_tc_et_summary_test.png')
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        print(f"WT/TC/ET test summary saved to: {outp}")
        plt.close()

    # Val summary
    req_val = {'val_wt', 'val_tc', 'val_et'}
    if req_val.issubset(set(results_df.columns)):
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        x = np.arange(len(models))
        width = 0.22
        means_wt = [results_df[results_df['model_name'] == m]['val_wt'].mean() for m in models]
        means_tc = [results_df[results_df['model_name'] == m]['val_tc'].mean() for m in models]
        means_et = [results_df[results_df['model_name'] == m]['val_et'].mean() for m in models]
        ax.bar(x - width, means_wt, width, label='WT')
        ax.bar(x,         means_tc, width, label='TC')
        ax.bar(x + width, means_et, width, label='ET')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], rotation=30)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Dice')
        ax.set_title('WT/TC/ET Validation Dice Summary (Best Epoch)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
        outp = os.path.join(results_dir, 'wt_tc_et_summary_val.png')
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        print(f"WT/TC/ET val summary saved to: {outp}")
        plt.close()

def create_parameter_efficiency_chart(results_df, results_dir):
    """파라미터 효율성 차트 생성"""
    if results_df.empty:
        print("⚠️  No results data available for parameter efficiency analysis")
        return
    
    # 모델별 평균 성능 계산
    model_stats = results_df.groupby('model_name').agg({
        'test_dice': 'mean',
        'val_dice': 'mean',
        'total_params': 'mean',
        'flops': 'mean'
    }).round(4)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('3D Segmentation Model Efficiency Analysis', fontsize=16)
    
    models = results_df['model_name'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Parameters vs Performance
    ax1 = axes[0]
    for i, model in enumerate(models):
        ax1.scatter(model_stats.loc[model, 'total_params'], 
                   model_stats.loc[model, 'test_dice'],
                   s=100, color=colors[i], alpha=0.7, label=model.upper())
    
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Test Dice Score')
    ax1.set_title('Parameters vs Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # FLOPs vs Performance
    ax2 = axes[1]
    for i, model in enumerate(models):
        ax2.scatter(model_stats.loc[model, 'flops'], 
                   model_stats.loc[model, 'test_dice'],
                   s=100, color=colors[i], alpha=0.7, label=model.upper())
    
    ax2.set_xlabel('FLOPs')
    ax2.set_ylabel('Test Dice Score')
    ax2.set_title('FLOPs vs Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 저장
    chart_path = os.path.join(results_dir, "parameter_efficiency.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Parameter efficiency chart saved to: {chart_path}")
    plt.close()

def create_3d_segmentation_visualization(model, test_loader, device='cuda', num_samples=3, results_dir=None):
    """3D 세그멘테이션 결과 시각화"""
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            
            # 첫 번째 샘플만 시각화
            if i == 0:
                # 3D 시각화를 위한 슬라이스 선택
                input_vol = inputs[0].cpu().numpy()  # (4, D, H, W)
                pred_vol = pred[0].cpu().numpy()      # (D, H, W)
                label_vol = labels[0].cpu().numpy()   # (D, H, W)
                
                # 중간 슬라이스들 선택
                d, h, w = pred_vol.shape
                slice_indices = [d//4, d//2, 3*d//4]
                
                # 슬라이스 시각화
                fig, axes = plt.subplots(3, 5, figsize=(20, 12))
                fig.suptitle('3D Segmentation Results Visualization', fontsize=16)
                
                modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
                
                for row, slice_idx in enumerate(slice_indices):
                    # 원본 이미지들
                    for col in range(4):
                        ax = axes[row, col]
                        slice_img = input_vol[col, slice_idx, :, :]
                        ax.imshow(slice_img, cmap='gray')
                        ax.set_title(f'{modality_names[col]} - Slice {slice_idx}')
                        ax.axis('off')
                    
                    # 예측 결과
                    ax = axes[row, 4]
                    slice_pred = pred_vol[slice_idx, :, :]
                    im = ax.imshow(slice_pred, cmap='tab10', vmin=0, vmax=3)
                    ax.set_title(f'Prediction - Slice {slice_idx}')
                    ax.axis('off')
                
                # 컬러바 추가
                fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, 
                           ticks=range(4), label='Classes')
                
                plt.tight_layout()
                
                if results_dir:
                    chart_path = os.path.join(results_dir, "segmentation_visualization.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    print(f"Segmentation visualization saved to: {chart_path}")
                else:
                    plt.savefig("segmentation_visualization.png", dpi=300, bbox_inches='tight')
                
                plt.close()
                break

def create_comprehensive_analysis(results_df, epochs_df, results_dir):
    """종합 분석 차트 생성"""
    print("\nCreating comprehensive analysis charts...")
    
    # 학습 곡선 차트
    if not epochs_df.empty:
        create_learning_curves_chart(epochs_df, results_dir)
        # WT/TC/ET 전용 학습 곡선(별도 파일)
        create_wt_tc_et_learning_curves(epochs_df, results_dir)
    
    # 모델 비교 차트
    if not results_df.empty:
        create_model_comparison_chart(results_df, results_dir)
        create_parameter_efficiency_chart(results_df, results_dir)
        # WT/TC/ET 요약 바차트(별도 파일)
        create_wt_tc_et_summary(results_df, results_dir)
    
    print("✅ Comprehensive analysis charts created successfully!")

def create_interactive_3d_plot(results_df, results_dir):
    """인터랙티브 3D 플롯 생성 (Plotly)
    
    Parameters, FLOPs, Test Dice Score의 3D 관계를 시각화합니다.
    """
    if results_df.empty:
        print("⚠️  No results data available for interactive plot")
        return
    
    # 필수 컬럼 확인
    required_cols = ['model_name', 'test_dice', 'total_params', 'flops']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"⚠️  Missing required columns for interactive plot: {missing_cols}")
        print(f"   Available columns: {list(results_df.columns)}")
        return
    
    # 모델별 평균 성능 계산 (필요한 컬럼만)
    agg_dict = {
        'test_dice': 'mean',
        'total_params': 'mean',
        'flops': 'mean'
    }
    
    # 선택적 컬럼 추가
    if 'val_dice' in results_df.columns:
        agg_dict['val_dice'] = 'mean'
    if 'precision' in results_df.columns:
        agg_dict['precision'] = 'mean'
    if 'recall' in results_df.columns:
        agg_dict['recall'] = 'mean'
    
    model_stats = results_df.groupby('model_name').agg(agg_dict).round(4)
    
    # NaN 값 제거 (모델별로 하나 이상의 유효한 값이 있어야 함)
    model_stats = model_stats.dropna(subset=['test_dice', 'total_params', 'flops'])
    
    if model_stats.empty:
        print("⚠️  No valid data after grouping and cleaning for interactive plot")
        return
    
    # 호버 텍스트 생성 (모든 메트릭 포함)
    hover_texts = []
    for model_name in model_stats.index:
        row = model_stats.loc[model_name]
        hover_parts = [
            f"<b>{model_name.upper()}</b>",
            f"Test Dice: {row['test_dice']:.4f}"
        ]
        if 'val_dice' in row:
            hover_parts.append(f"Val Dice: {row['val_dice']:.4f}")
        hover_parts.extend([
            f"Parameters: {row['total_params']:,.0f}",
            f"FLOPs: {row['flops']:,.0f}"
        ])
        if 'precision' in row:
            hover_parts.append(f"Precision: {row['precision']:.4f}")
        if 'recall' in row:
            hover_parts.append(f"Recall: {row['recall']:.4f}")
        hover_texts.append("<br>".join(hover_parts))
    
    # 3D 산점도 생성
    fig = go.Figure(data=go.Scatter3d(
        x=model_stats['total_params'],
        y=model_stats['flops'],
        z=model_stats['test_dice'],
        mode='markers+text',
        marker=dict(
            size=15,
            color=model_stats['test_dice'],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(
                title=dict(
                    text="Test Dice Score",
                    font=dict(size=12)
                )
            ),
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=model_stats.index,
        textposition="top center",
        textfont=dict(size=10, color='black'),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': '3D Model Performance Analysis: Parameters vs FLOPs vs Test Dice Score',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'family': 'Arial Black'}
        },
        scene=dict(
            xaxis_title=dict(text='Parameters', font=dict(size=12)),
            yaxis_title=dict(text='FLOPs', font=dict(size=12)),
            zaxis_title=dict(text='Test Dice Score', font=dict(size=12)),
            xaxis=dict(
                type='log' if model_stats['total_params'].max() / model_stats['total_params'].min() > 100 else 'linear',
                showspikes=False
            ),
            yaxis=dict(
                type='log' if model_stats['flops'].max() / model_stats['flops'].min() > 100 else 'linear',
                showspikes=False
            ),
            zaxis=dict(
                range=[max(0, model_stats['test_dice'].min() - 0.1), 
                       min(1, model_stats['test_dice'].max() + 0.1)],
                showspikes=False
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # 저장
    chart_path = os.path.join(results_dir, "interactive_3d_analysis.html")
    try:
        fig.write_html(chart_path)
        print(f"✅ Interactive 3D analysis saved to: {chart_path}")
    except Exception as e:
        print(f"⚠️  Error saving interactive 3D plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 테스트용 더미 데이터
    print("Testing visualization functions...")
    
    # 더미 결과 데이터 생성
    dummy_results = pd.DataFrame({
        'model_name': ['unet3d', 'unetr', 'swin_unetr'] * 2,
        'seed': [24, 24, 24, 42, 42, 42],
        'test_dice': [0.85, 0.87, 0.89, 0.84, 0.86, 0.88],
        'val_dice': [0.83, 0.85, 0.87, 0.82, 0.84, 0.86],
        'precision': [0.80, 0.82, 0.84, 0.79, 0.81, 0.83],
        'recall': [0.78, 0.80, 0.82, 0.77, 0.79, 0.81],
        'total_params': [1000000, 2000000, 3000000, 1000000, 2000000, 3000000],
        'flops': [5000000000, 10000000000, 15000000000, 5000000000, 10000000000, 15000000000]
    })
    
    # 더미 에포크 데이터 생성
    dummy_epochs = pd.DataFrame({
        'model_name': ['unet3d', 'unetr', 'swin_unetr'] * 10,
        'epoch': list(range(1, 11)) * 3,
        'train_loss': np.random.uniform(0.1, 0.5, 30),
        'val_dice': np.random.uniform(0.7, 0.9, 30),
        'test_dice': np.random.uniform(0.7, 0.9, 30)
    })
    
    # 테스트 디렉토리 생성
    test_dir = "test_visualization"
    os.makedirs(test_dir, exist_ok=True)
    
    # 차트 생성 테스트
    create_comprehensive_analysis(dummy_results, dummy_epochs, test_dir)
    create_interactive_3d_plot(dummy_results, test_dir)
    
    print("✅ Visualization test completed!")
