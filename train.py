import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import imageio

from data_loader_kaggle import BratsKaggleDataset, get_data_loaders_kaggle
from model_3d_unet import UNet3D_Simplified, combined_loss, calculate_dice_score
from gradcam_3d import GradCAM3D, analyze_gradcam
from visualization import comprehensive_analysis

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 손실 함수
        self.criterion = combined_loss
        
        # 메트릭 저장
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        self.best_dice = 0.0
        
    def train_epoch(self):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['segmentation'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
            
            # 메모리 정리 (더 자주)
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate(self):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['segmentation'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Dice score 계산
                dice_scores = calculate_dice_score(outputs, targets)
                avg_dice = dice_scores.mean().item()
                
                total_loss += loss.item()
                total_dice += avg_dice
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{avg_dice:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    def train(self, start_epoch=0):
        """전체 훈련 과정"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # 훈련
            train_loss = self.train_epoch()
            
            # 검증
            val_loss, val_dice = self.validate()
            
            # 스케줄러 업데이트
            self.scheduler.step(val_loss)
            
            # 메트릭 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_dice_scores.append(val_dice)
            
            # 결과 출력
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {val_dice:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 최고 성능 모델 저장
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint(epoch, val_dice, is_best=True)
                print(f"New best model saved! Dice: {val_dice:.4f}")
            
            # 정기 체크포인트 저장
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_dice, is_best=False)
            
            # 조기 종료 체크
            if self.config['early_stopping'] and epoch > 10:
                if self._check_early_stopping():
                    print("Early stopping triggered!")
                    break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        
        # 훈련 결과 플롯
        self.plot_training_history()
        
        # 종합 분석 실행
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS")
        print("="*60)
        self.comprehensive_analysis()
    
    def _check_early_stopping(self):
        """조기 종료 체크"""
        if len(self.val_dice_scores) < 10:
            return False
        
        recent_scores = self.val_dice_scores[-10:]
        if max(recent_scores) - min(recent_scores) < 0.001:
            return True
        return False
    
    def save_checkpoint(self, epoch, dice_score, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dice_score': dice_score,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores
        }
        
        # 체크포인트 디렉토리 생성
        os.makedirs('checkpoints', exist_ok=True)
        
        # 일반 체크포인트 저장
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = 'checkpoints/best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def plot_training_history(self):
        """훈련 히스토리 플롯"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss 플롯
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Dice Score 플롯
        ax2.plot(self.val_dice_scores, label='Validation Dice Score', color='green')
        ax2.set_title('Validation Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_analysis(self):
        """종합 분석 실행"""
        try:
            print("종합 분석 실행 중...")
            # comprehensive_analysis 함수 사용
            comprehensive_analysis(self.model, self.val_loader, self.device, num_samples=1)
            print("\n✅ 종합 분석 완료!")
            print("생성된 파일들:")
            print("  - training_history.png")
            print("  - segmentation_slices.png")
            print("  - volume_distribution.png")
            print("  - dice_scores_analysis.png")
            print("  - comparison_*.gif")
            print("  - segmentation_3d.html")
            
        except Exception as e:
            print(f"✗ 종합 분석 실패: {e}")
            import traceback
            traceback.print_exc()

def main():
    # 설정 (전체 데이터셋 훈련용)
    config = {
        'batch_size': 1,  # 메모리 제약으로 배치 크기 1
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 30,  # 30 에포크로 증가
        'save_interval': 5,  # 5 에포크마다 저장
        'early_stopping': True,
        'patience': 15,  # 15 에포크 동안 개선 없으면 조기 종료
        'data_dir': 'data'
    }
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 메모리 관리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {total_memory:.1f} GB")
        print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    
    # 데이터 로더 생성 (Kaggle 데이터 사용 - 안전한 크기부터)
    print("Loading Kaggle BraTS data...")
    train_loader, val_loader = get_data_loaders_kaggle(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=0,  # Windows에서 안정성을 위해 0으로 설정
        max_samples=5  # 안정성을 위해 5개로 다시 줄임
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 모델 생성
    model = UNet3D_Simplified(n_channels=4, n_classes=4).to(device)
    
    # 체크포인트에서 이어서 훈련 (있는 경우)
    start_epoch = 0
    if os.path.exists('checkpoints/checkpoint_epoch_20.pth'):
        print("체크포인트에서 이어서 훈련합니다...")
        checkpoint = torch.load('checkpoints/checkpoint_epoch_20.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"에포크 {start_epoch}부터 이어서 훈련합니다.")
    
    # 트레이너 생성 및 훈련 시작
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()

