#!/usr/bin/env python3
"""간단한 결과 분석"""

import torch
import os

def main():
    print("20 에포크까지의 훈련 결과 분석 시작...")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 체크포인트 확인
    checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('checkpoint_epoch_')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    print(f"발견된 체크포인트: {len(checkpoint_files)}개")
    for f in checkpoint_files:
        print(f"  - {f}")
    
    # 최신 체크포인트 로드
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"\n최신 체크포인트 로딩: {latest_checkpoint}")
        
        checkpoint_path = os.path.join('checkpoints', latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"체크포인트 에포크: {checkpoint['epoch']}")
        if 'dice_score' in checkpoint:
            print(f"체크포인트 Dice Score: {checkpoint['dice_score']:.4f}")
        
        # 훈련 히스토리 분석
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            val_dice_scores = checkpoint['val_dice_scores']
            
            print(f"\n훈련 히스토리:")
            print(f"  - 총 에포크: {len(train_losses)}")
            print(f"  - 최종 Train Loss: {train_losses[-1]:.4f}")
            print(f"  - 최종 Val Loss: {val_losses[-1]:.4f}")
            print(f"  - 최종 Val Dice: {val_dice_scores[-1]:.4f}")
            print(f"  - 최고 Val Dice: {max(val_dice_scores):.4f}")
            
            # 개선 추이 분석
            print(f"\n개선 추이:")
            if len(val_dice_scores) >= 5:
                recent_5 = val_dice_scores[-5:]
                print(f"  - 최근 5 에포크 평균 Dice: {sum(recent_5)/len(recent_5):.4f}")
            
            if len(val_dice_scores) >= 10:
                recent_10 = val_dice_scores[-10:]
                print(f"  - 최근 10 에포크 평균 Dice: {sum(recent_10)/len(recent_10):.4f}")
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()


