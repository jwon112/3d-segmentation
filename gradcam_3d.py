import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2

class GradCAM3D:
    """3D Grad-CAM 구현"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hook_layers()
        
    def hook_layers(self):
        """타겟 레이어에 훅 등록"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        # 타겟 레이어 찾기
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
                
        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found in model")
            
        # 훅 등록
        self.backward_hook = target_layer.register_backward_hook(backward_hook)
        self.forward_hook = target_layer.register_forward_hook(forward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Grad-CAM 생성"""
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            # 3D 세그멘테이션에서는 전체 볼륨의 평균 클래스 확률 사용
            avg_probs = output.mean(dim=(2, 3, 4))  # 공간 차원에 대해 평균
            class_idx = avg_probs.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        # 3D 세그멘테이션에서는 특정 클래스의 평균 활성화 사용
        class_output = output[0, class_idx].mean()
        class_output.backward()
        
        # Grad-CAM 계산
        gradients = self.gradients[0]  # [C, D, H, W]
        activations = self.activations[0]  # [C, D, H, W]
        
        # Global Average Pooling으로 가중치 계산
        weights = torch.mean(gradients, dim=(1, 2, 3))  # [C]
        
        # 가중합으로 CAM 생성 (같은 디바이스에서)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # [D, H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU 적용 (양수 부분만)
        cam = F.relu(cam)
        
        # 정규화
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        """훅 제거"""
        self.backward_hook.remove()
        self.forward_hook.remove()

class GradCAM3DVisualizer:
    """3D Grad-CAM 시각화 클래스"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def visualize_slices(self, image, cam, segmentation=None, slice_indices=None, 
                        modality_names=['FLAIR', 'T1', 'T1CE', 'T2'], 
                        class_names=['Background', 'NCR/NET', 'ED', 'ET']):
        """2D 슬라이스로 시각화"""
        
        if slice_indices is None:
            # 중간 슬라이스들 선택
            d, h, w = image.shape[1:]
            slice_indices = [d//4, d//2, 3*d//4]
        
        n_slices = len(slice_indices)
        n_modalities = image.shape[0]
        
        fig, axes = plt.subplots(n_slices, n_modalities + 2, figsize=(4*(n_modalities + 2), 4*n_slices))
        if n_slices == 1:
            axes = axes.reshape(1, -1)
        
        for i, slice_idx in enumerate(slice_indices):
            for j in range(n_modalities):
                ax = axes[i, j]
                slice_img = image[j, slice_idx, :, :]
                ax.imshow(slice_img, cmap='gray')
                ax.set_title(f'{modality_names[j]} - Slice {slice_idx}')
                ax.axis('off')
            
            # Grad-CAM 시각화
            ax = axes[i, n_modalities]
            slice_cam = cam[slice_idx, :, :]
            im = ax.imshow(slice_cam, cmap='jet', alpha=0.7)
            ax.set_title(f'Grad-CAM - Slice {slice_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Segmentation 마스크 (있는 경우)
            ax = axes[i, n_modalities + 1]
            if segmentation is not None:
                slice_seg = segmentation[slice_idx, :, :]
                im = ax.imshow(slice_seg, cmap='tab10', alpha=0.7)
                ax.set_title(f'Segmentation - Slice {slice_idx}')
            else:
                ax.text(0.5, 0.5, 'No Segmentation', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'No Segmentation - Slice {slice_idx}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('gradcam_slices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_3d_volume(self, image, cam, segmentation=None, threshold=0.3):
        """3D 볼륨 시각화 (Plotly 사용)"""
        
        # CAM에서 임계값 이상인 부분만 추출
        cam_binary = (cam > threshold).astype(np.uint8)
        
        # 3D 좌표 추출
        coords = np.where(cam_binary)
        if len(coords[0]) == 0:
            print("No significant regions found in Grad-CAM")
            return
            
        # Plotly 3D 시각화
        fig = go.Figure()
        
        # Grad-CAM 포인트들
        fig.add_trace(go.Scatter3d(
            x=coords[2],  # Width
            y=coords[1],  # Height  
            z=coords[0],  # Depth
            mode='markers',
            marker=dict(
                size=2,
                color=cam[coords],
                colorscale='Jet',
                opacity=0.6,
                colorbar=dict(title="Grad-CAM Intensity")
            ),
            name='Grad-CAM'
        ))
        
        # Segmentation 마스크 (있는 경우)
        if segmentation is not None:
            seg_coords = np.where(segmentation > 0)
            if len(seg_coords[0]) > 0:
                fig.add_trace(go.Scatter3d(
                    x=seg_coords[2],
                    y=seg_coords[1], 
                    z=seg_coords[0],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=segmentation[seg_coords],
                        colorscale='Viridis',
                        opacity=0.4
                    ),
                    name='Segmentation'
                ))
        
        fig.update_layout(
            title='3D Grad-CAM Visualization',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Depth'
            ),
            width=800,
            height=600
        )
        
        fig.write_html('gradcam_3d.html')
        # fig.show()  # 터미널 출력 방지를 위해 주석 처리
    
    def create_gif_animation(self, image, cam, output_path='gradcam_animation.gif', 
                           modality_idx=0, duration=100):
        """GIF 애니메이션 생성"""
        import imageio
        
        d, h, w = image.shape[1:]
        frames = []
        
        for slice_idx in range(0, d, 2):  # 2슬라이스마다
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 원본 이미지
            slice_img = image[modality_idx, slice_idx, :, :]
            ax1.imshow(slice_img, cmap='gray')
            ax1.set_title(f'Original - Slice {slice_idx}')
            ax1.axis('off')
            
            # Grad-CAM
            slice_cam = cam[slice_idx, :, :]
            im = ax2.imshow(slice_cam, cmap='jet', alpha=0.7)
            ax2.set_title(f'Grad-CAM - Slice {slice_idx}')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Figure를 이미지로 변환
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image_array)
            
            plt.close(fig)
        
        # GIF 생성
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"GIF animation saved to {output_path}")

def analyze_gradcam(model, data_loader, device, target_layer='bottleneck', 
                   num_samples=3, class_names=['Background', 'NCR/NET', 'ED', 'ET']):
    """Grad-CAM 분석 및 시각화"""
    
    model.eval()
    visualizer = GradCAM3DVisualizer(model, device)
    
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
            
        images = batch['image'].to(device)
        segmentations = batch['segmentation'].to(device)
        patient_id = batch.get('patient_id', [f'patient_{i}'])[0] if 'patient_id' in batch else f'patient_{i}'
        
        print(f"\nAnalyzing sample {i+1}: {patient_id}")
        
        # 예측 (gradient 필요 없음)
        with torch.no_grad():
            outputs = model(images)
            # 3D 세그멘테이션에서는 전체 볼륨의 평균 클래스 확률을 사용
            probs = F.softmax(outputs, dim=1)
            avg_probs = probs.mean(dim=(2, 3, 4))  # 공간 차원에 대해 평균
            predicted_class = torch.argmax(avg_probs, dim=1).item()
            confidence = avg_probs[0, predicted_class].item()
            
            print(f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        
        # Grad-CAM 생성 (gradient 필요)
        gradcam = GradCAM3D(model, target_layer)
        cam = gradcam.generate_cam(images, class_idx=predicted_class)
        gradcam.remove_hooks()
        
        # 시각화
        image_np = images[0].cpu().detach().numpy()
        seg_np = segmentations[0].cpu().detach().numpy()
        # cam은 이미 numpy 배열로 반환됨
        
        # 2D 슬라이스 시각화
        visualizer.visualize_slices(image_np, cam, seg_np)
        
        # 3D 볼륨 시각화
        visualizer.visualize_3d_volume(image_np, cam, seg_np)
        
        # GIF 애니메이션 생성
        visualizer.create_gif_animation(image_np, cam, f'gradcam_{patient_id}.gif')

if __name__ == "__main__":
    # 테스트
    from models.model_3d_unet import UNet3D_Small as UNet3D_Simplified
    from data_loader import get_data_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = UNet3D_Simplified(n_channels=4, n_classes=4).to(device)
    
    # 데이터 로더
    train_loader, val_loader = get_data_loaders('.', batch_size=1, num_workers=0)
    
    # Grad-CAM 분석
    analyze_gradcam(model, val_loader, device, target_layer='bottleneck', num_samples=2)

