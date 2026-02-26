"""Learning Rate Schedulers for training."""

from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    """nnUNet 스타일 Polynomial Learning Rate Scheduler
    
    학습률을 다항식 함수로 부드럽게 감소시킵니다.
    
    Formula: lr = initial_lr * (1 - current_step / max_steps) ** exponent
    
    Args:
        optimizer: Optimizer
        initial_lr: 초기 학습률
        max_steps: 전체 스텝 수 (epoch 수 * iterations_per_epoch)
        exponent: 지수 (기본값 0.9, nnUNet 표준)
        current_step: 현재 스텝 (체크포인트에서 재개할 때 사용)
    
    Example:
        >>> optimizer = optim.Adam(model.parameters(), lr=1e-2)
        >>> max_steps = epochs * len(train_loader)
        >>> scheduler = PolyLRScheduler(optimizer, initial_lr=1e-2, max_steps=max_steps)
        >>> for epoch in range(epochs):
        ...     for batch in train_loader:
        ...         scheduler.step()  # 매 iteration마다 호출
    """
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        """학습률 업데이트
        
        Args:
            current_step: 현재 스텝 (None이면 자동으로 증가)
        """
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """현재 학습률 반환"""
        return self._last_lr
