#!/usr/bin/env python3
"""
BraTS 데이터 로더 (하위 호환용 래퍼)

실제 구현은 `dataloaders` 패키지로 이동했습니다.
새 코드는 `from dataloaders import ...` 사용을 권장하며,
기존 `from data_loader import ...`는 그대로 동작합니다.
"""

from dataloaders import *  # noqa: F401,F403


if __name__ == "__main__":
    # 간단 테스트: 기본 get_data_loaders 호출
    train_loader, val_loader, test_loader, *_ = get_data_loaders('data', batch_size=1, max_samples=None)  # type: ignore[name-defined]

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for image, mask in train_loader:
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        break


