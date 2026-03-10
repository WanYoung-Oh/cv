"""
Mixup & CutMix 구현
데이터 증강 기법으로 과적합을 방지하고 일반화 성능을 향상시킵니다.

References:
- Mixup: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

log = logging.getLogger(__name__)


class MixupCutmix:
    """Mixup & CutMix 데이터 증강
    
    Args:
        mixup_alpha: Mixup beta 분포의 alpha 파라미터 (기본값: 0.8)
        cutmix_alpha: CutMix beta 분포의 alpha 파라미터 (기본값: 1.0)
        prob: Mixup/CutMix 적용 확률 (기본값: 0.5, 50% 배치에만 적용)
        switch_prob: Mixup vs CutMix 선택 확률 (기본값: 0.5, 50:50 비율)
        label_smoothing: Label smoothing 값 (기본값: 0.0, 사용 안 함)
        num_classes: 클래스 개수
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.0,
        num_classes: int = 17
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
        log.info(f"🎨 Mixup/CutMix 초기화:")
        log.info(f"   - mixup_alpha: {mixup_alpha}")
        log.info(f"   - cutmix_alpha: {cutmix_alpha}")
        log.info(f"   - prob: {prob} (배치 적용 확률)")
        log.info(f"   - switch_prob: {switch_prob} (Mixup vs CutMix)")
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        """Mixup/CutMix 적용
        
        Args:
            images: 배치 이미지 (B, C, H, W)
            labels: 배치 레이블 (B,)
        
        Returns:
            - 적용 안 됨: (images, labels, None, None)
            - 적용됨: (mixed_images, labels_a, labels_b, lam)
        """
        # prob 확률로 적용 여부 결정
        if np.random.rand() > self.prob:
            return images, labels, None, None
        
        batch_size = images.shape[0]
        
        # Mixup vs CutMix 선택
        if np.random.rand() < self.switch_prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(images.device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            
            return mixed_images, labels_a, labels_b, lam
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            index = torch.randperm(batch_size).to(images.device)
            
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape, lam)
            images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            # 실제 혼합 비율 조정 (bbox 크기 기반)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-1] * images.shape[-2]))
            
            labels_a, labels_b = labels, labels[index]
            return images, labels_a, labels_b, lam
    
    def _rand_bbox(
        self,
        size: Tuple[int, int, int, int],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """CutMix를 위한 랜덤 bounding box 생성
        
        Args:
            size: 이미지 크기 (B, C, H, W)
            lam: lambda 값 (혼합 비율)
        
        Returns:
            (bbx1, bby1, bbx2, bby2): bounding box 좌표
        """
        H, W = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 중심점 랜덤 선택
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box 계산 (이미지 경계 내로 clip)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Mixup/CutMix loss 계산
    
    Args:
        criterion: Loss 함수
        pred: 모델 예측 (logits)
        y_a: 첫 번째 레이블
        y_b: 두 번째 레이블
        lam: 혼합 비율
    
    Returns:
        Mixed loss
    """
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b
