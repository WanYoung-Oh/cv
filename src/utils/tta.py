"""
Test Time Augmentation (TTA) 구현
예측 시 여러 augmentation을 적용하여 결과의 안정성과 성능을 향상시킵니다.

tta_level 옵션:
  - "light"    : 5가지  (original, hflip, vflip, rot90, rot270)
  - "standard" : 8가지  (D4 완전 집합 — 모든 90° 배수 회전 × 뒤집기 조합) ← 기본값
  - "heavy"    : 11가지 (D4 + 스캔 품질 3종: brightness_up, brightness_down, sharpen)

D4 대칭군 (정사각형의 8가지 대칭):
  original, hflip, vflip, rot90, rot180, rot270, transpose, anti_transpose
  - transpose      = hflip(rot90_CW)  : 주대각선 기준 반전
  - anti_transpose = vflip(rot90_CW)  : 반대각선 기준 반전
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Callable, Optional
import logging

log = logging.getLogger(__name__)

# level별 변환 개수 참조용
TTA_LEVEL_SIZES = {"light": 5, "standard": 8, "heavy": 11}


# ─── 텐서 헬퍼 ───────────────────────────────────────────────────────────────

def _sharpen_tensor(x: torch.Tensor) -> torch.Tensor:
    """채널별 샤프닝 (3×3 언샤프 마스크 커널, depthwise conv)"""
    kernel = torch.tensor(
        [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3).contiguous()
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


# ─── numpy 헬퍼 ──────────────────────────────────────────────────────────────

def _sharpen_numpy(img: np.ndarray) -> np.ndarray:
    """numpy 이미지 샤프닝 (cv2.filter2D, 3×3 언샤프 마스크)"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)


# ─── numpy 기반 TTA ───────────────────────────────────────────────────────────

def get_tta_transforms(
    level: str = "standard",
) -> List[Tuple[str, Callable, Callable]]:
    """TTA 변환 함수 리스트 반환 (numpy 이미지 기준)

    Args:
        level: "light" | "standard" | "heavy"

    Returns:
        List of (name, transform_fn, inverse_transform_fn)
    """
    CW  = cv2.ROTATE_90_CLOCKWISE
    CCW = cv2.ROTATE_90_COUNTERCLOCKWISE
    R180 = cv2.ROTATE_180

    light = [
        ("original",
            lambda x: x,
            lambda x: x),
        ("hflip",
            lambda x: cv2.flip(x, 1),
            lambda x: cv2.flip(x, 1)),
        ("vflip",
            lambda x: cv2.flip(x, 0),
            lambda x: cv2.flip(x, 0)),
        ("rot90",
            lambda x: cv2.rotate(x, CW),
            lambda x: cv2.rotate(x, CCW)),
        ("rot270",
            lambda x: cv2.rotate(x, CCW),
            lambda x: cv2.rotate(x, CW)),
    ]

    standard_extra = [
        ("rot180",
            lambda x: cv2.rotate(x, R180),
            lambda x: cv2.rotate(x, R180)),
        # transpose: hflip(rot90_CW) — 주대각선 기준 반전 (자기 역원)
        ("transpose",
            lambda x: cv2.flip(cv2.rotate(x, CW), 1),
            lambda x: cv2.flip(cv2.rotate(x, CW), 1)),
        # anti_transpose: vflip(rot90_CW) — 반대각선 기준 반전 (자기 역원)
        ("anti_transpose",
            lambda x: cv2.flip(cv2.rotate(x, CW), 0),
            lambda x: cv2.flip(cv2.rotate(x, CW), 0)),
    ]

    quality_extra = [
        ("brightness_up",
            lambda x: np.clip(x.astype(np.float32) * 1.15, 0, 255).astype(np.uint8),
            lambda x: x),
        ("brightness_down",
            lambda x: np.clip(x.astype(np.float32) * 0.85, 0, 255).astype(np.uint8),
            lambda x: x),
        ("sharpen",
            _sharpen_numpy,
            lambda x: x),
    ]

    if level == "light":
        return light
    elif level == "standard":
        return light + standard_extra
    else:  # heavy
        return light + standard_extra + quality_extra


def predict_with_tta(
    model: torch.nn.Module,
    image: np.ndarray,
    transform,
    device: torch.device,
    tta_transforms: Optional[List[Tuple[str, Callable, Callable]]] = None,
    level: str = "standard",
    return_probs: bool = False,
) -> torch.Tensor:
    """단일 numpy 이미지에 TTA를 적용한 예측

    Args:
        model: PyTorch 모델
        image: 원본 이미지 (numpy array, RGB, H×W×3)
        transform: Albumentations transform (resize, normalize, toTensor 포함)
        device: 디바이스
        tta_transforms: 직접 지정 시 사용 (None이면 level 기준 자동 생성)
        level: "light" | "standard" | "heavy"
        return_probs: True이면 확률 반환, False이면 클래스 인덱스 반환

    Returns:
        평균 예측 확률 또는 클래스 인덱스
    """
    if tta_transforms is None:
        tta_transforms = get_tta_transforms(level)

    model.eval()
    preds = []

    with torch.no_grad():
        for name, aug_fn, _ in tta_transforms:
            aug_img = aug_fn(image.copy())
            img_tensor = transform(image=aug_img)['image']
            img_tensor = img_tensor.unsqueeze(0).to(device)
            logits = model(img_tensor)
            preds.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(preds).mean(dim=0)

    if return_probs:
        return avg_probs
    return avg_probs.argmax(dim=1)


# ─── 텐서 기반 배치 TTA ───────────────────────────────────────────────────────

def predict_batch_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    tta_transforms: Optional[List[Tuple[str, Callable, Callable]]] = None,
    level: str = "standard",
    return_probs: bool = False,
) -> torch.Tensor:
    """배치 이미지에 TTA를 적용한 예측 (이미 텐서로 변환된 이미지)

    Args:
        model: PyTorch 모델
        images: 배치 이미지 텐서 (B×C×H×W, already normalized)
        device: 디바이스
        tta_transforms: 직접 지정 시 사용 (None이면 level 기준 텐서 변환 자동 생성)
        level: "light" | "standard" | "heavy"
        return_probs: True이면 확률 반환, False이면 클래스 인덱스 반환

    Returns:
        평균 예측 확률 또는 클래스 인덱스
    """
    if tta_transforms is not None:
        # 외부에서 직접 지정한 경우 (numpy 스타일 튜플 호환)
        named_fns = [(name, fn) for name, fn, *_ in tta_transforms]
    else:
        # 텐서용 변환 함수 정의
        # torch.rot90(k=1, dims=[2,3]): 90° CCW
        #   transpose      = vflip(rot90_CCW) : (i,j) → (j, i)
        #   anti_transpose = hflip(rot90_CCW) : (i,j) → (N-1-j, N-1-i)
        _tensor_fns = {
            "original":        lambda x: x,
            "hflip":           lambda x: torch.flip(x, dims=[3]),
            "vflip":           lambda x: torch.flip(x, dims=[2]),
            "rot90":           lambda x: torch.rot90(x, k=1, dims=[2, 3]),
            "rot270":          lambda x: torch.rot90(x, k=3, dims=[2, 3]),
            "rot180":          lambda x: torch.rot90(x, k=2, dims=[2, 3]),
            "transpose":       lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2]),
            "anti_transpose":  lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3]),
            "brightness_up":   lambda x: torch.clamp(x * 1.15, -3.0, 3.0),
            "brightness_down": lambda x: torch.clamp(x * 0.85, -3.0, 3.0),
            "sharpen":         _sharpen_tensor,
        }
        _level_keys = {
            "light":    ["original", "hflip", "vflip", "rot90", "rot270"],
            "standard": ["original", "hflip", "vflip", "rot90", "rot270",
                         "rot180", "transpose", "anti_transpose"],
            "heavy":    ["original", "hflip", "vflip", "rot90", "rot270",
                         "rot180", "transpose", "anti_transpose",
                         "brightness_up", "brightness_down", "sharpen"],
        }
        named_fns = [(k, _tensor_fns[k]) for k in _level_keys[level]]

    model.eval()
    preds = []

    with torch.no_grad():
        for _, aug_fn in named_fns:
            aug_images = aug_fn(images)
            logits = model(aug_images)
            preds.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(preds).mean(dim=0)

    if return_probs:
        return avg_probs
    return avg_probs.argmax(dim=1)
