"""Device detection utility for PyTorch

사용 가능한 최적의 디바이스(CUDA, MPS, CPU)를 자동으로 감지합니다.
MPS에서 호환성 문제 발생 시 자동으로 CPU로 fallback합니다.
"""

import torch
import logging

log = logging.getLogger(__name__)


def get_device(model_name: str = None, force_cpu_for_transformers: bool = True):
    """사용 가능한 최적의 디바이스 자동 선택

    우선순위:
    1. CUDA (NVIDIA GPU)
    2. MPS (Mac Metal Performance Shaders - M1 이상)
       - Vision Transformer 계열은 MPS에서 stride 호환성 문제로 CPU로 자동 fallback
    3. CPU

    Args:
        model_name: 모델 이름 (Vision Transformer 감지용)
        force_cpu_for_transformers: Transformer 모델일 때 MPS를 CPU로 강제 전환 (기본: True)

    Returns:
        tuple: (device, accelerator, devices, device_info)
            - device: torch.device 객체
            - accelerator: PyTorch Lightning accelerator 문자열
            - devices: 사용할 디바이스 개수
            - device_info: 디바이스 정보 문자열
    """
    # Vision Transformer 계열 모델 감지
    is_transformer = False
    if model_name:
        transformer_keywords = ['vit', 'swin', 'deit', 'beit', 'cait', 'convnext', 'mixer']
        is_transformer = any(keyword in model_name.lower() for keyword in transformer_keywords)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
        devices = 1
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        # MPS + Transformer 조합은 stride 문제로 CPU로 fallback
        if is_transformer and force_cpu_for_transformers:
            log.warning(
                f"⚠️  Vision Transformer 모델 '{model_name}'은 MPS에서 stride 호환성 문제가 있습니다. "
                "CPU로 자동 전환합니다. (성능 저하 가능)"
            )
            device = torch.device("cpu")
            accelerator = "cpu"
            devices = "auto"
            device_info = "CPU (MPS fallback for Transformer compatibility)"
        else:
            # Mac M1 이상 (M1, M2, M3, M4, ...)
            device = torch.device("mps")
            accelerator = "mps"
            devices = 1
            device_info = "Apple MPS (Metal Performance Shaders)"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"
        devices = "auto"
        device_info = "CPU"

    return device, accelerator, devices, device_info


def get_simple_device(model_name: str = None) -> torch.device:
    """간단한 디바이스 객체만 반환 (inference/ensemble용)

    Args:
        model_name: 모델 이름 (Vision Transformer 감지용, optional)

    Returns:
        torch.device: 사용 가능한 최적의 디바이스
    """
    device, _, _, _ = get_device(model_name=model_name)
    return device
