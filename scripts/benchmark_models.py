"""
모델 성능 벤치마크 스크립트

빠르게 다양한 모델들의 성능을 비교하는 스크립트입니다.
1-2 에포크로 각 모델을 테스트하여 메모리, 속도, 초기 수렴 속도를 비교합니다.

사용법:
    python scripts/benchmark_models.py
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# 프로젝트 루트를 Python path에 추가 (어디서든 실행 가능하도록)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datamodule import DocumentImageDataModule
from src.models.module import DocumentClassifierModule

# 상수 정의
DEFAULT_NUM_CLASSES = 17
DEFAULT_NUM_WORKERS = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DATA_PATH = "datasets_fin"
BYTES_PER_FLOAT32 = 4
MB_DIVISOR = 1024 * 1024

# 벤치마크 모델 목록 (프로젝트 설정 파일 기준)
# configs/model/*.yaml 파일들과 동기화됨
BENCHMARK_MODELS = [
    # CNN 계열 (고해상도 768x768 - 문서 이미지 세부 정보 보존)
    {"name": "resnet34", "model_name": "resnet34", "pretrained": True, "batch_size": 8, "img_size": 768, "category": "CNN"},
    {"name": "resnet50", "model_name": "resnet50", "pretrained": True, "batch_size": 8, "img_size": 768, "category": "CNN"},
    {"name": "efficientnet_b4", "model_name": "efficientnet_b4", "pretrained": True, "batch_size": 4, "img_size": 768, "category": "CNN"},

    # Modern CNN (Hybrid - 표준 해상도)
    {"name": "convnext_base", "model_name": "convnext_base", "pretrained": True, "batch_size": 16, "img_size": 224, "category": "Hybrid"},

    # Transformer 계열 (고해상도 384x384)
    {"name": "swin_base_384", "model_name": "swin_base_patch4_window12_384", "pretrained": True, "batch_size": 8, "img_size": 384, "category": "Transformer"},
    {"name": "deit_base_384", "model_name": "deit_base_patch16_384", "pretrained": True, "batch_size": 8, "img_size": 384, "category": "Transformer"},
]


def get_device(model_name: str = "") -> str:
    """
    모델 종류와 환경에 따라 최적의 디바이스를 반환합니다.
    Swin, ConvNeXt, DiT 등 MPS 에러 유발 모델은 CPU로 우회합니다.
    """
    # 1. CUDA(NVIDIA)가 있으면 무조건 CUDA 사용 (가장 안정적)
    if torch.cuda.is_available():
        return "cuda"
    
    # 2. Apple Silicon(MPS) 환경일 경우
    if torch.backends.mps.is_available():
        # MPS에서 에러가 자주 발생하는 모델 키워드 목록
        problematic_models = ["swin", "convnext", "dit", "vit", "transformer"]
        
        # 모델 이름에 위 키워드가 포함되어 있다면 CPU로 반환
        if any(keyword in model_name.lower() for keyword in problematic_models):
            print(f"모델 '{model_name}'은(는) MPS 이슈로 인해 CPU 모드로 실행합니다.")
            return "cpu"
        
        return "mps"
    
    # 3. 그 외 기본값
    return "cpu"


def get_model_size(model: pl.LightningModule) -> int:
    """모델의 총 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters())


def get_memory_usage() -> float:
    """현재 프로세스의 메모리 사용량을 MB 단위로 반환합니다."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / MB_DIVISOR


class BenchmarkCallback(pl.Callback):
    """벤치마크 메트릭을 수집하는 콜백입니다."""

    def __init__(self):
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self.max_memory: float = 0.0
        self.epoch_start: Optional[float] = None
        self.batch_start: Optional[float] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.epoch_start is not None:
            self.epoch_times.append(time.time() - self.epoch_start)
        self.max_memory = max(self.max_memory, get_memory_usage())

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.batch_start = time.time()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.batch_start is not None:
            self.batch_times.append(time.time() - self.batch_start)


def benchmark_model(
    model_config: Dict[str, Any],
    data_root: str,
    num_epochs: int = 1,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_workers: int = DEFAULT_NUM_WORKERS
) -> Dict[str, Any]:
    """단일 모델에 대한 벤치마크를 수행합니다."""
    device = get_device(model_config['name'])
    separator = "=" * 70
    print(f"\n{separator}\n벤치마킹: {model_config['name']} (Device: {device})\n{separator}")

    results = {
        "model": model_config["name"],
        "category": model_config["category"],
        "config": model_config
    }

    try:
        data_module = DocumentImageDataModule(
            data_root=data_root,
            train_csv="train.csv",
            test_csv="test.csv",
            batch_size=model_config["batch_size"],
            num_workers=num_workers,
            img_size=model_config["img_size"],
        )
        data_module.setup()

        model = DocumentClassifierModule(
            model_name=model_config["model_name"],
            pretrained=model_config.get("pretrained", True),
            num_classes=num_classes,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=num_epochs,
        )

        num_params = get_model_size(model)
        model_size_mb = num_params * BYTES_PER_FLOAT32 / MB_DIVISOR
        results.update({
            "num_params": num_params,
            "model_size_mb": model_size_mb
        })

        benchmark_callback = BenchmarkCallback()
        logger = CSVLogger(save_dir=".benchmark_logs", name=model_config["name"])

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=device,
            devices=1,
            logger=logger,
            callbacks=[benchmark_callback],
            precision="32-true",
            enable_model_summary=False,
            log_every_n_steps=5
        )

        start_time = time.time()
        trainer.fit(model, datamodule=data_module)

        avg_epoch_time = (
            sum(benchmark_callback.epoch_times) / len(benchmark_callback.epoch_times)
            if benchmark_callback.epoch_times else 0.0
        )

        results.update({
            "total_train_time": time.time() - start_time,
            "avg_epoch_time": avg_epoch_time,
            "max_memory_mb": benchmark_callback.max_memory,
            "status": "success"
        })

    except Exception as e:
        print(f"오류 발생: {e}")
        results.update({
            "status": "failed",
            "error": str(e)
        })
        traceback.print_exc()

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    """벤치마크 결과를 출력합니다."""
    separator = "=" * 70
    print(f"\n{separator}\n최종 벤치마크 결과 요약\n{separator}")

    for r in results:
        if r["status"] == "success":
            print(
                f"  {r['model']:25s} | "
                f"파라미터: {r['num_params']:10,} | "
                f"시간: {r['total_train_time']:6.1f}s | "
                f"메모리: {r['max_memory_mb']:7.1f}MB"
            )
        else:
            error_msg = r.get('error', 'Unknown Error')
            print(f"  {r['model']:25s} | 실패: {error_msg}")


def save_results(results: List[Dict[str, Any]], output_dir: str = ".benchmark_results") -> None:
    """벤치마크 결과를 JSON 파일로 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    output_file = output_path / f"result_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n결과 저장됨: {output_file}")


def run_benchmark(
    data_path: str = DEFAULT_DATA_PATH,
    num_epochs: int = 2,
    models: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """전체 벤치마크를 실행합니다."""
    if models is None:
        models = BENCHMARK_MODELS

    all_results = []
    for config in models:
        result = benchmark_model(config, data_path, num_epochs)
        all_results.append(result)

    print_results(all_results)
    save_results(all_results)

    return all_results


if __name__ == "__main__":
    run_benchmark()
