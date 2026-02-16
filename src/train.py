"""
메인 학습 스크립트
Hydra + PyTorch Lightning + WanDB
"""

import logging
import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python path에 추가 (어디서든 실행 가능하도록)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv

from src.data.datamodule import DocumentImageDataModule
from src.models.module import DocumentClassifierModule
from src.utils.device import get_device


log = logging.getLogger(__name__)


def setup_seed(seed: int):
    """재현성을 위한 시드 설정"""
    import random
    import numpy as np

    random.seed(seed)
    # NumPy 1.26+에서는 np.random.seed 대신 Generator 사용 권장하지만
    # 하위 호환성을 위해 legacy API도 설정
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_experiment_dir(base_dir: str, model_name: str) -> tuple[Path, str]:
    """실험 디렉토리 생성 (날짜 + run_id)

    Args:
        base_dir: 베이스 체크포인트 디렉토리
        model_name: 모델 이름

    Returns:
        (experiment_dir, run_id): 실험 디렉토리와 run_id
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # 날짜 생성 (YYYYMMDD 형식)
    date_str = datetime.now().strftime("%Y%m%d")

    # 같은 날짜의 기존 실험 찾기
    existing_runs = list(base_path.glob(f"{date_str}_*"))

    if existing_runs:
        # 가장 큰 run 번호 찾기
        run_numbers = []
        for run_dir in existing_runs:
            try:
                # 예: 20260212_run_003 -> 3
                run_num = int(run_dir.name.split('_')[-1])
                run_numbers.append(run_num)
            except (ValueError, IndexError):
                continue

        next_run = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run = 1

    # run_id 생성
    run_id = f"{date_str}_run_{next_run:03d}"

    # 실험 디렉토리 생성
    experiment_dir = base_path / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"실험 ID: {run_id}")
    log.info(f"체크포인트 저장 경로: {experiment_dir}")

    return experiment_dir, run_id


def update_champion_model(
    current_checkpoint: Path,
    current_metric: float,
    champion_dir: Path,
    metric_name: str = "val_f1"
) -> bool:
    """최고 성능 모델(챔피언) 업데이트

    Args:
        current_checkpoint: 현재 체크포인트 경로
        current_metric: 현재 메트릭 값
        champion_dir: 챔피언 모델 저장 디렉토리
        metric_name: 메트릭 이름

    Returns:
        새로운 챔피언 여부
    """
    champion_dir.mkdir(parents=True, exist_ok=True)
    champion_info_path = champion_dir / "champion_info.json"

    # 기존 챔피언 정보 로드
    if champion_info_path.exists():
        with open(champion_info_path, 'r') as f:
            champion_info = json.load(f)

        best_metric = champion_info.get(metric_name, 0.0)
    else:
        best_metric = 0.0
        champion_info = {}

    # 현재 모델이 더 좋은지 확인
    if current_metric > best_metric:
        # 챔피언 모델 복사
        champion_checkpoint = champion_dir / "best_model.ckpt"
        shutil.copy2(current_checkpoint, champion_checkpoint)

        # 챔피언 정보 업데이트
        champion_info.update({
            metric_name: float(current_metric),
            "checkpoint_path": str(current_checkpoint),
            "updated_at": datetime.now().isoformat(),
            "model_name": current_checkpoint.parent.name
        })

        with open(champion_info_path, 'w') as f:
            json.dump(champion_info, f, indent=2)

        log.info("=" * 70)
        log.info("🏆 새로운 챔피언 모델!")
        log.info(f"   {metric_name}: {best_metric:.4f} → {current_metric:.4f}")
        log.info(f"   저장 경로: {champion_checkpoint}")
        log.info("=" * 70)

        return True

    return False


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 학습 함수"""

    # 환경 변수 로드 (.env 파일)
    load_dotenv()

    log.info("설정 정보:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # 시드 설정
    setup_seed(cfg.seed)
    
    # 디바이스 설정 (CUDA, MPS, CPU 자동 선택)
    # Vision Transformer 모델은 MPS에서 호환성 문제로 자동으로 CPU로 fallback
    device, accelerator, devices, device_info = get_device(model_name=cfg.model.model_name)
    log.info(f"사용 디바이스: {device_info}")
    
    # 데이터모듈 생성
    data_module = DocumentImageDataModule(
        data_root=cfg.data.root_path,
        train_csv=cfg.data.train_csv,
        test_csv=cfg.data.get('test_csv', None),
        train_image_dir=cfg.data.get('train_image_dir', 'train/'),
        test_image_dir=cfg.data.get('test_image_dir', 'test/'),
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        train_val_split=cfg.data.train_val_split,
        normalization=cfg.data.normalization,
        augmentation=cfg.data.augmentation,
        seed=cfg.seed,
        drop_last=cfg.training.get('drop_last', False),
    )
    
    # 데이터 설정 (클래스 가중치 계산)
    data_module.setup()
    
    # 모델 생성
    model = DocumentClassifierModule(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        class_weights=data_module.class_weights,
        warmup_epochs=cfg.training.warmup_epochs,
        epochs=cfg.training.epochs,
    )
    
    log.info(f"모델: {cfg.model.model_name}")
    log.info(f"총 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 실험 디렉토리 생성 (날짜 + run_id)
    experiment_dir, run_id = create_experiment_dir(
        base_dir=cfg.checkpoint_dir,
        model_name=cfg.model.model_name
    )

    # 실험 정보 저장
    experiment_info = {
        "run_id": run_id,
        "model_name": cfg.model.model_name,
        "started_at": datetime.now().isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    with open(experiment_dir / "experiment_info.json", 'w') as f:
        json.dump(experiment_info, f, indent=2)

    # 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(experiment_dir),
        filename="{epoch:02d}-{val_f1:.3f}",
        monitor=cfg.training.checkpoint.monitor,
        mode=cfg.training.checkpoint.mode,
        save_top_k=cfg.training.checkpoint.save_top_k,
    )
    
    # 조기 종료 콜백
    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.early_stopping.monitor,
        patience=cfg.training.early_stopping.patience,
        mode=cfg.training.early_stopping.mode,
    )
    
    # WanDB 로거
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.model.model_name}-{run_id}",
        log_model=cfg.wandb.log_model,
        mode=cfg.wandb.mode,
    )

    # WanDB에 전체 설정 저장
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))

    # Mixed Precision 설정
    precision = '16-mixed' if cfg.training.get('use_amp', False) else 32
    if precision == '16-mixed':
        log.info("✨ Mixed Precision (AMP) 활성화")

    # 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=cfg.training.log_interval,
        val_check_interval=cfg.training.val_check_interval,
        enable_progress_bar=True,
        precision=precision,
    )
    
    # 학습
    log.info("학습 시작...")
    trainer.fit(model, datamodule=data_module)

    # 테스트 (test.csv는 레이블이 더미 데이터이므로 제거)
    # 리더보드 제출용 추론은 inference.py 사용
    # log.info("테스트 시작...")
    # trainer.test(model, datamodule=data_module)

    # 최고 성능 체크포인트 찾기 및 챔피언 모델 업데이트
    best_checkpoint = checkpoint_callback.best_model_path
    if best_checkpoint and os.path.exists(best_checkpoint):
        # PyTorch Lightning의 best_model_score 사용 (파일명 파싱 불필요)
        try:
            val_f1 = checkpoint_callback.best_model_score.item()

            # 챔피언 디렉토리
            champion_dir = Path(cfg.checkpoint_dir) / "champion"

            # 챔피언 모델 업데이트
            is_new_champion = update_champion_model(
                current_checkpoint=Path(best_checkpoint),
                current_metric=val_f1,
                champion_dir=champion_dir,
                metric_name="val_f1"
            )

            # 실험 정보 업데이트
            experiment_info["best_checkpoint"] = str(best_checkpoint)
            experiment_info["val_f1"] = val_f1
            experiment_info["is_champion"] = is_new_champion
            experiment_info["completed_at"] = datetime.now().isoformat()

            with open(experiment_dir / "experiment_info.json", 'w') as f:
                json.dump(experiment_info, f, indent=2)

        except Exception as e:
            log.warning(f"챔피언 모델 업데이트 실패: {e}")

    log.info("학습 완료!")
    log.info(f"실험 ID: {run_id}")
    log.info(f"체크포인트: {experiment_dir}")
    log.info("💡 Inference를 수행하려면: python src/inference.py")


if __name__ == "__main__":
    main()
