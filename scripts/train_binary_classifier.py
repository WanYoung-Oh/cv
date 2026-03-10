"""
이진분류기 학습: class 3 (입퇴원확인서) vs class 7 (외래진료확인서)

- train.csv에서 class 3·7만 추출 → datasets_fin/binary_train.csv 생성
- Stratified 5-Fold K-Fold 학습
- 기존 DocumentClassifierModule(num_classes=2) 재사용
- 체크포인트: checkpoints/binary/fold_{i}/best.ckpt

실행 방법:
  python scripts/train_binary_classifier.py data=transformer_384

  # 모델 변경 (기본: efficientnet_b3)
  python scripts/train_binary_classifier.py data=transformer_384 binary.model_name=maxvit_base_tf_384.in1k

  # fold 수 변경
  python scripts/train_binary_classifier.py data=transformer_384 binary.n_folds=3
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
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

# 혼동 쌍 클래스 ID (메인 분류기 기준)
BINARY_CLASS3 = 3   # confirmation_of_admission_and_discharge
BINARY_CLASS7 = 7   # medical_outpatient_certificate


def prepare_binary_csv(train_csv_path: str, output_csv_path: str) -> int:
    """메인 train.csv에서 class 3·7만 추출 → binary_train.csv 생성

    DocumentImageDataModule이 sorted(unique values) 기반으로
    class_to_idx = {3: 0, 7: 1} 을 자동 생성하므로 별도 re-labeling 불필요.

    Returns:
        생성된 이진 데이터 샘플 수
    """
    df = pd.read_csv(train_csv_path)
    binary_df = df[df['target'].isin([BINARY_CLASS3, BINARY_CLASS7])].reset_index(drop=True)
    binary_df.to_csv(output_csv_path, index=False)

    n3 = (binary_df['target'] == BINARY_CLASS3).sum()
    n7 = (binary_df['target'] == BINARY_CLASS7).sum()
    log.info(
        f"이진분류기 학습 데이터 생성: 총 {len(binary_df)}개 "
        f"(class3={n3}, class7={n7}) → {output_csv_path}"
    )
    return len(binary_df)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """5-Fold 이진분류기 학습"""
    load_dotenv()

    binary_cfg = cfg.binary

    log.info("=" * 70)
    log.info("🔬 이진분류기 학습 시작 (class 3 vs class 7)")
    log.info("=" * 70)
    log.info(f"모델: {binary_cfg.model_name}")
    log.info(f"Folds: {binary_cfg.n_folds}")
    log.info(f"Epochs: {binary_cfg.epochs} (early stopping patience={binary_cfg.early_stopping_patience})")
    log.info(f"LR: {binary_cfg.learning_rate}, batch_size: {binary_cfg.batch_size}")

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.seed)

    # 이진 CSV 생성
    train_csv_path = os.path.join(cfg.data.root_path, cfg.data.train_csv)
    binary_csv_name = "binary_train.csv"
    binary_csv_path = os.path.join(cfg.data.root_path, binary_csv_name)
    prepare_binary_csv(train_csv_path, binary_csv_path)

    # 디바이스
    device, accelerator, devices, device_info = get_device(model_name=binary_cfg.model_name)
    log.info(f"디바이스: {device_info}")

    # 체크포인트 루트 디렉토리
    ckpt_root = Path(binary_cfg.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    n_folds = binary_cfg.n_folds
    fold_results = []

    for fold_idx in range(n_folds):
        log.info(f"\n{'=' * 70}")
        log.info(f"📂 Fold {fold_idx + 1}/{n_folds} 학습 시작")
        log.info(f"{'=' * 70}")

        # DataModule (이진 CSV + Stratified K-Fold)
        data_module = DocumentImageDataModule(
            data_root=cfg.data.root_path,
            train_csv=binary_csv_name,
            test_csv=None,
            train_image_dir=cfg.data.get('train_image_dir', 'train/'),
            test_image_dir=cfg.data.get('test_image_dir', 'test/'),
            img_size=cfg.data.img_size,
            batch_size=binary_cfg.batch_size,
            num_workers=binary_cfg.num_workers,
            train_val_split=0.8,
            normalization=cfg.data.normalization,
            augmentation=cfg.data.augmentation,
            seed=cfg.seed,
            use_kfold=True,
            n_folds=n_folds,
            fold_idx=fold_idx,
        )
        data_module.setup(stage='fit')

        # 이진분류기 모델
        model = DocumentClassifierModule(
            model_name=binary_cfg.model_name,
            pretrained=binary_cfg.pretrained,
            num_classes=binary_cfg.num_classes,
            learning_rate=binary_cfg.learning_rate,
            weight_decay=binary_cfg.weight_decay,
            class_weights=data_module.class_weights,
            warmup_epochs=binary_cfg.warmup_epochs,
            epochs=binary_cfg.epochs,
            dropout_rate=binary_cfg.dropout_rate,
            stochastic_depth=binary_cfg.stochastic_depth,
            label_smoothing=binary_cfg.get('label_smoothing', 0.1),
            use_mixup=binary_cfg.use_mixup,
            mixup_alpha=binary_cfg.mixup_alpha,
            cutmix_alpha=binary_cfg.cutmix_alpha,
            mixup_prob=binary_cfg.mixup_prob,
            switch_prob=binary_cfg.switch_prob,
            loss_type="cross_entropy",
            warmup_start_factor=binary_cfg.warmup_start_factor,
            scheduler_eta_min=binary_cfg.scheduler_eta_min,
            scheduler=binary_cfg.get('scheduler', 'cosine'),
        )

        # Fold별 체크포인트 디렉토리
        fold_dir = ckpt_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(fold_dir),
            filename="best",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        )

        early_stopping = EarlyStopping(
            monitor="val_f1",
            patience=binary_cfg.early_stopping_patience,
            mode="max",
        )

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"binary_fold{fold_idx}_{binary_cfg.model_name}",
            mode=cfg.wandb.mode,
        )

        precision = "16-mixed" if binary_cfg.use_amp else 32

        trainer = pl.Trainer(
            max_epochs=binary_cfg.epochs,
            accelerator=accelerator,
            devices=devices,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
            log_every_n_steps=5,
            enable_progress_bar=True,
            precision=precision,
            accumulate_grad_batches=binary_cfg.get('accumulate_grad_batches', 2),
        )

        trainer.fit(model, datamodule=data_module)

        best_ckpt = checkpoint_callback.best_model_path
        best_score = checkpoint_callback.best_model_score
        val_f1 = float(best_score) if best_score is not None else 0.0

        fold_results.append({
            "fold": fold_idx,
            "checkpoint": best_ckpt,
            "val_f1": val_f1,
        })
        log.info(f"✅ Fold {fold_idx} 완료: val_f1={val_f1:.4f} | {best_ckpt}")

    # 결과 요약
    log.info("\n" + "=" * 70)
    log.info("🏁 이진분류기 학습 완료 — 결과 요약")
    log.info("=" * 70)
    for r in fold_results:
        log.info(f"  Fold {r['fold']}: val_f1={r['val_f1']:.4f} | {r['checkpoint']}")
    avg_f1 = sum(r['val_f1'] for r in fold_results) / len(fold_results)
    log.info(f"  평균 val_f1: {avg_f1:.4f}")

    # 결과 JSON 저장
    result_info = {
        "model_name": binary_cfg.model_name,
        "n_folds": n_folds,
        "avg_val_f1": avg_f1,
        "trained_at": datetime.now().isoformat(),
        "folds": fold_results,
    }
    result_path = ckpt_root / "fold_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_info, f, indent=2)

    log.info(f"\n💾 결과 저장: {result_path}")
    log.info("\n📌 다음 단계: apply_binary_ensemble.py 실행")
    log.info("  configs/binary/apply_ensemble.yaml의 binary_checkpoints를")
    log.info("  위 결과의 checkpoint 경로로 업데이트 후:")
    log.info("  python scripts/apply_binary_ensemble.py data=transformer_384 binary=apply_ensemble")


if __name__ == "__main__":
    main()
