"""
Inference 스크립트
학습된 모델로 test 데이터셋에 대한 예측을 수행하고 pred.csv 생성
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, List

# 프로젝트 루트를 Python path에 추가 (어디서든 실행 가능하도록)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.datamodule import DocumentImageDataModule
from src.models.module import DocumentClassifierModule
from src.utils.device import get_simple_device


log = logging.getLogger(__name__)


def get_champion_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """챔피언 모델 체크포인트 찾기

    Args:
        checkpoint_dir: 체크포인트 베이스 디렉토리

    Returns:
        챔피언 체크포인트 경로 또는 None
    """
    champion_dir = checkpoint_dir / "champion"
    champion_checkpoint = champion_dir / "best_model.ckpt"
    champion_info_path = champion_dir / "champion_info.json"

    if champion_checkpoint.exists():
        # 챔피언 정보 로드
        if champion_info_path.exists():
            with open(champion_info_path, 'r') as f:
                champion_info = json.load(f)

            log.info("🏆 챔피언 모델 로드")
            log.info(f"   val_f1: {champion_info.get('val_f1', 'N/A')}")
            log.info(f"   원본 경로: {champion_info.get('checkpoint_path', 'N/A')}")
            log.info(f"   업데이트: {champion_info.get('updated_at', 'N/A')}")

        return champion_checkpoint

    return None


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """모든 실험 중 최고 성능 체크포인트 찾기

    Args:
        checkpoint_dir: 체크포인트 베이스 디렉토리

    Returns:
        최고 성능 체크포인트 경로 또는 None
    """
    best_checkpoint = None
    best_metric = 0.0

    # 모든 실험 디렉토리 탐색
    for exp_dir in checkpoint_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "champion":
            continue

        # 해당 실험의 체크포인트 찾기
        ckpt_files = list(exp_dir.glob("*.ckpt"))

        for ckpt_file in ckpt_files:
            try:
                # 파일명에서 val_f1 추출
                # 예: epoch=10-val_f1=0.950.ckpt -> 0.950
                filename = ckpt_file.stem
                if 'val_f1=' in filename:
                    val_f1_str = filename.split('val_f1=')[1]
                    val_f1 = float(val_f1_str)

                    if val_f1 > best_metric:
                        best_metric = val_f1
                        best_checkpoint = ckpt_file
            except (ValueError, IndexError):
                continue

    if best_checkpoint:
        log.info(f"최고 성능 체크포인트 발견: val_f1={best_metric:.4f}")
        log.info(f"경로: {best_checkpoint}")

    return best_checkpoint


def get_test_image_ids(test_csv_path: str) -> List[str]:
    """테스트 CSV에서 이미지 ID 추출

    Args:
        test_csv_path: 테스트 CSV 파일 경로

    Returns:
        이미지 ID 리스트
    """
    df = pd.read_csv(test_csv_path)
    # 첫 번째 컬럼이 이미지 파일명 또는 ID라고 가정
    return df.iloc[:, 0].tolist()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 inference 함수

    Hydra config로 inference 설정 관리:
        inference.checkpoint: 체크포인트 경로 (선택사항)
        inference.output: 출력 파일 경로 (기본값: pred.csv)
    """
    # Hydra config에서 inference 설정 읽기
    inference_cfg = cfg.get('inference', {})
    checkpoint_path = inference_cfg.get('checkpoint', None)
    output_path = inference_cfg.get('output', 'pred.csv')

    log.info("=" * 70)
    log.info("🔮 Inference 시작")
    log.info("=" * 70)

    # 체크포인트 경로 찾기
    if not checkpoint_path:
        checkpoint_dir = Path(cfg.checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"체크포인트 디렉토리 '{checkpoint_dir}'가 존재하지 않습니다."
            )

        # 1순위: 챔피언 모델
        champion_ckpt = get_champion_checkpoint(checkpoint_dir)
        if champion_ckpt:
            checkpoint_path = str(champion_ckpt)
            log.info("챔피언 모델 사용 ✓")
        else:
            # 2순위: 모든 실험 중 최고 성능 모델
            log.info("챔피언 모델이 없습니다. 최고 성능 모델 탐색 중...")
            best_ckpt = find_best_checkpoint(checkpoint_dir)

            if best_ckpt:
                checkpoint_path = str(best_ckpt)
            else:
                raise FileNotFoundError(
                    f"체크포인트를 찾을 수 없습니다.\n"
                    f"'{checkpoint_dir}' 디렉토리에 학습된 모델이 없습니다.\n"
                    f"먼저 'python src/train.py'로 모델을 학습하세요."
                )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

    log.info(f"사용 체크포인트: {checkpoint_path}")

    # 데이터모듈 생성
    test_csv_path = os.path.join(cfg.data.root_path, cfg.data.test_csv)

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(
            f"테스트 CSV 파일을 찾을 수 없습니다: {test_csv_path}\n"
            f"데이터셋을 먼저 준비해주세요."
        )

    log.info(f"테스트 데이터: {test_csv_path}")

    data_module = DocumentImageDataModule(
        data_root=cfg.data.root_path,
        train_csv=cfg.data.train_csv,
        test_csv=cfg.data.test_csv,
        train_image_dir=cfg.data.get('train_image_dir', 'train/'),
        test_image_dir=cfg.data.get('test_image_dir', 'test/'),
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        train_val_split=cfg.data.train_val_split,
        normalization=cfg.data.normalization,
        augmentation=cfg.data.augmentation,
    )

    data_module.setup()

    # 모델 로드
    log.info("모델 로드 중...")
    model = DocumentClassifierModule.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    # 디바이스 설정 (CUDA -> MPS -> CPU 자동 감지)
    device = get_simple_device()
    model = model.to(device)
    log.info(f"사용 디바이스: {device}")

    # Inference 수행
    log.info("Inference 수행 중...")
    predictions = []

    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images, _ = batch
            images = images.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            predictions.extend(preds.cpu().numpy().tolist())

    log.info(f"총 예측 수: {len(predictions)}")

    # 결과 저장
    # sample_submission.csv가 있으면 그 형식 따르기
    sample_submission_path = os.path.join(cfg.data.root_path, "sample_submission.csv")

    if os.path.exists(sample_submission_path):
        # sample_submission.csv 형식으로 저장
        log.info(f"sample_submission.csv 형식 사용: {sample_submission_path}")
        sample_df = pd.read_csv(sample_submission_path)
        sample_df.iloc[:, 1] = predictions[:len(sample_df)]
        result_df = sample_df
    else:
        # 기본 형식으로 저장 (id, target)
        log.info("기본 형식으로 저장 (id, target)")
        image_ids = get_test_image_ids(test_csv_path)
        result_df = pd.DataFrame({
            'id': image_ids[:len(predictions)],
            'target': predictions
        })

    # CSV 저장
    result_df.to_csv(output_path, index=False)

    log.info("=" * 70)
    log.info(f"✅ Inference 완료!")
    log.info(f"📄 결과 저장: {output_path}")
    log.info(f"📊 예측 샘플:")
    log.info(f"\n{result_df.head(10)}")
    log.info("=" * 70)

    # 클래스별 예측 분포 출력
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    log.info("\n📈 예측 클래스 분포:")
    for class_id, count in pred_counts.items():
        log.info(f"  클래스 {class_id}: {count:4d} ({count/len(predictions)*100:5.2f}%)")


if __name__ == "__main__":
    main()
