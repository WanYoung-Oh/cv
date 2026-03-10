"""
Pseudo-Labeling 스크립트

학습된 모델로 테스트 데이터의 pseudo-label을 생성합니다.
신뢰도 임계값(confidence threshold) 이상인 샘플만 선택합니다.

사용법:
    # 기본 (champion 모델, threshold=0.9)
    python scripts/pseudo_label.py

    # threshold 조정
    python scripts/pseudo_label.py pseudo.confidence_threshold=0.95

    # TTA 사용 (더 신뢰도 높은 pseudo-label)
    python scripts/pseudo_label.py pseudo.use_tta=true

    # 특정 run_id 사용
    python scripts/pseudo_label.py pseudo.run_id=20260221_run_001

    # 출력 경로 지정
    python scripts/pseudo_label.py pseudo.output_csv=datasets_fin/pseudo_labels.csv
"""

import os
import sys
import logging
import json
from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.datamodule import DocumentImageDataModule
from src.models.module import DocumentClassifierModule
from src.utils.device import get_simple_device
from src.utils.tta import predict_batch_with_tta


log = logging.getLogger(__name__)


def get_champion_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    champion_checkpoint = checkpoint_dir / "champion" / "best_model.ckpt"
    return champion_checkpoint if champion_checkpoint.exists() else None


def find_checkpoint_by_run_id(checkpoint_dir: Path, run_id: str) -> Optional[Path]:
    run_dir = checkpoint_dir / run_id
    if not run_dir.exists():
        return None

    exp_info_path = run_dir / "experiment_info.json"
    if exp_info_path.exists():
        with open(exp_info_path) as f:
            info = json.load(f)
        best_ckpt = info.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            return Path(best_ckpt)

    ckpt_files = list(run_dir.glob("*.ckpt"))
    if ckpt_files:
        return max(ckpt_files, key=lambda p: p.stat().st_mtime)

    return None


def load_class_mapping(data_root: str, train_csv: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """학습 데이터에서 클래스 매핑 로드

    Returns:
        (idx_to_class, class_to_idx) 딕셔너리
    """
    train_df = pd.read_csv(os.path.join(data_root, train_csv))
    all_classes = sorted(train_df.iloc[:, 1].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return idx_to_class, class_to_idx


def predict_with_confidence(
    model: DocumentClassifierModule,
    data_loader,
    device: torch.device,
    use_tta: bool = False,
    tta_level: str = "standard",
) -> Tuple[List[int], List[float]]:
    """모델 예측 + softmax 확률(신뢰도) 반환

    Args:
        model: 학습된 모델
        data_loader: 테스트 데이터 로더
        device: 사용 디바이스
        use_tta: TTA 사용 여부
        tta_level: TTA 레벨 ("light" | "standard" | "heavy")

    Returns:
        (predictions, confidences): 예측 클래스 인덱스와 최대 softmax 확률 리스트
    """
    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Pseudo-label 생성 중"):
            images, _ = batch
            images = images.to(device)

            if use_tta:
                avg_probs = predict_batch_with_tta(
                    model=model,
                    images=images,
                    device=device,
                    level=tta_level,
                    return_probs=True,
                )
            else:
                logits = model(images)
                avg_probs = F.softmax(logits.float(), dim=1)

            max_probs, pred_classes = avg_probs.max(dim=1)

            predictions.extend(pred_classes.cpu().numpy().tolist())
            confidences.extend(max_probs.cpu().numpy().tolist())

    return predictions, confidences


def generate_pseudo_labels(
    predictions: List[int],
    confidences: List[float],
    image_ids: List[str],
    idx_to_class: Dict[int, str],
    confidence_threshold: float,
    priority_class_ids: Optional[List[int]] = None,
    priority_min_quota: int = 20,
) -> pd.DataFrame:
    """신뢰도 임계값 이상인 샘플만 필터링하여 pseudo-label DataFrame 생성

    Args:
        predictions: 예측 클래스 인덱스 리스트
        confidences: softmax 최대 확률 리스트
        image_ids: 이미지 파일명 리스트
        idx_to_class: 인덱스 → 클래스명 매핑
        confidence_threshold: 신뢰도 임계값 (이 값 이상인 샘플만 사용)
        priority_class_ids: quota 보장이 필요한 우선 클래스 인덱스 리스트
        priority_min_quota: 우선 클래스별 최소 보장 샘플 수

    Returns:
        필터링된 pseudo-label DataFrame (컬럼: ID, target)
    """
    # 기본 threshold 필터링
    records = []
    for img_id, pred, conf in zip(image_ids, predictions, confidences):
        if conf >= confidence_threshold:
            records.append({
                "ID": img_id,
                "target": idx_to_class[pred],
                "confidence": conf,
            })

    # priority 클래스 quota 보장
    if priority_class_ids:
        used_ids = {rec["ID"] for rec in records}
        class_counts = Counter(rec["target"] for rec in records)

        for cls_id in priority_class_ids:
            cls_name = idx_to_class.get(cls_id)
            if cls_name is None:
                continue
            current_count = class_counts.get(cls_name, 0)
            if current_count < priority_min_quota:
                needed = priority_min_quota - current_count
                # confidence 내림차순으로 미선택 후보 수집
                candidates = sorted(
                    [
                        (img_id, conf)
                        for img_id, pred, conf in zip(image_ids, predictions, confidences)
                        if pred == cls_id and img_id not in used_ids
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                for img_id, conf in candidates[:needed]:
                    records.append({
                        "ID": img_id,
                        "target": cls_name,
                        "confidence": conf,
                    })
                    used_ids.add(img_id)
                added = min(len(candidates), needed)
                log.info(
                    f"Priority 클래스 {cls_id} ({cls_name}): "
                    f"기존 {current_count}개 → {current_count + added}개 (추가 {added}개)"
                )

    return pd.DataFrame(records)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Pseudo-labeling 메인 함수"""

    pseudo_cfg = cfg.get("pseudo", {})
    confidence_threshold = pseudo_cfg.get("confidence_threshold", 0.9)
    use_tta = pseudo_cfg.get("use_tta", False)
    tta_level = pseudo_cfg.get("tta_level", "standard")
    run_id = pseudo_cfg.get("run_id", None)
    output_csv = pseudo_cfg.get("output_csv", "datasets_fin/pseudo_labels.csv")
    priority_class_ids = list(pseudo_cfg.get("priority_class_ids") or []) or None
    priority_min_quota = pseudo_cfg.get("priority_min_quota", 20)

    log.info("=" * 70)
    log.info("🏷️  Pseudo-Labeling 시작")
    log.info("=" * 70)
    log.info(f"신뢰도 임계값: {confidence_threshold}")
    log.info(f"TTA 사용: {use_tta}")
    log.info(f"출력 경로: {output_csv}")

    # 체크포인트 탐색
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_path = None

    if run_id:
        checkpoint_path = find_checkpoint_by_run_id(checkpoint_dir, run_id)
        if checkpoint_path:
            log.info(f"✅ Run ID '{run_id}' 모델 사용")
        else:
            raise FileNotFoundError(f"Run ID '{run_id}' 체크포인트를 찾을 수 없습니다.")
    else:
        checkpoint_path = get_champion_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log.info("✅ 챔피언 모델 사용")
        else:
            raise FileNotFoundError(
                "챔피언 모델을 찾을 수 없습니다. "
                "먼저 'python src/train.py'로 모델을 학습하세요."
            )

    log.info(f"체크포인트: {checkpoint_path}")

    # 클래스 매핑 로드 (학습 데이터 기준)
    idx_to_class, class_to_idx = load_class_mapping(
        data_root=cfg.data.root_path,
        train_csv=cfg.data.train_csv,
    )
    log.info(f"클래스 수: {len(idx_to_class)}개")

    # 테스트 데이터 로드 (sample_submission.csv 기준)
    submission_csv = cfg.data.get("sample_submission_csv", cfg.data.get("test_csv", None))
    if not submission_csv:
        raise ValueError("sample_submission_csv 또는 test_csv 설정이 필요합니다.")

    submission_csv_path = os.path.join(cfg.data.root_path, submission_csv)
    test_df = pd.read_csv(submission_csv_path)
    image_ids = test_df.iloc[:, 0].tolist()
    log.info(f"테스트 이미지 수: {len(image_ids):,}개")

    # DataModule 생성 (test 데이터만 사용)
    data_module = DocumentImageDataModule(
        data_root=cfg.data.root_path,
        train_csv=cfg.data.train_csv,
        test_csv=submission_csv,
        train_image_dir=cfg.data.get("train_image_dir", "train/"),
        test_image_dir=cfg.data.get("test_image_dir", "test/"),
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        normalization=cfg.data.normalization,
        augmentation=cfg.data.augmentation,
        seed=cfg.get("seed", 42),
    )
    data_module.setup(stage="predict")
    test_loader = data_module.test_dataloader()

    # 모델 로드
    log.info("모델 로드 중...")
    model = DocumentClassifierModule.load_from_checkpoint(
        str(checkpoint_path), strict=False
    )
    model.eval()

    device = get_simple_device()
    model = model.to(device)
    log.info(f"디바이스: {device}")

    # 예측 + 신뢰도 계산
    predictions, confidences = predict_with_confidence(
        model=model,
        data_loader=test_loader,
        device=device,
        use_tta=use_tta,
        tta_level=tta_level,
    )

    # Pseudo-label 생성 (신뢰도 필터링 + priority quota)
    pseudo_df = generate_pseudo_labels(
        predictions=predictions,
        confidences=confidences,
        image_ids=image_ids,
        idx_to_class=idx_to_class,
        confidence_threshold=confidence_threshold,
        priority_class_ids=priority_class_ids,
        priority_min_quota=priority_min_quota,
    )

    total = len(image_ids)
    selected = len(pseudo_df)
    log.info("=" * 70)
    log.info(f"📊 Pseudo-label 생성 결과")
    log.info(f"  전체 테스트 이미지: {total:,}개")
    log.info(f"  선택된 이미지 (conf ≥ {confidence_threshold}): {selected:,}개 ({selected/total*100:.1f}%)")
    log.info(f"  제외된 이미지 (저신뢰도): {total-selected:,}개 ({(total-selected)/total*100:.1f}%)")

    # 클래스별 분포 출력
    if selected > 0:
        log.info("\n📈 클래스별 pseudo-label 분포:")
        class_dist = pseudo_df["target"].value_counts().sort_index()
        for cls, cnt in class_dist.items():
            log.info(f"  {cls}: {cnt:4d}개 ({cnt/selected*100:.1f}%)")

        # 신뢰도 통계
        log.info(f"\n📉 신뢰도 통계:")
        log.info(f"  평균: {pseudo_df['confidence'].mean():.4f}")
        log.info(f"  최소: {pseudo_df['confidence'].min():.4f}")
        log.info(f"  최대: {pseudo_df['confidence'].max():.4f}")
        log.info(f"  중앙값: {pseudo_df['confidence'].median():.4f}")

    # 저장 (학습에 사용할 컬럼만: ID, target)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if selected == 0:
        log.warning("⚠️  선택된 pseudo-label이 없습니다!")
        log.warning(f"   신뢰도 임계값({confidence_threshold})이 너무 높거나 모델 성능이 낮습니다.")
        log.warning(f"   임계값을 낮추거나 더 좋은 모델을 사용하세요.")
        log.info("=" * 70)
        return

    # train.csv 형식 맞춰 저장 (confidence는 별도 파일로)
    pseudo_df[["ID", "target"]].to_csv(output_path, index=False)

    # confidence 포함 상세 버전도 저장
    detail_path = output_path.parent / (output_path.stem + "_with_confidence.csv")
    pseudo_df.to_csv(detail_path, index=False)

    log.info("=" * 70)
    log.info(f"✅ Pseudo-label 저장 완료!")
    log.info(f"  학습용 CSV: {output_path}")
    log.info(f"  상세 CSV (confidence 포함): {detail_path}")
    log.info("=" * 70)
    log.info("")
    log.info("📌 다음 단계: Pseudo-label로 재학습")
    log.info(f"  python src/train.py data.pseudo_csv=pseudo_labels.csv")


if __name__ == "__main__":
    main()
