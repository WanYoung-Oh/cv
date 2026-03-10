"""
유틸리티 함수
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from src.data.datamodule import DocumentImageDataModule

log = logging.getLogger(__name__)


def save_json(data: Dict[str, Any], save_path: str) -> None:
    """데이터를 JSON으로 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_val_f1_from_filename(checkpoint_path: Path) -> Optional[float]:
    """
    체크포인트 파일명에서 val_f1 메트릭을 추출합니다.

    Args:
        checkpoint_path: 체크포인트 파일 경로 (예: epoch=10-val_f1=0.950.ckpt)

    Returns:
        추출된 val_f1 값, 실패 시 None

    Example:
        >>> extract_val_f1_from_filename(Path("epoch=10-val_f1=0.950.ckpt"))
        0.950
    """
    try:
        filename = checkpoint_path.stem
        if 'val_f1=' in filename:
            val_f1_str = filename.split('val_f1=')[1]
            return float(val_f1_str)
    except (ValueError, IndexError):
        pass
    return None


def create_datamodule_from_config(cfg: "DictConfig") -> "DocumentImageDataModule":
    """
    Hydra config에서 DocumentImageDataModule을 생성합니다.

    inference/ensemble 용도이므로 test_csv로 sample_submission_csv를 사용합니다.
    sample_submission_csv가 없으면 test_csv로 fallback합니다.

    Args:
        cfg: Hydra 설정 객체 (cfg.data, cfg.training 포함)

    Returns:
        설정된 DocumentImageDataModule 인스턴스
    """
    from src.data.datamodule import DocumentImageDataModule

    # inference/ensemble에서는 sample_submission_csv를 test 데이터 소스로 사용
    test_csv = cfg.data.get('sample_submission_csv', cfg.data.get('test_csv', None))

    return DocumentImageDataModule(
        data_root=cfg.data.root_path,
        train_csv=cfg.data.train_csv,
        test_csv=test_csv,
        train_image_dir=cfg.data.get('train_image_dir', 'train/'),
        test_image_dir=cfg.data.get('test_image_dir', 'test/'),
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        train_val_split=cfg.data.train_val_split,
        normalization=cfg.data.normalization,
        augmentation=cfg.data.augmentation,
        seed=cfg.get('seed', 42),
        drop_last=cfg.training.get('drop_last', False),
        pseudo_csv=cfg.data.get('pseudo_csv', None),
        pseudo_image_dir=cfg.data.get('pseudo_image_dir', 'test/'),
    )


def save_predictions_to_csv(
    predictions: List[int],
    output_path: str,
    data_root: str,
    test_csv_path: Optional[str] = None,
    task_name: str = "Inference",
) -> pd.DataFrame:
    """
    예측 결과를 CSV 파일로 저장하고 클래스 분포를 로깅합니다.

    Args:
        predictions: 예측 정수 인덱스 리스트
        output_path: 저장할 CSV 파일 경로
        data_root: 데이터 루트 경로
        test_csv_path: 테스트 CSV 경로 (optional)
        task_name: 작업 이름 (로깅용)

    Returns:
        저장된 DataFrame
    """
    pred_values = [int(p) for p in predictions]

    # sample_submission.csv가 있으면 그 형식 따르기
    sample_submission_path = os.path.join(data_root, "sample_submission.csv")

    if os.path.exists(sample_submission_path):
        log.info(f"sample_submission.csv 형식 사용: {sample_submission_path}")
        sample_df = pd.read_csv(sample_submission_path)
        col_name = sample_df.columns[1]
        sample_df[col_name] = pred_values[:len(sample_df)]
        result_df = sample_df
    else:
        # 기본 형식으로 저장 (id, target)
        log.info("기본 형식으로 저장 (id, target)")
        if test_csv_path and os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            image_ids = test_df.iloc[:, 0].tolist()
        else:
            image_ids = list(range(len(pred_values)))

        result_df = pd.DataFrame({
            'id': image_ids[:len(pred_values)],
            'target': pred_values
        })

    # 출력 디렉토리 생성 후 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    # 완료 로깅
    log.info("=" * 70)
    log.info(f"✅ {task_name} 완료!")
    log.info(f"📄 결과 저장: {output_path}")
    log.info("=" * 70)

    # 클래스별 예측 분포 출력
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    log.info(f"\n📈 예측 클래스 분포:")
    for class_id, count in pred_counts.items():
        log.info(f"  클래스 {class_id}: {count:4d} ({count/len(predictions)*100:5.2f}%)")

    return result_df
