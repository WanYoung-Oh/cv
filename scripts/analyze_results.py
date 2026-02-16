"""
결과 분석 스크립트
- Confusion Matrix 생성
- 클래스별 성능 분석
- 오분류 예시 출력
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
from typing import Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from src.models.module import DocumentClassifierModule
from src.data.datamodule import DocumentImageDataModule

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str) -> DocumentClassifierModule:
    """체크포인트에서 모델 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로

    Returns:
        로드된 모델
    """
    log.info(f"체크포인트 로드: {checkpoint_path}")
    model = DocumentClassifierModule.load_from_checkpoint(
        checkpoint_path,
        strict=False  # class_weights 등 추가 키 무시
    )
    model.eval()
    return model


def get_predictions(
    model: DocumentClassifierModule,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """데이터로더로부터 예측 결과 추출

    Args:
        model: 모델
        dataloader: 데이터로더
        device: 디바이스

    Returns:
        (predictions, labels, probabilities) tuple
    """
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="추론 중"):
            images, labels = batch
            images = images.to(device)

            # 예측
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probs)

    return predictions, labels, probabilities


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Optional[str] = None
):
    """Confusion Matrix 시각화

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        save_path: 저장 경로 (None이면 화면에만 표시)
    """
    cm = confusion_matrix(y_true, y_pred)

    # 정규화 (각 행의 합이 1이 되도록)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 원본 confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # 정규화된 confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Confusion Matrix 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str]
):
    """클래스별 성능 분석

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        y_probs: 예측 확률
        class_names: 클래스 이름 리스트
    """
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    # DataFrame으로 변환
    df_report = pd.DataFrame(report).transpose()

    # 클래스별 평균 확률
    class_probs = {}
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            avg_prob = y_probs[mask, i].mean()
            class_probs[class_name] = avg_prob

    # 결과 출력
    log.info("\n" + "=" * 80)
    log.info("클래스별 성능 분석")
    log.info("=" * 80)

    print("\n📊 Classification Report:")
    print(df_report.to_string())

    print("\n📈 클래스별 평균 확률 (정답인 경우):")
    for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {str(class_name):50s}: {prob:.4f}")

    # 가장 어려운 클래스 찾기 (F1 score 기준)
    class_f1 = df_report.loc[class_names, 'f1-score'].sort_values()

    print("\n⚠️  가장 어려운 클래스 (낮은 F1 순):")
    for class_name, f1 in class_f1.head(5).items():
        print(f"  {str(class_name):50s}: F1 = {f1:.4f}")

    print("\n✅ 가장 쉬운 클래스 (높은 F1 순):")
    for class_name, f1 in class_f1.tail(5).items():
        print(f"  {str(class_name):50s}: F1 = {f1:.4f}")

    log.info("=" * 80)


def find_misclassified_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    top_k: int = 10
):
    """오분류 예시 찾기

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        y_probs: 예측 확률
        class_names: 클래스 이름 리스트
        top_k: 출력할 개수
    """
    # 오분류 찾기
    misclassified = y_true != y_pred

    if misclassified.sum() == 0:
        log.info("✅ 오분류 없음!")
        return

    # 오분류 중 가장 확신한 것들 (높은 확률로 틀린 것)
    misclassified_probs = y_probs[misclassified].max(axis=1)
    misclassified_indices = np.where(misclassified)[0]

    # 확률 기준 정렬
    sorted_indices = np.argsort(misclassified_probs)[::-1]

    log.info("\n" + "=" * 80)
    log.info(f"오분류 예시 (총 {misclassified.sum()}개 중 상위 {min(top_k, len(sorted_indices))}개)")
    log.info("=" * 80)

    for rank, idx in enumerate(sorted_indices[:top_k], 1):
        orig_idx = misclassified_indices[idx]
        true_label = y_true[orig_idx]
        pred_label = y_pred[orig_idx]
        confidence = misclassified_probs[idx]

        print(f"\n{rank}. 샘플 #{orig_idx}")
        print(f"   실제: {str(class_names[true_label])}")
        print(f"   예측: {str(class_names[pred_label])} (확률: {confidence:.4f})")
        print(f"   Top-3 예측:")
        top3_indices = np.argsort(y_probs[orig_idx])[::-1][:3]
        for i, class_idx in enumerate(top3_indices, 1):
            print(f"     {i}. {str(class_names[class_idx]):50s}: {y_probs[orig_idx, class_idx]:.4f}")

    log.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="모델 결과 분석")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="체크포인트 파일 경로"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets_fin/",
        help="데이터 루트 디렉토리"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results/",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="디바이스 (cpu, cuda, mps)"
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model = load_checkpoint(args.checkpoint)

    # DataModule 생성 (baseline_aug 설정 사용)
    from omegaconf import OmegaConf

    # Config 로드
    config_path = Path("configs/data/baseline_aug.yaml")
    if config_path.exists():
        data_config = OmegaConf.load(config_path)
    else:
        log.warning("baseline_aug.yaml 없음, 기본 설정 사용")
        data_config = OmegaConf.create({
            "img_size": 768,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "train_val_split": 0.8
        })

    data_module = DocumentImageDataModule(
        data_root=args.data_root,
        train_csv="train.csv",
        test_csv="test.csv",
        img_size=data_config.img_size,
        batch_size=args.batch_size,
        num_workers=4,
        train_val_split=data_config.get("train_val_split", 0.8),
        normalization=data_config.get("normalization"),
        augmentation=data_config.get("augmentation"),
        seed=42
    )

    data_module.setup()

    # 클래스 이름
    class_names = data_module.train_dataset.classes
    log.info(f"클래스 개수: {len(class_names)}")

    # Validation 데이터로 예측
    log.info("Validation 데이터 분석 중...")
    val_dataloader = data_module.val_dataloader()
    y_pred, y_true, y_probs = get_predictions(model, val_dataloader, args.device)

    # Confusion Matrix 생성
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, str(cm_path))

    # 클래스별 성능 분석
    analyze_class_performance(y_true, y_pred, y_probs, class_names)

    # 오분류 예시
    find_misclassified_examples(y_true, y_pred, y_probs, class_names, top_k=10)

    # 전체 성능 요약
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    log.info("\n" + "=" * 80)
    log.info("전체 성능 요약")
    log.info("=" * 80)
    log.info(f"Accuracy:       {accuracy:.4f}")
    log.info(f"F1 (Macro):     {f1_macro:.4f}")
    log.info(f"F1 (Weighted):  {f1_weighted:.4f}")
    log.info(f"총 샘플 수:      {len(y_true)}")
    log.info(f"오분류 샘플:     {(y_true != y_pred).sum()}")
    log.info("=" * 80)

    log.info(f"\n✅ 분석 완료! 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
