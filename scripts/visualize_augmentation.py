#!/usr/bin/env python3
"""
baseline_aug.yaml 데이터 증강 설정이 datamodule에 의해 정상 작동하는지 확인합니다.
학습 이미지를 랜덤 샘플링하여 20개를 원본 vs 증강 비교 이미지로 출력합니다.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

from src.data.datamodule import DocumentImageDataModule


def denormalize(tensor, mean, std):
    """정규화된 텐서를 0-255 RGB 이미지로 복원"""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    mean = np.array(mean)
    std = np.array(std)
    img = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    # 1. baseline_aug.yaml 로드
    config_path = project_root / "configs" / "data" / "baseline_aug.yaml"
    data_cfg = OmegaConf.load(config_path)

    # 2. DataModule 생성 (datamodule과 동일한 방식으로 augmentation 파싱)
    dm = DocumentImageDataModule(
        data_root="datasets_fin",
        train_csv="train.csv",
        test_csv=None,
        train_image_dir="train/",
        test_image_dir="test/",
        img_size=data_cfg.img_size,
        batch_size=16,
        num_workers=0,  # 시각화는 싱글 스레드
        train_val_split=data_cfg.train_val_split,
        normalization=OmegaConf.to_container(data_cfg.normalization, resolve=True),
        augmentation=OmegaConf.to_container(data_cfg.augmentation, resolve=True),
        seed=42,
        pseudo_csv=None,
    )
    dm.setup(stage="fit")

    train_dataset = dm.train_dataset
    n_total = len(train_dataset)

    # 3. 20개 랜덤 샘플 인덱스
    np.random.seed(42)
    indices = np.random.choice(n_total, size=min(20, n_total), replace=False)

    mean = dm.normalization["mean"]
    std = dm.normalization["std"]

    # 4. 원본 vs 증강 이미지 수집
    originals = []
    augmenteds = []

    for idx in indices:
        # 원본: transform 적용 전 이미지 (DocumentImageDataset vs MixedDocumentDataset)
        if hasattr(train_dataset, "samples"):
            img_name, _, image_dir = train_dataset.samples[idx]
            img_path = os.path.join(dm.data_root, image_dir, img_name)
        else:
            row = train_dataset.df.iloc[idx]
            img_name = row.iloc[0]
            img_path = os.path.join(
                train_dataset.data_root,
                train_dataset.image_subdir,
                img_name,
            )
        orig_img = np.array(Image.open(img_path).convert("RGB"))
        originals.append(orig_img)

        # 증강: dataset __getitem__ 호출 (같은 인덱스에서 매번 다른 랜덤 증강)
        aug_tensor, _ = train_dataset[idx]
        aug_img = denormalize(aug_tensor, mean, std)
        augmenteds.append(aug_img)

    # 5. 그리드 시각화: 4행 x 10열 (상단 2행: 원본 20개, 하단 2행: 증강 20개)
    cell_h, cell_w = 100, 100
    n_cols = 10
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    fig.suptitle(
        "baseline_aug.yaml Augmentation Check (via datamodule)\n"
        "Top: Original | Bottom: Augmented (LongestMaxSize, PadIfNeeded, Flip, Rotate, CLAHE, etc.)",
        fontsize=11,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.axis("off")
            k = (i % 2) * n_cols + j  # 행 0,1→원본 / 행 2,3→증강, 각 10개씩
            if k >= 20:
                continue
            img = originals[k] if i < 2 else augmenteds[k]
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            ax.imshow(np.array(pil_img))
            if j == 0:
                label = "Original" if i < 2 else "Augmented"
                ax.set_ylabel(label, fontsize=9)

    plt.tight_layout()

    # 6. 저장
    output_dir = project_root / ".benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "augmentation_visualization.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n저장 완료: {output_path}")
    print(f"총 {len(indices)}개 샘플 (원본 | 증강) 비교 이미지를 생성했습니다.")
    print("데이터 증강이 datamodule을 통해 정상적으로 적용되는 것을 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
