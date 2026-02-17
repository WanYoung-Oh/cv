"""
PyTorch Lightning DataModule for document image classification
"""

import os
import logging
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


# Logger
log = logging.getLogger(__name__)

# ImageNet 정규화 상수
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DocumentImageDataset(Dataset):
    """문서 이미지 데이터셋"""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        image_subdir: str,
        transform=None,
        has_label: bool = True,
        class_to_idx: Optional[Dict] = None,
    ):
        """
        Args:
            df: DataFrame (이미지 이름, [클래스 레이블])
            data_root: 데이터 루트 디렉토리 (datasets_fin/)
            image_subdir: 이미지 하위 디렉토리 (train/ 또는 test/)
            transform: 이미지 변환
            has_label: label 컬럼 존재 여부 (test/submission용은 False)
            class_to_idx: 사전 빌드된 클래스→인덱스 매핑 (None이면 df에서 자동 생성)
        """
        self.df = df
        self.data_root = data_root
        self.image_subdir = image_subdir
        self.transform = transform
        self.has_label = has_label

        if self.has_label:
            if class_to_idx is not None:
                # DataModule에서 전체 학습 데이터 기준으로 빌드된 매핑 사용
                self.class_to_idx = class_to_idx
                self.classes = sorted(class_to_idx.keys())
            elif len(df.columns) >= 2:
                # 독립 사용 시 fallback: df에서 직접 빌드
                self.classes = sorted(self.df.iloc[:, 1].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            else:
                self.classes = []
                self.class_to_idx = {}
        else:
            self.classes = []
            self.class_to_idx = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img_name = row.iloc[0]

        img_path = os.path.join(self.data_root, self.image_subdir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)['image']

        if self.has_label and self.class_to_idx:
            label = self.class_to_idx[row.iloc[1]]
            return image, label

        # 레이블 없음 (inference용 submission CSV)
        return image, -1


class DocumentImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule"""

    def __init__(
        self,
        data_root: str,
        train_csv: str,
        test_csv: Optional[str] = None,
        train_image_dir: str = "train/",
        test_image_dir: str = "test/",
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        normalization: Optional[dict] = None,
        augmentation: Optional[dict] = None,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_csv = os.path.join(data_root, train_csv)
        self.test_csv = os.path.join(data_root, test_csv) if test_csv else None
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed if seed is not None else 42
        self.drop_last = drop_last

        self.normalization = normalization or {
            'mean': IMAGENET_MEAN,
            'std': IMAGENET_STD
        }
        self.augmentation_cfg = augmentation or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        self.class_names: Optional[List] = None
        self.idx_to_class: Optional[Dict] = None

        # fit/test 독립 초기화 플래그 (stage-aware)
        self._fit_done = False
        self._test_done = False

    def _parse_augmentation(self, aug_config: dict):
        """Augmentation config dict를 Albumentations 객체로 변환

        Args:
            aug_config: {'type': 'HorizontalFlip', 'p': 0.5} 형태의 dict

        Returns:
            Albumentations transform 객체
        """
        if isinstance(aug_config, DictConfig):
            aug_dict = OmegaConf.to_container(aug_config, resolve=True)
        else:
            aug_dict = aug_config.copy()

        aug_type = aug_dict.pop('type')

        if hasattr(A, aug_type):
            aug_class = getattr(A, aug_type)
            return aug_class(**aug_dict)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    def _get_transforms(self, is_train: bool = True):
        """데이터 변환 구성

        Config의 augmentation 설정을 파싱하여 동적으로 생성합니다.
        모든 augmentation 파싱이 실패한 경우 기본 Resize로 fallback합니다.
        """
        transforms = []

        if is_train:
            aug_list = self.augmentation_cfg.get('train_augmentations', [])
        else:
            aug_list = self.augmentation_cfg.get('val_augmentations', [])

        if aug_list and self.augmentation_cfg.get('enabled', True):
            failed_augmentations = []
            for aug_config in aug_list:
                try:
                    transforms.append(self._parse_augmentation(aug_config))
                except Exception as e:
                    log.error(
                        f"Failed to parse augmentation config: {aug_config}\n"
                        f"Error: {type(e).__name__}: {e}\n"
                        f"This augmentation will be SKIPPED."
                    )
                    failed_augmentations.append(aug_config.get('type', 'unknown'))

            if failed_augmentations:
                log.error(
                    f"⚠️  {len(failed_augmentations)} augmentation(s) failed: {failed_augmentations}"
                )

            # 모든 augmentation 파싱 실패 시 Resize fallback
            if not transforms:
                log.error(
                    "All augmentations failed. Falling back to basic Resize. "
                    "Check your augmentation config."
                )
                transforms.append(A.Resize(height=self.img_size, width=self.img_size))
        else:
            # aug_list가 없거나 enabled=False
            transforms.append(A.Resize(height=self.img_size, width=self.img_size))

        transforms.extend([
            A.Normalize(
                mean=self.normalization['mean'],
                std=self.normalization['std']
            ),
            ToTensorV2(),
        ])

        return A.Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        """데이터 로드 및 분할

        Args:
            stage: 'fit' (train/val), 'test' (test), 'predict', None (전체)
        """
        # Fit 단계: train/val 데이터셋 구성
        if stage in ('fit', 'train', None) and not self._fit_done:
            train_full_df = pd.read_csv(self.train_csv)

            train_idx, val_idx = train_test_split(
                range(len(train_full_df)),
                test_size=(1 - self.train_val_split),
                random_state=self.seed,
                stratify=train_full_df.iloc[:, 1]
            )

            train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_full_df.iloc[val_idx].reset_index(drop=True)

            # 클래스 매핑을 전체 학습 데이터 기준으로 한 번만 빌드
            all_classes = sorted(train_full_df.iloc[:, 1].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
            self.class_names = all_classes
            self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

            # 클래스 가중치 계산 (train_df 기준)
            class_counts = train_df.iloc[:, 1].value_counts()
            class_weights = 1.0 / class_counts.sort_index()
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            self.class_weights = torch.FloatTensor(class_weights.values)

            train_transform = self._get_transforms(is_train=True)
            val_transform = self._get_transforms(is_train=False)

            # 동일한 class_to_idx를 train/val 모두에 전달 → 인덱스 일관성 보장
            self.train_dataset = DocumentImageDataset(
                train_df,
                self.data_root,
                self.train_image_dir,
                transform=train_transform,
                has_label=True,
                class_to_idx=class_to_idx,
            )
            self.val_dataset = DocumentImageDataset(
                val_df,
                self.data_root,
                self.train_image_dir,
                transform=val_transform,
                has_label=True,
                class_to_idx=class_to_idx,
            )

            self._fit_done = True

        # Test 단계: submission CSV 기반 테스트 데이터셋 구성
        if stage in ('test', 'predict', None) and not self._test_done:
            if self.test_csv and os.path.exists(self.test_csv):
                test_df = pd.read_csv(self.test_csv)
                val_transform = self._get_transforms(is_train=False)
                # submission 데이터는 레이블 없음 (has_label=False 고정)
                self.test_dataset = DocumentImageDataset(
                    test_df,
                    self.data_root,
                    self.test_image_dir,
                    transform=val_transform,
                    has_label=False,
                )
                log.info(
                    f"테스트 데이터셋 로드: {len(self.test_dataset)}개 이미지 "
                    f"({self.test_csv})"
                )
            else:
                log.info("test_csv 미설정 또는 파일 없음 - 테스트 데이터셋 미생성")

            self._test_done = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError(
                "test_dataset이 초기화되지 않았습니다. "
                "test_csv 경로를 확인하거나 setup()을 먼저 호출하세요."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
