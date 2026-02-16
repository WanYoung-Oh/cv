"""
PyTorch Lightning DataModule for document image classification
"""

import os
import logging
from typing import Optional

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
        has_label: bool = True
    ):
        """
        Args:
            df: DataFrame (이미지 이름, [클래스 레이블])
            data_root: 데이터 루트 디렉토리 (datasets_fin/)
            image_subdir: 이미지 하위 디렉토리 (train/ 또는 test/)
            transform: 이미지 변환
            has_label: label 컬럼 존재 여부 (test.csv는 False)
        """
        self.df = df
        self.data_root = data_root
        self.image_subdir = image_subdir
        self.transform = transform
        self.has_label = has_label

        # 클래스 인코딩 (label이 있는 경우만)
        if self.has_label and len(df.columns) >= 2:
            self.classes = sorted(self.df.iloc[:, 1].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.classes = []
            self.class_to_idx = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img_name = row.iloc[0]

        # 이미지 경로: datasets_fin/train/image.jpg 또는 datasets_fin/test/image.jpg
        img_path = os.path.join(self.data_root, self.image_subdir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)['image']

        if self.has_label and self.class_to_idx:
            label = self.class_to_idx[row.iloc[1]]
            return image, label

        # label이 없는 경우 (inference용 test.csv)
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
        self.seed = seed if seed is not None else 42  # Config에서 전달된 seed 사용 (기본값 42)

        # 정규화 설정
        self.normalization = normalization or {
            'mean': IMAGENET_MEAN,
            'std': IMAGENET_STD
        }

        self.augmentation_cfg = augmentation or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        self._setup_done = False

    def _parse_augmentation(self, aug_config: dict):
        """Augmentation config dict를 Albumentations 객체로 변환

        Args:
            aug_config: {'type': 'HorizontalFlip', 'p': 0.5} 형태의 dict

        Returns:
            Albumentations transform 객체
        """
        # DictConfig를 일반 dict로 변환 (OmegaConf struct mode 대응)
        if isinstance(aug_config, DictConfig):
            aug_dict = OmegaConf.to_container(aug_config, resolve=True)
        else:
            aug_dict = aug_config.copy()

        aug_type = aug_dict.pop('type')

        # Albumentations 클래스 동적 로드
        if hasattr(A, aug_type):
            aug_class = getattr(A, aug_type)
            return aug_class(**aug_dict)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    def _get_transforms(self, is_train: bool = True):
        """데이터 변환 구성

        Config의 augmentation 설정을 파싱하여 동적으로 생성합니다.
        """
        transforms = []

        # Config에서 augmentation 리스트 가져오기
        if is_train:
            aug_list = self.augmentation_cfg.get('train_augmentations', [])
        else:
            aug_list = self.augmentation_cfg.get('val_augmentations', [])

        # Config의 augmentation을 파싱하여 적용
        if aug_list and self.augmentation_cfg.get('enabled', True):
            for aug_config in aug_list:
                try:
                    transforms.append(self._parse_augmentation(aug_config))
                except Exception as e:
                    # augmentation 파싱 실패 시 경고 후 건너뜀
                    log.warning(f"Failed to parse augmentation: {aug_config}, error: {e}")
        else:
            # Config에 augmentation이 없으면 기본 Resize만 적용
            transforms.append(A.Resize(height=self.img_size, width=self.img_size))

        # 정규화 및 Tensor 변환 (항상 마지막에 적용)
        transforms.extend([
            A.Normalize(
                mean=self.normalization['mean'],
                std=self.normalization['std']
            ),
            ToTensorV2(),
        ])

        return A.Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        """데이터 로드 및 분할"""
        if self._setup_done:
            return

        # Train 데이터 로드 및 Train/Val 분할
        train_full_df = pd.read_csv(self.train_csv)

        train_idx, val_idx = train_test_split(
            range(len(train_full_df)),
            test_size=(1 - self.train_val_split),
            random_state=self.seed,
            stratify=train_full_df.iloc[:, 1]
        )

        train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_full_df.iloc[val_idx].reset_index(drop=True)

        # 클래스 가중치 계산
        class_counts = train_df.iloc[:, 1].value_counts()
        class_weights = 1.0 / class_counts.sort_index()
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        self.class_weights = torch.FloatTensor(class_weights.values)

        # 데이터셋 생성 (DataFrame 직접 전달)
        train_transform = self._get_transforms(is_train=True)
        val_transform = self._get_transforms(is_train=False)

        self.train_dataset = DocumentImageDataset(
            train_df,
            self.data_root,
            self.train_image_dir,
            transform=train_transform,
            has_label=True
        )

        self.val_dataset = DocumentImageDataset(
            val_df,
            self.data_root,
            self.train_image_dir,
            transform=val_transform,
            has_label=True
        )

        # Test 데이터셋 (별도 파일로 제공)
        if self.test_csv and os.path.exists(self.test_csv):
            test_df = pd.read_csv(self.test_csv)
            # label 컬럼 존재 여부 확인
            has_label = len(test_df.columns) >= 2
            self.test_dataset = DocumentImageDataset(
                test_df,
                self.data_root,
                self.test_image_dir,
                transform=val_transform,
                has_label=has_label
            )

        self._setup_done = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
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
            return self.val_dataloader()

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
