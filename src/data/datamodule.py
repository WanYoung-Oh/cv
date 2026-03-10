"""
PyTorch Lightning DataModule for document image classification
"""

import os
import logging
from typing import Optional, Dict, List, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split, StratifiedKFold


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
        confusion_pair_class_ids: Optional[List[int]] = None,
        confusion_pair_transform=None,
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
        self.confusion_pair_class_ids = confusion_pair_class_ids
        self.confusion_pair_transform = confusion_pair_transform

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

        # 혼동 쌍 클래스에 추가 증강 적용 (numpy 단계, main transform 전)
        if (self.has_label and self.confusion_pair_transform is not None
                and self.confusion_pair_class_ids is not None):
            class_name = row.iloc[1]
            if class_name in self.class_to_idx:
                if self.class_to_idx[class_name] in self.confusion_pair_class_ids:
                    image = self.confusion_pair_transform(image=image)['image']

        if self.transform:
            image = self.transform(image=image)['image']

        if self.has_label and self.class_to_idx:
            label = self.class_to_idx[row.iloc[1]]
            return image, label

        # 레이블 없음 (inference용 submission CSV)
        return image, -1


class MixedDocumentDataset(Dataset):
    """원본 학습 데이터 + Pseudo-label 테스트 데이터를 합친 데이터셋

    원본 이미지는 train/ 디렉토리에서, pseudo-label 이미지는 test/ 디렉토리에서 로드합니다.
    """

    def __init__(
        self,
        orig_df: pd.DataFrame,
        orig_image_dir: str,
        pseudo_df: pd.DataFrame,
        pseudo_image_dir: str,
        data_root: str,
        transform=None,
        class_to_idx: Optional[Dict] = None,
        confusion_pair_class_ids: Optional[List[int]] = None,
        confusion_pair_transform=None,
    ):
        self.data_root = data_root
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        self.confusion_pair_class_ids = confusion_pair_class_ids
        self.confusion_pair_transform = confusion_pair_transform

        # 각 샘플에 (이미지명, 클래스명, 이미지_디렉토리) 저장
        self.samples: List[Tuple[str, str, str]] = []

        for _, row in orig_df.iterrows():
            self.samples.append((row.iloc[0], row.iloc[1], orig_image_dir))

        for _, row in pseudo_df.iterrows():
            self.samples.append((row.iloc[0], row.iloc[1], pseudo_image_dir))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_name, class_name, image_dir = self.samples[idx]
        img_path = os.path.join(self.data_root, image_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        # 혼동 쌍 클래스에 추가 증강 적용 (numpy 단계, main transform 전)
        if (self.confusion_pair_transform is not None
                and self.confusion_pair_class_ids is not None
                and class_name in self.class_to_idx
                and self.class_to_idx[class_name] in self.confusion_pair_class_ids):
            image = self.confusion_pair_transform(image=image)['image']

        if self.transform:
            image = self.transform(image=image)["image"]

        label = self.class_to_idx[class_name]
        return image, label


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
        # K-Fold 관련 파라미터
        use_kfold: bool = False,
        n_folds: int = 5,
        fold_idx: int = 0,
        # Pseudo-Labeling 관련 파라미터
        pseudo_csv: Optional[str] = None,
        pseudo_image_dir: str = "test/",
        # 소수 클래스 Oversampling
        oversample_minority_classes: bool = False,
        minority_class_ids: Optional[List[int]] = None,
        minority_oversample_repeat: int = 1,
        minority_oversample_threshold: Optional[int] = None,
        # 혼동 쌍 클래스 가중치 보정
        confusion_pair_class_ids: Optional[List[int]] = None,
        confusion_pair_extra_weight: float = 1.0,
        # 클래스 가중치 소스
        class_weights_source: str = "auto",
        class_weights_csv: Optional[str] = None,
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
        
        # K-Fold 설정
        self.use_kfold = use_kfold
        self.n_folds = n_folds
        self.fold_idx = fold_idx

        # Pseudo-Labeling 설정
        # pseudo_csv는 datasets_fin/ 내 상대 경로 (예: "pseudo_labels.csv")
        self.pseudo_csv = os.path.join(data_root, pseudo_csv) if pseudo_csv else None
        self.pseudo_image_dir = pseudo_image_dir

        # 소수 클래스 Oversampling
        self.oversample_minority_classes = oversample_minority_classes
        self.minority_class_ids = minority_class_ids
        self.minority_oversample_repeat = minority_oversample_repeat
        self.minority_oversample_threshold = minority_oversample_threshold

        # 혼동 쌍 클래스 가중치 보정
        self.confusion_pair_class_ids = confusion_pair_class_ids
        self.confusion_pair_extra_weight = confusion_pair_extra_weight

        # 클래스 가중치 소스
        self.class_weights_source = class_weights_source
        self.class_weights_csv = class_weights_csv

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

    # 중첩 transforms를 지원하는 컨테이너 타입 목록
    _CONTAINER_TYPES = {"SomeOf", "OneOf", "Sequential"}

    def _parse_augmentation(self, aug_config: dict):
        """Augmentation config dict를 Albumentations 객체로 변환

        단일 변환과 컨테이너 변환(SomeOf, OneOf, Sequential)을 모두 지원합니다.
        컨테이너 변환의 경우 내부 transforms 리스트를 재귀적으로 파싱합니다.

        Args:
            aug_config: {'type': 'HorizontalFlip', 'p': 0.5} 형태의 dict
                        또는 중첩 구조:
                        {'type': 'SomeOf', 'n': 2, 'transforms': [...], 'p': 1.0}

        Returns:
            Albumentations transform 객체
        """
        if isinstance(aug_config, DictConfig):
            aug_dict = OmegaConf.to_container(aug_config, resolve=True)
        else:
            aug_dict = aug_config.copy()

        aug_type = aug_dict.pop('type')

        if not hasattr(A, aug_type):
            raise ValueError(f"Unknown augmentation type: {aug_type}")

        aug_class = getattr(A, aug_type)

        # 컨테이너 타입: 내부 transforms를 재귀적으로 파싱
        if aug_type in self._CONTAINER_TYPES:
            inner_configs = aug_dict.pop('transforms', [])
            inner_transforms = [
                self._parse_augmentation(cfg) for cfg in inner_configs
            ]
            return aug_class(inner_transforms, **aug_dict)

        # 단일 변환
        return aug_class(**aug_dict)

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

    def _get_confusion_pair_transform(self):
        """혼동 쌍 클래스용 추가 증강 생성 (Normalize/ToTensor 제외)"""
        confusion_aug_list = self.augmentation_cfg.get('confusion_pair_extra_augmentations', None)
        if confusion_aug_list:
            transforms = [self._parse_augmentation(a) for a in confusion_aug_list]
        else:
            # 기본 강화 세트
            transforms = [
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.3),
            ]
        return A.Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        """데이터 로드 및 분할

        Args:
            stage: 'fit' (train/val), 'test' (test), 'predict', None (전체)
        """
        # Fit 단계: train/val 데이터셋 구성
        if stage in ('fit', 'train', None) and not self._fit_done:
            train_full_df = pd.read_csv(self.train_csv)

            # Pseudo-label 데이터 통합 (val 분리 전에 원본 학습 데이터에만 추가)
            pseudo_df = None
            if self.pseudo_csv and os.path.exists(self.pseudo_csv):
                pseudo_df = pd.read_csv(self.pseudo_csv)
                log.info(
                    f"🏷️  Pseudo-label 데이터 로드: {len(pseudo_df):,}개 "
                    f"({self.pseudo_csv})"
                )

            if self.use_kfold:
                # Stratified K-Fold 사용
                log.info(f"📊 Stratified K-Fold 사용: {self.n_folds}-Fold, 현재 Fold {self.fold_idx + 1}/{self.n_folds}")
                
                skf = StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=True,
                    random_state=self.seed
                )
                
                labels = train_full_df.iloc[:, 1].values
                folds = list(skf.split(np.zeros(len(labels)), labels))
                train_idx, val_idx = folds[self.fold_idx]
                
                log.info(f"   Train: {len(train_idx):,}개, Val: {len(val_idx):,}개")
            else:
                # stratify split: 샘플이 1개뿐인 클래스는 train에만 포함 (iloc 위치 인덱스 기준)
                labels = train_full_df.iloc[:, 1].values
                cls_counts = pd.Series(labels).value_counts()
                singleton_mask = np.isin(labels, cls_counts[cls_counts < 2].index.tolist())

                singleton_pos = list(np.where(singleton_mask)[0])
                splittable_pos = list(np.where(~singleton_mask)[0])
                splittable_labels = labels[splittable_pos]

                if singleton_pos:
                    log.warning(
                        f"⚠️  샘플 1개 클래스 {list(cls_counts[cls_counts < 2].index)} → "
                        f"train 전용으로 처리 ({len(singleton_pos)}개)"
                    )

                split_train_local, split_val_local = train_test_split(
                    range(len(splittable_pos)),
                    test_size=(1 - self.train_val_split),
                    random_state=self.seed,
                    stratify=splittable_labels
                )
                train_idx = singleton_pos + [splittable_pos[i] for i in split_train_local]
                val_idx = [splittable_pos[i] for i in split_val_local]

            train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_full_df.iloc[val_idx].reset_index(drop=True)

            # 클래스 매핑을 전체 학습 데이터 기준으로 한 번만 빌드
            all_classes = sorted(train_full_df.iloc[:, 1].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
            self.class_names = all_classes
            self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

            # 원본 train_df 저장 (class_weights 계산용 — oversampling 전)
            train_df_original = train_df.copy()

            # (A) 소수 클래스 Oversampling
            if self.oversample_minority_classes:
                if self.minority_class_ids is not None:
                    target_classes = {
                        self.idx_to_class[i] for i in self.minority_class_ids
                        if i in self.idx_to_class
                    }
                elif self.minority_oversample_threshold is not None:
                    cls_counts = train_df.iloc[:, 1].value_counts()
                    target_classes = {
                        cls for cls, cnt in cls_counts.items()
                        if cnt < self.minority_oversample_threshold
                    }
                else:
                    target_classes = set()

                if target_classes:
                    extra_rows = []
                    for cls_name in target_classes:
                        rows = train_df[train_df.iloc[:, 1] == cls_name]
                        for _ in range(self.minority_oversample_repeat):
                            extra_rows.append(rows)
                    if extra_rows:
                        train_df = pd.concat([train_df] + extra_rows, ignore_index=True)
                        train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                        log.info(f"소수 클래스 oversampling 완료: 총 {len(train_df):,}개")

            # (B) 클래스 가중치 계산
            if self.class_weights_source == 'csv' and self.class_weights_csv:
                weights_path = self.class_weights_csv
                if not os.path.isabs(weights_path):
                    weights_path = os.path.join(self.data_root, weights_path)
                weights_df = pd.read_csv(weights_path).sort_values('class_id')
                self.class_weights = torch.FloatTensor(weights_df['weight'].values)
                log.info(f"클래스 가중치 CSV 로드: {weights_path}")
            else:
                # auto 계산 (역빈도, oversampling 전 원본 기준)
                class_counts = train_df_original.iloc[:, 1].value_counts()
                class_weights_series = 1.0 / class_counts.sort_index()
                class_weights_series = class_weights_series / class_weights_series.sum() * len(class_weights_series)
                self.class_weights = torch.FloatTensor(class_weights_series.values)

            # (C) 혼동 쌍 가중치 보정
            if (self.confusion_pair_class_ids
                    and self.confusion_pair_extra_weight > 1.0
                    and self.class_weights is not None):
                for idx in self.confusion_pair_class_ids:
                    if 0 <= idx < len(self.class_weights):
                        self.class_weights[idx] *= self.confusion_pair_extra_weight
                log.info(
                    f"혼동 쌍 클래스 {self.confusion_pair_class_ids} "
                    f"가중치 {self.confusion_pair_extra_weight}배 적용"
                )

            # oversampling이 반영된 train_df 저장 (pseudo merge 전)
            # MixedDocumentDataset의 orig_df로 사용 — oversampling 효과 보존
            train_df_for_orig = train_df.copy()

            train_transform = self._get_transforms(is_train=True)
            val_transform = self._get_transforms(is_train=False)

            # 혼동 쌍 augmentation 설정
            confusion_pair_transform = None
            if (self.augmentation_cfg.get('augmentation_confusion_pair_enabled', False)
                    and self.confusion_pair_class_ids):
                confusion_pair_transform = self._get_confusion_pair_transform()
                log.info(f"혼동 쌍 클래스 {self.confusion_pair_class_ids} 전용 추가 증강 활성화")

            # Pseudo-label 데이터를 train_df에 통합 (val에는 추가 안 함)
            if pseudo_df is not None and len(pseudo_df) > 0:
                # pseudo_df 컬럼을 train_df 형식(ID, target)으로 맞춤
                pseudo_df_aligned = pseudo_df[["ID", "target"]].copy()
                pseudo_df_aligned.columns = train_full_df.columns[:2]

                # 알 수 없는 클래스 필터링 (안전장치)
                known_classes = set(class_to_idx.keys())
                valid_mask = pseudo_df_aligned.iloc[:, 1].isin(known_classes)
                pseudo_df_aligned = pseudo_df_aligned[valid_mask].reset_index(drop=True)

                orig_size = len(train_df)
                train_df = pd.concat(
                    [train_df, pseudo_df_aligned], ignore_index=True
                )
                log.info(
                    f"🏷️  Pseudo-label 통합 완료: "
                    f"원본 {orig_size:,}개 + pseudo {len(pseudo_df_aligned):,}개 "
                    f"= 총 {len(train_df):,}개"
                )

            # 동일한 class_to_idx를 train/val 모두에 전달 → 인덱스 일관성 보장
            # pseudo-label 이미지는 test 디렉토리에 있으므로 MixedDataset 사용
            if pseudo_df is not None and len(pseudo_df) > 0:
                self.train_dataset = MixedDocumentDataset(
                    orig_df=train_df_for_orig,
                    orig_image_dir=self.train_image_dir,
                    pseudo_df=pseudo_df_aligned,
                    pseudo_image_dir=self.pseudo_image_dir,
                    data_root=self.data_root,
                    transform=train_transform,
                    class_to_idx=class_to_idx,
                    confusion_pair_class_ids=self.confusion_pair_class_ids,
                    confusion_pair_transform=confusion_pair_transform,
                )
            else:
                self.train_dataset = DocumentImageDataset(
                    train_df,
                    self.data_root,
                    self.train_image_dir,
                    transform=train_transform,
                    has_label=True,
                    class_to_idx=class_to_idx,
                    confusion_pair_class_ids=self.confusion_pair_class_ids,
                    confusion_pair_transform=confusion_pair_transform,
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

    def analyze_dataset_statistics(
        self, 
        dataset_type: str = 'test',
        sample_size: Optional[int] = None,
        compute_rgb_stats: bool = True
    ) -> Dict:
        """데이터셋의 통계 정보를 분석합니다.
        
        Args:
            dataset_type: 분석할 데이터셋 타입 ('train', 'val', 'test')
            sample_size: 샘플링할 이미지 개수 (None이면 전체, 큰 데이터셋에서는 샘플링 권장)
            compute_rgb_stats: RGB 채널별 통계 계산 여부 (시간이 오래 걸림)
        
        Returns:
            통계 정보를 담은 딕셔너리
        """
        # 데이터셋 선택
        if dataset_type == 'train':
            if self.train_dataset is None:
                raise RuntimeError("train_dataset이 초기화되지 않았습니다. setup()을 먼저 호출하세요.")
            dataset = self.train_dataset
        elif dataset_type == 'val':
            if self.val_dataset is None:
                raise RuntimeError("val_dataset이 초기화되지 않았습니다. setup()을 먼저 호출하세요.")
            dataset = self.val_dataset
        elif dataset_type == 'test':
            if self.test_dataset is None:
                raise RuntimeError("test_dataset이 초기화되지 않았습니다. setup()을 먼저 호출하세요.")
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Use 'train', 'val', or 'test'.")

        log.info("=" * 70)
        log.info(f"📊 {dataset_type.upper()} 데이터셋 분석 시작")
        log.info("=" * 70)
        
        total_samples = len(dataset)
        log.info(f"총 샘플 수: {total_samples:,}개")
        
        # 샘플링
        if sample_size and sample_size < total_samples:
            log.info(f"샘플링: {sample_size:,}개 이미지로 분석 (전체의 {sample_size/total_samples*100:.1f}%)")
            indices = np.random.choice(total_samples, sample_size, replace=False)
        else:
            log.info("전체 데이터셋 분석")
            indices = range(total_samples)
        
        # 통계 수집
        image_sizes = []
        aspect_ratios = []
        file_sizes_kb = []
        
        if compute_rgb_stats:
            rgb_means = []
            rgb_stds = []
            brightness_values = []
        
        log.info("이미지 메타데이터 수집 중...")
        
        for idx in indices:
            if hasattr(dataset, 'df'):
                row = dataset.df.iloc[idx]
                img_name = row.iloc[0]
                img_path = os.path.join(dataset.data_root, dataset.image_subdir, img_name)
            else:  # MixedDocumentDataset
                img_name, _, image_dir = dataset.samples[idx]
                img_path = os.path.join(dataset.data_root, image_dir, img_name)
            
            try:
                # 파일 크기
                file_size_kb = os.path.getsize(img_path) / 1024
                file_sizes_kb.append(file_size_kb)
                
                # 이미지 로드 (원본)
                img = Image.open(img_path).convert('RGB')
                width, height = img.size
                
                # 크기 정보
                image_sizes.append((width, height))
                aspect_ratios.append(width / height)
                
                # RGB 통계 (선택적)
                if compute_rgb_stats:
                    img_array = np.array(img).astype(np.float32) / 255.0  # 0-1 정규화
                    
                    # 채널별 평균/표준편차
                    rgb_mean = img_array.mean(axis=(0, 1))  # (R, G, B)
                    rgb_std = img_array.std(axis=(0, 1))
                    rgb_means.append(rgb_mean)
                    rgb_stds.append(rgb_std)
                    
                    # 밝기 (그레이스케일 변환)
                    brightness = img_array.mean()
                    brightness_values.append(brightness)
                    
            except Exception as e:
                log.warning(f"이미지 로드 실패: {img_name} - {e}")
                continue
        
        # 통계 계산
        widths = [w for w, h in image_sizes]
        heights = [h for w, h in image_sizes]
        areas = [w * h for w, h in image_sizes]
        
        stats = {
            'total_samples': total_samples,
            'analyzed_samples': len(image_sizes),
            'image_size': {
                'width': {
                    'min': int(np.min(widths)),
                    'max': int(np.max(widths)),
                    'mean': float(np.mean(widths)),
                    'median': float(np.median(widths)),
                    'std': float(np.std(widths)),
                },
                'height': {
                    'min': int(np.min(heights)),
                    'max': int(np.max(heights)),
                    'mean': float(np.mean(heights)),
                    'median': float(np.median(heights)),
                    'std': float(np.std(heights)),
                },
                'area_megapixels': {
                    'min': float(np.min(areas) / 1e6),
                    'max': float(np.max(areas) / 1e6),
                    'mean': float(np.mean(areas) / 1e6),
                    'median': float(np.median(areas) / 1e6),
                },
            },
            'aspect_ratio': {
                'min': float(np.min(aspect_ratios)),
                'max': float(np.max(aspect_ratios)),
                'mean': float(np.mean(aspect_ratios)),
                'median': float(np.median(aspect_ratios)),
                'std': float(np.std(aspect_ratios)),
            },
            'file_size_kb': {
                'min': float(np.min(file_sizes_kb)),
                'max': float(np.max(file_sizes_kb)),
                'mean': float(np.mean(file_sizes_kb)),
                'median': float(np.median(file_sizes_kb)),
                'total_mb': float(np.sum(file_sizes_kb) / 1024),
            },
        }
        
        if compute_rgb_stats:
            rgb_means_array = np.array(rgb_means)
            rgb_stds_array = np.array(rgb_stds)
            
            stats['rgb_statistics'] = {
                'mean': {
                    'R': float(rgb_means_array[:, 0].mean()),
                    'G': float(rgb_means_array[:, 1].mean()),
                    'B': float(rgb_means_array[:, 2].mean()),
                },
                'std': {
                    'R': float(rgb_stds_array[:, 0].mean()),
                    'G': float(rgb_stds_array[:, 1].mean()),
                    'B': float(rgb_stds_array[:, 2].mean()),
                },
                'brightness': {
                    'min': float(np.min(brightness_values)),
                    'max': float(np.max(brightness_values)),
                    'mean': float(np.mean(brightness_values)),
                    'median': float(np.median(brightness_values)),
                },
            }
        
        # 결과 출력
        self._print_statistics(stats, dataset_type)
        
        return stats
    
    def _print_statistics(self, stats: Dict, dataset_type: str):
        """통계 정보를 보기 좋게 출력합니다."""
        log.info("")
        log.info("=" * 70)
        log.info(f"📈 {dataset_type.upper()} 데이터셋 통계")
        log.info("=" * 70)
        
        log.info(f"\n📦 샘플 정보:")
        log.info(f"  총 샘플 수: {stats['total_samples']:,}개")
        log.info(f"  분석된 샘플: {stats['analyzed_samples']:,}개")
        
        size_stats = stats['image_size']
        log.info(f"\n📐 이미지 크기:")
        log.info(f"  Width:  {size_stats['width']['min']:4d} ~ {size_stats['width']['max']:4d} "
                f"(평균: {size_stats['width']['mean']:.0f}, 중앙값: {size_stats['width']['median']:.0f})")
        log.info(f"  Height: {size_stats['height']['min']:4d} ~ {size_stats['height']['max']:4d} "
                f"(평균: {size_stats['height']['mean']:.0f}, 중앙값: {size_stats['height']['median']:.0f})")
        log.info(f"  Area:   {size_stats['area_megapixels']['min']:.2f} ~ {size_stats['area_megapixels']['max']:.2f} MP "
                f"(평균: {size_stats['area_megapixels']['mean']:.2f} MP)")
        
        ar_stats = stats['aspect_ratio']
        log.info(f"\n📏 Aspect Ratio:")
        log.info(f"  범위: {ar_stats['min']:.2f} ~ {ar_stats['max']:.2f}")
        log.info(f"  평균: {ar_stats['mean']:.2f} (표준편차: {ar_stats['std']:.2f})")
        
        file_stats = stats['file_size_kb']
        log.info(f"\n💾 파일 크기:")
        log.info(f"  범위: {file_stats['min']:.1f} ~ {file_stats['max']:.1f} KB")
        log.info(f"  평균: {file_stats['mean']:.1f} KB (중앙값: {file_stats['median']:.1f} KB)")
        log.info(f"  전체: {file_stats['total_mb']:.1f} MB")
        
        if 'rgb_statistics' in stats:
            rgb_stats = stats['rgb_statistics']
            log.info(f"\n🎨 RGB 채널 통계:")
            log.info(f"  Mean: R={rgb_stats['mean']['R']:.3f}, "
                    f"G={rgb_stats['mean']['G']:.3f}, "
                    f"B={rgb_stats['mean']['B']:.3f}")
            log.info(f"  Std:  R={rgb_stats['std']['R']:.3f}, "
                    f"G={rgb_stats['std']['G']:.3f}, "
                    f"B={rgb_stats['std']['B']:.3f}")
            
            brightness = rgb_stats['brightness']
            log.info(f"\n💡 밝기:")
            log.info(f"  범위: {brightness['min']:.3f} ~ {brightness['max']:.3f}")
            log.info(f"  평균: {brightness['mean']:.3f} (중앙값: {brightness['median']:.3f})")
        
        log.info("\n" + "=" * 70)
    
    def get_dataset_info(self) -> Dict:
        """데이터셋의 기본 정보를 반환합니다.
        
        Returns:
            데이터셋 정보 딕셔너리
        """
        info = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'normalization': self.normalization,
        }
        
        if self.train_dataset:
            info['train'] = {
                'size': len(self.train_dataset),
                'num_classes': len(self.class_names) if self.class_names else 0,
                'class_names': self.class_names,
            }
        
        if self.val_dataset:
            info['val'] = {
                'size': len(self.val_dataset),
            }
        
        if self.test_dataset:
            info['test'] = {
                'size': len(self.test_dataset),
            }
        
        return info
