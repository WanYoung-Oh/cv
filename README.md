# 문서 이미지 분류 프로젝트

> PyTorch Lightning + Hydra + WanDB 기반 문서 이미지 분류 시스템

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-2.4-purple.svg)](https://lightning.ai/)
[![F1 Goal](https://img.shields.io/badge/F1--Goal-0.95-success.svg)](https://github.com)

---

## 🎯 프로젝트 목표

**목표**: F1-Macro **0.95** (퍼블릭 리더보드 기준)

| Metric | 목표 | 현재 | 비고 |
|--------|------|------|------|
| **F1-Macro** | 0.95+ | - | 퍼블릭 리더보드 기준 |
| **Val F1 (Champion)** | - | **0.974** | 20260301_run_007 (MaxViT-384, K-Fold) |
| **퍼블릭 리더보드** | - | **0.9441** | MaxViT-Base-384 앙상블 |

**현재 과제**: Val F1(0.974)과 리더보드(0.9441) 간 **Generalization Gap** 축소
→ TTA, K-Fold 앙상블, Binary Ensemble 보정 등 진행 중

---

## 🚀 빠른 시작 (5분)

### 1. 환경 설정
```bash
conda activate pytorch_test
pip install -r requirements.txt
```

### 2. WanDB 설정 (선택사항)
```bash
# .env 파일 생성
echo "WANDB_API_KEY=your-api-key" > .env
echo "WANDB_PROJECT=doc_image_classification" >> .env

# 또는 WanDB 없이 실행
export WANDB_MODE=disabled
```

### 3. 학습
```bash
# MaxViT-Base-384 + K-Fold (현재 베스트)
python src/train.py model=maxvit_base_384 data=transformer_384 training=transformer data.use_kfold=true data.fold_idx=0

# ResNet34 + baseline_aug (경량 베이스라인)
python src/train.py data=baseline_aug model=resnet34 training=baseline_512
```

### 4. Inference (리더보드 제출)
```bash
python src/inference.py
# 출력: datasets_fin/submission/submission_{model_name}.csv
```

### 5. 앙상블
```bash
python src/ensemble.py ensemble=MaxViT_kfold_5models
```

---

## 📊 데이터셋

### 구조
```
datasets_fin/
├── train.csv               (1,570개, 레이블 있음)
├── sample_submission.csv   (3,140개, 리더보드 제출 형식)
├── meta.csv                (17개 클래스 정보)
├── train/                  (훈련 이미지)
├── test/                   (테스트 이미지, 리더보드 제출용)
├── pseudo_labels.csv       (Pseudo Labeling 결과, confidence ≥ 0.9)
└── submission/             (inference/ensemble 결과 자동 저장)
```

### 클래스 정보
- **17개 클래스**: 이력서, 여권, 운전면허증 등 문서 타입
- **불균형**: 상위 3개 클래스가 전체의 50%
- **해결**: Class Weights 자동 계산, Oversampling (minority class)

---

## 🤖 사용 가능한 모델

### CNN 모델 (512×512)
| 모델 | Config | Data Config | 비고 |
|------|--------|-------------|------|
| ResNet34 | `resnet34` | `baseline_aug` | 메모리 효율 |
| ResNet50 | `resnet50` | `baseline_aug` | 안정적 |
| EfficientNet-B4 | `efficientnet_b4` | `baseline_aug` | batch_size=8 권장 |
| ConvNeXt-Base | `convnext_base` | `baseline_aug` | 최신 CNN |

### Transformer 모델
| 모델 | Config | Data Config | 입력 크기 |
|------|--------|-------------|-----------|
| Swin-Base-224 | `swin_base_224` | `transformer_224` | 224×224 |
| Swin-Base-384 | `swin_base_384` | `transformer_384` | 384×384 |
| DeiT-Base-224 | `deit_base_224` | `transformer_224` | 224×224 |
| DeiT-Base-384 | `deit_base_384` | `transformer_384` | 384×384 |
| **MaxViT-Base-384** | `maxvit_base_384` | `transformer_384` | 384×384 ⭐현재 베스트 |

---

## 📁 프로젝트 구조

```
CV/
├── readme.md                
├── requirements.txt         # 의존성
│
├── configs/                 # Hydra 설정
│   ├── config.yaml          # 메인 설정 (기본 조합)
│   ├── data/                # 데이터 설정 (이미지 크기, Augmentation)
│   │   ├── baseline_aug.yaml    # CNN 512×512 + 강한 Augmentation
│   │   ├── transformer_384.yaml # Transformer 384×384
│   │   └── transformer_224.yaml # Transformer 224×224
│   ├── model/               # 모델 아키텍처 설정
│   ├── training/            # 훈련 하이퍼파라미터
│   │   ├── default.yaml         # ResNet/ConvNeXt 기본값
│   │   ├── baseline_512.yaml    # 512px CNN용
│   │   ├── transformer.yaml     # Transformer용 (AMP 포함)
│   │   └── efficientnet.yaml    # EfficientNet용
│   ├── ensemble/            # 앙상블 조합 (15개 사전 정의)
│   ├── binary/              # Binary Classifier 설정
│   ├── inference/           # Inference 설정
│
├── src/
│   ├── train.py             # 훈련 (Hydra + Lightning + WanDB)
│   ├── inference.py         # 추론 (TTA 옵션 포함)
│   ├── ensemble.py          # 앙상블 (Soft/Hard Voting, Weighted)
│   ├── data/
│   │   └── datamodule.py    # DataModule (K-Fold, Oversampling, Pseudo)
│   ├── models/
│   │   └── module.py        # LightningModule (FocalLoss, Mixup/CutMix)
│   └── utils/
│       ├── tta.py           # TTA (light/standard/heavy, D4 대칭군)
│       ├── binary_ensemble.py # Binary Ensemble 보정 로직
│       ├── helpers.py       # 공통 유틸리티
│       ├── mixup.py         # Mixup 유틸리티
│       └── device.py        # CUDA/MPS/CPU 자동 선택
│
├── scripts/
│   ├── train_folds.sh           # K-Fold 전체 학습 (5 fold 순차 실행)
│   ├── binary_ensemble.sh       # Binary Ensemble 파이프라인
│   ├── analyze_dataset.py       # 데이터셋 통계 분석
│   ├── analyze_test_dataset.py  # 테스트셋 분석
│   ├── analyze_results.py       # Confusion Matrix 시각화
│   ├── benchmark_models.py      # 모델 속도/메모리 벤치마크
│   ├── train_binary_classifier.py # Binary Classifier (class 3·7) 학습
│   ├── apply_binary_ensemble.py # Binary Ensemble 적용 (Grid Search)
│   └── visualize_augmentation.py # Augmentation 미리보기
│
├── docs/
│   ├── OPERATION.md         # ⭐ 운영 매뉴얼 (상세 가이드)
│   ├── research.md          # 연구 노트
│   └── modify.md            # 변경 이력
│
├── analysis/                # 분석 결과 (augmentation 전략 등)
├── datasets_fin/            # 데이터셋 (gitignore)
├── checkpoints/             # 모델 체크포인트 (gitignore)
│   └── champion/            # 최고 성능 모델 자동 저장
└── outputs/                 # Hydra 실행 로그
```

---

## 💡 주요 명령어

### 학습
```bash
# MaxViT-384 + K-Fold (베스트 조합)
python src/train.py model=maxvit_base_384 data=transformer_384 training=transformer \
  data.use_kfold=true data.fold_idx=0

# K-Fold 5개 fold 순차 실행 (셸 스크립트)
bash scripts/train_folds.sh

# ConvNeXt-Base
python src/train.py model=convnext_base data=baseline_aug training=default
```

### Inference
```bash
# 단일 모델 추론
python src/inference.py

# TTA 적용 추론 (standard: D4 8가지 대칭)
python src/inference.py inference.use_tta=true inference.tta_level=standard

# 특정 체크포인트 지정
python src/inference.py inference.checkpoint=checkpoints/champion/best_model.ckpt
```

### 앙상블
```bash
# 사전 정의된 앙상블 설정 사용
python src/ensemble.py ensemble=MaxViT_kfold_5models
python src/ensemble.py ensemble=ensemble_384_3models_0302

# TTA 포함 앙상블
python src/ensemble.py ensemble=MaxViT_kfold_5models ensemble.use_tta=true
```

### 분석 및 유틸리티
```bash
# Confusion Matrix 시각화
python scripts/analyze_results.py --checkpoint "checkpoints/champion/best_model.ckpt"

# Pseudo Label 생성 (confidence ≥ 0.9)
python scripts/pseudo_label.py

# Binary Classifier 학습 (class 3·7 혼동 보정)
bash scripts/binary_ensemble.sh

# Augmentation 미리보기
python scripts/visualize_augmentation.py

# 데이터셋 통계 분석
python scripts/analyze_dataset.py
```

---

## ⚙️ 고급 기능

### K-Fold Cross Validation
```bash
# 5-fold 중 fold 0 학습
python src/train.py data.use_kfold=true data.n_folds=5 data.fold_idx=0

# 전체 fold 학습 후 앙상블
bash scripts/train_folds.sh
python src/ensemble.py ensemble=MaxViT_kfold_5models
```

### TTA (Test Time Augmentation)
| 레벨 | 횟수 | 설명 |
|------|------|------|
| `light` | 5가지 | original, hflip, vflip, rot90, rot270 |
| `standard` | 8가지 | D4 대칭군 완전 집합 (기본값) |
| `heavy` | 11가지 | D4 + brightness_up, brightness_down, sharpen |


### Binary Ensemble (class 3·7 보정)
class 3(운전면허증)과 class 7(기타)의 혼동을 Binary Classifier로 보정:
```bash
# Binary Classifier 학습 (5-fold K-Fold)
python scripts/train_binary_classifier.py

# 보정 적용 (Grid Search로 최적 threshold α, θ 탐색)
python scripts/apply_binary_ensemble.py
```

---

## 📚 상세 문서

**[docs/OPERATION.md](docs/OPERATION.md)** - 운영 매뉴얼 (필독 ⭐)

- 학습/추론 파이프라인 전체 흐름
- 환경별 추천 조합 (CUDA 서버)
- 모델별 실행 명령어 및 하이퍼파라미터
- 앙상블 방법 (Soft/Hard Voting, Weighted)
- Binary Ensemble Phase 1/2 상세 설명
- 과적합 완화 전략 (TTA, Mixup/CutMix, K-Fold, Label Smoothing)
- 트러블슈팅

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **Framework** | PyTorch Lightning 2.4 |
| **설정 관리** | Hydra + OmegaConf |
| **실험 추적** | WanDB |
| **모델** | timm (MaxViT, Swin, DeiT, ConvNeXt 등) |
| **데이터 증강** | Albumentations |
| **Mixup/CutMix** | timm.data.Mixup |
| **메트릭** | torchmetrics |

---

<div align="center">

**[운영 매뉴얼](docs/OPERATION.md)** | **[개발 가이드](CLAUDE.md)**

Made with ❤️ using PyTorch Lightning

**프로젝트 목표**: F1-Macro 0.95 (퍼블릭 리더보드)
**Val F1 Champion**: 0.974 (MaxViT-Base-384, K-Fold, 20260301_run_007)
**퍼블릭 베스트**: 0.9441 (앙상블)

</div>
