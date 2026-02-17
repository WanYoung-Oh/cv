# 문서 이미지 분류 프로젝트 ✅

> PyTorch Lightning + Hydra + WanDB 기반 고성능 문서 이미지 분류 시스템

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-2.4-purple.svg)](https://lightning.ai/)
[![F1 Score](https://img.shields.io/badge/F1--Score-0.993-success.svg)](https://github.com)

---

## 🎉 프로젝트 완료

**목표 달성**: F1-Macro **0.993** (목표 0.88 대비 **+13% 초과 달성**)

| Metric | 목표 | 달성 | 상태 |
|--------|------|------|------|
| **F1-Macro** | 0.88+ | **0.993** | ✅ +13% |
| **Accuracy** | - | **0.994** | ✅ 우수 |
| **Val F1** | - | **0.993** | ✅ 안정적 |

**Best 모델**: ResNet34 + baseline_aug (768×768)
**체크포인트**: `checkpoints/champion/best_model.ckpt`

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

### 3. Best 모델로 Inference (리더보드 제출)
```bash
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt
# 출력: datasets_fin/submission/submission_{model_name}.csv
```

### 4. 새로운 모델 훈련 (선택사항)
```bash
# ResNet34 (Best 모델 재현)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768

# Transformer 모델
python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768
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
└── submission/             (inference 결과 자동 저장)
```

### 클래스 정보
- **17개 클래스**: 이력서, 여권, 운전면허증 등 문서 타입
- **불균형**: 상위 3개 클래스가 전체의 50%
- **해결**: Class Weights 적용

---

## 🤖 사용 가능한 모델

### CNN 모델 (768×768)
| 모델 | Config | 예상 성능 | 비고 |
|------|--------|-----------|------|
| **ResNet34** ✅ | resnet34 | **F1 0.993** | Best 모델 |
| ResNet50 | resnet50 | F1 0.96~0.98 | 안정적 |
| EfficientNet-B4 | efficientnet_b4 | F1 0.96~0.98 | batch_size=8 |
| ConvNeXt-Base | convnext_base | F1 0.96~0.98 | 최신 CNN |

### Transformer 모델 (384×384)
| 모델 | Config | 예상 성능 | 비고 |
|------|--------|-----------|------|
| Swin-Base-384 | swin_base_384 | F1 0.95~0.97 | Window 12 |
| DeiT-Base-384 | deit_base_384 | F1 0.94~0.96 | ViT 개선 |

**Data Config**:
- CNN: `data=baseline_aug` (768×768)
- Transformer: `data=transformer_384` (384×384)

---

## 📁 프로젝트 구조

```
CV/
├── README.md                      # 👈 이 문서
├── CLAUDE.md                      # 개발 가이드라인
├── requirements.txt               # 의존성
│
├── configs/                       # Hydra 설정
│   ├── config.yaml               # 메인 설정
│   ├── data/                     # Data Config
│   │   ├── baseline_aug.yaml     # ⭐ CNN용 (768×768)
│   │   └── transformer_384.yaml  # Transformer용 (384×384)
│   ├── model/                    # Model Config (6종)
│   │   ├── resnet34.yaml         # ⭐ Best 모델
│   │   ├── resnet50.yaml
│   │   ├── efficientnet_b4.yaml
│   │   ├── convnext_base.yaml
│   │   ├── swin_base_384.yaml
│   │   └── deit_base_384.yaml
│   └── training/                 # Training Config
│       └── baseline_768.yaml     # ⭐ Best 설정
│
├── src/                          # 소스 코드
│   ├── train.py                  # ⭐ 훈련 스크립트
│   ├── inference.py              # 🔮 추론 (리더보드 제출)
│   ├── ensemble.py               # 🎲 앙상블
│   ├── data/
│   │   └── datamodule.py        # DataModule
│   ├── models/
│   │   └── module.py            # LightningModule
│   └── utils/
│       ├── device.py            # 디바이스 자동 선택
│       └── helpers.py           # 유틸리티
│
├── scripts/
│   └── analyze_results.py        # 📊 결과 분석 (Confusion Matrix)
│
├── docs/                         # 📚 문서
│   ├── README.md                 # 프로젝트 개요
│   ├── PROJECT_GUIDE.md          # ⭐ 완료 가이드 (필독)
│   └── archive/2026-02/CV/      # PDCA 문서 아카이브
│       ├── design.md            # 설계 문서
│       ├── analysis.md          # Gap Analysis (95%)
│       └── report.md            # 완료 보고서
│
├── datasets_fin/                 # 데이터셋
│   ├── train.csv                # 학습 데이터 (레이블 있음)
│   ├── sample_submission.csv    # 제출 형식 (inference 입력/출력 기준)
│   ├── train/ (1,570장)
│   ├── test/ (3,140장)
│   └── submission/              # inference 결과 저장 (자동 생성)
│
├── checkpoints/                  # 모델 체크포인트
│   ├── YYYYMMDD_run_XXX/         # 실험별 디렉토리
│   │   ├── experiment_info.json # 실험 정보
│   │   └── epoch=XX-val_f1=X.XXX.ckpt
│   └── champion/                 # ⭐ Best 모델 (F1 0.993)
│       ├── best_model.ckpt
│       └── champion_info.json
│
├── outputs/                      # Hydra 실행 로그 (single run)
│   └── YYYY-MM-DD/HH-MM-SS/
│       └── .hydra/              # Config 스냅샷
│
└── analysis_results/             # 분석 결과
    └── confusion_matrix.png
```

---

## 💡 사용 예시

### 훈련

#### ResNet34 (Best 모델 재현)
```bash
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768
```

#### Swin-Base-384 (Transformer)
```bash
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768
```

#### EfficientNet-B4 (메모리 부족 시)
```bash
python src/train.py \
  data=baseline_aug \
  model=efficientnet_b4 \
  training=baseline_768 \
  training.batch_size=8
```

### Inference
```bash
# Champion 모델 (기본)
python src/inference.py
# 출력: datasets_fin/submission/submission_{model_name}.csv

# 특정 run_id 사용
python src/inference.py inference.run_id=20260216_run_001

# 특정 체크포인트 직접 지정
python src/inference.py inference.checkpoint=checkpoints/20260215_run_002/epoch=10-val_f1=0.993.ckpt
```

### 결과 분석
```bash
python scripts/analyze_results.py --checkpoint checkpoints/champion/best_model.ckpt
# 출력: analysis_results/confusion_matrix.png
```

### Hydra Multi-Run (하이퍼파라미터 스윕)
```bash
# 여러 모델 동시 실험
python src/train.py --multirun \
  model=resnet34,resnet50 \
  data=baseline_aug,transformer_384

# 결과: multirun/YYYY-MM-DD/HH-MM-SS/{0,1,2,3}/
# 각 실험마다 .hydra/config.yaml에 설정 저장 (재현성)
```

---

## 🎯 핵심 성공 요인

1. **고해상도 입력 (768×768)** - 문서 세부 정보 보존
2. **Aspect Ratio 보존** - LongestMaxSize + PadIfNeeded 2단계 전략
3. **문서 특화 Augmentation** - CLAHE, Perspective, Sharpen 등 7종
4. **Class Weights** - 불균형 데이터 처리 성공

---

## 💡 주요 기능

### 1. Aspect Ratio 보존 전략 ⭐

직사각형 이미지를 왜곡 없이 처리하는 핵심 기술:

```yaml
# configs/data/baseline_aug.yaml
train_augmentations:
  # 1단계: 긴 쪽을 768로 맞춤 (비율 유지)
  - type: LongestMaxSize
    max_size: 768

  # 2단계: 부족한 부분 흰색 패딩
  - type: PadIfNeeded
    min_height: 768
    min_width: 768
    value: [255, 255, 255]  # 문서 배경색
```

**효과**: 정보 손실 최소화 → F1 0.993 달성의 핵심 요인

### 2. Hydra Multi-Run 지원

하이퍼파라미터 스윕 자동화:

```bash
# 여러 설정 동시 실험
python src/train.py --multirun \
  model=resnet34,resnet50,swin_base_384 \
  data.img_size=384,768

# 결과 자동 저장
# multirun/YYYY-MM-DD/HH-MM-SS/
#   ├── 0/  (resnet34 + 384)
#   ├── 1/  (resnet34 + 768)
#   ├── 2/  (resnet50 + 384)
#   └── 3/  (resnet50 + 768)
```

각 실험의 config가 `.hydra/`에 자동 저장되어 **완벽한 재현성** 보장

### 3. Champion 모델 자동 추적

최고 성능 모델을 자동으로 관리:

```json
// checkpoints/champion/champion_info.json
{
  "val_f1": 0.993,
  "checkpoint_path": "checkpoints/20260215_run_002/...",
  "updated_at": "2026-02-15T18:00:00",
  "model_name": "resnet34"
}
```

새로운 모델이 더 좋은 성능을 내면 자동으로 `champion/` 업데이트

### 4. 실험 자동 관리

날짜 + run_id 시스템으로 체계적 관리:

```
checkpoints/
├── 20260215_run_001/  (ResNet34, F1 0.993)
├── 20260215_run_002/  (ResNet50, F1 0.975)
└── champion/          → run_001 (자동 링크)
```

---

## 📚 상세 문서

- **[PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)** - 프로젝트 완료 가이드 ⭐
  - 달성 성과
  - 사용 가능한 모델
  - 훈련/Inference 방법
  - Config 구조
  - 핵심 성공 요인
  - 다음 단계 (선택사항)

- **[CLAUDE.md](CLAUDE.md)** - 개발 가이드라인
  - Package Management
  - Coding Conventions
  - 프로젝트 구조
  - Hydra/WanDB 사용법

---

## 🔧 트러블슈팅

### 메모리 부족
```bash
# Batch size 감소
python src/train.py training.batch_size=8

# 작은 모델 사용
python src/train.py model=resnet34
```

### WanDB 로그인
```bash
wandb login
# 또는
echo "WANDB_MODE=disabled" > .env
```

### Config 오버라이드
```bash
# CLI에서 모든 설정 변경 가능
python src/train.py \
  training.learning_rate=5e-4 \
  training.epochs=30 \
  training.batch_size=8
```

### Hydra 출력 디렉토리 정리
```bash
# outputs/, multirun/ 디렉토리는 Hydra가 자동 생성
# 필요 없으면 삭제 가능 (.gitignore에 포함됨)
rm -rf outputs/ multirun/
```

---

## 🛠️ 기술 스택

| 분류 | 기술 | 버전 |
|------|------|------|
| **Framework** | PyTorch Lightning | 2.4+ |
| **설정 관리** | Hydra | 1.3+ |
| **실험 추적** | WanDB | 0.18+ |
| **모델** | timm | 1.0+ |
| **데이터 증강** | Albumentations | 1.4+ |
| **메트릭** | torchmetrics | 1.4+ |

---

## 📈 다음 단계 (선택사항)

### 옵션 A: 리더보드 제출 (추천)
```bash
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt
# submission.csv 파일 제출
```

### 옵션 B: TTA + Ensemble
- 목표: F1 0.995+
- ROI: 낮음 (+0.2~0.4%)

### 옵션 C: Transformer 모델 실험
- Swin-384, DeiT-384
- ROI: 매우 낮음

자세한 내용은 **[PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**를 참고하세요.

---

## 📞 도움말

### 문서
1. **[PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)** - 메인 가이드 ⭐
2. **[CLAUDE.md](CLAUDE.md)** - 개발 가이드라인

### 참고 자료
- [PyTorch Lightning 문서](https://lightning.ai/docs/pytorch/stable/)
- [Hydra 문서](https://hydra.cc/docs/intro/)
- [WanDB 문서](https://docs.wandb.ai/)
- [timm 문서](https://huggingface.co/docs/timm)

---

## 📝 주요 명령어

```bash
# 훈련 (Best 모델)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768

# Inference (출력: datasets_fin/submission/submission_{model_name}.csv)
python src/inference.py

# 결과 분석
python scripts/analyze_results.py --checkpoint checkpoints/champion/best_model.ckpt
```

---

<div align="center">

**[프로젝트 가이드](docs/PROJECT_GUIDE.md)** | **[개발 가이드](CLAUDE.md)**

Made with ❤️ using PyTorch Lightning

**프로젝트 완료일**: 2026-02-15
**최종 성과**: F1 0.993 / Accuracy 0.994

</div>
