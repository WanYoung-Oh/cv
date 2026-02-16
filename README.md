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
# 출력: submission.csv
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
├── train.csv          (1,570개, 레이블 있음)
├── test.csv           (3,141개, 레이블 더미 0, 리더보드 제출용)
├── meta.csv           (17개 클래스 정보)
├── train/             (훈련 이미지)
└── test/              (테스트 이미지)
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
├── README.md                 # 👈 이 문서
├── CLAUDE.md                 # 개발 가이드라인
├── requirements.txt          # 의존성
│
├── configs/                  # Hydra 설정
│   ├── data/                # baseline_aug, transformer_384
│   ├── model/               # resnet34, swin_base_384, deit_base_384 등
│   └── training/            # baseline_768
│
├── src/                     # 소스 코드
│   ├── train.py             # ⭐ 훈련
│   ├── inference.py         # 🔮 추론
│   ├── ensemble.py          # 🎲 앙상블
│   ├── data/                # DataModule
│   ├── models/              # LightningModule
│   └── utils/               # 유틸리티
│
├── scripts/
│   └── analyze_results.py   # 📊 결과 분석
│
├── docs/                    # 📚 문서
│   ├── PROJECT_GUIDE.md     # ⭐ 메인 가이드 (필독)
│   └── archive/             # PDCA 문서 아카이브
│
├── datasets_fin/            # 데이터셋
│
├── checkpoints/             # 체크포인트
│   └── champion/            # Best 모델 ⭐
│
└── analysis_results/        # 분석 결과
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
# Champion 모델
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt

# 특정 체크포인트
python src/inference.py checkpoint=checkpoints/20260215_run_002/best_model.ckpt
```

### 결과 분석
```bash
python scripts/analyze_results.py --checkpoint checkpoints/champion/best_model.ckpt
# 출력: analysis_results/confusion_matrix.png
```

---

## 🎯 핵심 성공 요인

1. **고해상도 입력 (768×768)** - 문서 세부 정보 보존
2. **Aspect Ratio 보존** - LongestMaxSize + PadIfNeeded 2단계 전략
3. **문서 특화 Augmentation** - CLAHE, Perspective, Sharpen 등 7종
4. **Class Weights** - 불균형 데이터 처리 성공

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

# Inference
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt

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
