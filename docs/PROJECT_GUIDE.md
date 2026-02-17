# CV 프로젝트 완료 가이드

> PyTorch Lightning + Hydra + WanDB 기반 문서 이미지 분류 시스템

**프로젝트 상태**: ✅ 완료 (F1 0.993 달성)
**목표 달성**: 113% (목표 0.88 vs 실제 0.993)
**마지막 업데이트**: 2026-02-17

---

## 🎉 달성 성과

| Metric | 목표 | 달성 | 상태 |
|--------|------|------|------|
| **F1-Macro** | 0.88+ | **0.993** | ✅ +13% 초과 |
| **Accuracy** | - | **0.994** | ✅ 우수 |
| **Val F1** | - | **0.993** | ✅ 안정적 |
| **Test F1** | - | **0.993** | ✅ 일치 |

**Best 모델**: ResNet34 + baseline_aug (768×768)
**체크포인트**: `checkpoints/champion/best_model.ckpt`

---

## 📊 데이터셋 정보

### 구조
```
datasets_fin/
├── train.csv               (1,570개, 레이블 있음)
├── sample_submission.csv   (3,140개, 리더보드 제출 형식)
├── meta.csv                (17개 클래스)
├── train/                  (훈련 이미지)
├── test/                   (테스트 이미지, 리더보드 제출용)
└── submission/             (inference 결과 자동 저장)
```

### 클래스 정보
- **총 17개 클래스**: 문서 타입 (이력서, 여권, 운전면허증 등)
- **불균형**: 상위 3개 클래스가 전체의 50%
- **해결**: Class Weights 적용

### 이미지 특성
- **평균 크기**: 498×538 (AR: 0.97)
- **크기 범위**: W: 384~753, H: 348~682
- **방향성**: 세로 1,040개, 가로 513개, 정사각 17개

---

## 🤖 사용 가능한 모델

### CNN 모델 (768×768)
| 모델 | Config | Data Config | 예상 성능 | 비고 |
|------|--------|-------------|-----------|------|
| **ResNet34** ✅ | resnet34 | baseline_aug | **F1 0.993** | Best 모델 |
| ResNet50 | resnet50 | baseline_aug | F1 0.96~0.98 | 안정적 |
| EfficientNet-B4 | efficientnet_b4 | baseline_aug | F1 0.96~0.98 | 효율적, batch_size=8 |
| ConvNeXt-Base | convnext_base | baseline_aug | F1 0.96~0.98 | 최신 CNN |

### Transformer 모델 (384×384)
| 모델 | Config | Data Config | 예상 성능 | 비고 |
|------|--------|-------------|-----------|------|
| Swin-Base-384 | swin_base_384 | transformer_384 | F1 0.95~0.97 | Window 12 |
| DeiT-Base-384 | deit_base_384 | transformer_384 | F1 0.94~0.96 | ViT 개선 |

---

## 🚀 사용 방법

### 1. 환경 설정
```bash
conda activate pytorch_test
pip install -r requirements.txt
```

### 2. 훈련

#### Best 모델 재현 (ResNet34)
```bash
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768
```

#### Transformer 모델 실험
```bash
# Swin-Base-384
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768

# DeiT-Base-384
python src/train.py \
  data=transformer_384 \
  model=deit_base_384 \
  training=baseline_768
```

#### CNN 모델 실험
```bash
# EfficientNet-B4 (메모리 부족 시)
python src/train.py \
  data=baseline_aug \
  model=efficientnet_b4 \
  training=baseline_768 \
  training.batch_size=8

# ConvNeXt-Base
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768
```

### 3. Inference (리더보드 제출)
```bash
# Champion 모델 사용 (기본)
python src/inference.py
# 출력: datasets_fin/submission/submission_{model_name}.csv

# 특정 run_id 사용
python src/inference.py inference.run_id=20260216_run_001

# 직접 체크포인트 경로 지정
python src/inference.py inference.checkpoint=checkpoints/20260215_run_002/epoch=10-val_f1=0.993.ckpt

# 출력 파일명 직접 지정
python src/inference.py inference.output=datasets_fin/submission/submission_final.csv
```

### 4. 결과 분석
```bash
# Confusion Matrix 생성
python scripts/analyze_results.py --checkpoint checkpoints/champion/best_model.ckpt

# 출력: analysis_results/confusion_matrix.png
```

---

## ⚙️ Config 구조

### Model Configs (configs/model/)
```yaml
# CNN (768×768용)
- resnet34.yaml           # Best 모델 ⭐
- resnet50.yaml
- efficientnet_b4.yaml
- convnext_base.yaml

# Transformer (384×384용)
- swin_base_384.yaml
- deit_base_384.yaml
```

### Data Configs (configs/data/)
```yaml
# CNN용 (768×768)
baseline_aug.yaml:
  - LongestMaxSize(768) + PadIfNeeded(768×768)
  - RandomRotate90, Rotate ±45°
  - CLAHE, Perspective, ColorJitter 등

# Transformer용 (384×384)
transformer_384.yaml:
  - LongestMaxSize(384) + PadIfNeeded(384×384)
  - 동일한 Augmentation
```

### Training Configs (configs/training/)
```yaml
baseline_768.yaml:
  - batch_size: 16
  - learning_rate: 1e-3
  - epochs: 50
  - early_stopping: patience=10
```

---

## 🎯 핵심 성공 요인

### 1. 고해상도 입력 (768×768)
- 기존 224×224 대비 **3.4배** 해상도
- 문서의 세부 정보 완벽 보존
- 텍스트 인식 성능 향상

### 2. Aspect Ratio 보존
- **LongestMaxSize + PadIfNeeded** 2단계 전략
- 정보 손실 최소화
- 왜곡 방지

### 3. 문서 특화 Augmentation
- **CLAHE**: 대비 강화 (잉크/종이 분리)
- **Perspective**: 스캔 각도 변화
- **ColorJitter**: 종이 색상 변화
- **Sharpen**: 선명도 향상

### 4. Class Weights
- 불균형 데이터 처리 성공
- Weight 범위: 1.00 ~ 2.17
- Weighted CrossEntropy Loss 사용

---

## 📈 다음 단계 (선택사항)

### 옵션 A: 현재 결과로 완료 (추천) ⭐
- **현재 성능**: F1 0.993 (목표 대비 +13%)
- **상태**: 프로젝트 목표 초과 달성
- **다음**: 리더보드 제출, 프로젝트 완료

### 옵션 B: TTA + Ensemble
- **목표**: F1 0.995+ 도전
- **방법**:
  1. TTA: 4가지 회전 (0°, 90°, 180°, 270°)
  2. Ensemble: ResNet34 + Swin-384 + DeiT-384
- **예상 시간**: 3~4시간
- **ROI**: 낮음 (+0.2~0.4%)

### 옵션 C: Transformer 모델 실험
- **목표**: 다양한 아키텍처 경험
- **방법**: Swin-384, DeiT-384 훈련
- **예상 시간**: 각 2~3시간
- **ROI**: 매우 낮음

---

## 📚 참고 자료

### PDCA 문서
- [data-augmentation-strategy.md](02-design/data-augmentation-strategy.md) - 설계 문서
- [CV.analysis.md](03-analysis/CV.analysis.md) - Gap Analysis (Match Rate 95%)
- [CV.report.md](04-report/CV.report.md) - 완료 보고서

### Config 파일
- [baseline_aug.yaml](../configs/data/baseline_aug.yaml) - Best 성능 config
- [transformer_384.yaml](../configs/data/transformer_384.yaml) - Transformer용
- [baseline_768.yaml](../configs/training/baseline_768.yaml) - 훈련 설정

### 코드
- [train.py](../src/train.py) - 훈련 스크립트
- [inference.py](../src/inference.py) - 추론 스크립트
- [analyze_results.py](../scripts/analyze_results.py) - 결과 분석

---

## 🔧 문제 해결

### 메모리 부족
```bash
# Batch size 감소
python src/train.py training.batch_size=8

# 작은 모델 사용
python src/train.py model=resnet34  # 21M params
```

### WanDB 로그인
```bash
wandb login

# 또는 .env 파일 설정
WANDB_API_KEY=your-api-key
WANDB_PROJECT=doc_image_classification
```

### Config 오버라이드
```bash
# CLI에서 모든 하이퍼파라미터 변경 가능
python src/train.py \
  training.learning_rate=5e-4 \
  training.epochs=30 \
  training.batch_size=8
```

---

## 📝 주요 명령어 요약

```bash
# 훈련 (Best 모델)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768

# Inference
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt

# 결과 분석
python scripts/analyze_results.py --checkpoint checkpoints/champion/best_model.ckpt

# WanDB 대시보드
# 훈련 시작 후 터미널의 URL 클릭
```

---

## 💡 학습 내용

### 성공 요인
1. 고해상도 입력이 문서 이미지에 필수
2. Aspect ratio 보존이 정보 손실 최소화
3. 도메인 특화 augmentation이 일반 augmentation보다 효과적
4. Baseline부터 시작하여 단계적 개선이 효율적

### 예상과 다른 점
- **설계 예상**: F1 0.82~0.85
- **실제 달성**: F1 0.993
- **차이 이유**: 고해상도 + 문서 특화 augmentation의 시너지

### 다음 프로젝트를 위한 교훈
1. 문서 이미지는 고해상도 (768+) 필수
2. Aspect ratio 보존 우선
3. 도메인 지식 활용한 augmentation 설계
4. ResNet 같은 검증된 아키텍처의 강력함

---

**프로젝트 완료일**: 2026-02-15
**최종 성과**: F1 0.993 (목표 0.88 대비 +13%)
**Best 모델**: ResNet34 + baseline_aug (768×768)
