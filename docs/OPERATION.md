# 훈련 작업 가이드 (Operation Manual)

> 모든 모델/데이터/학습 조합의 실행 스크립트 및 설명

**최종 업데이트**: 2026-02-16
**프로젝트**: 문서 이미지 분류 (F1 0.993 달성)

---

## 🖥️ 사용 가능한 환경

### 환경 1: CUDA x86 Server
- **GPU**: CUDA 지원
- **RAM**: 128 GB
- **상태**: ✅ 모든 모델 조합 가능
- **특징**: 대용량 메모리, 고성능 훈련

### 환경 2: Mac mini M4 Pro
- **GPU**: Apple MPS (Metal)
- **RAM**: 24 GB
- **상태**: ✅ 대부분 모델 가능 (일부 제약)
- **특징**: 중간 용량, 안정적 훈련

---

## 📋 목차

1. [빠른 참조](#빠른-참조)
2. [환경별 추천 조합](#환경별-추천-조합)
3. [CUDA 서버 전체 조합](#cuda-서버-전체-조합)
4. [Mac mini M4 Pro 조합](#mac-mini-m4-pro-조합)
5. [모델별 상세 가이드](#모델별-상세-가이드)
6. [성능 비교 매트릭스](#성능-비교-매트릭스)
7. [트러블슈팅](#트러블슈팅)

---

## 🚀 빠른 참조

### Best 모델 (검증됨)

```bash
# ResNet34 + baseline_aug + baseline_768
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768

# 성과: F1 0.993, Accuracy 0.994
# 메모리: ~8 GB (Apple MPS 안전)
# 시간: 2~3시간
```

### 사용 가능한 설정

| 카테고리 | 옵션 | 설명 |
|----------|------|------|
| **모델** | resnet34, resnet50, efficientnet_b4, convnext_base, swin_base_384, deit_base_384 | 6종 |
| **데이터** | baseline_aug (768×768), transformer_384 (384×384) | 2종 |
| **학습** | baseline_768, default, efficientnet, transformer | 4종 |

---

## ⭐ 환경별 추천 조합

### 🖥️ CUDA 서버 (128GB RAM) - 추천 전략

#### 전략 1: 최고 성능 추구 (병렬 실험)

```bash
# Hydra Multi-Run으로 여러 모델 동시 훈련
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base \
  data=baseline_aug \
  training=baseline_768

# 결과: multirun/YYYY-MM-DD/HH-MM-SS/{0,1,2,3}/
```

**장점**:
- ✅ 128GB RAM으로 동시 훈련 가능
- ✅ 빠른 실험 반복
- ✅ 최적 모델 자동 선정

#### 전략 2: 대용량 Batch Size (최고 성능)

```bash
# ConvNeXt-Base + 큰 batch size
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768 \
  training.batch_size=32

# 또는 더 크게
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768 \
  training.batch_size=64
```

**장점**:
- ✅ 안정적인 gradient 업데이트
- ✅ 더 나은 수렴 성능
- ✅ 최신 CNN 아키텍처 활용

#### 전략 3: 모든 모델 벤치마크

```bash
# CNN 모델 전체
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base \
  data=baseline_aug

# Transformer 모델 전체
python src/train.py --multirun \
  model=swin_base_384,deit_base_384 \
  data=transformer_384

# 모든 조합
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base,swin_base_384,deit_base_384 \
  data=baseline_aug,transformer_384
```

---

### 💻 Mac mini M4 Pro (24GB) - 추천 전략

#### 전략 1: Best 모델 재현

```bash
# ResNet34 (검증됨: F1 0.993)
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768
```

**장점**:
- ✅ 메모리 안전 (~8 GB)
- ✅ 검증된 성능
- ✅ 빠른 훈련

#### 전략 2: 중형 모델 실험

```bash
# ResNet50
python src/train.py \
  data=baseline_aug \
  model=resnet50 \
  training=baseline_768

# EfficientNet-B4 (24GB에서 가능)
python src/train.py \
  data=baseline_aug \
  model=efficientnet_b4 \
  training=baseline_768 \
  training.batch_size=8
```

**장점**:
- ✅ 24GB로 충분
- ✅ 다양한 아키텍처 경험
- ✅ 안정적 훈련

#### 전략 3: Transformer 실험

```bash
# Swin-Base-384 (안정적)
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768 \
  training.batch_size=16

# DeiT-Base-384
python src/train.py \
  data=transformer_384 \
  model=deit_base_384 \
  training=baseline_768 \
  training.batch_size=16
```

**장점**:
- ✅ 24GB로 안전
- ✅ Transformer 경험
- ✅ 다양한 실험 가능

---

## 🖥️ CUDA 서버 전체 조합

### CNN 모델 (768×768) - 전체 가능 ✅

| 모델 | 데이터 | Batch Size | 명령어 | 예상 성능 |
|------|--------|------------|--------|-----------|
| **ResNet34** | baseline_aug | 32 | `python src/train.py data=baseline_aug model=resnet34 training=baseline_768 training.batch_size=32` | F1 0.99+ |
| **ResNet50** | baseline_aug | 32 | `python src/train.py data=baseline_aug model=resnet50 training=baseline_768 training.batch_size=32` | F1 0.96~0.98 |
| **EfficientNet-B4** | baseline_aug | 32 | `python src/train.py data=baseline_aug model=efficientnet_b4 training=baseline_768 training.batch_size=32` | F1 0.96~0.98 |
| **ConvNeXt-Base** | baseline_aug | 32 | `python src/train.py data=baseline_aug model=convnext_base training=baseline_768 training.batch_size=32` | F1 0.96~0.98 |
| **ConvNeXt-Base** | baseline_aug | 64 | `python src/train.py data=baseline_aug model=convnext_base training=baseline_768 training.batch_size=64` | F1 0.97~0.99 |

### Transformer 모델 (384×384) - 전체 가능 ✅

| 모델 | 데이터 | Batch Size | 명령어 | 예상 성능 |
|------|--------|------------|--------|-----------|
| **Swin-Base-384** | transformer_384 | 32 | `python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768 training.batch_size=32` | F1 0.95~0.97 |
| **DeiT-Base-384** | transformer_384 | 32 | `python src/train.py data=transformer_384 model=deit_base_384 training=baseline_768 training.batch_size=32` | F1 0.94~0.96 |

### Multi-Run 조합 (병렬 실험) ✅

```bash
# 전체 CNN 모델 비교
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base \
  data=baseline_aug \
  training.batch_size=32

# 전체 Transformer 모델 비교
python src/train.py --multirun \
  model=swin_base_384,deit_base_384 \
  data=transformer_384 \
  training.batch_size=32

# 모든 모델 벤치마크 (6개 모델 × 2개 데이터 = 12개 실험)
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base,swin_base_384,deit_base_384 \
  data=baseline_aug,transformer_384
```

**CUDA 서버 장점**:
- ✅ 모든 모델 조합 가능
- ✅ 큰 batch size (32~64) 사용 가능
- ✅ Multi-Run으로 동시 실험 가능
- ✅ 메모리 제약 없음

---

## 💻 Mac mini M4 Pro 조합

### CNN 모델 (768×768)

| 모델 | 데이터 | Batch Size | 메모리 | 상태 | 명령어 |
|------|--------|------------|--------|------|--------|
| **ResNet34** ⭐ | baseline_aug | 16 | ~8 GB | ✅ 안전 | `python src/train.py data=baseline_aug model=resnet34 training=baseline_768` |
| **ResNet50** | baseline_aug | 16 | ~10 GB | ✅ 안전 | `python src/train.py data=baseline_aug model=resnet50 training=baseline_768` |
| **EfficientNet-B4** | baseline_aug | 8 | ~19 GB | ✅ 가능 | `python src/train.py data=baseline_aug model=efficientnet_b4 training=baseline_768 training.batch_size=8` |
| **EfficientNet-B4** | baseline_aug | 12 | ~22 GB | ⚠️ 경계 | `python src/train.py data=baseline_aug model=efficientnet_b4 training=baseline_768 training.batch_size=12` |
| **ConvNeXt-Base** | baseline_aug | 4 | ~20 GB | ⚠️ 위험 | `python src/train.py data=baseline_aug model=convnext_base training=baseline_768 training.batch_size=4 data.img_size=512` |

### Transformer 모델 (384×384)

| 모델 | 데이터 | Batch Size | 메모리 | 상태 | 명령어 |
|------|--------|------------|--------|------|--------|
| **Swin-Base-384** | transformer_384 | 16 | ~12 GB | ✅ 안전 | `python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768` |
| **Swin-Base-384** | transformer_384 | 20 | ~15 GB | ✅ 가능 | `python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768 training.batch_size=20` |
| **DeiT-Base-384** | transformer_384 | 16 | ~12 GB | ✅ 안전 | `python src/train.py data=transformer_384 model=deit_base_384 training=baseline_768` |
| **DeiT-Base-384** | transformer_384 | 20 | ~15 GB | ✅ 가능 | `python src/train.py data=transformer_384 model=deit_base_384 training=baseline_768 training.batch_size=20` |

### Multi-Run 조합 (순차 실험)

```bash
# 안전한 CNN 모델 비교 (ResNet 계열)
python src/train.py --multirun \
  model=resnet34,resnet50 \
  data=baseline_aug

# Transformer 모델 비교
python src/train.py --multirun \
  model=swin_base_384,deit_base_384 \
  data=transformer_384

# 안전한 모든 모델 (EfficientNet 제외)
python src/train.py --multirun \
  model=resnet34,resnet50,swin_base_384,deit_base_384 \
  data=baseline_aug,transformer_384
```

**Mac mini M4 Pro 장점**:
- ✅ 대부분 모델 가능 (24GB)
- ✅ EfficientNet-B4 가능 (batch_size 조정)
- ✅ Transformer 모델 안정적
- ⚠️ ConvNeXt-Base는 여전히 제한적

---

## 📊 환경별 모델 호환성 매트릭스

### 전체 비교 테이블

| 모델 | Parameters | 768×768 메모리 | CUDA 서버 | Mac M4 Pro 24GB | 예상 F1 |
|------|------------|----------------|-----------|-----------------|---------|
| **ResNet34** | 21M | ~8 GB | ✅ (bs=32) | ✅ (bs=16) | **0.993** |
| **ResNet50** | 25M | ~10 GB | ✅ (bs=32) | ✅ (bs=16) | 0.96~0.98 |
| **EfficientNet-B4** | 17.6M | ~19 GB | ✅ (bs=32) | ✅ (bs=8-12) | 0.96~0.98 |
| **ConvNeXt-Base** | 88M | ~25 GB+ | ✅ (bs=32-64) | ⚠️ (bs=2-4, 512px) | 0.96~0.98 |
| **Swin-Base-384** | 88M | ~12 GB (384px) | ✅ (bs=32) | ✅ (bs=16-20) | 0.95~0.97 |
| **DeiT-Base-384** | 86M | ~12 GB (384px) | ✅ (bs=32) | ✅ (bs=16-20) | 0.94~0.96 |

**범례**:
- ✅ 안전 사용 가능
- ⚠️ 주의 필요 (작은 batch size 또는 이미지 크기)
- ❌ 사용 불가능
- bs = batch size

---

## 🎯 환경별 권장 워크플로우

### CUDA 서버 워크플로우

#### Phase 1: 빠른 벤치마크 (병렬)

```bash
# 모든 모델 동시 실험 (Multi-Run)
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base \
  data=baseline_aug \
  training.batch_size=32

# 예상 시간: 각 모델 2~3시간 (병렬 실행)
```

#### Phase 2: Top 모델 재훈련 (큰 batch size)

```bash
# 최고 성능 모델을 더 큰 batch size로
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768 \
  training.batch_size=64
```

#### Phase 3: Ensemble

```bash
# 여러 모델의 결과를 앙상블
python src/ensemble.py \
  --checkpoints \
    multirun/.../0/best.ckpt \
    multirun/.../1/best.ckpt \
    multirun/.../2/best.ckpt \
  --method soft_voting
```

---

### Mac mini M4 Pro 워크플로우

#### Phase 1: Best 모델 재현

```bash
# ResNet34 (검증됨)
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768

# 예상 시간: 2~3시간
```

#### Phase 2: 추가 실험 (순차)

```bash
# ResNet50
python src/train.py \
  data=baseline_aug \
  model=resnet50 \
  training=baseline_768

# Swin-Base-384
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768
```

#### Phase 3: Inference

```bash
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt
```

---

## 🚀 실전 추천 조합

### 1. 최고 성능 추구 (CUDA 서버)

```bash
# ConvNeXt-Base + 큰 batch size
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768 \
  training.batch_size=64
```

### 2. 빠른 프로토타입 (Mac M4 Pro)

```bash
# ResNet34 (검증됨)
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768
```

### 3. 전체 벤치마크 (CUDA 서버)

```bash
# 모든 모델 동시 실험
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base,swin_base_384,deit_base_384 \
  data=baseline_aug,transformer_384 \
  training.batch_size=32
```

### 4. Transformer 비교 (Mac M4 Pro)

```bash
# Swin vs DeiT
python src/train.py --multirun \
  model=swin_base_384,deit_base_384 \
  data=transformer_384
```

---

## ⭐ 기존 추천 조합 (범용)

### 1. ResNet34 + baseline_aug (Best) 🏆

```bash
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768
```

**특징**:
- ✅ **검증된 성능**: F1 0.993
- ✅ **메모리 효율**: ~8 GB
- ✅ **고해상도**: 768×768
- ✅ **Apple MPS 안전**

**예상 결과**:
- F1-Macro: 0.99+
- Accuracy: 0.99+
- 훈련 시간: 2~3시간

---

### 2. ResNet50 + baseline_aug (안정적)

```bash
python src/train.py \
  data=baseline_aug \
  model=resnet50 \
  training=baseline_768
```

**특징**:
- ✅ **안정적 성능**: F1 0.96~0.98 예상
- ✅ **메모리**: ~10 GB
- ✅ **고해상도**: 768×768
- ✅ **Apple MPS 안전**

**사용 사례**: ResNet34보다 약간 더 큰 용량이 필요할 때

---

### 3. Swin-Base-384 (Transformer)

```bash
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768
```

**특징**:
- ✅ **Transformer 아키텍처**
- ✅ **Window Attention**: Window 12
- ⚠️ **메모리**: ~12 GB
- ⚠️ **성능**: F1 0.95~0.97 예상

**사용 사례**: Transformer 실험, 다양한 아키텍처 비교

---

### 4. DeiT-Base-384 (Transformer)

```bash
python src/train.py \
  data=transformer_384 \
  model=deit_base_384 \
  training=baseline_768
```

**특징**:
- ✅ **ViT 개선 버전**
- ✅ **Distillation Training**
- ⚠️ **메모리**: ~12 GB
- ⚠️ **성능**: F1 0.94~0.96 예상

**사용 사례**: ViT 실험, Knowledge Distillation 연구

---

## 📊 모델별 상세 가이드

### CNN 모델 (768×768 권장)

#### ResNet34 (Best Model) ⭐⭐⭐

```bash
# 기본 설정 (추천)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768

# 다른 데이터 설정 (384×384)
python src/train.py data=transformer_384 model=resnet34 training=baseline_768

# 커스텀 batch size
python src/train.py data=baseline_aug model=resnet34 training=baseline_768 training.batch_size=32
```

**사양**:
- Parameters: 21M
- 메모리 (768×768, batch=16): ~8 GB
- Apple MPS: ✅ 안전

**예상 성능**:
- F1-Macro: 0.99+
- Accuracy: 0.99+

---

#### ResNet50 ⭐⭐

```bash
# 기본 설정
python src/train.py data=baseline_aug model=resnet50 training=baseline_768

# 메모리 절약
python src/train.py data=baseline_aug model=resnet50 training=baseline_768 training.batch_size=8
```

**사양**:
- Parameters: 25M
- 메모리 (768×768, batch=16): ~10 GB
- Apple MPS: ✅ 안전

**예상 성능**:
- F1-Macro: 0.96~0.98
- Accuracy: 0.97~0.99

---

#### EfficientNet-B4 ⚠️

```bash
# 작은 batch size (필수)
python src/train.py \
  data=baseline_aug \
  model=efficientnet_b4 \
  training=baseline_768 \
  training.batch_size=4

# 더 작은 이미지
python src/train.py \
  data=baseline_aug \
  model=efficientnet_b4 \
  training=baseline_768 \
  training.batch_size=2 \
  data.img_size=384
```

**사양**:
- Parameters: 17.6M
- 메모리 (768×768, batch=8): ~19 GB
- Apple MPS: ❌ **OOM 위험 높음**

**경고**:
- ⚠️ Apple MPS (20GB)에서 OOM 발생 가능성 높음
- ⚠️ batch_size=2, img_size=384로도 OOM 가능
- 🚫 **Apple MPS에서는 사용 비추천**

**예상 성능** (메모리 허용 시):
- F1-Macro: 0.96~0.98
- Accuracy: 0.97~0.99

---

#### ConvNeXt-Base ❌

```bash
# 극도로 작은 설정 (성공 확률 낮음)
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768 \
  training.batch_size=2 \
  data.img_size=384
```

**사양**:
- Parameters: 88M
- 메모리 (768×768, batch=16): ~25 GB+
- Apple MPS: ❌ **사용 불가능**

**경고**:
- 🚫 **Apple MPS (20GB)에서 사용 불가능**
- 🚫 작은 설정으로도 OOM 발생 가능성 99%
- ✅ CUDA GPU (24GB+)에서만 사용 권장

**예상 성능** (CUDA GPU):
- F1-Macro: 0.96~0.98
- Accuracy: 0.97~0.99

---

### Transformer 모델 (384×384 권장)

#### Swin-Base-384 ⭐

```bash
# 기본 설정 (추천)
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768

# 메모리 절약
python src/train.py \
  data=transformer_384 \
  model=swin_base_384 \
  training=baseline_768 \
  training.batch_size=8
```

**사양**:
- Parameters: ~88M
- 메모리 (384×384, batch=16): ~12 GB
- Apple MPS: ⚠️ **주의 필요**

**특징**:
- Window-based Self-Attention (Window 12)
- Hierarchical architecture
- 384×384 입력 최적화

**예상 성능**:
- F1-Macro: 0.95~0.97
- Accuracy: 0.96~0.98

---

#### DeiT-Base-384 ⭐

```bash
# 기본 설정 (추천)
python src/train.py \
  data=transformer_384 \
  model=deit_base_384 \
  training=baseline_768

# 메모리 절약
python src/train.py \
  data=transformer_384 \
  model=deit_base_384 \
  training=baseline_768 \
  training.batch_size=8
```

**사양**:
- Parameters: ~86M
- 메모리 (384×384, batch=16): ~12 GB
- Apple MPS: ⚠️ **주의 필요**

**특징**:
- Data-efficient Image Transformer
- Knowledge Distillation
- 384×384 입력 최적화

**예상 성능**:
- F1-Macro: 0.94~0.96
- Accuracy: 0.95~0.97

---

## 📋 전체 조합 매트릭스

### CNN 모델 조합

| 모델 | 데이터 | 학습 | 명령어 | 메모리 | Apple MPS |
|------|--------|------|--------|--------|-----------|
| **ResNet34** | baseline_aug | baseline_768 | `python src/train.py data=baseline_aug model=resnet34 training=baseline_768` | ~8 GB | ✅ |
| **ResNet34** | transformer_384 | baseline_768 | `python src/train.py data=transformer_384 model=resnet34 training=baseline_768` | ~6 GB | ✅ |
| **ResNet50** | baseline_aug | baseline_768 | `python src/train.py data=baseline_aug model=resnet50 training=baseline_768` | ~10 GB | ✅ |
| **ResNet50** | transformer_384 | baseline_768 | `python src/train.py data=transformer_384 model=resnet50 training=baseline_768` | ~8 GB | ✅ |
| EfficientNet-B4 | baseline_aug | baseline_768 | `python src/train.py data=baseline_aug model=efficientnet_b4 training=baseline_768 training.batch_size=4` | ~19 GB | ❌ |
| EfficientNet-B4 | transformer_384 | baseline_768 | `python src/train.py data=transformer_384 model=efficientnet_b4 training=baseline_768 training.batch_size=8` | ~15 GB | ⚠️ |
| ConvNeXt-Base | baseline_aug | baseline_768 | `python src/train.py data=baseline_aug model=convnext_base training=baseline_768 training.batch_size=2 data.img_size=384` | ~25 GB+ | ❌ |
| ConvNeXt-Base | transformer_384 | baseline_768 | `python src/train.py data=transformer_384 model=convnext_base training=baseline_768 training.batch_size=4` | ~20 GB+ | ❌ |

### Transformer 모델 조합

| 모델 | 데이터 | 학습 | 명령어 | 메모리 | Apple MPS |
|------|--------|------|--------|--------|-----------|
| **Swin-Base-384** | transformer_384 | baseline_768 | `python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768` | ~12 GB | ⚠️ |
| Swin-Base-384 | baseline_aug | baseline_768 | `python src/train.py data=baseline_aug model=swin_base_384 training=baseline_768` | ~18 GB | ❌ |
| **DeiT-Base-384** | transformer_384 | baseline_768 | `python src/train.py data=transformer_384 model=deit_base_384 training=baseline_768` | ~12 GB | ⚠️ |
| DeiT-Base-384 | baseline_aug | baseline_768 | `python src/train.py data=baseline_aug model=deit_base_384 training=baseline_768` | ~18 GB | ❌ |

---

## 🎯 사용 사례별 추천

### 사례 1: 최고 성능 필요 (리더보드 제출)

```bash
# ResNet34 (검증됨: F1 0.993)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768
```

---

### 사례 2: Transformer 실험

```bash
# Swin-Base-384
python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768

# DeiT-Base-384
python src/train.py data=transformer_384 model=deit_base_384 training=baseline_768
```

---

### 사례 3: 다양한 CNN 비교

```bash
# ResNet 계열
python src/train.py data=baseline_aug model=resnet34 training=baseline_768
python src/train.py data=baseline_aug model=resnet50 training=baseline_768

# 최신 CNN (메모리 주의)
python src/train.py data=baseline_aug model=convnext_base training=baseline_768 training.batch_size=2 data.img_size=384
```

---

### 사례 4: Hydra Multi-Run (하이퍼파라미터 스윕)

```bash
# 여러 모델 동시 실험
python src/train.py --multirun \
  model=resnet34,resnet50 \
  data=baseline_aug,transformer_384

# 결과: multirun/YYYY-MM-DD/HH-MM-SS/{0,1,2,3}/
```

---

### 사례 5: 메모리 제약 환경

```bash
# 작은 batch size
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768 \
  training.batch_size=8

# 작은 이미지
python src/train.py \
  data=transformer_384 \
  model=resnet34 \
  training=baseline_768
```

---

## 🔧 트러블슈팅

### 환경별 트러블슈팅

#### CUDA 서버

##### GPU 메모리 확인

```bash
# GPU 메모리 사용량 모니터링
watch -n 1 nvidia-smi

# 특정 GPU 사용
CUDA_VISIBLE_DEVICES=0 python src/train.py ...
```

##### Multi-GPU 사용 (가능한 경우)

```bash
# PyTorch Lightning은 자동으로 사용 가능한 GPU 감지
python src/train.py \
  data=baseline_aug \
  model=convnext_base \
  training=baseline_768
```

##### 큰 Batch Size 최적화

```bash
# Batch size를 늘려 효율성 향상
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768 \
  training.batch_size=64  # 또는 128
```

---

#### Mac mini M4 Pro

##### MPS 메모리 모니터링

```bash
# Activity Monitor에서 메모리 확인
# 또는 터미널에서
top -l 1 | grep "PhysMem"
```

##### 메모리 절약 팁

```bash
# 1. Batch size 줄이기
python src/train.py ... training.batch_size=8

# 2. num_workers 줄이기 (메모리 절약)
python src/train.py ... training.num_workers=2

# 3. 작은 이미지 크기
python src/train.py ... data.img_size=512
```

---

### OOM (Out of Memory) 발생 시

**증상**:
- CUDA: `RuntimeError: CUDA out of memory`
- MPS: `RuntimeError: MPS backend out of memory`

**환경별 해결책**:

1. **Batch size 줄이기**
   ```bash
   python src/train.py ... training.batch_size=8  # 기본 16 → 8
   python src/train.py ... training.batch_size=4  # → 4
   python src/train.py ... training.batch_size=2  # → 2
   ```

2. **이미지 크기 줄이기**
   ```bash
   python src/train.py ... data.img_size=512  # 768 → 512
   python src/train.py ... data.img_size=384  # 768 → 384
   ```

3. **작은 모델 사용**
   ```bash
   # EfficientNet-B4/ConvNeXt-Base → ResNet34
   python src/train.py data=baseline_aug model=resnet34 training=baseline_768
   ```

---

### WanDB 로그인 문제

```bash
# 방법 1: 환경 변수 설정
echo "WANDB_MODE=disabled" > .env

# 방법 2: 로그인
wandb login

# 방법 3: 실행 시 비활성화
export WANDB_MODE=disabled
python src/train.py ...
```

---

### Hydra 경고 메시지

**경고**: `Defaults list is missing _self_`

**해결**: 무시 가능 (기능에 영향 없음)

---

### Augmentation 경고

**경고**: `Argument(s) 'value' are not valid for transform PadIfNeeded`

**해결**: 무시 가능 (Albumentations 버전 차이, 기능 정상 작동)

---

## 📊 성능 비교 요약

### CUDA 서버 (128GB RAM)

| 모델 | 입력 크기 | Batch Size | F1 Score | 메모리 | 훈련 시간 | 상태 |
|------|-----------|------------|----------|--------|-----------|------|
| **ResNet34** ⭐ | 768×768 | 32 | **0.993** | ~8 GB | 2~3h | ✅ |
| ResNet50 | 768×768 | 32 | 0.96~0.98 | ~10 GB | 3~4h | ✅ |
| EfficientNet-B4 | 768×768 | 32 | 0.96~0.98 | ~20 GB | 3~4h | ✅ |
| **ConvNeXt-Base** | 768×768 | 32 | 0.96~0.98 | ~28 GB | 4~5h | ✅ |
| **ConvNeXt-Base** | 768×768 | 64 | 0.97~0.99 | ~50 GB | 4~5h | ✅ |
| Swin-Base-384 | 384×384 | 32 | 0.95~0.97 | ~15 GB | 2~3h | ✅ |
| DeiT-Base-384 | 384×384 | 32 | 0.94~0.96 | ~15 GB | 2~3h | ✅ |

### Mac mini M4 Pro (24GB)

| 모델 | 입력 크기 | Batch Size | F1 Score | 메모리 | 훈련 시간 | 상태 |
|------|-----------|------------|----------|--------|-----------|------|
| **ResNet34** ⭐ | 768×768 | 16 | **0.993** | ~8 GB | 2~3h | ✅ |
| ResNet50 | 768×768 | 16 | 0.96~0.98 | ~10 GB | 3~4h | ✅ |
| **EfficientNet-B4** | 768×768 | 8 | 0.96~0.98 | ~19 GB | 3~4h | ✅ |
| EfficientNet-B4 | 768×768 | 12 | 0.96~0.98 | ~22 GB | 3~4h | ⚠️ |
| ConvNeXt-Base | 512×512 | 4 | 0.94~0.96 | ~20 GB | 4~5h | ⚠️ |
| **Swin-Base-384** | 384×384 | 16 | 0.95~0.97 | ~12 GB | 2~3h | ✅ |
| Swin-Base-384 | 384×384 | 20 | 0.95~0.97 | ~15 GB | 2~3h | ✅ |
| **DeiT-Base-384** | 384×384 | 16 | 0.94~0.96 | ~12 GB | 2~3h | ✅ |
| DeiT-Base-384 | 384×384 | 20 | 0.94~0.96 | ~15 GB | 2~3h | ✅ |

**범례**:
- ✅ 안전 사용 가능
- ⚠️ 주의 필요 (메모리 모니터링 권장)
- ❌ 사용 불가능 (OOM)

---

## 💡 추천 워크플로우

### 1단계: Best 모델로 Baseline 확립

```bash
python src/train.py data=baseline_aug model=resnet34 training=baseline_768
```

### 2단계: 다른 모델 실험 (선택사항)

```bash
# ResNet50
python src/train.py data=baseline_aug model=resnet50 training=baseline_768

# Transformer
python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768
```

### 3단계: Inference (리더보드 제출)

```bash
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt
# 출력: submission.csv
```

### 4단계: 결과 분석

```bash
python scripts/analyze_results.py --checkpoint checkpoints/champion/best_model.ckpt
# 출력: analysis_results/confusion_matrix.png
```

---

## 📚 관련 문서

- [README.md](../README.md) - 프로젝트 개요
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) - 완료 가이드
- [CLAUDE.md](../CLAUDE.md) - 개발 가이드라인

---

## 🎯 Quick Reference

### CUDA 서버 - 복사해서 사용

```bash
# Best 모델 (큰 batch size)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768 training.batch_size=32

# ConvNeXt-Base (최신 CNN)
python src/train.py data=baseline_aug model=convnext_base training=baseline_768 training.batch_size=32

# 전체 벤치마크 (병렬)
python src/train.py --multirun \
  model=resnet34,resnet50,efficientnet_b4,convnext_base \
  data=baseline_aug \
  training.batch_size=32

# Transformer 비교
python src/train.py --multirun \
  model=swin_base_384,deit_base_384 \
  data=transformer_384 \
  training.batch_size=32
```

### Mac mini M4 Pro - 복사해서 사용

```bash
# Best 모델 (검증됨)
python src/train.py data=baseline_aug model=resnet34 training=baseline_768

# ResNet50
python src/train.py data=baseline_aug model=resnet50 training=baseline_768

# EfficientNet-B4 (24GB 가능)
python src/train.py data=baseline_aug model=efficientnet_b4 training=baseline_768 training.batch_size=8

# Swin-Base-384
python src/train.py data=transformer_384 model=swin_base_384 training=baseline_768

# 안전한 Multi-Run
python src/train.py --multirun \
  model=resnet34,resnet50 \
  data=baseline_aug
```

---

## 📊 환경 선택 가이드

| 목적 | 권장 환경 | 이유 |
|------|----------|------|
| **최고 성능** | CUDA 서버 | ConvNeXt-Base, 큰 batch size |
| **빠른 프로토타입** | Mac M4 Pro | ResNet34, 즉시 시작 |
| **병렬 실험** | CUDA 서버 | Multi-Run 동시 실행 |
| **Transformer 실험** | 둘 다 | 양쪽 모두 가능 |
| **대용량 Batch** | CUDA 서버 | 128GB RAM 활용 |
| **휴대성/편의성** | Mac M4 Pro | 로컬 환경 |

---

## 🎓 학습 내용

### CUDA 서버 활용법
- ✅ 모든 모델 조합 가능
- ✅ Multi-Run으로 병렬 실험
- ✅ 큰 batch size로 안정적 학습
- ✅ ConvNeXt-Base 같은 대형 모델 활용

### Mac mini M4 Pro 활용법
- ✅ 24GB로 대부분 모델 가능
- ✅ EfficientNet-B4도 가능 (batch_size 조정)
- ✅ Transformer 모델 안정적
- ✅ 빠른 프로토타입 및 실험

---

**마지막 업데이트**: 2026-02-16
**프로젝트 상태**: 완료 (F1 0.993)
**Best 모델**: ResNet34 + baseline_aug (768×768)
**환경**: CUDA 서버 (128GB) + Mac mini M4 Pro (24GB)
