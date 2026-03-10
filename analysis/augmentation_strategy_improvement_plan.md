# 데이터 증강 전략 종합 개선 계획

> 학습 데이터 분석 기반 Config 최적화 + 과적합 해결 (Stratified K-Fold, TTA, Mixup, 정규화 강화) + Test 데이터 증강 전략 + Pseudo-Labeling을 통합한 종합 개선 계획

**작성일**: 2026-02-21
**최종 업데이트**: 2026-02-21
**상태**: ✅ 구현 완료 (Phase 1~4)

---

## 📊 문제 상황 분석

### 현재 상태
- **Val F1**: 0.96~0.988 (매우 높음)
- **Public LB**: 0.81~0.9085 (낮음)
- **Gap**: 0.06~0.18 (심각한 과적합 또는 train/test 분포 불일치)

### 데이터 분석 결과

#### 학습 데이터 (.benchmark_results)
- **총 1,570개**
- **평균 크기**: 498×538px
- **클래스 불균형**: 최대 2.17배 (class_1: 46개)
- **Aspect Ratio**: 평균 0.97 (거의 정사각형)
- **Portrait 66.2%**, Landscape 32.7%

#### 테스트 데이터 (test_detailed_analysis.json)
- **총 3,140개** (학습의 2배!)
- **구김/왜곡**: 66% high (학습 데이터보다 훨씬 어려움)
- **그림자**: 24.1%
- **비정상 방향**: 36.2%
- **흑백+저채도**: 87.2%

#### 현재 Train/Val Split
- 방식: `train_test_split` (stratify 사용 중)
- 비율: 80/20
- **개선 필요**: K-Fold 적용으로 안정적인 평가

---

## 🎯 해결 방안: 3단계 통합 전략

### Phase 1: 학습 데이터 분석 기반 Config 최적화 ✅

#### 1-1. 이미지 크기 최적화 (512 적용)

**문제점 (기존 고해상도):**
- 평균 이미지 크기: 498×538px
- 과도한 upscaling 시 불필요한 보간 아티팩트, 학습 속도 저하

**적용 완료 (512px):**
- `configs/data/baseline_aug.yaml`: img_size 512
- `configs/data/default.yaml`: img_size 512
- 모든 관련 파라미터 조정 (LongestMaxSize, PadIfNeeded, CoarseDropout)

**기대 효과:**
- 픽셀 수 감소로 학습 속도·메모리 효율 향상
- 학습 속도 향상 (약 30-40%)
- 메모리 효율 → 배치 크기 증가 가능
- 원본 데이터 특성 보존

#### 1-2. 클래스 가중치 문서화

**적용 완료:**
```yaml
# 실제 클래스 가중치 (.benchmark_results/class_weights.csv):
# - class_1 (임신의료비 지급신청서): 2.17배 (46개)
# - class_13 (이력서): 1.35배 (74개)
# - class_14 (소견서): 2.00배 (50개)
# - 나머지 클래스: 1.00배 (100개)
# 최대 불균형 비율: 2.17배
```

---

### Phase 2: 과적합 해결 - Stratified K-Fold ✅

#### 2-1. DataModule에 K-Fold 기능 추가

**적용 완료:**
- `src/data/datamodule.py`: K-Fold 로직 추가
  - `use_kfold`, `n_folds`, `fold_idx` 파라미터
  - StratifiedKFold로 클래스 비율 보존
- `configs/config.yaml`: K-Fold 설정 추가

**사용 방법:**
```bash
# 단일 모델 (기존 방식)
python src/train.py model.model_name=resnet50

# K-Fold (Fold 0~4)
for fold in {0..4}; do
    python src/train.py \
        model.model_name=resnet50 \
        data.use_kfold=true \
        data.fold_idx=$fold
done
```

**효과:**
- **클래스 비율 보존**: 각 폴드에서 클래스 분포 동일
- **안정적인 평가**: 5개 모델의 평균 성능으로 신뢰도 향상
- **과적합 감소**: 다양한 train/val 조합으로 일반화
- **LB 스코어 향상**: Val과 Test 분포 불일치 완화

---

### Phase 3: 과적합 해결 - TTA & Mixup & 정규화 강화 ✅

#### 3-1. TTA (Test Time Augmentation) 구현

**적용 완료:**
- `src/utils/tta.py`: TTA 유틸리티 구현
- `src/inference.py`: TTA 적용 로직 추가
- `configs/inference/default.yaml`: use_tta 설정 추가

**TTA 변환:**
1. Original
2. Horizontal Flip
3. Vertical Flip
4. Rotate 90°
5. Rotate 270°

**사용 방법:**
```bash
python src/inference.py inference.use_tta=true
```

**효과:**
- 방향 다양성: 테스트 데이터의 36.2% 비정상 방향 대응
- 강건성 향상: 약간의 변화에도 일관된 예측
- **LB +0.02~0.05 향상**: 일반적으로 2~5% 성능 향상

#### 3-2. Mixup/CutMix 추가

**적용 완료:**
- `src/utils/mixup.py`: Mixup/CutMix 구현
- `src/models/module.py`: training_step에 Mixup 로직 추가
- `configs/training/default.yaml`: Mixup 설정 추가

**사용 방법:**
```bash
python src/train.py training.use_mixup=true
```

**효과:**
- 데이터 다양성 증가
- 과적합 방지
- 일반화 성능 향상

#### 3-3. 정규화 강화

**적용 완료:**
- `configs/training/default.yaml`:
  - `weight_decay`: 1e-4 → 0.01 (100배 증가)
  - `dropout_rate`: 0.0 → 0.3
  - `label_smoothing_value`: 0.0 → 0.1
  - `learning_rate`: 1e-3 → 5e-4 (안정적 학습)
  - `early_stopping.patience`: 5 → 10

**효과:**
- 과적합 방지
- 안정적인 학습
- Val/LB Gap 감소

---

## 📋 수정 파일 요약

| 파일 | 변경 내용 | 상태 |
|---|---|---|
| `configs/data/baseline_aug.yaml` | img_size: 512, 클래스 가중치 주석 | ✅ 완료 |
| `configs/data/default.yaml` | img_size: 512, pseudo_csv 옵션 | ✅ 완료 |
| `src/data/datamodule.py` | K-Fold 기능, MixedDocumentDataset, pseudo_csv 지원 | ✅ 완료 |
| `configs/config.yaml` | K-Fold, pseudo 설정 추가 | ✅ 완료 |
| `src/utils/tta.py` | TTA 구현 (신규) | ✅ 완료 |
| `src/inference.py` | TTA 적용 | ✅ 완료 |
| `configs/inference/default.yaml` | use_tta 설정 추가 | ✅ 완료 |
| `src/utils/mixup.py` | Mixup/CutMix 구현 (신규) | ✅ 완료 |
| `src/models/module.py` | Mixup, Dropout, Label Smoothing 적용 | ✅ 완료 |
| `configs/training/default.yaml` | 정규화 파라미터 강화 | ✅ 완료 |
| `scripts/pseudo_label.py` | Pseudo-label 생성 스크립트 (신규) | ✅ 완료 |
| `src/train.py` | pseudo_csv 파라미터 전달 | ✅ 완료 |
| `src/utils/helpers.py` | create_datamodule_from_config pseudo_csv 지원 | ✅ 완료 |
| `configs/pseudo.yaml` | Pseudo-label 생성 설정 (신규) | ✅ 완료 |

---

## 🚀 실행 순서 & 예상 성능

### Step 1: Config 최적화 (즉시)
```bash
# 512px로 단일 실험
python src/train.py \
    data=baseline_aug \
    model.model_name=resnet50

# 예상: Val F1 0.96 → LB 0.87 (Gap 감소)
```

### Step 2: K-Fold (1일 소요)
```bash
# 5-Fold 학습
for fold in {0..4}; do
    python src/train.py \
        data=baseline_aug \
        model.model_name=resnet50 \
        data.use_kfold=true \
        data.fold_idx=$fold
done

# 예상: Local CV 0.95 ± 0.01 → LB 0.90 (Gap 0.05)
```

### Step 3: TTA 적용 (즉시)
```bash
python src/inference.py \
    inference.use_tta=true \
    inference.run_id=champion

# 예상: LB +0.02~0.03 → 0.92~0.93
```

### Step 4: Mixup + 정규화 (1일 소요)
```bash
python src/train.py \
    data=baseline_aug \
    model.model_name=resnet50 \
    data.use_kfold=true \
    training.use_mixup=true \
    training.dropout_rate=0.3 \
    training.weight_decay=0.01

# 예상: Local CV 0.93 ± 0.015 → LB 0.93~0.95 (Gap 0.00~0.02)
```

### Step 5: Pseudo-Labeling (2단계, 2일 소요)

```bash
# Step 5-1: 기존 best 모델로 pseudo-label 생성 (TTA 권장)
python scripts/pseudo_label.py \
    pseudo.use_tta=true \
    pseudo.confidence_threshold=0.9

# Step 5-2: Pseudo-label 포함 재학습
python src/train.py \
    data=baseline_aug \
    data.pseudo_csv=pseudo_labels.csv \
    data.use_kfold=true \
    training.use_mixup=true

# 예상: LB 0.95~0.97 (Gap ≈ 0)
```

### 최종 예상 성능

| 단계 | Local CV | Public LB | Gap | 개선폭 |
|---|---|---|---|---|
| 현재 | 0.96~0.988 | 0.81~0.9085 | 0.15 | - |
| + 512px | 0.96 | 0.87 | 0.09 | +0.06 |
| + K-Fold | 0.95±0.01 | 0.90 | 0.05 | +0.03 |
| + TTA | 0.95±0.01 | 0.92 | 0.03 | +0.02 |
| + Mixup+정규화 | 0.93±0.015 | 0.93~0.95 | 0.00~0.02 | +0.01~0.03 |
| + **Pseudo-Labeling** | 0.94±0.01 | **0.95~0.97** | ≈0 | **+0.03~0.07** |

**목표: Gap 0.15 → 0 달성!**

---

---

### Phase 4: Pseudo-Labeling ✅

#### 4-1. 개요 및 근거

**왜 Pseudo-Labeling인가:**
- 테스트 데이터(3,140개)가 학습 데이터(1,570개)의 **정확히 2배**
- 고신뢰도 예측만 선별하면 추가 학습 데이터로 활용 가능
- 특히 Val/LB Gap의 원인이 train/test **분포 불일치**이므로, 테스트 도메인 데이터를 직접 학습에 포함하면 Gap을 줄일 수 있음

**구현 전략:**
- Softmax 확률의 최댓값을 신뢰도로 사용
- `confidence_threshold` 이상인 샘플만 pseudo-label로 채택 (기본값: 0.9)
- Pseudo-label 이미지는 `test/` 디렉토리, 학습 이미지는 `train/` 디렉토리에서 로드
- Val 셋에는 포함하지 않음 (검증 데이터 오염 방지)

#### 4-2. 적용 완료 내역

| 파일 | 변경 내용 |
|---|---|
| `scripts/pseudo_label.py` | pseudo-label 생성 스크립트 (신규) |
| `src/data/datamodule.py` | `MixedDocumentDataset` 추가, `pseudo_csv` 파라미터 지원 |
| `src/train.py` | `pseudo_csv`, `pseudo_image_dir` 파라미터 전달 |
| `src/utils/helpers.py` | `create_datamodule_from_config`에 pseudo_csv 지원 추가 |
| `configs/pseudo.yaml` | pseudo-label 생성 설정 (신규) |
| `configs/data/default.yaml` | `pseudo_csv`, `pseudo_image_dir` 옵션 추가 |
| `configs/data/baseline_aug.yaml` | `pseudo_csv`, `pseudo_image_dir` 옵션 추가 |
| `configs/config.yaml` | pseudo 설정 그룹 추가 |

#### 4-3. 사용 방법

**Step 1: Pseudo-label 생성**
```bash
# 기본 (champion 모델, confidence ≥ 0.9)
python scripts/pseudo_label.py

# TTA로 더 신뢰도 높은 pseudo-label 생성 (권장)
python scripts/pseudo_label.py pseudo.use_tta=true

# 임계값 조정 (더 보수적: 높은 품질)
python scripts/pseudo_label.py pseudo.confidence_threshold=0.95

# 특정 모델 사용
python scripts/pseudo_label.py pseudo.run_id=20260221_run_001
```

생성 결과:
- `datasets_fin/pseudo_labels.csv` — 학습용 (ID, target)
- `datasets_fin/pseudo_labels_with_confidence.csv` — 상세 (ID, target, confidence)

**Step 2: Pseudo-label 포함 재학습**
```bash
# Pseudo-label 추가 학습 (기본)
python src/train.py data.pseudo_csv=pseudo_labels.csv

# K-Fold + Pseudo-label 조합 (권장)
python src/train.py \
    data=baseline_aug \
    data.pseudo_csv=pseudo_labels.csv \
    data.use_kfold=true \
    data.fold_idx=0
```

#### 4-4. 신뢰도 임계값 가이드

| 임계값 | 예상 선택 비율 | 특징 |
|---|---|---|
| 0.95 | ~30~50% (940~1,570개) | 고품질, 소량 |
| 0.90 | ~50~70% (1,570~2,200개) | **권장** (품질/수량 균형) |
| 0.80 | ~70~90% (2,200~2,820개) | 수량 우선, 노이즈 위험 |

#### 4-5. 예상 효과

- 학습 데이터 **최대 2배 증가** (1,570 → 최대 3,140+개)
- 테스트 도메인 분포를 직접 학습에 반영 → Val/LB Gap 감소
- **LB +0.03~0.07 향상** 기대

---

## 📌 Test 데이터 증강 전략 (기존 완료)

### 테스트 데이터 특성 반영
- **구김/왜곡 66%**: GridDistortion p=0.5, ElasticTransform p=0.3
- **그림자 24.1%**: RandomShadow p=0.2
- **비정상 방향 36.2%**: RandomRotate90 p=0.5, Rotate±45 p=0.3
- **낮은 대비**: CLAHE p=0.65
- **흑백 중심**: ColorJitter saturation/hue 축소

---

## 🎯 다음 단계

1. **512px 단일 실험**: 성능 검증 ✅
2. **K-Fold 5개 학습**: 안정적인 CV 구축 ✅
3. **TTA 적용**: 즉시 성능 향상 ✅
4. **Mixup + 정규화**: 최종 과적합 해결 ✅
5. **Pseudo-Labeling**: 테스트 데이터 활용 ✅
   - `python scripts/pseudo_label.py pseudo.use_tta=true` — pseudo-label 생성
   - `python src/train.py data.pseudo_csv=pseudo_labels.csv` — 재학습
6. **앙상블**: K-Fold 5개 + Pseudo-label 모델 앙상블

---

## 📚 참고 자료

- 학습 데이터 분석: `.benchmark_results/`
- 테스트 데이터 분석: `analysis/test_detailed_analysis.json`
- 증강 전략 추천: `analysis/test_augmentation_recommendations.json`
