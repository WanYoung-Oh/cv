# 혼동 클래스 [3, 7] 분류 개선 리서치

> **작성일**: 2026-02-28
> **목적**: 입퇴원확인서(class 3) vs 외래진료확인서(class 7) 혼동 문제 해결 방안 연구
> **구현 여부**: 검토 단계 — 개발 미착수

---

## 1. 문제 정의

### 1.1 혼동 클래스 현황

| 항목 | 내용 |
|------|------|
| 클래스 3 | `confirmation_of_admission_and_discharge` (입퇴원확인서) |
| 클래스 7 | `medical_outpatient_certificate` (외래진료확인서) |
| 훈련 샘플 수 | 각 99장 / 101장 (합계 **200장**) |
| 전체 훈련 데이터 | 1,570장 (17클래스) |

### 1.2 왜 혼동되는가

두 문서는 한국의 의료 서식으로, 아래 특성으로 인해 시각적 유사성이 높음:

- **공통 시각 요소**: 병원 도장/직인, 의사 서명, 한글 텍스트 레이아웃, 표 구조
- **형식 유사성**: 환자 정보(성명·주민등록번호·병명)를 동일한 위치에 기재
- **핵심 차이**: 입원/퇴원 날짜 필드(class 3) vs 통원 날짜 필드(class 7) — 텍스트 내용 차이이지 레이아웃 차이가 아님
- **스캔 품질 영향**: 69% grayscale, 66% high distortion 환경에서 텍스트 가독성 저하 → 픽셀 레벨 분류기가 놓치기 쉬움

### 1.3 기존 시도 대비 한계

현재 메인 파이프라인에서 적용한 혼동 클래스 대응:

| 기존 방법 | 효과 | 한계 |
|-----------|------|------|
| `confusion_pair_extra_weight` (5~10×) | 손실 함수 재조정 | 다른 클래스 성능 트레이드오프 |
| Focal Loss (γ=2.0~3.0) | 어려운 샘플 집중 | 전체 클래스에 동일 적용 |
| 3·7 전용 데이터 증강 | 다양성 증대 | 메인 모델 수용량 한계 |
| 클래스 가중치 보정 | 불균형 완화 | 근본적 혼동 해결 아님 |

**결론**: 기존 방법은 메인 분류기의 최적화 방향 조정에 불과하며, 혼동 자체를 직접 해결하지 못함.

---

## 2. 접근법 분류

크게 **3가지 전략**으로 분류 가능:

```
A. 두 단계 분류 (Two-Stage Hierarchical)
   └─ 메인 분류기 → 불확실 시 전용 이진분류기로 재분류

B. 확률 보정 앙상블 (Probability Correction Ensemble)
   └─ 메인 확률 + 이진분류기 확률을 결합하여 최종 예측 보정

C. 표현 학습 개선 (Representation Learning)
   └─ Contrastive / Metric Learning으로 두 클래스 임베딩 거리 확대
```

---

## 3. 방법론 상세 분석

### 3.1 방법 A: Two-Stage 계층적 분류

#### 개념
```
입력 이미지
    │
    ▼
[Stage 1] 메인 17클래스 분류기
    │
    ├─ 예측 클래스 ∈ {3, 7} 또는 p(3)+p(7) > θ
    │         └──→ [Stage 2] 이진분류기 (3 vs 7)
    │
    └─ 그 외 클래스 → 직접 출력
```

#### 세부 구현 방식

**A-1. Confidence-Based Routing (신뢰도 기반 라우팅)**
- 메인 분류기가 클래스 3 또는 7을 예측할 때 → 이진분류기로 재분류
- 또는 p(class3) + p(class7) > θ (예: 0.3) 조건으로 라우팅
- **장점**: 불확실한 경우만 처리, 다른 클래스 성능 유지
- **단점**: θ 임계값 튜닝 필요, 라우팅 실수가 오류 전파

**A-2. Always-On Specialist (상시 전문가 모델)**
- 모든 샘플에 이진분류기 실행 후, 메인 분류기 결과와 결합
- 단, 이진분류기 출력은 클래스 3·7에만 영향
- **장점**: 라우팅 오류 없음, 안정적
- **단점**: 추론 비용 2배

#### 논문·문헌 근거

- **"Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks"** (Bertinetto et al., CVPR 2020)
  → 계층적 레이블 구조를 활용한 손실 함수 개선. 혼동하기 쉬운 클래스를 상위 노드로 묶어 계층적 분류 수행. ImageNet에서 오분류 비용 감소.

- **"Hierarchical Image Classification using Entailment Cone Embeddings"** (Dhall et al., CVPR 2020)
  → 계층 구조를 임베딩 공간에 반영. 유사 클래스끼리의 혼동을 계층적으로 처리.

- **Kaggle 실전 사례 (문서 분류)**:
  RVL-CDIP 등 문서 분류 대회에서 혼동 클래스 쌍에 전용 이진분류기를 추가하여 성능 향상 보고 (Accuracy +1~3%).

#### 실효성 평가

| 항목 | 평가 |
|------|------|
| 혼동 해결 직접성 | ★★★★★ (가장 직접적) |
| 구현 복잡도 | ★★★☆☆ (중간) |
| 소규모 데이터(200장) 대응 | ★★★☆☆ (K-Fold 필수) |
| 다른 클래스 성능 영향 | 없음 (독립적) |
| 추론 시간 | 약 1.3~1.5배 증가 |

---

### 3.2 방법 B: 확률 보정 앙상블

#### 개념
메인 분류기의 클래스 3·7 확률을 이진분류기 출력으로 재배분(re-weighting).

#### 세부 방식

**B-1. 비례 재배분 (Proportional Redistribution)**
```
p_main: 메인 분류기의 17클래스 확률 벡터
p_bin:  이진분류기의 (class3, class7) 확률

pool = p_main[3] + p_main[7]   # 두 클래스의 총 확률

final[3] = p_bin[3] * pool
final[7] = p_bin[7] * pool
final[other] = p_main[other]   # 나머지 클래스는 그대로
```
→ 메인 분류기가 3·7 중 하나로 확신하는 경우에도 이진분류기가 교정 가능

**B-2. 가중 평균 (Weighted Average)**
```
α = 보정 강도 (0.0~1.0, 튜닝 파라미터)

final[3] = (1-α) * p_main[3] + α * p_bin[3] * pool / (p_bin[3] + p_bin[7])
final[7] = (1-α) * p_main[7] + α * p_bin[7] * pool / (p_bin[3] + p_bin[7])
```

**B-3. Log-Odds Correction (Post-hoc Calibration)**
```
# Platt Scaling 응용
log_odds_main = log(p_main[3] / p_main[7])
log_odds_bin  = log(p_bin[3]  / p_bin[7])

combined_log_odds = w1 * log_odds_main + w2 * log_odds_bin
final_prob_3 = sigmoid(combined_log_odds)
final_prob_7 = 1 - final_prob_3
```
→ w1, w2는 validation set으로 학습 가능 (Platt Scaling과 동일 원리)

#### 논문·문헌 근거

- **"Probability Calibration for Knowledge Graph Embedding Models"** (Tabacof & Costabello, 2019)
  → 확률 교정 기법의 이론적 기반. Temperature Scaling, Platt Scaling.

- **"On Calibration of Modern Neural Networks"** (Guo et al., ICML 2017)
  → 딥러닝 모델이 과신(overconfident)하는 문제. Temperature Scaling이 간단하면서 효과적.

- **"Specialised Ensemble Methods for Imbalanced and Difficult Problems"** (다수 ML 경진대회 사례)
  → 전문가 모델(specialist)의 출력으로 기본 모델(generalist)을 보정하는 앙상블 전략. 특히 혼동이 높은 클래스 쌍에 효과적.

#### 실효성 평가

| 항목 | 평가 |
|------|------|
| 혼동 해결 직접성 | ★★★★☆ |
| 구현 복잡도 | ★★☆☆☆ (가장 간단) |
| 소규모 데이터 대응 | ★★★★☆ (메인 확률 보정만이라 강건) |
| 다른 클래스 성능 영향 | 최소 (pool 재배분) |
| 추론 시간 | 약 1.3~1.5배 증가 |

---

### 3.3 방법 C: Contrastive / Metric Learning

#### 개념
단순 cross-entropy 분류 대신, 클래스 3·7의 임베딩을 **최대한 멀리** 떨어지도록 훈련.

#### 세부 방식

**C-1. Supervised Contrastive Learning (SupCon)**
```
손실 함수:
L_SupCon = -log( Σ_{j∈P(i)} exp(z_i·z_j/τ) / Σ_{k≠i} exp(z_i·z_k/τ) )

P(i) = 같은 클래스의 다른 샘플들
→ class 3과 7이 hard negative로 작용 → 임베딩 거리 극대화
```
- 논문: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
  → ImageNet에서 Cross-Entropy 단독 대비 Top-1 +1.6% 향상, fine-grained 태스크에서 더 큰 효과

**C-2. ArcFace (Angular Margin Loss)**
```
L = -log( e^{s·cos(θ_yi + m)} / (e^{s·cos(θ_yi + m)} + Σ_{j≠yi} e^{s·cos(θ_j)}) )

m = angular margin (예: 0.5)
→ 결정 경계에 여유를 두어 유사 클래스 구분력 강화
```
- 원래 얼굴 인식(face recognition)에서 탁월한 성능
- 문서 분류에도 적용 사례 있음 (RVL-CDIP 등)

**C-3. Triplet Loss with Hard Negative Mining**
```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

hard negative: class 3 ↔ class 7 쌍을 의도적으로 hard negative로 지정
```
- 소규모 데이터에서 triplet mining 비효율 → SupCon이 더 권장됨

#### 실효성 평가

| 항목 | 평가 |
|------|------|
| 혼동 해결 직접성 | ★★★★☆ (표현 수준에서 해결) |
| 구현 복잡도 | ★★★★★ (가장 복잡) |
| 소규모 데이터 대응 | ★★☆☆☆ (200장은 contrastive에 불리) |
| 다른 클래스 성능 영향 | 있을 수 있음 (전체 훈련 방식 변경) |
| 추론 시간 | 변화 없음 (훈련 단계 변경) |

---

## 4. 이진분류기 설계 상세

### 4.1 학습 데이터 구성

| 전략 | 설명 | 장/단점 |
|------|------|---------|
| **클래스 3·7만 추출** | 200장으로 독립 학습 | 클린하지만 극소규모 |
| **17클래스 → 이진 재라벨링** | 클래스 3→0, 7→1, 나머지 제외 | 동일 |
| **부정 샘플(hard negative) 포함** | 유사 의료 문서(class 6 medical_bill_receipts, class 12 prescription 등) 추가 | 경계 강화, 데이터 증가 |
| **전체 데이터 유지 + 이진 헤드** | 백본 공유, 출력 헤드만 변경 | 데이터 효율, 지식 공유 |

**권장**: `클래스 3·7만 추출` (200장) + **K-Fold 5분할** 사용 (fold당 160장 훈련, 40장 검증)

### 4.2 모델 아키텍처 선택

| 아키텍처 | 파라미터 수 | 권장 이유 |
|----------|------------|----------|
| EfficientNet-B0 (224) | 5.3M | 소규모 데이터에 과적합 최소, 빠른 훈련 |
| **EfficientNet-B3 (384)** | 12M | 메인과 동일 해상도, 균형적 선택 ⭐ |
| MaxViT-Base-384 (메인과 동일) | 119M | 과적합 위험 (200장 대비 과대), 메인 ckpt 초기화 가능 |
| Swin-Tiny (224) | 28M | 중간 선택지 |

**초기화 전략**:
- **옵션 1**: ImageNet pretrained으로 새로 파인튜닝 (독립 학습)
- **옵션 2**: 메인 17클래스 모델 체크포인트에서 백본 가중치 전이 (transfer) → 분류 헤드만 교체
  → 이미 문서 이미지를 학습한 가중치이므로 빠른 수렴 기대 ⭐

### 4.3 학습 전략 (소규모 데이터 과적합 방지)

| 기법 | 설정값 권장 | 근거 |
|------|------------|------|
| Dropout | 0.3~0.5 | 메인(0.1)보다 강화 |
| Stochastic Depth | 0.1~0.2 | 정규화 |
| Weight Decay | 0.05~0.1 | AdamW |
| Label Smoothing | 0.1~0.2 | 이진분류기에도 효과 |
| Early Stopping | patience=15~20 | 과적합 조기 탐지 |
| **K-Fold (5-fold)** | fold 5개 | 200장 최대 활용, 앙상블 효과 ⭐ |
| Learning Rate | 5e-5 ~ 1e-4 | 소규모 데이터 안정적 수렴 |
| Epochs | 50~100 | Early stopping과 병행 |
| Warmup | 5 epochs | 안정적 초기 수렴 |
| **Mixup** | alpha=0.2 | 이진분류기에서도 경계 강화 효과 |

### 4.4 데이터 증강 전략

#### 기본 전략 (메인 파이프라인과 동일)
- RandomRotate90, HorizontalFlip, VerticalFlip
- GridDistortion, ElasticTransform (스캔 왜곡 모사)
- RandomBrightnessContrast, GaussianBlur (스캔 품질 다양화)
- CoarseDropout (문서 일부 가림)

#### 이진분류기 특화 추가 전략

**전략 1: 텍스트 영역 마스킹 (RandomErasing / CoarseDropout 강화)**
- 목표: 특정 단어("입원", "퇴원", "외래")에 의존하지 않고 레이아웃 구조로 분류 강제
- 방법: 이미지 면적의 20~40%를 랜덤하게 가림
- 기대 효과: 구조 기반 분류 능력 강화 → 텍스트 열화 환경에서도 견고함

**전략 2: 강한 해상도 저하 (Downscale + Upscale)**
- 스캔 품질 불량 시뮬레이션
- `Downscale(scale_min=0.5, scale_max=0.8, p=0.3)`

**전략 3: 문서 기울기 시뮬레이션 (Perspective Transform 강화)**
- 실제 스캔에서 각도 오차 모사
- `Perspective(scale=(0.05, 0.2), p=0.5)` (메인보다 강화)

**전략 4: 모폴로지 변환 (선택적)**
- 텍스트 두께 변환 (팽창/침식) → 다양한 인쇄 품질 모사

### 4.5 TTA 전략

이진분류기의 추론 시 TTA는 **heavy level (11가지)** 권장:
- 소규모 데이터 → 단일 추론이 불안정할 수 있음
- 11가지 변환 평균 → 안정적 확률 출력
- K-Fold 5모델 × TTA 11가지 = **55회 추론 평균** → 매우 안정적

```
이진분류기 최종 확률 = mean(5개 fold 모델 × 11가지 TTA 변환)
```

---

## 5. 앙상블 전략 상세

### 5.1 추천 앙상블 아키텍처

```
[추론 시간]

입력 이미지 x
    │
    ├─→ [메인 분류기] (MaxViT-384, TTA standard)
    │       └─ p_main: 17차원 확률 벡터
    │
    └─→ [이진분류기 앙상블] (5-fold × TTA heavy)
            └─ p_bin: (p3, p7) 확률 쌍

[결합]
pool = p_main[3] + p_main[7]
final[3] = p_bin_ensemble[3] * pool        ← 이진분류기가 3·7 비율 결정
final[7] = p_bin_ensemble[7] * pool        ← 이진분류기가 3·7 비율 결정
final[k] = p_main[k]  (k ≠ 3, 7)          ← 나머지 클래스는 메인 그대로

argmax(final) → 최종 예측 클래스
```

### 5.2 앙상블 방식 비교

| 방식 | 공식 | 장점 | 단점 |
|------|------|------|------|
| **비례 재배분** (추천) | `final[3] = p_bin[3] * (p_main[3]+p_main[7])` | 직관적, 안정적 | α 튜닝 없음 |
| 가중 평균 | `final[3] = (1-α)*p_main[3] + α*p_bin[3]*pool` | 유연한 보정 강도 | α 튜닝 필요 |
| Log-Odds 보정 | Platt Scaling 응용 | 이론적 우수 | validation set 필요 |
| Two-Stage Routing | p(3)+p(7)>θ → 전용 이진분류기 결과 사용 | 가장 직접적 | θ 튜닝 필요 |

**비례 재배분 방식 선택 이유**:
- 메인 분류기가 클래스 3·7에 할당한 총 확률(`pool`)을 보존하면서
- 그 안에서의 배분 비율만 이진분류기로 결정
- 다른 클래스의 확률은 전혀 영향 받지 않음
- 별도 하이퍼파라미터 튜닝 불필요

### 5.3 예시 시나리오

**시나리오 A: 메인이 틀리고 이진분류기가 맞는 경우**
```
입력: 실제 클래스 7 (외래진료확인서)

p_main: [0.02, 0.0, ..., 0.55, ..., 0.40, ..., 0.03]
                           ↑class3=0.55    ↑class7=0.40
→ 메인 예측: class 3 (오답)

p_bin:  [class3=0.18, class7=0.82]
pool = 0.55 + 0.40 = 0.95

final[3] = 0.18 × 0.95 = 0.171
final[7] = 0.82 × 0.95 = 0.779
→ 최종 예측: class 7 ✅ (보정 성공)
```

**시나리오 B: 메인이 맞고 이진분류기가 오히려 틀리는 경우**
```
입력: 실제 클래스 3

p_main: [0.01, ..., 0.85, ..., 0.10, ...]
                    ↑class3=0.85    ↑class7=0.10
→ 메인 예측: class 3 (정답)

p_bin:  [class3=0.42, class7=0.58]  ← 이진분류기 오답
pool = 0.85 + 0.10 = 0.95

final[3] = 0.42 × 0.95 = 0.399
final[7] = 0.58 × 0.95 = 0.551
→ 최종 예측: class 7 ❌ (보정 실패)
```

→ **위험 시나리오 B에 대응**: 메인이 한 클래스에 대해 매우 높은 확신(p > 0.7)을 가질 때는 보정 강도를 줄이는 `α` 파라미터 도입 검토

---

## 6. 방법론 종합 비교

### 6.1 전략별 성능 잠재력

| 전략 | 혼동 해결 기대 효과 | 구현 복잡도 | 소규모 데이터 적합성 | 추론 시간 | 추천 순위 |
|------|---------------------|------------|---------------------|-----------|-----------|
| **B-1. 비례 재배분 앙상블** | ★★★★☆ | ★★☆☆☆ | ★★★★★ | 1.5배 | **1위** ⭐ |
| **A-1. Confidence Routing** | ★★★★★ | ★★★☆☆ | ★★★★☆ | 1.3배 | **2위** |
| B-2. 가중 평균 앙상블 | ★★★☆☆ | ★★☆☆☆ | ★★★★★ | 1.5배 | 3위 |
| B-3. Log-Odds 보정 | ★★★★☆ | ★★★☆☆ | ★★★★☆ | 1.5배 | 4위 |
| C-1. SupCon (재훈련) | ★★★★☆ | ★★★★★ | ★★☆☆☆ | 변화없음 | 5위 |
| C-2. ArcFace (재훈련) | ★★★★☆ | ★★★★★ | ★★☆☆☆ | 변화없음 | 6위 |

### 6.2 성능 향상 기대 수치 (추정)

현재 베스트: 퍼블릭 F1 **0.9321**, Val F1 **0.961**

| 전략 조합 | F1 개선 추정 (클래스 3·7) | 전체 F1 개선 추정 |
|-----------|--------------------------|-------------------|
| 이진분류기 단독 (5-fold + heavy TTA) | +3~8% (해당 클래스) | +0.5~1.5% |
| 비례 재배분 앙상블 | +2~5% (해당 클래스) | +0.3~1.0% |
| Routing + 비례 재배분 조합 | +4~8% (해당 클래스) | +0.5~1.5% |

> ⚠️ 수치는 200장 소규모 데이터 기반 추정값. 실제 결과는 이진분류기 품질에 크게 의존.

---

## 7. 구현 계획 (제안)

### 7.1 최소 구현 경로 (Phase 1 — 권장)

```
목표: 비례 재배분 앙상블 구현 (가장 낮은 위험, 합리적 성능 기대)

1. scripts/train_binary_classifier.py 작성
   - 클래스 3·7만 추출 (200장)
   - 5-Fold K-Fold 학습
   - EfficientNet-B3-384 또는 메인 MaxViT 백본 초기화
   - Heavy augmentation + Mixup

2. scripts/binary_inference.py 작성
   - 5-fold 모델 앙상블 + heavy TTA
   - 출력: p_bin (class3_prob, class7_prob)

3. src/ensemble.py 확장 (또는 신규 스크립트)
   - 메인 분류기 예측 + 이진분류기 확률 결합
   - 비례 재배분 방식 적용

4. 검증
   - Validation set에서 class 3·7 F1 변화 확인
   - 전체 F1 변화 확인 (regression 없는지)
```

### 7.2 확장 구현 경로 (Phase 2 — 선택)

```
목표: Confidence Routing으로 정밀도 향상

1. Phase 1 완료 후 진행
2. ensemble.py에 routing 로직 추가
   - p_main[3] + p_main[7] > θ 조건 분기
   - θ 최적값: validation set grid search (0.1~0.8)
3. α 파라미터 도입 (메인 확신도에 따른 보정 강도 조절)
```

### 7.3 신규 파일 목록 (예상)

```
CV/
├── scripts/
│   ├── train_binary_classifier.py   # 이진분류기 학습
│   └── binary_inference.py          # 이진분류기 추론 + 앙상블
├── configs/
│   ├── binary/
│   │   └── default.yaml             # 이진분류기 하이퍼파라미터
│   └── ensemble/
│       └── with_binary.yaml         # 이진+메인 앙상블 설정
└── src/
    └── utils/
        └── binary_ensemble.py       # 비례 재배분 로직
```

---

## 8. 위험 요소 및 완화 방안

| 위험 | 원인 | 완화 방안 |
|------|------|-----------|
| 이진분류기 과적합 | 200장 소규모 | K-Fold 5분할 + heavy augmentation + Dropout 0.4 |
| 보정으로 다른 클래스 성능 하락 | pool 외 확률 영향 | 비례 재배분: 다른 클래스 확률 불변 |
| 이진분류기가 메인보다 못한 경우 | 소규모 데이터 품질 | α 파라미터로 보정 강도 0으로 조정 가능 |
| 추론 시간 증가 | 모델 2개 실행 | EfficientNet-B0 사용으로 경량화 |
| Validation leak | K-Fold split 오류 | DataModule의 kfold split과 동일 seed 사용 |

---

## 9. 결론 및 권장 사항

### 9.1 최종 권장 전략

> **1순위**: 비례 재배분 앙상블 (B-1)
> **2순위**: Confidence Routing + 비례 재배분 조합 (A-1 + B-1)

**권장 이유**:
1. **200장이라는 소규모 데이터**에서 Contrastive Learning(C-1, C-2)은 충분한 배치 다양성 확보 어려움
2. **비례 재배분**은 이진분류기의 품질에 비례하여 성능이 결정됨 — 낮은 위험
3. **K-Fold 5분할** 이진분류기 앙상블로 소규모 데이터의 분산을 최소화
4. 메인 파이프라인을 **전혀 변경하지 않아도** 추론 단계에서만 적용 가능

### 9.2 구현 전 체크리스트

- [ ] 클래스 3·7 훈련 이미지 시각적 검토 (실제로 어떻게 다른지 확인)
- [ ] Validation set 기준으로 메인 분류기의 class 3·7 혼동 수 파악 (혼동행렬)
- [ ] 이진분류기 validation F1이 메인 분류기 class3·7 부분 accuracy보다 높은지 검증
- [ ] α=0 (보정 없음) 대비 성능 비교 (regression test)

### 9.3 예상 최종 성능

메인 파이프라인 (현재): 퍼블릭 **0.9321**
비례 재배분 앙상블 적용 후 (기대): **0.935~0.945**

> 클래스 수(17)가 많아 개별 클래스 2개의 혼동 해결이 전체 Macro F1에 미치는 수치적 영향은 제한적이나, 리더보드 순위에는 유의미한 차이를 가져올 수 있음.

---

## 10. 참고 자료

- Khosla et al. (2020). "Supervised Contrastive Learning." NeurIPS 2020.
- Bertinetto et al. (2020). "Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks." CVPR 2020.
- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML 2017.
- Deng et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
- Müller et al. (2019). "When Does Label Smoothing Help?" NeurIPS 2019.
- timm 라이브러리 문서 (huggingface/pytorch-image-models)
- RVL-CDIP Document Classification Benchmark — 문서 이미지 분류 선행 연구

---

*이 문서는 리서치/검토 단계의 자료입니다. 개발 착수는 검토 승인 후 진행합니다.*
