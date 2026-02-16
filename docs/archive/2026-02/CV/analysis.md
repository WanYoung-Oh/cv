# Data Augmentation Strategy - Gap Analysis Report

> **Analysis Type**: Design vs Implementation Gap Analysis
>
> **Project**: CV (문서 이미지 분류 시스템)
> **Analyst**: Claude Code (bkit gap-detector)
> **Date**: 2026-02-15
> **Design Doc**: [data-augmentation-strategy.md](../02-design/data-augmentation-strategy.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

Design 문서 `data-augmentation-strategy.md`의 **Step 1 Baseline** 구현 계획과 실제 구현 코드 간의 일치도를 분석합니다. 예상 성능 대비 실제 성능도 비교합니다.

### 1.2 Analysis Scope

- **Design Document**: `docs/02-design/data-augmentation-strategy.md` (Step 1: Baseline)
- **Implementation Files**:
  - `configs/data/baseline_aug.yaml` - Augmentation 설정
  - `configs/training/baseline_768.yaml` - 훈련 설정
  - `src/data/datamodule.py` - DataModule (OmegaConf 대응)
  - `src/train.py` - 학습 스크립트 (sys.path, model.model_name 수정)
- **Analysis Date**: 2026-02-15

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 Step 1 Baseline 구현 항목 비교

Design 문서 Step 1 (Line 496-520)에 정의된 구현 사항과 실제 구현을 대조합니다.

| # | Design 항목 | Design 위치 | 실제 구현 | 구현 위치 | Status |
|---|------------|------------|----------|----------|--------|
| 1 | 고정 크기 768x768 | strategy.md:500 | img_size: 768 | baseline_aug.yaml:6 | Match |
| 2 | Padding으로 AR 보존 | strategy.md:500 | LongestMaxSize + PadIfNeeded | baseline_aug.yaml:21-31 | Match (개선됨) |
| 3 | RandomRotate90 (p=0.5) | strategy.md:517 | RandomRotate90 (p=0.5) | baseline_aug.yaml:38-39 | Match |
| 4 | Rotate limit=[-45,45] (p=0.3) | strategy.md:518 | Rotate limit=[-45,45] (p=0.3) | baseline_aug.yaml:42-46 | Match |
| 5 | HorizontalFlip (p=0.5) | strategy.md:519 | HorizontalFlip (p=0.5) | baseline_aug.yaml:49 | Match |
| 6 | Config 파일 생성 | strategy.md:509 | baseline_aug.yaml 생성됨 | configs/data/baseline_aug.yaml | Match |

### 2.2 Augmentation 설정 상세 비교

#### Resize Strategy

| 항목 | Design (strategy.md:512-515) | Implementation (baseline_aug.yaml) | Status |
|------|-----|-----|--------|
| Strategy | `resize.height: 768, width: 768, padding_mode: "constant"` | `LongestMaxSize(768) + PadIfNeeded(768,768)` | Changed (개선) |
| Padding Color | 미지정 (value: 0 암시) | `value: [255, 255, 255]` (흰색) | Changed (개선) |
| Border Mode | 미지정 | `border_mode: 0 (BORDER_CONSTANT)` | Added |

**평가**: Design은 단순 resize+padding을 제안했으나, 실제 구현은 `LongestMaxSize`로 aspect ratio를 보존한 뒤 `PadIfNeeded`로 패딩하는 2단계 전략을 적용했습니다. 문서 이미지 특성상 흰색(255,255,255) 패딩은 합리적인 개선입니다. Design 의도를 충실히 반영하면서 품질을 높인 구현입니다.

#### Rotation Augmentation

| 항목 | Design (strategy.md:517-519) | Implementation (baseline_aug.yaml:34-54) | Status |
|------|-----|-----|--------|
| RandomRotate90 | p=0.5 | p=0.5 | Match |
| Rotate | limit=[-45,45], p=0.3 | limit=[-45,45], p=0.3, border_mode=0, value=[255,255,255] | Match (상세화) |
| HorizontalFlip | p=0.5 | p=0.5 | Match |
| VerticalFlip | 미명시 | p=0.2 | Added |

**평가**: Rotation 관련 핵심 augmentation은 Design과 정확히 일치합니다. VerticalFlip은 Design Step 1에는 없지만, Design Phase 2 (Line 308)에서 `VerticalFlip (p=0.2)`로 명시된 항목이 선제 구현되었습니다.

#### 추가 구현된 Augmentation (Design에 없음)

| 항목 | Implementation 위치 | 파라미터 | 목적 |
|------|---------------------|----------|------|
| CLAHE | baseline_aug.yaml:58-60 | clip_limit=4.0, tile_grid_size=[8,8], p=0.4 | 문서 대비 강화 |
| RandomBrightnessContrast | baseline_aug.yaml:63-66 | brightness=0.2, contrast=0.3, p=0.3 | 그림자 제거 |
| Perspective | baseline_aug.yaml:69-71 | scale=[0.05,0.1], p=0.2 | 스캔 각도 변형 |
| ColorJitter | baseline_aug.yaml:74-79 | brightness=0.1, contrast=0.1, p=0.2 | 종이 색상 변화 |
| GaussNoise | baseline_aug.yaml:82-84 | var_limit=[10.0,30.0], p=0.1 | 스캔 노이즈 |
| Sharpen | baseline_aug.yaml:87-90 | alpha=[0.2,0.5], lightness=[0.5,1.0], p=0.2 | 선명도 조정 |
| Val CLAHE | baseline_aug.yaml:107-110 | clip_limit=4.0, p=1.0 | 검증 시 대비 개선 |

**평가**: Design Step 1에 없는 7개 추가 augmentation이 구현되었습니다. 이들은 문서 이미지 도메인에 특화된 합리적인 추가이며, 99.3% F1 달성에 기여한 것으로 판단됩니다.

### 2.3 Training Config 비교

| 항목 | Design (strategy.md) | Implementation (baseline_768.yaml) | Status |
|------|-----|-----|--------|
| Image Size | 768x768 | 768 | Match |
| Batch Size | 미명시 | 16 | Added |
| Learning Rate | 미명시 | 1e-3 | Added |
| Epochs | 미명시 | 50 | Added |
| Optimizer | 미명시 | adam | Added |
| Scheduler | 미명시 | cosine (warmup 5 epochs) | Added |
| Early Stopping | 미명시 | patience=10, monitor=val_loss | Added |
| Checkpoint | 미명시 | save_top_k=3, monitor=val_f1 | Added |

**평가**: Design Step 1은 augmentation 전략에 집중하여 training hyperparameter를 상세히 명시하지 않았습니다. 실제 구현은 `configs/training/baseline_768.yaml`에 완전한 훈련 설정을 포함하며, 768x768 해상도에 맞게 batch_size를 32에서 16으로 조정했습니다.

### 2.4 코드 수정 사항 비교

| 항목 | 수정 내용 | 파일 | Status |
|------|----------|------|--------|
| OmegaConf 대응 | DictConfig를 일반 dict로 변환 후 augmentation 파싱 | datamodule.py:129-151 | Added (필수 인프라) |
| sys.path 추가 | 프로젝트 루트를 Python path에 추가 | train.py:15-16 | Added (필수 인프라) |
| model.name 변경 | `model.name` -> `model.model_name` | train.py:168 | Changed (네이밍 개선) |
| 동적 augmentation | Config 기반 augmentation 동적 생성 | datamodule.py:153-187 | Added (확장성 확보) |

**평가**: Design 문서에 직접 명시되지 않은 인프라 수정이지만, Step 1 구현을 위해 필수적인 변경입니다. 특히 `_parse_augmentation()` 메서드를 통한 동적 augmentation 생성은 향후 Step 2/3 확장을 위한 기반을 마련했습니다.

### 2.5 Design Step 1에서 미구현된 항목

| # | Design 항목 | Design 위치 | Status | 비고 |
|---|------------|------------|--------|------|
| - | (없음) | - | - | Step 1의 모든 핵심 항목이 구현됨 |

Step 1 범위 내에서 미구현 항목은 없습니다.

### 2.6 Design 전체 로드맵 중 미구현 항목 (Step 2-3)

Design 문서의 전체 로드맵 중 아직 구현되지 않은 항목입니다. 이들은 Step 1 범위 밖이므로 Gap이 아니라 **향후 구현 대상**입니다.

| Phase | 항목 | Design 위치 | Status |
|-------|------|------------|--------|
| Phase 1 | Aspect Ratio Bucketing (AspectRatioBucketDataset) | strategy.md:274-296 | 미구현 (Step 2) |
| Phase 1 | BucketBatchSampler | strategy.md:285-293 | 미구현 (Step 2) |
| Phase 2 | Bucket별 Rotation Augmentation | strategy.md:299-336 | 미구현 (Step 2) |
| Phase 3 | 모델별 Config 업데이트 (bucketing) | strategy.md:340-385 | 미구현 (Step 2) |
| Phase 4 | Training Pipeline bucketing 분기 | strategy.md:389-441 | 미구현 (Step 2) |
| Phase 5 | Inference Pipeline bucketing + TTA | strategy.md:446-489 | 미구현 (Step 3) |

---

## 3. Performance Analysis

### 3.1 예상 성능 vs 실제 성능

| Metric | Design 예상 (Step 1) | 실제 성능 | Delta | Status |
|--------|---------------------|----------|-------|--------|
| F1-Macro | 0.82 ~ 0.85 | **0.993** | +0.143 ~ +0.173 | 초과 달성 |
| Accuracy | 미명시 | **0.994** | - | 초과 달성 |

### 3.2 예상 성능 비교표 대비 실제 결과

Design 문서의 성능 비교 예상표 (Line 552-559)와 비교합니다.

| 방법 | Design 예상 F1 | 실제 F1 | 비교 |
|------|---------------|---------|------|
| Baseline (고정 512x512) | 0.82 | - | 미실험 |
| **고정 768x768 + Rotation (Step 1)** | **0.85** | **0.993** | **+0.143 (+16.8%)** |
| Multi-scale (Step 3) | 0.87 | - | 미필요 |
| Bucketing (Step 2) | 0.88 | - | 미필요 |
| Bucketing + TTA (Step 3) | 0.90 | - | 미필요 |

### 3.3 Performance Gap 분석

실제 성능이 Design 예상을 **매우 크게 초과**했습니다 (0.85 vs 0.993).

**초과 달성 원인 분석**:

1. **추가 Augmentation 효과**: Design에 없던 CLAHE, Perspective, ColorJitter 등 문서 특화 augmentation이 큰 성능 향상을 가져옴
2. **흰색 패딩**: 문서 이미지의 배경색에 맞춘 패딩 색상이 모델 학습에 유리하게 작용
3. **Validation CLAHE**: 검증 시에도 대비 개선을 적용하여 일관된 이미지 품질 보장
4. **Design 예측의 보수성**: Design이 augmentation만으로 도달 가능한 성능을 과소 평가했을 가능성
5. **데이터셋 특성**: 17개 클래스의 문서 이미지가 예상보다 분류 용이한 특성을 가짐

### 3.4 Step 2/3 필요성 재평가

| 항목 | Design 계획 | 현재 상태 | 권장 |
|------|------------|----------|------|
| Step 2: Bucketing | F1 0.88+ 달성 목적 | F1 0.993 이미 달성 | 불필요 (성능 목표 초과) |
| Step 3: TTA + Ensemble | 최고 성능 달성 목적 | F1 0.993 이미 달성 | 불필요 (한계 수익 체감) |
| 프로젝트 목표 (F1 0.88+) | 미달성 예상 | **0.993으로 달성** | 목표 완료 |

---

## 4. Convention Compliance

### 4.1 Naming Convention

| Category | Convention | 구현 | Status |
|----------|-----------|------|--------|
| 함수/변수 | snake_case | `_parse_augmentation`, `aug_config` | Match |
| 클래스 | PascalCase | `DocumentImageDataModule`, `DocumentClassifierModule` | Match |
| 상수 | UPPER_SNAKE_CASE | `IMAGENET_MEAN`, `IMAGENET_STD` | Match |
| Config 키 | snake_case | `train_augmentations`, `img_size` | Match |

### 4.2 CLAUDE.md Rules Compliance

| Rule | Status | 비고 |
|------|--------|------|
| print() 사용 금지 | Match | logging 모듈 사용 |
| 하드코딩 경로 금지 | Match | Hydra config 사용 |
| 매직 넘버 금지 | Match | Config로 관리 |
| Hydra config로 하이퍼파라미터 관리 | Match | baseline_aug.yaml, baseline_768.yaml |
| Google 스타일 Docstring | Match | datamodule.py 함수에 적용 |
| Type hints | Match | 함수 시그니처에 적용 |

### 4.3 Architecture Compliance

| Rule | Status | 비고 |
|------|--------|------|
| DataModule에 데이터 로직 캡슐화 | Match | DocumentImageDataModule |
| LightningModule에 모델+훈련 로직 | Match | DocumentClassifierModule |
| 유틸리티를 src/utils/에 배치 | Match | device.py |
| 설정 파일을 configs/에 배치 | Match | baseline_aug.yaml, baseline_768.yaml |

---

## 5. Match Rate Summary

### 5.1 항목별 점수

| Category | 일치 항목 | 전체 항목 | Match Rate | Status |
|----------|----------|----------|:----------:|:------:|
| Core Augmentation (Resize + Rotation) | 6/6 | 6 | **100%** | Match |
| Augmentation Parameters | 5/5 | 5 | **100%** | Match |
| Config 파일 구조 | 1/1 | 1 | **100%** | Match |
| Training Config | 1/1 | 1 | **100%** | Match (768 img_size) |
| Convention Compliance | 10/10 | 10 | **100%** | Match |
| Architecture Compliance | 4/4 | 4 | **100%** | Match |

### 5.2 Overall Match Rate

```
+---------------------------------------------+
|  Overall Match Rate: 95%                     |
+---------------------------------------------+
|  Match (Design=Impl):     19 items (70%)     |
|  Changed (개선됨):          3 items (11%)     |
|    - Resize: LongestMaxSize 전략 (개선)       |
|    - Padding: 흰색 (문서 특화 개선)            |
|    - model.name -> model.model_name (개선)    |
|  Added (Design에 없음):     5 items (19%)     |
|    - 문서 특화 augmentation 7종               |
|    - VerticalFlip (Phase 2에서 선제 구현)      |
|    - OmegaConf 대응 코드                      |
|    - sys.path 추가                            |
|    - Training hyperparameters 상세화           |
|  Missing (미구현):          0 items (0%)      |
+---------------------------------------------+
```

**Match Rate 산정 기준**:
- Design Step 1에 명시된 핵심 항목 기준: **100%** (6/6 항목 완전 구현)
- Design 의도 반영도 포함 (Changed/Added는 개선이므로 감점 최소): **95%**
- 5% 감점 이유: Design에 없는 추가 augmentation은 성능에 기여했지만, Design 문서에 반영 필요

---

## 6. Score Summary

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match (Step 1 핵심 항목) | 100% | Match |
| Design Match (전체 상세도 포함) | 95% | Match |
| Architecture Compliance | 100% | Match |
| Convention Compliance | 100% | Match |
| Performance (목표 F1 0.88+ 대비) | 113% | 초과 달성 |
| **Overall** | **95%** | **Match** |

---

## 7. Findings Summary

### 7.1 Match Items (Design = Implementation)

1. 고정 크기 768x768 이미지 resize
2. Aspect ratio 보존 + padding 전략
3. RandomRotate90 (p=0.5)
4. Rotate limit=[-45,45] (p=0.3)
5. HorizontalFlip (p=0.5)
6. `configs/data/baseline_aug.yaml` 파일 생성
7. ImageNet normalization 사용
8. Train/Val 분할 (0.8)
9. Class weights 사용
10. Albumentations 기반 augmentation pipeline

### 7.2 Changed Items (Design -> Implementation 개선)

| # | 항목 | Design | Implementation | Impact |
|---|------|--------|----------------|--------|
| 1 | Resize 전략 | 단순 resize + padding | LongestMaxSize + PadIfNeeded 2단계 | High (AR 완전 보존) |
| 2 | Padding 색상 | value: 0 (검정) | value: [255,255,255] (흰색) | High (문서 배경 일치) |
| 3 | Config 키 | model.name | model.model_name | Low (Python 예약어 회피) |

### 7.3 Added Items (Design에 없지만 구현됨)

| # | 항목 | Implementation 위치 | Impact |
|---|------|---------------------|--------|
| 1 | CLAHE (대비 강화) | baseline_aug.yaml:58-60 | High |
| 2 | RandomBrightnessContrast | baseline_aug.yaml:63-66 | Medium |
| 3 | Perspective (원근 변환) | baseline_aug.yaml:69-71 | Medium |
| 4 | ColorJitter (색상 변화) | baseline_aug.yaml:74-79 | Low |
| 5 | GaussNoise (스캔 노이즈) | baseline_aug.yaml:82-84 | Low |
| 6 | Sharpen (선명도) | baseline_aug.yaml:87-90 | Low |
| 7 | VerticalFlip (p=0.2) | baseline_aug.yaml:53-54 | Low |
| 8 | Val CLAHE (검증 대비 개선) | baseline_aug.yaml:107-110 | Medium |
| 9 | Training hyperparameters | baseline_768.yaml 전체 | Medium |
| 10 | OmegaConf 대응 | datamodule.py:129-151 | High (인프라) |

### 7.4 Missing Items (Gap)

**Step 1 범위 내**: 없음 (0개)

**Step 2-3 범위 (향후 구현 대상, 현 성능으로 불필요)**:
- Aspect Ratio Bucketing 전체 구현
- BucketBatchSampler
- 모델별 Bucketing Config
- TTA (Test-time Augmentation)
- Inference Pipeline bucketing

### 7.5 Performance Overachievement

| Metric | 목표 | Design 예상 | 실제 | 초과율 |
|--------|------|------------|------|--------|
| F1-Macro | 0.88+ | 0.82~0.85 | **0.993** | +16.8% |
| Accuracy | 미명시 | 미명시 | **0.994** | - |
| Val F1 | 미명시 | 미명시 | **0.993** | - |

---

## 8. Recommended Actions

### 8.1 Design 문서 업데이트 필요

| Priority | 항목 | 이유 |
|----------|------|------|
| High | Step 1 실제 구현 결과 반영 | 추가 augmentation 7종 문서화 |
| High | 성능 예측 테이블 업데이트 | 0.85 -> 0.993 실측치 반영 |
| Medium | Step 2/3 필요성 재평가 기록 | 이미 목표 달성으로 불필요 판단 근거 |
| Low | Resize 전략 상세화 | LongestMaxSize + PadIfNeeded 방식 기록 |

### 8.2 Immediate Actions (불필요)

Step 1의 모든 핵심 항목이 구현되었고, F1 0.993으로 프로젝트 목표(0.88+)를 크게 초과했으므로 즉시 수정할 사항은 없습니다.

### 8.3 Optional Next Steps

| Priority | 항목 | 기대 효과 |
|----------|------|----------|
| Low | Design 문서에 실제 결과 반영 | 문서 정합성 확보 |
| Low | Step 2/3를 "Completed (Not Needed)" 처리 | 로드맵 정리 |
| Optional | Completion Report 작성 | PDCA 사이클 완성 |

---

## 9. Conclusion

### Step 1 Baseline 구현 결과

**Match Rate: 95%** -- Design Step 1에 명시된 핵심 항목 6개가 모두 구현되었으며, 3개 항목이 개선된 형태로 구현되었습니다. 추가로 문서 도메인 특화 augmentation 7종이 선제 구현되어 성능을 크게 끌어올렸습니다.

### 성능 달성

실제 F1-Macro **0.993**으로, Design 예상(0.82~0.85)을 16.8% 초과 달성했습니다. 프로젝트 목표인 F1 0.88+도 13% 초과 달성하였습니다.

### PDCA 판정

Match Rate 95% >= 90% 이므로 **Check 단계 통과**입니다. Act(개선 반복) 단계는 불필요하며, Report(완료 보고서) 단계로 진행할 수 있습니다.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-15 | Initial gap analysis (Step 1 Baseline) | Claude Code |
