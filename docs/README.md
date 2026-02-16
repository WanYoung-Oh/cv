# 문서 이미지 분류 프로젝트

> PyTorch Lightning + Hydra + WanDB 기반 고성능 문서 이미지 분류 시스템

## 🎉 프로젝트 완료

**달성 성과**: F1-Macro **0.993** (목표 0.88 대비 **+13% 초과 달성**)

| Metric | 목표 | 달성 | 상태 |
|--------|------|------|------|
| F1-Macro | 0.88+ | **0.993** | ✅ +13% |
| Accuracy | - | **0.994** | ✅ |
| Val F1 | - | **0.993** | ✅ |

**Best 모델**: ResNet34 + baseline_aug (768×768)

---

## 📚 문서 구조

### 주요 가이드
- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - 프로젝트 완료 가이드 (필수 읽기 ⭐)
  - 달성 성과
  - 사용 가능한 모델
  - 훈련/Inference 방법
  - 핵심 성공 요인
  - 다음 단계 (선택사항)

### PDCA 문서 (완료)
- **[data-augmentation-strategy.md](02-design/data-augmentation-strategy.md)** - 설계 문서
- **[CV.analysis.md](03-analysis/CV.analysis.md)** - Gap Analysis (Match Rate 95%)
- **[CV.report.md](04-report/CV.report.md)** - 완료 보고서

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
conda activate pytorch_test
pip install -r requirements.txt
```

### 2. Best 모델 훈련 (ResNet34)
```bash
python src/train.py \
  data=baseline_aug \
  model=resnet34 \
  training=baseline_768
```

### 3. Inference (리더보드 제출)
```bash
python src/inference.py checkpoint=checkpoints/champion/best_model.ckpt
# 출력: submission.csv
```

---

## 🎯 핵심 성공 요인

1. **고해상도 입력** (768×768) - 문서 세부 정보 보존
2. **Aspect Ratio 보존** - LongestMaxSize + PadIfNeeded
3. **문서 특화 Augmentation** - CLAHE, Perspective, Sharpen 등
4. **Class Weights** - 불균형 데이터 처리

---

## 📊 프로젝트 구조

```
CV/
├── configs/          # Hydra 설정
│   ├── data/        # baseline_aug.yaml, transformer_384.yaml
│   ├── model/       # resnet34, swin_base_384, deit_base_384 등
│   └── training/    # baseline_768.yaml
├── src/
│   ├── data/        # DataModule
│   ├── models/      # LightningModule
│   ├── train.py     # 훈련
│   └── inference.py # 추론
├── scripts/
│   └── analyze_results.py  # 결과 분석
├── checkpoints/     # 모델 체크포인트
│   └── champion/    # Best 모델
└── docs/            # 문서
    ├── PROJECT_GUIDE.md    # 메인 가이드 ⭐
    ├── 02-design/          # 설계 문서
    ├── 03-analysis/        # Gap Analysis
    └── 04-report/          # 완료 보고서
```

---

## 💡 다음 단계

### 옵션 A: 완료 (추천)
- F1 0.993으로 목표 초과 달성
- 리더보드 제출

### 옵션 B: TTA + Ensemble (선택)
- 예상 성능: F1 0.995+
- ROI: 낮음 (+0.2~0.4%)

### 옵션 C: Transformer 실험 (선택)
- Swin-384, DeiT-384
- ROI: 매우 낮음

자세한 내용은 **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**를 참고하세요.

---

**프로젝트 완료일**: 2026-02-15
**최종 성과**: F1 0.993 / Accuracy 0.994
