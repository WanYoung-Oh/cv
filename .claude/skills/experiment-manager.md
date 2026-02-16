# Experiment Manager Skill

> 딥러닝 실험 관리 전문가

## 역할

PyTorch Lightning + Hydra + WanDB 환경에서 실험을 설계하고 추적합니다.

---

## 핵심 지식

### 1. 실험 설계 패턴

```python
# 실험 네이밍 규칙
{date}_{model}_{image_size}_{augmentation}_{특징}

# 예시
20260213_efficientnet_b1_384_cutout_baseline
```

### 2. Hydra 설정 오버라이드

```bash
# 기본 템플릿
python src/train.py \
  model={model_config} \
  data.image_size={size} \
  data.batch_size={batch} \
  training.max_epochs={epochs}

# 예시
python src/train.py \
  model=efficientnet_b1 \
  data.image_size=384 \
  data.batch_size=16 \
  training.max_epochs=50
```

### 3. WanDB 태그 전략

```python
tags = [
    f"model:{cfg.model.name}",
    f"size:{cfg.data.image_size}",
    f"aug:{cfg.data.augmentation}",
    "baseline" if is_baseline else "experiment"
]
```

### 4. 실험 체크리스트

- [ ] 가설 명확히 정의
- [ ] Baseline 성능 확인
- [ ] 하이퍼파라미터 범위 설정
- [ ] 예상 결과 기록
- [ ] WanDB 프로젝트 설정
- [ ] .env 파일 확인
- [ ] Git 커밋 (실험 전)

---

## 실험 타입

### Type 1: Baseline
- **목적**: 기본 성능 측정
- **변수**: 모델만 변경
- **메트릭**: F1-Macro, Accuracy

### Type 2: Augmentation
- **목적**: 데이터 증강 효과 검증
- **변수**: Augmentation 기법
- **비교**: Baseline vs New

### Type 3: Ensemble
- **목적**: 앙상블 성능 향상
- **변수**: Voting 방식 (Hard/Soft/Rank)
- **요구사항**: 최소 3개 모델

### Type 4: Multi-scale
- **목적**: 이미지 크기 영향 분석
- **변수**: Image size
- **범위**: 224, 384, 512, 640

---

## 실험 문서화 템플릿

```markdown
# 실험: {실험명}

## 메타데이터
- **날짜**: YYYY-MM-DD
- **타입**: [Baseline/Augmentation/Ensemble/Multi-scale]
- **WanDB Run**: [링크]

## 가설
{무엇을 검증하려고 하는가?}

## 설정
\`\`\`yaml
model:
  name: efficientnet_b1
  pretrained: true

data:
  image_size: 384
  batch_size: 16
  augmentation: cutout

training:
  max_epochs: 50
  optimizer: adamw
  lr: 1e-4
\`\`\`

## 예상 결과
- Baseline F1: 0.86
- Expected F1: 0.88+
- 근거: [이유]

## 실제 결과
- F1-Macro: {결과}
- Accuracy: {결과}
- Best Epoch: {결과}

## 분석
{결과가 예상과 다른 이유, 인사이트}

## 다음 단계
- [ ] {후속 실험}
- [ ] {개선 방향}
```

---

## 자주 사용하는 명령어

### 실험 시작
```bash
# 1. 실험 브랜치 생성
git checkout -b exp/efficientnet-b1-384

# 2. 설정 파일 수정
# configs/model/efficientnet_b1.yaml

# 3. 훈련 실행
python src/train.py model=efficientnet_b1 data.image_size=384
```

### 실험 비교 (WanDB)
```python
import wandb

api = wandb.Api()
runs = api.runs("username/doc_image_classification")

# F1 점수로 정렬
sorted_runs = sorted(runs, key=lambda x: x.summary.get("val_f1", 0), reverse=True)
```

### 체크포인트 관리
```bash
# 최고 성능 모델 찾기
ls -lt checkpoints/ | head -5

# 챔피언 모델 복사
cp checkpoints/epoch=42-val_f1=0.885.ckpt checkpoints/champion.ckpt
```

---

## 주의사항

### ❌ 하지 말아야 할 것
- 실험 중간에 코드 변경 (재현 불가)
- WanDB 로깅 없이 훈련 (결과 추적 불가)
- Git 커밋 없이 여러 실험 (버전 관리 불가)
- .env 파일 커밋 (보안 위험)

### ✅ 해야 할 것
- 실험마다 별도 브랜치 생성
- 결과를 docs/04-report/에 문서화
- 챔피언 모델 주기적으로 업데이트
- 실패한 실험도 기록 (중복 방지)

---

## 트러블슈팅

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python src/train.py data.batch_size=8

# Gradient accumulation 사용
python src/train.py training.accumulate_grad_batches=4
```

### WanDB 연결 실패
```bash
# API 키 재설정
wandb login

# .env 파일 확인
cat .env | grep WANDB
```

### 체크포인트 로딩 오류
```python
# 모델 구조 변경 시 strict=False
model = Model.load_from_checkpoint(ckpt_path, strict=False)
```

---

## 성능 목표

| 모델 | F1-Macro | Accuracy | 비고 |
|------|----------|----------|------|
| EfficientNet-B0 (224) | 0.84 | 0.85 | Baseline |
| EfficientNet-B1 (384) | 0.86 | 0.87 | Target |
| EfficientNet-B3 (512) | 0.88+ | 0.89+ | Stretch |
| Ensemble (3 models) | 0.90+ | 0.91+ | Final |

---

**이 Skill은 실험 관련 요청 시 자동으로 활성화됩니다.**
