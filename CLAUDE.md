# CV 프로젝트 개발 가이드

> PyTorch Lightning + Hydra + WanDB 기반 문서 이미지 분류 시스템

---

## 🎯 프로젝트 목표
- F1-Macro Score **0.88+** 달성
- Production-ready 코드 품질 유지
- 실험 재현 가능성 보장

---

## 📦 Package Management

### 환경 관리
- **가상환경**: `conda activate pytorch_test`
- **의존성 설치**: `pip install -r requirements.txt`
- **버전 고정**: requirements.txt에 명시된 버전 범위 준수

### 규칙
- ✅ pip로 패키지 설치
- ✅ 새 패키지 추가 시 requirements.txt 업데이트
- ❌ conda install 사용 금지 (pip 우선)
- ❌ 버전 명시 없이 패키지 추가 금지

---

## 💻 Coding Conventions

### 코드 스타일
- **포맷터**: black (설치 후 적용 권장)
- **타입 힌트**: 함수 시그니처에 필수 적용
- **Docstring**: Google 스타일 권장

```python
def train_model(
    config: DictConfig,
    datamodule: LightningDataModule,
    model: LightningModule
) -> dict:
    """모델을 훈련하고 결과를 반환합니다.

    Args:
        config: Hydra 설정 객체
        datamodule: 데이터 모듈
        model: Lightning 모델

    Returns:
        훈련 결과 딕셔너리 (metrics, checkpoints 등)
    """
    pass
```

### 네이밍 규칙
- **함수/변수**: snake_case (예: `train_model`, `best_f1_score`)
- **클래스**: PascalCase (예: `DocumentDataModule`, `EfficientNetClassifier`)
- **상수**: UPPER_SNAKE_CASE (예: `MAX_EPOCHS`, `NUM_CLASSES`)
- **Private**: underscore prefix (예: `_internal_method`)

---

## 📁 프로젝트 구조

```
CV/
├── configs/          # Hydra 설정 파일
│   ├── config.yaml  # 메인 설정
│   ├── data/        # 데이터셋 설정
│   ├── model/       # 모델 아키텍처 설정
│   └── training/    # 훈련 하이퍼파라미터
├── src/
│   ├── data/        # 데이터 로딩 (DataModule)
│   ├── models/      # 모델 정의 (LightningModule)
│   ├── utils/       # 유틸리티 함수
│   ├── train.py     # 훈련 스크립트
│   ├── inference.py # 추론 스크립트
│   └── ensemble.py  # 앙상블 스크립트
├── scripts/         # 분석/전처리 스크립트
├── datasets_fin/    # 데이터셋 (gitignore)
├── checkpoints/     # 모델 체크포인트 (gitignore)
└── logs/            # 로그 파일 (gitignore)
```

### 파일 위치 규칙
- **새로운 모델**: `src/models/` 에 추가
- **새로운 데이터 로더**: `src/data/` 에 추가
- **유틸리티 함수**: `src/utils/` 에 추가
- **실험 스크립트**: `scripts/` 에 추가
- **설정 파일**: `configs/` 에 추가

---

## ⚙️ Configuration Management (Hydra)

### 설정 파일 수정
- **직접 수정**: `configs/*.yaml` 파일 편집
- **CLI 오버라이드**: `python src/train.py model.name=efficientnet_b0`
- **새 설정 추가**: 기존 그룹 구조 유지

### 설정 접근
```python
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # cfg.model.name
    # cfg.data.batch_size
    # cfg.training.max_epochs
    pass
```

### 규칙
- ✅ 모든 하이퍼파라미터는 Hydra config로 관리
- ✅ 환경 변수는 .env 파일 사용
- ❌ 코드에 하드코딩된 경로/값 금지
- ❌ argparse 사용 금지 (Hydra로 통일)

---

## 📊 Experiment Tracking (WanDB)

### 환경 변수 설정
```bash
# .env 파일에 추가
WANDB_API_KEY=your-api-key-here
WANDB_PROJECT=doc_image_classification
WANDB_ENTITY=your-username
```

### Logger 사용
```python
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project=cfg.wandb.project,
    name=f"{cfg.model.name}_{cfg.data.image_size}",
    config=OmegaConf.to_container(cfg, resolve=True)
)

trainer = Trainer(logger=wandb_logger)
```

### 규칙
- ✅ 모든 실험은 WanDB에 로깅
- ✅ Run name은 `{model}_{size}_{특징}` 형식
- ✅ Config는 완전히 로깅 (재현성)
- ❌ .env 파일은 git에 커밋 금지

---

## 🚫 Prohibited

### 코드
- ❌ `print()` 사용 금지 → `logging` 또는 `rich.print()` 사용
- ❌ `global` 변수 사용 금지
- ❌ 하드코딩된 경로 금지 → config 사용
- ❌ 매직 넘버 금지 → 상수로 정의
- ❌ Try-except 남발 금지 → 필요한 곳에만 사용

### 데이터
- ❌ `datasets_fin/` 직접 수정 금지
- ❌ 원본 데이터 덮어쓰기 금지
- ❌ .csv 파일 직접 편집 금지 → 스크립트로 관리

### Git
- ❌ `.env` 파일 커밋 금지
- ❌ `checkpoints/` 커밋 금지
- ❌ `datasets_fin/` 커밋 금지
- ❌ `__pycache__/` 커밋 금지

---

## 🔧 개발 워크플로우

### 1. 새로운 실험 시작
```bash
# 1. 설정 파일 수정 (configs/)
# 2. 코드 수정 (src/)
# 3. 훈련 실행
python src/train.py

# 4. 결과 확인 (WanDB)
# 5. 체크포인트 확인 (checkpoints/)
```

### 2. 코드 수정 시
```bash
# 1. 기능 브랜치 생성
git checkout -b feature/new-augmentation

# 2. 코드 작성 + 테스트
# 3. (Optional) 포맷팅
black src/

# 4. 커밋
git add .
git commit -m "Add new augmentation strategy"
```

### 3. 실험 재현
```bash
# WanDB에서 config 확인
# config.yaml 동일하게 설정
python src/train.py
```

---

## 📝 Notes

### PyTorch Lightning 패턴
- **DataModule**: 데이터 로딩 로직 캡슐화
- **LightningModule**: 모델 + 훈련/검증 로직
- **Trainer**: 훈련 루프 자동화
- **Callbacks**: EarlyStopping, ModelCheckpoint 등

### Hydra 패턴
- **Compositional**: 설정을 조합하여 사용
- **Override**: CLI에서 동적으로 변경
- **Structured**: OmegaConf로 타입 안전성 보장

### 성능 최적화
- `torch.compile()` 사용 (PyTorch 2.0+)
- Mixed precision training (AMP)
- DataLoader num_workers 조정
- Gradient accumulation (메모리 부족 시)

---

## 🤖 AI 협업 가이드

### Claude에게 요청할 때
- ✅ "Hydra config에 새로운 augmentation 추가해줘"
- ✅ "EfficientNet-B1 모델 추가하고 WanDB 로깅 설정해줘"
- ✅ "src/data/transforms.py에 Cutout augmentation 구현해줘"
- ❌ "빨리 코드 짜줘" (구체적이지 않음)

---

### Claude Behavioral guidelines

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

**1. Think Before Coding**

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**2. Simplicity First**

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**3. Surgical Changes**

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

**4. Goal-Driven Execution**

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---
### Claude가 실수할 때
- 이 문서에 규칙 추가
- 팀원과 공유
- 반복 방지
---

**마지막 업데이트**: 2026-02-14
