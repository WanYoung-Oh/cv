# /experiment - 실험 설정 생성

새로운 실험을 위한 설정 파일을 생성합니다.

## 입력 받기

1. **모델 선택**
   - efficientnet_b0, efficientnet_b1, resnet50 등

2. **이미지 크기**
   - 224, 384, 512 등

3. **실험 목적**
   - baseline, augmentation, ensemble 등

## 작업 순서

1. **날짜 기반 폴더 생성**
   - `configs/experiments/YYYYMMDD_실험명/` 생성

2. **설정 파일 복사**
   - 기본 config.yaml 복사
   - 사용자 입력에 맞게 수정

3. **실행 명령어 생성**
   - Hydra override 명령어 제공

4. **문서화**
   - `docs/01-plan/실험명.md` 생성
   - 실험 목적, 가설, 예상 결과 작성

## 출력 형식

```markdown
# 실험: {실험명}

**날짜**: 2026-02-13
**모델**: EfficientNet-B1
**이미지 크기**: 384

## 가설
- 더 큰 이미지 크기로 세밀한 문서 구조 인식 향상

## 실행 명령어
\`\`\`bash
python src/train.py \
  model=efficientnet_b1 \
  data.image_size=384 \
  training.max_epochs=50
\`\`\`

## 예상 결과
- F1-Macro: 0.88+ (기존 0.86 대비 향상)
```
