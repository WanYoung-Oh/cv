# /format - 코드 포맷팅

Black 포맷터로 전체 코드를 포맷팅합니다.

## 작업 순서

1. **Black 설치 확인**
   - `pip list | grep black` 실행
   - 없으면 설치 안내

2. **포맷팅 실행**
   - `black src/ scripts/` 실행
   - 변경된 파일 목록 표시

3. **결과 확인**
   - 포맷팅된 파일 수 보고
   - Git diff로 변경 내용 확인 제안

## 실행 예시

```bash
# 전체 포맷팅
black src/ scripts/

# 특정 파일만
black src/train.py

# Dry-run (변경 미리보기)
black --check src/
```

## 설정

프로젝트 루트에 `pyproject.toml` 생성 권장:

```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
```
