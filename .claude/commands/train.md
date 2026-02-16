# /train - 모델 훈련 실행

새로운 모델을 훈련합니다.

## 작업 순서

1. **설정 파일 확인**
   - `configs/config.yaml` 읽기
   - 사용자에게 현재 설정 요약 보여주기

2. **환경 변수 확인**
   - `.env` 파일 존재 확인
   - WANDB_API_KEY 설정 여부 확인

3. **훈련 실행**
   - `python src/train.py` 실행
   - 로그 출력 모니터링

4. **결과 확인**
   - WanDB 링크 제공
   - 체크포인트 저장 위치 안내

## 실행 예시

```bash
# 기본 훈련
python src/train.py

# 특정 모델로 훈련
python src/train.py model=efficientnet_b1

# 배치 크기 변경
python src/train.py data.batch_size=32
```

## 주의사항

- .env 파일이 없으면 먼저 생성 안내
- GPU 메모리 부족 시 배치 크기 줄이기 권장
- WanDB 로그인 필요 시 안내
