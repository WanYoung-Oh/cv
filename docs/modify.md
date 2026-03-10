기능 수정 및 문서(OPERATION.md) 업데이트 대상 

## 1. Ensemble TTA: tta_level 미지원

- **문서**: `+ensemble.tta_level=heavy` 예시
- **실제**: ensemble.py가 tta_level을 읽지 않음 → 항상 standard

**조치**: tta_level 지원 기능 추가 및 문서 업데이트

---

## 2. analyze_results.py 인자 OPERATION.md 문서에 추가 

- **실제**: --checkpoint-dir, --data-root, --batch-size 등 추가 인자 존재
- **조치**: 전체 인자 목록 추가

---

## 3. Ensemble config 지정 방법 OPERATION.md 문서에 추가 

- **조치**: `ensemble=ensemble_384_3models` 등 config 기반 지정 예시 추가

---

## 4. Local Minimum 탈출을 위해 CosineAnnealingWarmRestarts 반영

- **조치**: configs/training/transformer.yaml : scheduler 옵션(기본은 cosine), T_0, T_mult 추가
- **조치**: src/models/module.py :CosineAnnealingWarmRestarts 지원 추가
- **조치**: src/train.py : T_0, T_mult를 모듈에 전달 

- **조치**: OPERATION.md 문서에 실행방법 및 설정 반영 내용 추가
  (T_mult=1: 모든 주기 동일 길이, T_mult=2: 후반부 주기가 길어져 fine-tuning에 유리, T_0: 너무 작으면 학습이 불안정해질 수 있어 10~20 정도를 권장 등 설정 참고사항 추가)

---