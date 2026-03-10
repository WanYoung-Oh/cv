#!/bin/bash

# 1. 가상환경 활성화 (본인의 경로에 맞게 수정하세요)
# 예: venv인 경우
source /data/ephemeral/home/py310/bin/activate

# 2. 학습할 폴드 리스트
FOR_FOLDS="0 1 2 3 4"

for fold in $FOR_FOLDS; do
  echo "=========================================="
  echo "Starting Training for Fold: $fold"
  echo "Active Environment: $VIRTUAL_ENV" # 현재 활성화된 환경 확인용
  echo "=========================================="

  python /data/ephemeral/home/src/train.py \
    data=transformer_384 \
    data.oversample_minority_classes=true \
    data.minority_class_ids=[1,13,14] \
    data.use_kfold=true \
    data.fold_idx=$fold \
    data.pseudo_csv=pseudo_labels.csv \
    model=maxvit_base_384 \
    training=transformer || exit 1

  echo "Finished Fold: $fold successfully."
done
