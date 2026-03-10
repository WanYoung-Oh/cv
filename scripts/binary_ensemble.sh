#!/usr/bin/env bash
# =============================================================================
# binary_ensemble.sh  —  이진분류기 전체 파이프라인
#
# Step 1: train_binary_classifier.py   — class 3·7 이진분류기 5-Fold 학습
# Step 2: apply_binary_ensemble.py     — 이진분류기로 메인 분류기 확률 보정
#
# 사용법:
#   ./binary_ensemble.sh                     # 전체 파이프라인 (기본: proportional)
#   ./binary_ensemble.sh --skip-train        # Step 1 건너뜀 (기존 체크포인트 사용)
#   ./binary_ensemble.sh --mode=routing      # Phase 2 routing 모드
#   ./binary_ensemble.sh --grid-search       # val set θ·α grid search 후 적용
#   ./binary_ensemble.sh --no-tta            # TTA 비활성화
#
# SSH/VPN 종료 후에도 실행 지속:
#   스크립트 실행 시 자동으로 nohup 백그라운드로 전환됩니다.
#   로그는 logs/binary_ensemble_YYYYMMDD_HHMMSS.log에 저장됩니다.
# =============================================================================

set -euo pipefail

# ── 설정 변수 ─────────────────────────────────────────────────────────────────
PYTHON="/data/ephemeral/home/py310/bin/python"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"

# Hydra config group 이름
DATA_CONFIG="transformer_384"
BINARY_TRAIN_CONFIG="default"
BINARY_APPLY_CONFIG="apply_ensemble"
ENSEMBLE_CONFIG="ensemble_binary_MaxVit_kfold"

# 보정 파라미터 기본값
MODE="proportional"          # proportional | weighted | routing
ALPHA="0.8"
THETA="0.3"
DYNAMIC_ALPHA="true"
USE_TTA="true"
TTA_LEVEL="light"            # light | standard | heavy
GRID_SEARCH="false"

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
SKIP_TRAIN=false

for arg in "$@"; do
    case $arg in
        --skip-train)    SKIP_TRAIN=true                ;;
        --mode=*)        MODE="${arg#*=}"               ;;
        --alpha=*)       ALPHA="${arg#*=}"              ;;
        --theta=*)       THETA="${arg#*=}"              ;;
        --tta-level=*)   TTA_LEVEL="${arg#*=}"          ;;
        --no-tta)        USE_TTA="false"                ;;
        --grid-search)   GRID_SEARCH="true"             ;;
        --ensemble=*)    ENSEMBLE_CONFIG="${arg#*=}"    ;;
        --help|-h)
            sed -n '3,16p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
    esac
done

# ── 자동 nohup  (SSH/VPN 종료 후에도 프로세스 유지) ──────────────────────────
# 환경 변수 _BINARY_BG가 없으면 자기 자신을 nohup으로 재실행하고 종료.
if [ -z "${_BINARY_BG:-}" ]; then
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/binary_ensemble_$(date +%Y%m%d_%H%M%S).log"
    export _BINARY_BG=1
    nohup bash "${BASH_SOURCE[0]}" "$@" >> "${LOG_FILE}" 2>&1 &
    BG_PID=$!
    echo "========================================================"
    echo "  이진분류기 파이프라인 — 백그라운드 실행"
    echo "  PID    : ${BG_PID}"
    echo "  로그   : ${LOG_FILE}"
    echo "  실시간 확인 : tail -f ${LOG_FILE}"
    echo "  강제 종료   : kill ${BG_PID}"
    echo "========================================================"
    exit 0
fi

# =============================================================================
# 이하: nohup 컨텍스트에서 실행 (SSH 끊겨도 지속)
# =============================================================================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

PIPELINE_START=$(date +%s)

log "========================================================"
log "  이진분류기 파이프라인 시작"
log "  PROJECT_ROOT : ${PROJECT_ROOT}"
log "  PYTHON       : ${PYTHON}"
log "  SKIP_TRAIN   : ${SKIP_TRAIN}"
log "  MODE         : ${MODE}  ALPHA=${ALPHA}  THETA=${THETA}"
log "  USE_TTA      : ${USE_TTA} (${TTA_LEVEL})"
log "  GRID_SEARCH  : ${GRID_SEARCH}"
log "  ENSEMBLE     : ${ENSEMBLE_CONFIG}"
log "========================================================"

cd "${PROJECT_ROOT}"

# ── Step 1: 이진분류기 5-Fold 학습 ───────────────────────────────────────────
if [ "${SKIP_TRAIN}" = false ]; then
    log ""
    log "▶ [Step 1/2] 이진분류기 5-Fold 학습 시작"
    log "  python scripts/train_binary_classifier.py data=${DATA_CONFIG} binary=${BINARY_TRAIN_CONFIG}"
    log ""

    STEP1_START=$(date +%s)

    "${PYTHON}" scripts/train_binary_classifier.py \
        data="${DATA_CONFIG}" \
        binary="${BINARY_TRAIN_CONFIG}"

    STEP1_END=$(date +%s)
    STEP1_MIN=$(( (STEP1_END - STEP1_START) / 60 ))
    STEP1_SEC=$(( (STEP1_END - STEP1_START) % 60 ))

    log ""
    log "✅ [Step 1/2] 이진분류기 학습 완료 (${STEP1_MIN}분 ${STEP1_SEC}초)"
    log "   체크포인트 : checkpoints/binary/fold_*/best*.ckpt"
    log "   결과 요약  : checkpoints/binary/fold_results.json"
else
    log "⏭  [Step 1/2] 건너뜀 (--skip-train)"
fi

# ── 체크포인트 경로 자동 수집 (fold_results.json → Hydra CLI override) ────────
FOLD_RESULTS="${PROJECT_ROOT}/checkpoints/binary/fold_results.json"
BINARY_CKPT_OVERRIDE=""

if [ -f "${FOLD_RESULTS}" ]; then
    log ""
    log "📋 fold_results.json에서 체크포인트 경로 읽는 중..."

    CKPT_PATHS=$("${PYTHON}" -c "
import json, sys
with open('${FOLD_RESULTS}') as f:
    data = json.load(f)
paths = [r['checkpoint'] for r in data.get('folds', []) if r.get('checkpoint')]
if not paths:
    sys.exit(1)
print(','.join(paths))
" 2>/dev/null || echo "")

    if [ -n "${CKPT_PATHS}" ]; then
        BINARY_CKPT_OVERRIDE="binary.binary_checkpoints=[${CKPT_PATHS}]"
        log "  체크포인트: ${CKPT_PATHS/,/, }"
    else
        log "  ⚠️ 유효한 체크포인트 없음 → apply_ensemble.yaml 기본값 사용"
    fi
else
    log ""
    log "  ℹ️ fold_results.json 없음 → apply_ensemble.yaml 기본값 사용"
fi

# ── Step 2: 보정 앙상블 적용 ─────────────────────────────────────────────────
log ""
log "▶ [Step 2/2] 이진분류기 보정 앙상블 적용 (mode=${MODE})"
log ""

STEP2_START=$(date +%s)

# 명령 배열로 구성 (경로에 공백 없음을 가정)
CMD=(
    "${PYTHON}" scripts/apply_binary_ensemble.py
    data="${DATA_CONFIG}"
    binary="${BINARY_APPLY_CONFIG}"
    ensemble="${ENSEMBLE_CONFIG}"
    binary.mode="${MODE}"
    binary.alpha="${ALPHA}"
    binary.theta="${THETA}"
    binary.dynamic_alpha="${DYNAMIC_ALPHA}"
    binary.use_tta="${USE_TTA}"
    binary.tta_level="${TTA_LEVEL}"
    binary.grid_search="${GRID_SEARCH}"
)

# 학습된 체크포인트 경로가 수집된 경우에만 override 추가
if [ -n "${BINARY_CKPT_OVERRIDE}" ]; then
    CMD+=("${BINARY_CKPT_OVERRIDE}")
fi

"${CMD[@]}"

STEP2_END=$(date +%s)
STEP2_MIN=$(( (STEP2_END - STEP2_START) / 60 ))
STEP2_SEC=$(( (STEP2_END - STEP2_START) % 60 ))

log ""
log "✅ [Step 2/2] 보정 앙상블 완료 (${STEP2_MIN}분 ${STEP2_SEC}초)"

# ── 파이프라인 완료 ───────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - PIPELINE_START) / 60 ))
TOTAL_SEC=$(( (TOTAL_END - PIPELINE_START) % 60 ))

log ""
log "========================================================"
log "  파이프라인 완료"
log "  총 소요 시간: ${TOTAL_MIN}분 ${TOTAL_SEC}초"
log "========================================================"
