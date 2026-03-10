"""
Ensemble 시스템
- Voting (Hard/Soft)
- Weighted Averaging
- Stacking
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# 프로젝트 루트를 Python path에 추가 (어디서든 실행 가능하도록)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from scipy import stats

from src.data.datamodule import DocumentImageDataModule
from src.models.module import DocumentClassifierModule
from src.utils.device import get_simple_device
from src.utils.helpers import create_datamodule_from_config, save_predictions_to_csv
from src.utils.tta import predict_batch_with_tta

log = logging.getLogger(__name__)


def discover_all_best_checkpoints(checkpoint_dir: Path, img_size: int) -> Tuple[List[str], List[float]]:
    """checkpoint_dir 아래 모든 run의 experiment_info.json에서 best_checkpoint 수집 (동일 img_size만).

    Returns:
        (checkpoint_paths, val_f1_list): 경로 리스트와 val_f1 리스트 (가중치용)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return [], []

    paths = []
    val_f1s = []
    for run_dir in sorted(checkpoint_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "champion":
            # champion은 심볼릭/복사이므로 원본 run 기준으로 수집된 것만 사용
            continue
        info_path = run_dir / "experiment_info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path) as f:
                info = json.load(f)
        except Exception:
            continue
        cfg = info.get("config") or {}
        data_cfg = cfg.get("data") or {}
        run_img_size = data_cfg.get("img_size")
        if run_img_size is not None and run_img_size != img_size:
            continue
        best_ckpt = info.get("best_checkpoint")
        if not best_ckpt:
            # 상대 경로일 수 있음
            for ckpt in run_dir.glob("*.ckpt"):
                best_ckpt = str(ckpt)
                break
        if not best_ckpt or not os.path.exists(best_ckpt):
            continue
        paths.append(best_ckpt)
        val_f1s.append(float(info.get("val_f1", 0.0)))

    return paths, val_f1s


def load_models(checkpoint_paths: List[str]) -> List[DocumentClassifierModule]:
    """여러 체크포인트 로드"""
    models = []
    for ckpt_path in checkpoint_paths:
        if not os.path.exists(ckpt_path):
            log.warning(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
            continue

        log.info(f"로드 중: {ckpt_path}")
        model = DocumentClassifierModule.load_from_checkpoint(ckpt_path, strict=False)
        model.eval()
        models.append(model)

    return models


def predict_single_model(
    model: DocumentClassifierModule,
    data_loader,
    device: torch.device,
    return_probs: bool = True,
    use_tta: bool = False,
    tta_level: str = "standard",
) -> np.ndarray:
    """단일 모델로 예측 (use_tta=True 시 TTA 적용)"""
    all_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            images, _ = batch
            images = images.to(device)

            if use_tta:
                probs = predict_batch_with_tta(model, images, device, level=tta_level, return_probs=True)
                all_outputs.append(probs.cpu().numpy())
            else:
                logits = model(images)
                if return_probs:
                    probs = torch.softmax(logits, dim=1)
                    all_outputs.append(probs.cpu().numpy())
                else:
                    preds = logits.argmax(dim=1)
                    all_outputs.append(preds.cpu().numpy())

    return np.concatenate(all_outputs, axis=0)


def hard_voting(predictions: List[np.ndarray]) -> np.ndarray:
    """Hard Voting: 다수결"""
    # predictions: List[N,] -> (num_models, N)
    votes = np.stack(predictions, axis=0)
    # 각 샘플에 대해 최빈값
    final_preds, _ = stats.mode(votes, axis=0, keepdims=False)
    return final_preds


def soft_voting(probabilities: List[np.ndarray], weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Soft Voting: 확률 평균"""
    # probabilities: List[N, C] -> (num_models, N, C)
    probs = np.stack(probabilities, axis=0)

    if weights is None:
        # 균등 가중치
        avg_probs = np.mean(probs, axis=0)
    else:
        # 가중 평균
        weights = np.array(weights).reshape(-1, 1, 1)  # (num_models, 1, 1)
        avg_probs = np.sum(probs * weights, axis=0) / np.sum(weights)

    # 최종 예측
    final_preds = np.argmax(avg_probs, axis=1)
    return final_preds, avg_probs


def rank_averaging(probabilities: List[np.ndarray]) -> np.ndarray:
    """Rank Averaging: 순위 기반 앙상블"""
    # 각 모델의 확률을 rank로 변환
    ranks = []
    for probs in probabilities:
        # 각 샘플에 대해 클래스별 순위
        rank = np.argsort(np.argsort(-probs, axis=1), axis=1)  # 낮은 rank = 높은 확률
        ranks.append(rank)

    # 평균 rank
    avg_rank = np.mean(np.stack(ranks, axis=0), axis=0)

    # rank가 가장 낮은 클래스 선택
    final_preds = np.argmin(avg_rank, axis=1)
    return final_preds


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 ensemble 함수

    Hydra config로 앙상블 설정 관리:
        ensemble.checkpoints: 체크포인트 경로 리스트
        ensemble.method: "hard_voting" | "soft_voting" | "rank_averaging"
        ensemble.weights: 모델 가중치 (선택사항)
        ensemble.output: 출력 파일 경로
    """
    # Hydra config에서 앙상블 설정 읽기
    ensemble_cfg = cfg.get('ensemble', {})
    use_tta = ensemble_cfg.get('use_tta', False)
    tta_level = ensemble_cfg.get('tta_level', 'standard')
    use_all_runs = ensemble_cfg.get('use_all_runs', False)
    checkpoints = list(ensemble_cfg.get('checkpoints', []))
    method = ensemble_cfg.get('method', 'soft_voting')
    weights = ensemble_cfg.get('weights', None)

    # use_all_runs: 현재 data img_size와 같은 모든 run의 best 체크포인트 자동 수집
    if use_all_runs:
        checkpoint_dir = Path(cfg.get('checkpoint_dir', 'checkpoints'))
        img_size = cfg.data.get('img_size', 384)
        discovered_paths, discovered_val_f1s = discover_all_best_checkpoints(checkpoint_dir, img_size)
        if discovered_paths:
            checkpoints = discovered_paths
            if method == 'soft_voting' and discovered_val_f1s:
                weights = discovered_val_f1s
            log.info(f"use_all_runs: img_size={img_size} 기준 {len(checkpoints)}개 체크포인트 수집")
        else:
            log.warning("use_all_runs=true이지만 수집된 체크포인트가 없습니다. checkpoints를 사용합니다.")

    # 출력 경로: datasets_fin/submission/submission_ensemble_{method}.csv
    submission_dir = os.path.join(cfg.data.root_path, "submission")
    default_output = os.path.join(submission_dir, f"submission_ensemble_{method}.csv")
    output = ensemble_cfg.get('output', default_output)

    if not checkpoints:
        raise ValueError(
            "ensemble.checkpoints가 비어 있고 use_all_runs로도 수집되지 않았습니다. "
            "configs/ensemble/default.yaml을 확인하거나 CLI로 전달하세요: "
            "python src/ensemble.py ensemble.checkpoints=[path1,path2]"
        )

    log.info("=" * 70)
    log.info("🔮 Ensemble Inference 시작")
    log.info("=" * 70)
    log.info(f"방법: {method}")
    log.info(f"모델 수: {len(checkpoints)}")

    # 모델 로드
    models = load_models(checkpoints)

    if len(models) == 0:
        raise ValueError("로드된 모델이 없습니다.")

    if len(models) == 1:
        log.warning("모델이 1개만 로드되었습니다. 앙상블 효과 없음.")

    # 데이터 로드 (sample_submission.csv를 테스트 데이터 소스로 사용)
    submission_csv = cfg.data.get('sample_submission_csv', cfg.data.get('test_csv', None))
    if not submission_csv:
        raise ValueError("cfg.data.sample_submission_csv 또는 cfg.data.test_csv가 필요합니다.")

    submission_csv_path = os.path.join(cfg.data.root_path, submission_csv)
    if not os.path.exists(submission_csv_path):
        raise FileNotFoundError(f"Submission CSV를 찾을 수 없습니다: {submission_csv_path}")

    # DataModule 생성 (팩토리 함수 사용, sample_submission_csv를 test_csv로 전달)
    data_module = create_datamodule_from_config(cfg)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # 디바이스 설정 (CUDA -> MPS -> CPU 자동 감지)
    device = get_simple_device()
    log.info(f"사용 디바이스: {device}")

    # 각 모델로 예측
    log.info("\n📊 모델별 예측 수행 중...")
    all_predictions = []
    all_probabilities = []

    if use_tta:
        log.info(f"🔄 TTA 활성화 (level={tta_level})")

    for i, model in enumerate(models):
        model = model.to(device)
        log.info(f"\n모델 {i+1}/{len(models)} 예측 중...")

        if method == "hard_voting":
            preds = predict_single_model(model, test_loader, device, return_probs=False, use_tta=use_tta, tta_level=tta_level)
            all_predictions.append(preds)
        else:
            probs = predict_single_model(model, test_loader, device, return_probs=True, use_tta=use_tta, tta_level=tta_level)
            all_probabilities.append(probs)

    # 앙상블
    log.info(f"\n🔄 {method} 적용 중...")

    if method == "hard_voting":
        final_preds = hard_voting(all_predictions)
        final_probs = None

    elif method == "soft_voting":
        final_preds, final_probs = soft_voting(all_probabilities, weights)

    elif method == "rank_averaging":
        final_preds = rank_averaging(all_probabilities)
        final_probs = None

    else:
        raise ValueError(
            f"Unknown ensemble method: {method}. "
            f"Supported methods: hard_voting, soft_voting, rank_averaging"
        )

    log.info(f"총 예측 수: {len(final_preds)}")

    # 결과 저장 (유틸리티 함수 사용)
    result_df = save_predictions_to_csv(
        predictions=final_preds,
        output_path=output,
        data_root=cfg.data.root_path,
        test_csv_path=submission_csv_path,
        task_name="Ensemble",
    )

    # 앙상블 정보 저장 (use_all_runs 시 checkpoints가 이미 list라 OmegaConf.to_container 불가)
    from omegaconf import OmegaConf, ListConfig

    def _to_list(obj):
        if obj is None:
            return None
        if isinstance(obj, list):
            return obj
        if isinstance(obj, ListConfig):
            return list(obj)
        return OmegaConf.to_container(obj, resolve=True)

    checkpoints_list = _to_list(checkpoints) or []
    weights_list = _to_list(weights)
    
    ensemble_info = {
        "method": method,
        "num_models": len(models),
        "checkpoints": checkpoints_list,
        "weights": weights_list,
        "output": output
    }

    info_path = output.replace(".csv", "_info.json")
    with open(info_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2)

    log.info(f"\n💾 앙상블 정보 저장: {info_path}")


if __name__ == "__main__":
    main()
