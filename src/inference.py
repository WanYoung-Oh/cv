"""
Inference 스크립트
학습된 모델로 test 데이터셋에 대한 예측을 수행하고 pred.csv 생성
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, List

# 프로젝트 루트를 Python path에 추가 (어디서든 실행 가능하도록)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.datamodule import DocumentImageDataModule
from src.models.module import DocumentClassifierModule
from src.utils.device import get_simple_device
from src.utils.tta import predict_batch_with_tta, get_tta_transforms
from src.utils.helpers import (
    extract_val_f1_from_filename,
    create_datamodule_from_config,
    save_predictions_to_csv
)


log = logging.getLogger(__name__)


def get_champion_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """챔피언 모델 체크포인트 찾기

    Args:
        checkpoint_dir: 체크포인트 베이스 디렉토리

    Returns:
        챔피언 체크포인트 경로 또는 None
    """
    champion_dir = checkpoint_dir / "champion"
    champion_checkpoint = champion_dir / "best_model.ckpt"
    champion_info_path = champion_dir / "champion_info.json"

    if champion_checkpoint.exists():
        # 챔피언 정보 로드
        if champion_info_path.exists():
            with open(champion_info_path, 'r') as f:
                champion_info = json.load(f)

            log.info("🏆 챔피언 모델 로드")
            log.info(f"   val_f1: {champion_info.get('val_f1', 'N/A')}")
            log.info(f"   원본 경로: {champion_info.get('checkpoint_path', 'N/A')}")
            log.info(f"   업데이트: {champion_info.get('updated_at', 'N/A')}")

        return champion_checkpoint

    return None


def find_checkpoint_by_run_id(checkpoint_dir: Path, run_id: str) -> Optional[Path]:
    """특정 run_id의 best checkpoint 찾기

    Args:
        checkpoint_dir: 체크포인트 베이스 디렉토리
        run_id: 실험 run ID (예: 20260216_run_001)

    Returns:
        해당 run의 best checkpoint 경로 또는 None
    """
    run_dir = checkpoint_dir / run_id

    if not run_dir.exists():
        log.error(f"Run ID '{run_id}'가 존재하지 않습니다: {run_dir}")
        log.info("\n사용 가능한 Run ID 목록:")
        for exp_dir in checkpoint_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != "champion":
                log.info(f"  - {exp_dir.name}")
        return None

    # experiment_info.json에서 best_checkpoint 정보 읽기
    exp_info_path = run_dir / "experiment_info.json"
    if exp_info_path.exists():
        with open(exp_info_path, 'r') as f:
            exp_info = json.load(f)

        log.info(f"📋 Run ID '{run_id}' 정보:")
        log.info(f"   모델: {exp_info.get('model_name', 'N/A')}")
        log.info(f"   시작: {exp_info.get('started_at', 'N/A')}")
        log.info(f"   val_f1: {exp_info.get('val_f1', 'N/A')}")

        best_ckpt_path = exp_info.get('best_checkpoint')
        if best_ckpt_path and Path(best_ckpt_path).exists():
            return Path(best_ckpt_path)

    # experiment_info가 없거나 best_checkpoint 정보가 없으면
    # 파일명에서 가장 높은 val_f1을 가진 checkpoint 찾기
    log.info("experiment_info.json에서 정보를 찾을 수 없습니다. 파일명에서 탐색 중...")
    ckpt_files = list(run_dir.glob("*.ckpt"))
    best_checkpoint = None
    best_metric = 0.0

    for ckpt_file in ckpt_files:
        val_f1 = extract_val_f1_from_filename(ckpt_file)
        if val_f1 is not None and val_f1 > best_metric:
            best_metric = val_f1
            best_checkpoint = ckpt_file

    if best_checkpoint:
        log.info(f"최고 성능 체크포인트 발견: val_f1={best_metric:.4f}")

    return best_checkpoint


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """모든 실험 중 최고 성능 체크포인트 찾기

    Args:
        checkpoint_dir: 체크포인트 베이스 디렉토리

    Returns:
        최고 성능 체크포인트 경로 또는 None
    """
    best_checkpoint = None
    best_metric = 0.0

    # 모든 실험 디렉토리 탐색
    for exp_dir in checkpoint_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "champion":
            continue

        # 해당 실험의 체크포인트 찾기
        ckpt_files = list(exp_dir.glob("*.ckpt"))

        for ckpt_file in ckpt_files:
            val_f1 = extract_val_f1_from_filename(ckpt_file)
            if val_f1 is not None and val_f1 > best_metric:
                best_metric = val_f1
                best_checkpoint = ckpt_file

    if best_checkpoint:
        log.info(f"최고 성능 체크포인트 발견: val_f1={best_metric:.4f}")
        log.info(f"경로: {best_checkpoint}")

    return best_checkpoint


def get_test_image_ids(test_csv_path: str) -> List[str]:
    """테스트 CSV에서 이미지 ID 추출

    Args:
        test_csv_path: 테스트 CSV 파일 경로

    Returns:
        이미지 ID 리스트
    """
    df = pd.read_csv(test_csv_path)
    # 첫 번째 컬럼이 이미지 파일명 또는 ID라고 가정
    return df.iloc[:, 0].tolist()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 inference 함수

    Hydra config로 inference 설정 관리:
        inference.checkpoint: 체크포인트 경로 (선택사항, 최우선)
        inference.run_id: 실험 run ID (선택사항, 2순위)
        inference.output: 출력 파일 경로 (기본값: datasets_fin/submission/{model_name}_{run_id}.csv)
        inference.use_tta: TTA 사용 여부 (기본값: false)
    
    체크포인트 지정 시 해당 학습의 config를 자동으로 로드합니다.
    챔피언 모델 사용 시 현재 config를 사용합니다.

    사용 예시:
        # Champion 모델 사용 (기본, 현재 config 사용)
        # 출력: datasets_fin/submission/{model_name}_champion.csv
        python src/inference.py

        # TTA 사용 (예측 성능 향상, 5배 느림)
        python src/inference.py inference.use_tta=true
        
        # 특정 run_id 사용 (해당 run의 config 자동 로드)
        # 출력: datasets_fin/submission/{model_name}_20260216_run_001.csv
        python src/inference.py inference.run_id=20260216_run_001

        # 직접 checkpoint 경로 지정 (해당 run의 config 자동 로드)
        # 경로에 '='가 있으면 반드시 따옴표로 감싸서 복사 붙여넣기 가능
        # 출력: datasets_fin/submission/{model_name}_20260216_run_001.csv
        python src/inference.py inference.checkpoint="checkpoints/20260216_run_001/epoch=10-val_f1=0.950.ckpt"
        
        # 출력 파일명 직접 지정
        python src/inference.py inference.output=datasets_fin/submission/my_prediction.csv
    """
    from omegaconf import OmegaConf
    
    # Hydra config에서 inference 설정 읽기
    inference_cfg = cfg.get('inference', {})
    checkpoint_path = inference_cfg.get('checkpoint', None)
    run_id = inference_cfg.get('run_id', None)
    use_tta = inference_cfg.get('use_tta', False)
    tta_level = inference_cfg.get('tta_level', 'standard')

    log.info("=" * 70)
    log.info("🔮 Inference 시작")
    log.info("=" * 70)

    if use_tta:
        from src.utils.tta import TTA_LEVEL_SIZES
        n = TTA_LEVEL_SIZES.get(tta_level, 8)
        log.info(f"🔄 TTA 활성화: level={tta_level} ({n}가지 변환, 약 {n}× 느림)")

    # 체크포인트 경로 찾기
    use_champion = False
    actual_run_id = None  # 실제 사용된 run_id 추적
    
    if not checkpoint_path:
        checkpoint_dir = Path(cfg.checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"체크포인트 디렉토리 '{checkpoint_dir}'가 존재하지 않습니다."
            )

        # 1순위: run_id 지정
        if run_id:
            log.info(f"Run ID '{run_id}'의 모델 탐색 중...")
            run_ckpt = find_checkpoint_by_run_id(checkpoint_dir, run_id)

            if run_ckpt:
                checkpoint_path = str(run_ckpt)
                actual_run_id = run_id
                log.info(f"✅ Run ID '{run_id}' 모델 사용")
            else:
                raise FileNotFoundError(
                    f"Run ID '{run_id}'의 체크포인트를 찾을 수 없습니다.\n"
                    f"'{checkpoint_dir}' 디렉토리를 확인하세요."
                )
        else:
            # 2순위: 챔피언 모델
            champion_ckpt = get_champion_checkpoint(checkpoint_dir)
            if champion_ckpt:
                checkpoint_path = str(champion_ckpt)
                use_champion = True
                actual_run_id = "champion"
                log.info("✅ 챔피언 모델 사용 (현재 config 사용)")
            else:
                # 3순위: 모든 실험 중 최고 성능 모델
                log.info("챔피언 모델이 없습니다. 최고 성능 모델 탐색 중...")
                best_ckpt = find_best_checkpoint(checkpoint_dir)

                if best_ckpt:
                    checkpoint_path = str(best_ckpt)
                    # best_ckpt에서 run_id 추출 (예: checkpoints/20260216_run_001/...)
                    actual_run_id = best_ckpt.parent.name
                    log.info("✅ 최고 성능 모델 사용")
                else:
                    raise FileNotFoundError(
                        f"체크포인트를 찾을 수 없습니다.\n"
                        f"'{checkpoint_dir}' 디렉토리에 학습된 모델이 없습니다.\n"
                        f"먼저 'python src/train.py'로 모델을 학습하세요."
                    )
    else:
        # checkpoint 경로가 직접 지정된 경우, 경로에서 run_id 추출
        checkpoint_path_obj = Path(checkpoint_path)
        if checkpoint_path_obj.parent.name == "champion":
            actual_run_id = "champion"
            use_champion = True
        else:
            actual_run_id = checkpoint_path_obj.parent.name

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

    log.info(f"사용 체크포인트: {checkpoint_path}")
    
    # 체크포인트의 실제 모델명 가져오기
    model_name = cfg.model.model_name  # 기본값
    checkpoint_dir_path = Path(checkpoint_path).parent
    
    # 챔피언 모델 사용 시
    if use_champion:
        champion_info_path = Path(cfg.checkpoint_dir) / "champion" / "champion_info.json"
        if champion_info_path.exists():
            with open(champion_info_path, 'r') as f:
                champion_info = json.load(f)
                champion_run_id = champion_info.get('model_name', 'champion')
                
                # champion_run_id로 experiment_info.json 찾기
                experiment_dir = Path(cfg.checkpoint_dir) / champion_run_id
                experiment_info_path = experiment_dir / "experiment_info.json"
                
                if experiment_info_path.exists():
                    with open(experiment_info_path, 'r') as ef:
                        experiment_info = json.load(ef)
                        model_name = experiment_info.get('model_name', model_name)
                        log.info(f"📋 챔피언 모델명: {model_name}")
                        checkpoint_dir_path = experiment_dir
    else:
        # 일반 run_id 또는 checkpoint 경로 지정 시에도 실제 모델명 가져오기
        experiment_info_path = checkpoint_dir_path / "experiment_info.json"
        if experiment_info_path.exists():
            with open(experiment_info_path, 'r') as f:
                experiment_info = json.load(f)
                model_name = experiment_info.get('model_name', model_name)
                log.info(f"📋 실제 모델명: {model_name}")
    
    # 출력 경로 결정: datasets_fin/submission/{model_name}_{run_id}.csv
    submission_dir = os.path.join(cfg.data.root_path, "submission")
    default_output = os.path.join(submission_dir, f"{model_name}_{actual_run_id}.csv")
    output_path = inference_cfg.get('output') or default_output
    
    # 체크포인트의 학습 config 로드 (챔피언이 아닌 경우에만)
    if not use_champion:
        experiment_info_path = checkpoint_dir_path / "experiment_info.json"
        
        if experiment_info_path.exists():
            log.info(f"📋 학습 config 로드 중: {experiment_info_path}")
            with open(experiment_info_path, 'r') as f:
                experiment_info = json.load(f)
                original_config = experiment_info.get('config', {})
                
                # data config 덮어쓰기 (img_size, normalization, augmentation)
                if 'data' in original_config:
                    log.info("   - data config (img_size, normalization 적용)")
                    cfg.data.img_size = original_config['data'].get('img_size', cfg.data.img_size)
                    cfg.data.normalization = original_config['data'].get('normalization', cfg.data.normalization)
                    
                    # val_augmentations만 사용 (DL 원칙)
                    if 'augmentation' in original_config['data']:
                        log.info("   - validation augmentation 적용 (DL 원칙: Resize + Pad + CLAHE)")
                        cfg.data.augmentation = original_config['data']['augmentation']
        else:
            log.warning(f"⚠️  experiment_info.json을 찾을 수 없습니다: {experiment_info_path}")
            log.warning(f"   현재 config를 사용합니다.")

    # 데이터모듈 생성
    # inference는 sample_submission.csv를 테스트 데이터 소스로 사용
    submission_csv = cfg.data.get('sample_submission_csv', cfg.data.get('test_csv', None))
    if not submission_csv:
        raise ValueError("cfg.data.sample_submission_csv 또는 cfg.data.test_csv가 필요합니다.")

    submission_csv_path = os.path.join(cfg.data.root_path, submission_csv)
    if not os.path.exists(submission_csv_path):
        raise FileNotFoundError(
            f"Submission CSV 파일을 찾을 수 없습니다: {submission_csv_path}\n"
            f"데이터셋을 먼저 준비해주세요."
        )

    log.info(f"테스트 데이터 (submission): {submission_csv_path}")

    # DataModule 생성 (팩토리 함수 사용, sample_submission_csv를 test_csv로 전달)
    data_module = create_datamodule_from_config(cfg)
    data_module.setup(stage='test')

    # 모델 로드
    log.info("모델 로드 중...")
    model = DocumentClassifierModule.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    # 디바이스 설정 (CUDA -> MPS -> CPU 자동 감지)
    device = get_simple_device()
    model = model.to(device)
    log.info(f"사용 디바이스: {device}")

    # Inference 수행
    log.info("Inference 수행 중...")
    predictions = []

    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images, _ = batch
            images = images.to(device)

            if use_tta:
                preds = predict_batch_with_tta(
                    model=model,
                    images=images,
                    device=device,
                    level=tta_level,
                    return_probs=False,
                )
            else:
                # 기본 예측
                logits = model(images)
                preds = logits.argmax(dim=1)

            predictions.extend(preds.cpu().numpy().tolist())

    log.info(f"총 예측 수: {len(predictions)}")

    # 결과 저장 (유틸리티 함수 사용)
    result_df = save_predictions_to_csv(
        predictions=predictions,
        output_path=output_path,
        data_root=cfg.data.root_path,
        test_csv_path=submission_csv_path,
        task_name="Inference",
    )

    # 예측 샘플 출력
    log.info(f"📊 예측 샘플:")
    log.info(f"\n{result_df.head(10)}")


if __name__ == "__main__":
    main()
