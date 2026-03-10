"""
이진분류기 기반 클래스 3·7 보정 앙상블 (Phase 1 + Phase 2)

이진분류기(class 3 vs 7) 출력으로 메인 분류기(17클래스) 확률을 보정합니다.

모드 선택 (binary.mode):
  proportional — Phase 1: 비례 재배분 (α, θ 무시)
    final[3] = p_bin[3] * pool
    final[7] = p_bin[7] * pool

  weighted     — Phase 2: α 가중 평균 (θ 무시)
    final[3] = (1-α)*p_main[3] + α*p_bin[3]*pool
    final[7] = (1-α)*p_main[7] + α*p_bin[7]*pool

  routing      — Phase 2: pool>θ인 샘플에만 weighted 보정 적용 (권장)

α 파라미터 (binary.alpha):
  - 0.0: 보정 없음, 1.0: 완전 비례 재배분
  - dynamic_alpha=true이면 메인 확신도 > confidence_threshold 시 α 선형 감소
    → 메인이 맞고 이진분류기가 틀리는 시나리오 B 피해 방지 (research.md §5.3)

실행 방법:
  # Phase 1 비례 재배분 (기본)
  python scripts/apply_binary_ensemble.py \\
    data=transformer_384 \\
    binary=apply_ensemble

  # Phase 2 Confidence Routing + 동적 α
  python scripts/apply_binary_ensemble.py \\
    data=transformer_384 \\
    binary=apply_ensemble \\
    binary.mode=routing \\
    binary.alpha=0.8 \\
    binary.theta=0.3 \\
    binary.dynamic_alpha=true

  # heavy TTA 적용
  python scripts/apply_binary_ensemble.py \\
    data=transformer_384 \\
    binary=apply_ensemble \\
    binary.use_tta=true \\
    binary.tta_level=heavy \\
    binary.mode=routing

References:
    docs/research.md §5.3, §7.2
    src/utils/binary_ensemble.py
"""

import json
import os
import sys
import logging
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.module import DocumentClassifierModule
from src.utils.binary_ensemble import binary_correction, grid_search_params, make_fallback_p_bin
from src.utils.device import get_simple_device
from src.utils.helpers import create_datamodule_from_config, save_predictions_to_csv
from src.utils.tta import predict_batch_with_tta

log = logging.getLogger(__name__)


def _predict_probs(
    checkpoints: list,
    data_loader,
    device: torch.device,
    use_tta: bool = False,
    tta_level: str = "standard",
    label: str = "",
) -> np.ndarray:
    """체크포인트 목록으로 추론 후 확률 행렬 반환 (모델 평균).

    Args:
        checkpoints: 체크포인트 경로 리스트
        data_loader: 테스트 DataLoader
        device: 추론 디바이스
        use_tta: TTA 사용 여부
        tta_level: TTA 강도 ("light" | "standard" | "heavy")
        label: 로깅용 레이블

    Returns:
        (N, num_classes) 확률 행렬 (모든 모델의 평균)
    """
    all_model_probs = []

    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            log.warning(f"체크포인트 없음 (건너뜀): {ckpt_path}")
            continue

        log.info(f"  [{label}] 로드: {ckpt_path}")
        model = DocumentClassifierModule.load_from_checkpoint(ckpt_path, strict=False)
        model.eval()
        model = model.to(device)

        batch_probs = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=Path(ckpt_path).stem, leave=False):
                images, _ = batch
                images = images.to(device)

                if use_tta:
                    probs = predict_batch_with_tta(
                        model, images, device, level=tta_level, return_probs=True
                    )
                else:
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)

                batch_probs.append(probs.cpu().numpy())

        all_model_probs.append(np.concatenate(batch_probs, axis=0))
        model.cpu()
        del model

    if not all_model_probs:
        raise ValueError(f"[{label}] 로드된 체크포인트가 없습니다. 경로를 확인하세요.")

    return np.mean(all_model_probs, axis=0)   # (N, C)


def _plot_grid_search_heatmap(gs_result: dict, output_path: str, mode: str) -> None:
    """Grid Search 결과를 히트맵(routing) 또는 막대 그래프(weighted)로 시각화.

    Args:
        gs_result: grid_search_params() 반환값
        output_path: 저장할 PNG 파일 경로
        mode: "routing" | "weighted"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results = gs_result["results"]
    best = gs_result["best"]
    baseline_f1 = gs_result["baseline_f1"]
    df = pd.DataFrame(results)

    if mode == "routing":
        pivot = df.pivot(index="alpha", columns="theta", values="f1")
        pivot = pivot.sort_index(ascending=False)  # α 내림차순 (위쪽이 높은 α)

        n_cols, n_rows = len(pivot.columns), len(pivot.index)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.9 + 2), max(6, n_rows * 0.75 + 2)))

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            annot=True,
            fmt=".4f",
            linewidths=0.5,
            cbar_kws={"label": "F1-Macro"},
            annot_kws={"size": 8},
        )

        # 최적 포인트: 파란 테두리 사각형
        best_row = list(pivot.index).index(best["alpha"])
        best_col = list(pivot.columns).index(best["theta"])
        ax.add_patch(mpatches.Rectangle(
            (best_col, best_row), 1, 1,
            fill=False, edgecolor="blue", linewidth=2.5, clip_on=False,
        ))

        ax.set_xlabel("θ  (routing threshold)", fontsize=11)
        ax.set_ylabel("α  (correction strength)", fontsize=11)
        ax.set_xticklabels([f"{float(c):.2f}" for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticklabels([f"{float(r):.1f}" for r in pivot.index], rotation=0)
        ax.set_title(
            f"Grid Search — F1-Macro Heatmap  (mode=routing)\n"
            f"baseline={baseline_f1:.4f}  |  best: θ={best['theta']:.2f}, α={best['alpha']:.1f}, "
            f"F1={best['f1']:.4f}  (Δ={best['f1'] - baseline_f1:+.4f})",
            fontsize=12,
        )

    else:  # weighted: α만
        df_sorted = df.sort_values("alpha").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(max(6, len(df_sorted) * 0.8 + 2), 5))

        colors = ["orangered" if row["alpha"] == best["alpha"] else "steelblue"
                  for _, row in df_sorted.iterrows()]
        ax.bar(range(len(df_sorted)), df_sorted["f1"].values,
               color=colors, edgecolor="white", linewidth=0.5)

        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(i, row["f1"] + 5e-5, f"{row['f1']:.4f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{r['alpha']:.1f}" for _, r in df_sorted.iterrows()])
        ax.set_xlabel("α  (correction strength)", fontsize=11)
        ax.set_ylabel("F1-Macro", fontsize=11)
        ax.axhline(baseline_f1, color="gray", linestyle="--",
                   label=f"baseline={baseline_f1:.4f}")
        ax.legend()
        ax.set_title(
            f"Grid Search — F1-Macro by α  (mode=weighted)\n"
            f"best: α={best['alpha']:.1f}, F1={best['f1']:.4f}  (Δ={best['f1'] - baseline_f1:+.4f})",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  💾 히트맵 저장: {output_path}")


def _collect_labels(data_loader) -> np.ndarray:
    """DataLoader에서 정답 레이블만 수집 (grid search용).

    Returns:
        (N,) int array
    """
    labels = []
    for _, label in data_loader:
        labels.append(label.numpy() if hasattr(label, "numpy") else np.array(label))
    return np.concatenate(labels, axis=0)


def apply_proportional_redistribution(
    p_main: np.ndarray,
    p_bin: np.ndarray,
    class3_idx: int = 3,
    class7_idx: int = 7,
) -> np.ndarray:
    """B-1 비례 재배분 적용.

    메인 분류기의 class 3·7 총 확률(pool)을 보존하면서,
    이진분류기가 결정한 비율로 재배분합니다.
    나머지 15개 클래스의 확률은 변경되지 않습니다.

    Args:
        p_main: 메인 분류기 확률 행렬 (N, 17)
        p_bin:  이진분류기 확률 행렬 (N, 2) — [:, 0]=class3, [:, 1]=class7
        class3_idx: 메인 분류기에서 class 3의 인덱스
        class7_idx: 메인 분류기에서 class 7의 인덱스

    Returns:
        (N, 17) 보정된 확률 행렬
    """
    final = p_main.copy()
    pool = p_main[:, class3_idx] + p_main[:, class7_idx]   # (N,)
    final[:, class3_idx] = p_bin[:, 0] * pool
    final[:, class7_idx] = p_bin[:, 1] * pool
    return final


def _log_correction_stats(
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    class3_idx: int,
    class7_idx: int,
) -> None:
    """보정 전후 예측 변화 통계 출력"""
    changed_mask = pred_before != pred_after
    n_changed = changed_mask.sum()
    log.info(f"  보정된 샘플: {n_changed}개 / 전체 {len(pred_before)}개")

    if n_changed == 0:
        return

    # 변경 내역 (class 3·7 관련만)
    changed_before = pred_before[changed_mask]
    changed_after = pred_after[changed_mask]
    changes = Counter(zip(changed_before.tolist(), changed_after.tolist()))

    log.info("  변경 내역 (before → after):")
    for (b, a), cnt in changes.most_common():
        marker = " ← 혼동쌍 관련" if b in (class3_idx, class7_idx) or a in (class3_idx, class7_idx) else ""
        log.info(f"    class {b:2d} → class {a:2d}: {cnt:4d}건{marker}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """이진분류기 보정 앙상블 실행 (Phase 1 + Phase 2)"""
    binary_cfg = cfg.binary
    ensemble_cfg = cfg.get('ensemble', {})

    # ── 공통 설정 ─────────────────────────────────────────────────────────────
    use_tta = binary_cfg.get('use_tta', False)
    tta_level = binary_cfg.get('tta_level', 'standard')
    class3_idx = binary_cfg.get('class3_idx', 3)
    class7_idx = binary_cfg.get('class7_idx', 7)
    output = binary_cfg.get('output', 'datasets_fin/submission/submission_binary_ensemble.csv')

    # ── Phase 2 파라미터 ──────────────────────────────────────────────────────
    mode = binary_cfg.get('mode', 'proportional')
    alpha = float(binary_cfg.get('alpha', 0.8))
    theta = float(binary_cfg.get('theta', 0.3))
    dynamic_alpha = bool(binary_cfg.get('dynamic_alpha', True))
    confidence_threshold = float(binary_cfg.get('confidence_threshold', 0.7))

    # ── Grid Search 설정 ──────────────────────────────────────────────────────
    do_grid_search = bool(binary_cfg.get('grid_search', False))
    gs_theta_min = float(binary_cfg.get('grid_search_theta_min', 0.05))
    gs_theta_max = float(binary_cfg.get('grid_search_theta_max', 0.80))
    gs_theta_step = float(binary_cfg.get('grid_search_theta_step', 0.05))
    gs_alpha_min = float(binary_cfg.get('grid_search_alpha_min', 0.3))
    gs_alpha_max = float(binary_cfg.get('grid_search_alpha_max', 1.0))
    gs_alpha_step = float(binary_cfg.get('grid_search_alpha_step', 0.1))

    # ── 체크포인트 수집 ───────────────────────────────────────────────────────
    main_checkpoints = list(binary_cfg.get('main_checkpoints', []))
    if not main_checkpoints:
        main_checkpoints = list(ensemble_cfg.get('checkpoints', []))
    if not main_checkpoints:
        raise ValueError(
            "메인 체크포인트가 없습니다. "
            "binary.main_checkpoints 또는 ensemble.checkpoints를 지정하세요."
        )

    binary_checkpoints = list(binary_cfg.get('binary_checkpoints', []))
    # proportional 모드는 이진분류기 필수; weighted/routing은 fallback 허용
    if not binary_checkpoints and mode == 'proportional':
        raise ValueError(
            "이진분류기 체크포인트가 없습니다. "
            "binary.binary_checkpoints를 지정하거나 mode=weighted/routing을 사용하세요."
        )

    log.info("=" * 70)
    log.info(f"🔀 이진분류기 보정 앙상블 (mode={mode})")
    log.info("=" * 70)
    log.info(f"메인 체크포인트: {len(main_checkpoints)}개")
    log.info(f"이진 체크포인트: {len(binary_checkpoints)}개" + (" (없음 → fallback)" if not binary_checkpoints else ""))
    log.info(f"TTA: {use_tta}" + (f" (level={tta_level})" if use_tta else ""))
    if mode != 'proportional':
        log.info(f"alpha={alpha}, theta={theta}, dynamic_alpha={dynamic_alpha}, confidence_threshold={confidence_threshold}")
    log.info(f"출력: {output}")

    # 데이터 로더 구성
    data_module = create_datamodule_from_config(cfg)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    device = get_simple_device()
    log.info(f"디바이스: {device}")

    # 1. 메인 분류기 확률 (N, 17)
    log.info("\n📊 메인 분류기 추론 중...")
    p_main = _predict_probs(
        main_checkpoints, test_loader, device,
        use_tta=use_tta, tta_level=tta_level, label="main"
    )
    log.info(f"  p_main shape: {p_main.shape}")

    # 2. 이진분류기 확률 (N, 2)
    if binary_checkpoints:
        log.info("\n🔬 이진분류기 추론 중...")
        p_bin = _predict_probs(
            binary_checkpoints, test_loader, device,
            use_tta=use_tta, tta_level=tta_level, label="binary"
        )
        log.info(f"  p_bin shape: {p_bin.shape}")
    else:
        log.info("\n🔬 이진분류기 없음 → 메인 class3·7 확률로 fallback p_bin 생성")
        p_bin = make_fallback_p_bin(p_main, class3_idx, class7_idx)
        log.info(f"  p_bin shape: {p_bin.shape} (fallback)")

    if p_main.shape[0] != p_bin.shape[0]:
        raise ValueError(
            f"샘플 수 불일치: p_main={p_main.shape[0]}, p_bin={p_bin.shape[0]}"
        )

    # 2-b. Grid Search (binary.grid_search=true일 때)
    if do_grid_search:
        if mode not in ('weighted', 'routing'):
            log.warning(f"grid_search는 mode=weighted/routing에서만 의미 있습니다. (현재 mode={mode})")
        else:
            log.info("\n🔍 Grid Search 시작 (validation set 기준)...")
            val_loader = data_module.val_dataloader()

            log.info("  validation set 메인 분류기 추론 중...")
            p_main_val = _predict_probs(
                main_checkpoints, val_loader, device,
                use_tta=use_tta, tta_level=tta_level, label="main/val"
            )
            if binary_checkpoints:
                log.info("  validation set 이진분류기 추론 중...")
                p_bin_val = _predict_probs(
                    binary_checkpoints, val_loader, device,
                    use_tta=use_tta, tta_level=tta_level, label="binary/val"
                )
            else:
                p_bin_val = make_fallback_p_bin(p_main_val, class3_idx, class7_idx)

            y_val = _collect_labels(val_loader)
            log.info(f"  validation 샘플 수: {len(y_val)}")

            gs_result = grid_search_params(
                p_main_val=p_main_val,
                p_bin_val=p_bin_val,
                y_val=y_val,
                mode=mode,
                theta_min=gs_theta_min,
                theta_max=gs_theta_max,
                theta_step=gs_theta_step,
                alpha_min=gs_alpha_min,
                alpha_max=gs_alpha_max,
                alpha_step=gs_alpha_step,
                dynamic_alpha=dynamic_alpha,
                confidence_threshold=confidence_threshold,
                class_3_idx=class3_idx,
                class_7_idx=class7_idx,
            )

            best = gs_result["best"]
            log.info(f"\n  📌 Grid Search 결과 (baseline F1={gs_result['baseline_f1']:.4f})")
            log.info(f"     최적 θ={best['theta']}, α={best['alpha']}, F1={best['f1']:.4f} (Δ={best['f1']-gs_result['baseline_f1']:+.4f})")

            heatmap_path = output.replace(".csv", "_grid_search.png")
            _plot_grid_search_heatmap(gs_result, heatmap_path, mode)

            # 최적 파라미터를 현재 실행에 반영
            theta = best["theta"]
            alpha = best["alpha"]
            log.info(f"  → theta={theta}, alpha={alpha}로 보정 진행")

            # 결과 저장
            gs_output = output.replace(".csv", "_grid_search.json")
            with open(gs_output, "w") as f:
                json.dump(gs_result, f, indent=2, ensure_ascii=False)
            log.info(f"  💾 Grid Search 결과 저장: {gs_output}")

    # 3. 보정 적용
    log.info(f"\n🔧 보정 적용 중 (mode={mode})...")
    pred_before = p_main.argmax(axis=1)

    p_corrected = binary_correction(
        p_main=p_main,
        p_bin=p_bin,
        mode=mode,
        alpha=alpha,
        theta=theta,
        dynamic_alpha=dynamic_alpha,
        confidence_threshold=confidence_threshold,
        class_3_idx=class3_idx,
        class_7_idx=class7_idx,
    )
    pred_after = p_corrected.argmax(axis=1)

    # routing 모드: 라우팅 대상 샘플 수 출력
    if mode == 'routing':
        pool = p_main[:, class3_idx] + p_main[:, class7_idx]
        n_routed = int((pool > theta).sum())
        log.info(f"  라우팅 대상: {n_routed}/{len(pred_before)}개 (pool>θ={theta})")

    _log_correction_stats(pred_before, pred_after, class3_idx, class7_idx)

    # 4. 결과 저장
    submission_csv = cfg.data.get('sample_submission_csv', cfg.data.get('test_csv', None))
    submission_csv_path = os.path.join(cfg.data.root_path, submission_csv)

    save_predictions_to_csv(
        predictions=pred_after.tolist(),
        output_path=output,
        data_root=cfg.data.root_path,
        test_csv_path=submission_csv_path,
        task_name=f"Binary Ensemble ({mode})",
    )


if __name__ == "__main__":
    main()
