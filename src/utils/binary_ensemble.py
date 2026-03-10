"""
이진분류기 기반 클래스 3·7 보정 로직

Phase 1: 비례 재배분 (proportional redistribution)
  - pool = p_main[3] + p_main[7]
  - final[3] = p_bin[3] * pool, final[7] = p_bin[7] * pool

Phase 2: α 파라미터 + Confidence Routing
  - weighted: (1-α) * p_main + α * p_bin_scaled  (고정 또는 동적 α)
  - routing:  pool > θ인 샘플에만 weighted 보정 적용
  - dynamic_alpha: 메인 확신도가 confidence_threshold 초과 시 α 선형 감소

References:
  research.md §5.3 (시나리오 B 대응), §7.2 (Phase 2 구현 계획)
"""

from typing import Literal
import numpy as np


# 보정 대상 클래스 인덱스 (17클래스 기준)
_CLASS_3 = 3  # confirmation_of_admission_and_discharge (입퇴원확인서)
_CLASS_7 = 7  # medical_outpatient_certificate (외래진료확인서)


def proportional_redistribution(
    p_main: np.ndarray,
    p_bin: np.ndarray,
    class_3_idx: int = _CLASS_3,
    class_7_idx: int = _CLASS_7,
) -> np.ndarray:
    """Phase 1: 비례 재배분.

    메인 분류기의 class3·7 합산 확률(pool)을 보존하면서,
    그 안에서의 비율만 이진분류기로 재결정.

    final[3] = p_bin[3] * pool
    final[7] = p_bin[7] * pool
    final[k] = p_main[k]  (k ≠ 3, 7)

    Args:
        p_main: (N, C) 메인 분류기 확률 벡터
        p_bin:  (N, 2) 이진분류기 확률 ([:, 0]=class3_prob, [:, 1]=class7_prob)
        class_3_idx: p_main에서 class 3의 인덱스
        class_7_idx: p_main에서 class 7의 인덱스

    Returns:
        (N, C) 보정된 확률 벡터
    """
    final = p_main.copy()
    pool = p_main[:, class_3_idx] + p_main[:, class_7_idx]  # (N,)
    final[:, class_3_idx] = p_bin[:, 0] * pool
    final[:, class_7_idx] = p_bin[:, 1] * pool
    return final


def _compute_dynamic_alpha(
    p_main: np.ndarray,
    alpha_base: float,
    class_3_idx: int,
    class_7_idx: int,
    confidence_threshold: float,
) -> np.ndarray:
    """메인 확신도에 따른 동적 α 계산 (Phase 2).

    메인 분류기가 class 3 또는 7에 대해 강한 확신을 가질 때
    이진분류기의 영향(α)을 줄여 시나리오 B(메인 맞고 이진 틀림) 피해 방지.

    - max_conf <= confidence_threshold: alpha_base 유지
    - max_conf >  confidence_threshold: 선형 감소
      → max_conf = 1.0일 때 α = 0 (완전 보정 안 함)

    Args:
        p_main: (N, C) 메인 분류기 확률
        alpha_base: 기본 보정 강도 (0.0~1.0)
        confidence_threshold: α 감소 시작 임계값 (예: 0.7)

    Returns:
        (N,) 샘플별 α 값
    """
    max_conf = np.maximum(p_main[:, class_3_idx], p_main[:, class_7_idx])  # (N,)
    scale = np.where(
        max_conf > confidence_threshold,
        (1.0 - max_conf) / (1.0 - confidence_threshold + 1e-8),
        1.0,
    )
    return alpha_base * np.clip(scale, 0.0, 1.0)  # (N,)


def weighted_correction(
    p_main: np.ndarray,
    p_bin: np.ndarray,
    alpha: float = 0.5,
    class_3_idx: int = _CLASS_3,
    class_7_idx: int = _CLASS_7,
    dynamic_alpha: bool = False,
    confidence_threshold: float = 0.7,
) -> np.ndarray:
    """Phase 2: 가중 평균 보정 (α 파라미터).

    alpha=1.0이면 비례 재배분과 동일.
    alpha=0.0이면 보정 없음 (p_main 그대로).
    dynamic_alpha=True이면 메인 확신도에 따라 α를 자동 조절.

    final[3] = (1-α) * p_main[3] + α * p_bin[3] * pool
    final[7] = (1-α) * p_main[7] + α * p_bin[7] * pool
    final[k] = p_main[k]  (k ≠ 3, 7)

    Args:
        p_main: (N, C) 메인 분류기 확률
        p_bin:  (N, 2) 이진분류기 확률 ([:, 0]=class3, [:, 1]=class7)
        alpha:  보정 강도 (0.0=보정 없음, 1.0=완전 비례 재배분)
        dynamic_alpha: True이면 확신도에 따라 α 자동 조절
        confidence_threshold: 동적 α 감소 시작 임계값

    Returns:
        (N, C) 보정된 확률 벡터
    """
    pool = p_main[:, class_3_idx] + p_main[:, class_7_idx]  # (N,)

    if dynamic_alpha:
        alpha_arr = _compute_dynamic_alpha(
            p_main, alpha, class_3_idx, class_7_idx, confidence_threshold
        )  # (N,)
    else:
        alpha_arr = np.full(len(p_main), alpha)  # (N,)

    final = p_main.copy()
    final[:, class_3_idx] = (1 - alpha_arr) * p_main[:, class_3_idx] + alpha_arr * p_bin[:, 0] * pool
    final[:, class_7_idx] = (1 - alpha_arr) * p_main[:, class_7_idx] + alpha_arr * p_bin[:, 1] * pool
    return final


def binary_correction(
    p_main: np.ndarray,
    p_bin: np.ndarray,
    mode: Literal["proportional", "weighted", "routing"] = "routing",
    alpha: float = 0.8,
    theta: float = 0.3,
    dynamic_alpha: bool = True,
    confidence_threshold: float = 0.7,
    class_3_idx: int = _CLASS_3,
    class_7_idx: int = _CLASS_7,
) -> np.ndarray:
    """통합 이진분류기 보정 함수 (Phase 1 + Phase 2).

    Args:
        p_main: (N, C) 메인 분류기 확률
        p_bin:  (N, 2) 이진분류기 확률 ([:, 0]=class3, [:, 1]=class7)
        mode:
            "proportional" — Phase 1 비례 재배분 (α, θ 무시)
            "weighted"     — Phase 2 α 가중 평균 (θ 무시)
            "routing"      — Phase 2 θ 조건 분기 + α 가중 평균
        alpha: 보정 강도 (0.0~1.0). dynamic_alpha=True이면 기준값으로 사용.
        theta: 라우팅 임계값. pool(=p_main[3]+p_main[7]) > theta인 샘플에만 보정.
        dynamic_alpha: True이면 메인 확신도가 confidence_threshold 초과 시 α 감소.
        confidence_threshold: 동적 α 감소 시작 기준 (research.md §5.3 p>0.7 대응).

    Returns:
        (N, C) 보정된 확률 벡터
    """
    if mode == "proportional":
        return proportional_redistribution(p_main, p_bin, class_3_idx, class_7_idx)

    if mode == "weighted":
        return weighted_correction(
            p_main, p_bin, alpha, class_3_idx, class_7_idx,
            dynamic_alpha, confidence_threshold,
        )

    if mode == "routing":
        pool = p_main[:, class_3_idx] + p_main[:, class_7_idx]  # (N,)
        routed = pool > theta  # (N,) bool

        final = p_main.copy()
        if routed.any():
            corrected = weighted_correction(
                p_main[routed], p_bin[routed], alpha,
                class_3_idx, class_7_idx, dynamic_alpha, confidence_threshold,
            )
            final[routed] = corrected
        return final

    raise ValueError(f"Unknown mode: '{mode}'. Use 'proportional', 'weighted', or 'routing'.")


def grid_search_params(
    p_main_val: np.ndarray,
    p_bin_val: np.ndarray,
    y_val: np.ndarray,
    mode: Literal["weighted", "routing"] = "routing",
    theta_min: float = 0.05,
    theta_max: float = 0.80,
    theta_step: float = 0.05,
    alpha_min: float = 0.3,
    alpha_max: float = 1.0,
    alpha_step: float = 0.1,
    dynamic_alpha: bool = True,
    confidence_threshold: float = 0.7,
    class_3_idx: int = _CLASS_3,
    class_7_idx: int = _CLASS_7,
) -> dict:
    """θ, α grid search로 최적 파라미터 탐색.

    validation set에서 F1-Macro를 최대화하는 (θ, α) 조합을 탐색합니다.
    - mode='routing' : θ × α 2D grid search
    - mode='weighted': α만 탐색 (θ 의미 없음 → theta_min 고정)

    Args:
        p_main_val: (N, C) validation set 메인 분류기 확률
        p_bin_val:  (N, 2) validation set 이진분류기 확률
        y_val:      (N,)   validation set 정답 레이블
        mode: "weighted" | "routing"
        theta_min/max/step: θ 탐색 범위 및 간격
        alpha_min/max/step: α 탐색 범위 및 간격
        dynamic_alpha: 동적 α 사용 여부
        confidence_threshold: 동적 α 임계값

    Returns:
        {
          "best": {"theta": float, "alpha": float, "f1": float},
          "results": [{"theta", "alpha", "f1"}, ...] (F1 내림차순),
          "baseline_f1": float  # 보정 없을 때 F1
        }
    """
    from sklearn.metrics import f1_score

    # 보정 없는 baseline
    baseline_f1 = f1_score(y_val, p_main_val.argmax(axis=1), average="macro", zero_division=0)

    thetas = np.arange(theta_min, theta_max + theta_step / 2, theta_step).round(4)
    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step).round(4)

    # weighted 모드는 θ가 무의미하므로 1개만 순회
    if mode == "weighted":
        thetas = np.array([0.0])

    results = []
    best = {"theta": None, "alpha": None, "f1": -1.0}

    for alpha in alphas:
        for theta in thetas:
            p_corr = binary_correction(
                p_main_val, p_bin_val,
                mode=mode,
                alpha=float(alpha),
                theta=float(theta),
                dynamic_alpha=dynamic_alpha,
                confidence_threshold=confidence_threshold,
                class_3_idx=class_3_idx,
                class_7_idx=class_7_idx,
            )
            f1 = float(f1_score(y_val, p_corr.argmax(axis=1), average="macro", zero_division=0))
            results.append({"theta": float(theta), "alpha": float(alpha), "f1": round(f1, 6)})

            if f1 > best["f1"]:
                best = {"theta": float(theta), "alpha": float(alpha), "f1": round(f1, 6)}

    results.sort(key=lambda x: -x["f1"])
    return {"best": best, "results": results, "baseline_f1": round(baseline_f1, 6)}


def make_fallback_p_bin(p_main: np.ndarray, class_3_idx: int = _CLASS_3, class_7_idx: int = _CLASS_7) -> np.ndarray:
    """이진분류기 체크포인트 없을 때 메인 확률에서 p_bin 생성 (fallback).

    pool = p_main[3] + p_main[7]로 재정규화.
    이진분류기가 없어도 routing/weighted 모드 동작 가능.

    Returns:
        (N, 2) — [:, 0]=class3 상대 확률, [:, 1]=class7 상대 확률
    """
    pool = p_main[:, class_3_idx] + p_main[:, class_7_idx] + 1e-8  # (N,)
    p_bin = np.stack([
        p_main[:, class_3_idx] / pool,
        p_main[:, class_7_idx] / pool,
    ], axis=1)  # (N, 2)
    return p_bin
