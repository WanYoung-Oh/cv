"""
데이터셋 분석 스크립트
- 이미지 크기 분포, 클래스 분포, 메타 정보
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_image_info(img_path: Path) -> Tuple[int, int, float] | None:
    """이미지 크기 정보 반환 (w, h, aspect_ratio)"""
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            return w, h, w / h
    except Exception:
        return None


def analyze_class_sizes(train_df: pd.DataFrame, img_dir: Path, class_names: Dict) -> Dict:
    """Class별 이미지 크기 분석

    Args:
        train_df: 학습 데이터 DataFrame (image_name, label)
        img_dir: 이미지 디렉토리 경로
        class_names: Class ID to name mapping

    Returns:
        Dict: Class별 크기 통계 {class_id: {min/max/mean width/height}}
    """
    class_stats = {}

    print("\n[Class별 이미지 크기 분석 중...]")

    for class_id in tqdm(sorted(train_df.iloc[:, 1].unique())):
        # 해당 클래스의 이미지만 필터링
        class_df = train_df[train_df.iloc[:, 1] == class_id]
        img_paths = [img_dir / name for name in class_df.iloc[:, 0]]

        # 이미지 정보 수집
        widths, heights = [], []
        for img_path in img_paths:
            info = get_image_info(img_path)
            if info:
                widths.append(info[0])
                heights.append(info[1])

        if widths and heights:
            class_stats[class_id] = {
                'count': len(widths),
                'min_width': min(widths),
                'max_width': max(widths),
                'mean_width': np.mean(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'mean_height': np.mean(heights),
                'class_name': class_names.get(class_id, f'Class_{class_id}')
            }

    return class_stats


def recommend_input_size(class_stats: Dict) -> None:
    """CNN vs Transformer 모델을 위한 입력 크기 전략 추천

    Args:
        class_stats: Class별 크기 통계
    """
    # 전체 데이터 기준 통계
    all_min_width = min(s['min_width'] for s in class_stats.values())
    all_max_width = max(s['max_width'] for s in class_stats.values())
    all_min_height = min(s['min_height'] for s in class_stats.values())
    all_max_height = max(s['max_height'] for s in class_stats.values())
    all_mean_width = np.mean([s['mean_width'] for s in class_stats.values()])
    all_mean_height = np.mean([s['mean_height'] for s in class_stats.values()])

    print("\n" + "="*70)
    print("📊 입력 크기 전략 권장사항 (CNN vs Transformer)")
    print("="*70)

    print(f"\n전체 데이터셋 크기 범위:")
    print(f"  Width:  {all_min_width} ~ {all_max_width} (평균: {all_mean_width:.0f})")
    print(f"  Height: {all_min_height} ~ {all_max_height} (평균: {all_mean_height:.0f})")

    # CNN 모델 권장사항
    print(f"\n🔷 CNN 모델 (ResNet, EfficientNet)")
    cnn_sizes = [224, 256, 384, 512]
    recommended_cnn = min(cnn_sizes, key=lambda x: abs(x - all_mean_width))
    print(f"  권장 크기: {recommended_cnn}x{recommended_cnn}")
    print(f"  이유: 평균 크기({all_mean_width:.0f})에 가장 근접")
    print(f"  옵션: {cnn_sizes}")

    # Transformer 모델 권장사항
    print(f"\n🔶 Vision Transformer (ViT, Swin, DeiT)")
    vit_sizes = [224, 384, 512]
    recommended_vit = min(vit_sizes, key=lambda x: abs(x - all_mean_height))
    print(f"  권장 크기: {recommended_vit}x{recommended_vit}")
    print(f"  이유: ViT는 patch 단위 처리, 384+ 크기에서 성능 우수")
    print(f"  옵션: {vit_sizes}")

    # 추가 권장사항
    print(f"\n💡 추가 전략:")
    if all_max_width > 1000 or all_max_height > 1000:
        print(f"  ⚠️  큰 이미지 존재 → Multi-scale training 고려")
    if all_max_width / all_min_width > 3:
        print(f"  ⚠️  크기 편차 큼 → Adaptive pooling 또는 padding 전략 필요")

    aspect_ratio = all_mean_width / all_mean_height
    if aspect_ratio < 0.8 or aspect_ratio > 1.2:
        print(f"  ⚠️  종횡비 불균형 ({aspect_ratio:.2f}) → 정사각 resize 시 왜곡 주의")

    print("="*70)


def load_images_parallel(img_paths: List[Path], max_workers: int = 8) -> Tuple[List, List, List]:
    """병렬로 이미지 정보 로딩"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(get_image_info, img_paths), total=len(img_paths)))

    widths, heights, ratios = [], [], []
    for res in results:
        if res:
            widths.append(res[0])
            heights.append(res[1])
            ratios.append(res[2])

    return widths, heights, ratios


def compute_stats(widths: List, heights: List, ratios: List) -> Dict:
    """이미지 통계 계산"""
    return {
        "count": len(widths),
        "w_mean": np.mean(widths),
        "h_mean": np.mean(heights),
        "ar_mean": np.mean(ratios),
        "portrait": sum(r < 0.9 for r in ratios),
        "landscape": sum(r > 1.1 for r in ratios),
        "square": sum(0.9 <= r <= 1.1 for r in ratios)
    }


def plot_analysis(widths: List, heights: List, ratios: List,
                  class_counts: pd.Series, class_names: Dict, output_dir: str = ".benchmark_results"):
    """분석 시각화 및 저장"""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 이미지 크기 산점도
    axes[0].scatter(widths, heights, alpha=0.3, c='blue')
    axes[0].set_title("Image Size Distribution")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")

    # 클래스 분포
    labels = [class_names.get(i, f"C{i}") for i in class_counts.index]
    sns.barplot(x=class_counts.values, y=labels, hue=labels, palette="viridis", legend=False, ax=axes[1])
    axes[1].set_title("Class Distribution")

    # 종횡비 히스토그램
    axes[2].hist(ratios, bins=30, color='green', alpha=0.7)
    axes[2].axvline(1.0, color='red', linestyle='--', label='Square')
    axes[2].set_title("Aspect Ratio Distribution")
    axes[2].set_xlabel("Ratio (W/H)")

    plt.tight_layout()
    output_path = Path(output_dir) / "dataset_analysis.png"
    plt.savefig(output_path)
    print(f"\n[차트 저장: {output_path}]")


def calculate_class_weights(class_counts: pd.Series) -> pd.Series:
    """클래스 불균형 보정 가중치 계산"""
    weights = 1.0 / (class_counts / class_counts.sum())
    return weights / weights.min()


def save_to_csv(data: Dict | pd.DataFrame, filename: str, output_dir: str = ".benchmark_results") -> None:
    """데이터를 CSV 파일로 저장"""
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / filename

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[CSV 저장: {output_path}]")


def main():
    data_root = Path("datasets_fin")
    output_dir = ".benchmark_results"

    # 메타 정보 로드
    class_names = {}
    if (meta_path := data_root / "meta.csv").exists():
        meta_df = pd.read_csv(meta_path)
        class_names = dict(zip(meta_df.iloc[:, 0], meta_df.iloc[:, 1]))

    # 학습 데이터 로드
    train_df = pd.read_csv(data_root / "train.csv")
    img_paths = [data_root / "train" / name for name in train_df.iloc[:, 0]]

    # 이미지 분석
    print("\n[이미지 분석 중...]")
    widths, heights, ratios = load_images_parallel(img_paths)
    stats = compute_stats(widths, heights, ratios)

    print(f"\n[통계]")
    print(f"  평균 크기: {stats['w_mean']:.0f}x{stats['h_mean']:.0f} (AR: {stats['ar_mean']:.2f})")
    print(f"  방향성: 세로 {stats['portrait']}, 가로 {stats['landscape']}, 정사각 {stats['square']}")

    # 통계를 CSV로 저장
    stats_df = pd.DataFrame([{
        'total_images': stats['count'],
        'mean_width': f"{stats['w_mean']:.0f}",
        'mean_height': f"{stats['h_mean']:.0f}",
        'mean_aspect_ratio': f"{stats['ar_mean']:.2f}",
        'portrait_count': stats['portrait'],
        'landscape_count': stats['landscape'],
        'square_count': stats['square']
    }])
    save_to_csv(stats_df, 'dataset_statistics.csv', output_dir)

    # 클래스 분석
    class_counts = train_df.iloc[:, 1].value_counts().sort_index()
    weights = calculate_class_weights(class_counts)

    print("\n[클래스 가중치]")
    for idx, w in weights.items():
        print(f"  {class_names.get(idx, 'Unknown'):20s}: {w:.2f}")

    # 클래스 가중치를 CSV로 저장
    weights_df = pd.DataFrame([
        {
            'class_id': idx,
            'class_name': class_names.get(idx, 'Unknown'),
            'count': class_counts.get(idx, 0),
            'weight': f"{w:.2f}"
        }
        for idx, w in weights.items()
    ])
    save_to_csv(weights_df, 'class_weights.csv', output_dir)

    # 시각화
    plot_analysis(widths, heights, ratios, class_counts, class_names, output_dir)

    # 권장사항
    recommendations = []
    print("\n[권장사항]")
    if stats['ar_mean'] < 0.8:
        rec = "세로로 긴 이미지가 많음 → Padding 포함 Resize 권장"
        print(f"  - {rec}")
        recommendations.append(rec)
    if (imbalance := class_counts.max() / class_counts.min()) > 2.0:
        rec = f"클래스 불균형 {imbalance:.1f}배 → Focal Loss 또는 가중치 적용 권장"
        print(f"  - {rec}")
        recommendations.append(rec)

    # 권장사항을 CSV로 저장
    if recommendations:
        rec_df = pd.DataFrame([{'recommendation': rec} for rec in recommendations])
        save_to_csv(rec_df, 'recommendations.csv', output_dir)

    # Class별 이미지 크기 분석 (NEW)
    class_stats = analyze_class_sizes(train_df, data_root / "train", class_names)

    print("\n" + "="*70)
    print("📦 Class별 이미지 크기 분포")
    print("="*70)

    # 테이블 헤더
    print(f"{'Class':<20s} {'Count':>6s} {'Min W':>7s} {'Max W':>7s} {'Avg W':>7s} {'Min H':>7s} {'Max H':>7s} {'Avg H':>7s}")
    print("-" * 70)

    # Class별 통계를 DataFrame으로 변환
    class_stats_list = []

    # 각 Class 통계 출력
    for class_id in sorted(class_stats.keys()):
        s = class_stats[class_id]
        print(
            f"{s['class_name']:<20s} "
            f"{s['count']:>6d} "
            f"{s['min_width']:>7d} "
            f"{s['max_width']:>7d} "
            f"{s['mean_width']:>7.0f} "
            f"{s['min_height']:>7d} "
            f"{s['max_height']:>7d} "
            f"{s['mean_height']:>7.0f}"
        )

        # CSV용 데이터 수집
        class_stats_list.append({
            'class_id': class_id,
            'class_name': s['class_name'],
            'count': s['count'],
            'min_width': s['min_width'],
            'max_width': s['max_width'],
            'mean_width': f"{s['mean_width']:.0f}",
            'min_height': s['min_height'],
            'max_height': s['max_height'],
            'mean_height': f"{s['mean_height']:.0f}"
        })

    print("="*70)

    # Class별 크기 분포를 CSV로 저장
    class_stats_df = pd.DataFrame(class_stats_list)
    save_to_csv(class_stats_df, 'class_size_distribution.csv', output_dir)

    # CNN vs Transformer 입력 크기 전략 권장
    recommend_input_size(class_stats)

    # 입력 크기 전략 권장사항도 CSV로 저장
    all_mean_width = np.mean([s['mean_width'] for s in class_stats.values()])
    all_mean_height = np.mean([s['mean_height'] for s in class_stats.values()])

    cnn_sizes = [224, 256, 384, 512]
    recommended_cnn = min(cnn_sizes, key=lambda x: abs(x - all_mean_width))

    vit_sizes = [224, 384, 512]
    recommended_vit = min(vit_sizes, key=lambda x: abs(x - all_mean_height))

    input_size_rec = pd.DataFrame([
        {
            'model_type': 'CNN (ResNet, EfficientNet)',
            'recommended_size': f"{recommended_cnn}x{recommended_cnn}",
            'reason': f'평균 크기({all_mean_width:.0f})에 가장 근접',
            'options': ', '.join(map(str, cnn_sizes))
        },
        {
            'model_type': 'Transformer (ViT, Swin, DeiT)',
            'recommended_size': f"{recommended_vit}x{recommended_vit}",
            'reason': 'ViT는 patch 단위 처리, 384+ 크기에서 성능 우수',
            'options': ', '.join(map(str, vit_sizes))
        }
    ])
    save_to_csv(input_size_rec, 'input_size_recommendations.csv', output_dir)

    print(f"\n✅ 모든 결과가 '{output_dir}/' 디렉토리에 저장되었습니다.")


if __name__ == "__main__":
    main()
