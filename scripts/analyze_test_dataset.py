"""
데이터셋 분석 스크립트
학습/검증/테스트 데이터의 분포와 특성을 분석합니다.

데이터 증강 전략 수립을 위한 상세 분석:
- 좌우/상하 반전 상태
- 회전 정도 (0/90/180/270도)
- Ink/Paper 분리 정도
- 그림자 존재 여부
- 색상 변화 (흑백/컬러)
- 스캔 노이즈 수준
- 선명도 (Blur 정도)
- 구김/왜곡 (Distortion)
- 일부 가림 (Occlusion)
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from src.data.datamodule import DocumentImageDataModule


def analyze_image_characteristics(image_path: str) -> dict:
    """이미지의 다양한 특성을 분석합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        dict: 이미지 특성 분석 결과
    """
    try:
        # 이미지 로드 (OpenCV와 PIL 모두 사용)
        img_cv = cv2.imread(str(image_path))
        img_pil = Image.open(image_path)
        
        if img_cv is None or img_pil is None:
            return None
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # ===== 1. 방향성 분석 (Orientation) =====
        # 텍스트 방향 추정 (수평/수직 엣지 비율)
        edges = cv2.Canny(gray, 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        horizontal_edges = np.sum(np.abs(sobel_y) > 50)
        vertical_edges = np.sum(np.abs(sobel_x) > 50)
        orientation_ratio = horizontal_edges / (vertical_edges + 1e-6)
        
        # 방향 추정 (1.0 근처면 정상, 매우 크거나 작으면 회전 가능성)
        if 0.7 < orientation_ratio < 1.3:
            orientation = "normal"
        elif orientation_ratio > 1.3:
            orientation = "horizontal_dominant"  # 90도 회전 가능성
        else:
            orientation = "vertical_dominant"    # 270도 회전 가능성
            
        # ===== 2. Ink/Paper 대비 분석 =====
        # 밝은 영역(종이)과 어두운 영역(잉크)의 분리도
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink_ratio = np.sum(binary == 0) / (h * w)  # 어두운 픽셀 비율
        
        # 히스토그램 분석 (두 개의 피크가 명확하면 대비가 좋음)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # 대비 점수 계산
        contrast_score = np.std(gray) / 128.0  # 표준편차를 0-1로 정규화
        
        # ===== 3. 그림자 감지 =====
        # 이미지를 4등분하여 밝기 차이 분석
        h_mid, w_mid = h // 2, w // 2
        quadrants = [
            gray[:h_mid, :w_mid],      # 좌상
            gray[:h_mid, w_mid:],      # 우상
            gray[h_mid:, :w_mid],      # 좌하
            gray[h_mid:, w_mid:]       # 우하
        ]
        
        mean_brightness = [np.mean(q) for q in quadrants]
        brightness_std = np.std(mean_brightness)
        
        # 밝기 차이가 크면 그림자 존재 가능성
        has_shadow = brightness_std > 20  # 임계값은 조정 가능
        shadow_score = min(brightness_std / 50.0, 1.0)  # 0-1 정규화
        
        # ===== 4. 색상 분석 =====
        # 흑백인지 컬러인지 판단
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        
        if mean_saturation < 15:
            color_type = "grayscale"
        elif mean_saturation < 30:
            color_type = "low_saturation"
        else:
            color_type = "colored"
            
        # 종이 색상 추정 (가장 밝은 픽셀들의 평균)
        top_bright = np.percentile(gray, 95)
        paper_color = "white" if top_bright > 220 else "off_white" if top_bright > 180 else "tinted"
        
        # ===== 5. 노이즈 분석 =====
        # 라플라시안 변화량으로 노이즈 추정
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_score = np.var(laplacian) / 10000.0  # 정규화
        noise_score = min(noise_score, 1.0)
        
        if noise_score < 0.1:
            noise_level = "clean"
        elif noise_score < 0.3:
            noise_level = "low"
        elif noise_score < 0.6:
            noise_level = "medium"
        else:
            noise_level = "high"
            
        # ===== 6. 선명도 분석 (Blur Detection) =====
        # 라플라시안 분산으로 선명도 측정
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            sharpness = "blurry"
        elif laplacian_var < 500:
            sharpness = "moderate"
        else:
            sharpness = "sharp"
            
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        
        # ===== 7. 구김/왜곡 분석 =====
        # 직선 검출 (Hough Transform)
        edges_for_lines = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges_for_lines, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # 직선들의 각도 분포 분석
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                angles.append(angle)
            
            # 각도가 0, 90도에서 크게 벗어나면 왜곡 가능성
            angles_std = np.std(angles) if len(angles) > 0 else 0
            distortion_score = min(angles_std / 45.0, 1.0)
        else:
            distortion_score = 0.0
            
        if distortion_score < 0.2:
            distortion_level = "none"
        elif distortion_score < 0.5:
            distortion_level = "low"
        else:
            distortion_level = "high"
            
        # ===== 8. 가림 분석 (Occlusion) =====
        # 이미지 가장자리의 검은 영역이나 비정상적인 영역 감지
        border_size = 20
        border_pixels = np.concatenate([
            gray[:border_size, :].flatten(),
            gray[-border_size:, :].flatten(),
            gray[:, :border_size].flatten(),
            gray[:, -border_size:].flatten()
        ])
        
        dark_border_ratio = np.sum(border_pixels < 50) / len(border_pixels)
        
        # 중앙 영역의 비정상적인 밝기 영역 감지
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        abnormal_bright = np.sum(center_region > 250) / center_region.size
        abnormal_dark = np.sum(center_region < 30) / center_region.size
        
        occlusion_score = max(dark_border_ratio, abnormal_bright, abnormal_dark)
        
        if occlusion_score < 0.05:
            occlusion_level = "none"
        elif occlusion_score < 0.15:
            occlusion_level = "low"
        else:
            occlusion_level = "high"
        
        return {
            "orientation": orientation,
            "orientation_ratio": float(orientation_ratio),
            "ink_paper_contrast": float(contrast_score),
            "ink_ratio": float(ink_ratio),
            "shadow_detected": bool(has_shadow),
            "shadow_score": float(shadow_score),
            "color_type": color_type,
            "paper_color": paper_color,
            "mean_saturation": float(mean_saturation),
            "noise_level": noise_level,
            "noise_score": float(noise_score),
            "sharpness": sharpness,
            "sharpness_score": float(sharpness_score),
            "distortion_level": distortion_level,
            "distortion_score": float(distortion_score),
            "occlusion_level": occlusion_level,
            "occlusion_score": float(occlusion_score),
            "image_size": f"{w}x{h}"
        }
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None


def summarize_characteristics(characteristics_list: list) -> dict:
    """이미지 특성들을 요약합니다.
    
    Args:
        characteristics_list: 각 이미지의 특성 딕셔너리 리스트
        
    Returns:
        dict: 전체 데이터셋의 특성 요약
    """
    summary = {
        "total_images": len(characteristics_list),
        "orientation": defaultdict(int),
        "color_type": defaultdict(int),
        "paper_color": defaultdict(int),
        "noise_level": defaultdict(int),
        "sharpness": defaultdict(int),
        "distortion_level": defaultdict(int),
        "occlusion_level": defaultdict(int),
        "shadow_detected": {"yes": 0, "no": 0},
        "statistics": {
            "ink_paper_contrast": [],
            "shadow_score": [],
            "noise_score": [],
            "sharpness_score": [],
            "distortion_score": [],
            "occlusion_score": [],
            "orientation_ratio": []
        }
    }
    
    for char in characteristics_list:
        if char is None:
            continue
            
        # 카테고리 집계
        summary["orientation"][char["orientation"]] += 1
        summary["color_type"][char["color_type"]] += 1
        summary["paper_color"][char["paper_color"]] += 1
        summary["noise_level"][char["noise_level"]] += 1
        summary["sharpness"][char["sharpness"]] += 1
        summary["distortion_level"][char["distortion_level"]] += 1
        summary["occlusion_level"][char["occlusion_level"]] += 1
        summary["shadow_detected"]["yes" if char["shadow_detected"] else "no"] += 1
        
        # 통계값 수집
        for key in summary["statistics"].keys():
            summary["statistics"][key].append(char[key])
    
    # 통계 요약 계산
    for key, values in summary["statistics"].items():
        if values:
            summary["statistics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }
    
    # 기본 딕셔너리를 일반 딕셔너리로 변환
    summary["orientation"] = dict(summary["orientation"])
    summary["color_type"] = dict(summary["color_type"])
    summary["paper_color"] = dict(summary["paper_color"])
    summary["noise_level"] = dict(summary["noise_level"])
    summary["sharpness"] = dict(summary["sharpness"])
    summary["distortion_level"] = dict(summary["distortion_level"])
    summary["occlusion_level"] = dict(summary["occlusion_level"])
    
    return summary


def generate_augmentation_recommendations(summary: dict) -> dict:
    """분석 결과를 바탕으로 데이터 증강 전략을 제안합니다.
    
    Args:
        summary: 데이터셋 특성 요약
        
    Returns:
        dict: 증강 전략 제안
    """
    recommendations = {
        "essential": [],  # 필수 증강
        "recommended": [],  # 권장 증강
        "optional": [],  # 선택적 증강
        "not_needed": []  # 불필요한 증강
    }
    
    total = summary["total_images"]
    
    # 1. 방향성 분석
    normal_ratio = summary["orientation"].get("normal", 0) / total
    if normal_ratio > 0.8:
        recommendations["essential"].append({
            "type": "Rotation (RandomRotate90, Rotate)",
            "reason": f"대부분 정상 방향 ({normal_ratio*100:.1f}%), 회전 다양성 필요"
        })
    else:
        recommendations["not_needed"].append({
            "type": "Rotation",
            "reason": "이미 다양한 방향 존재"
        })
    
    # 2. 좌우/상하 반전
    recommendations["recommended"].append({
        "type": "HorizontalFlip, VerticalFlip",
        "reason": "문서 방향 다양화 (표준 증강)"
    })
    
    # 3. Ink/Paper 대비
    contrast_mean = summary["statistics"]["ink_paper_contrast"]["mean"]
    if contrast_mean < 0.4:
        recommendations["essential"].append({
            "type": "CLAHE, RandomBrightnessContrast",
            "reason": f"낮은 대비 ({contrast_mean:.2f}), 대비 향상 필수"
        })
    elif contrast_mean < 0.6:
        recommendations["recommended"].append({
            "type": "CLAHE",
            "reason": f"보통 대비 ({contrast_mean:.2f}), 대비 향상 권장"
        })
    
    # 4. 그림자
    shadow_ratio = summary["shadow_detected"]["yes"] / total
    if shadow_ratio > 0.3:
        recommendations["essential"].append({
            "type": "RandomBrightnessContrast, Shadow removal",
            "reason": f"그림자 있음 ({shadow_ratio*100:.1f}%), 밝기 조정 필수"
        })
    elif shadow_ratio > 0.1:
        recommendations["recommended"].append({
            "type": "RandomBrightnessContrast",
            "reason": f"일부 그림자 ({shadow_ratio*100:.1f}%)"
        })
    
    # 5. 색상
    colored_ratio = summary["color_type"].get("colored", 0) / total
    if colored_ratio > 0.3:
        recommendations["recommended"].append({
            "type": "ColorJitter",
            "reason": f"컬러 이미지 많음 ({colored_ratio*100:.1f}%), 색상 다양화"
        })
    
    # 6. 노이즈
    high_noise = summary["noise_level"].get("high", 0) + summary["noise_level"].get("medium", 0)
    noise_ratio = high_noise / total
    if noise_ratio > 0.3:
        recommendations["essential"].append({
            "type": "GaussNoise, ISONoise",
            "reason": f"노이즈 많음 ({noise_ratio*100:.1f}%), 노이즈 증강 필수"
        })
    elif noise_ratio > 0.1:
        recommendations["optional"].append({
            "type": "GaussNoise",
            "reason": f"일부 노이즈 ({noise_ratio*100:.1f}%)"
        })
    
    # 7. 선명도
    blurry_ratio = summary["sharpness"].get("blurry", 0) / total
    if blurry_ratio > 0.2:
        recommendations["recommended"].append({
            "type": "Blur, MotionBlur",
            "reason": f"흐린 이미지 많음 ({blurry_ratio*100:.1f}%), Blur 증강 권장"
        })
    else:
        recommendations["optional"].append({
            "type": "Sharpen",
            "reason": "대부분 선명함, 선명도 증강 선택적"
        })
    
    # 8. 구김/왜곡
    high_distortion = summary["distortion_level"].get("high", 0)
    distortion_ratio = high_distortion / total
    if distortion_ratio > 0.2:
        recommendations["essential"].append({
            "type": "GridDistortion, ElasticTransform",
            "reason": f"왜곡 많음 ({distortion_ratio*100:.1f}%), 구김 증강 필수"
        })
    elif distortion_ratio > 0.05:
        recommendations["recommended"].append({
            "type": "GridDistortion",
            "reason": f"일부 왜곡 ({distortion_ratio*100:.1f}%)"
        })
    
    # 9. 가림
    high_occlusion = summary["occlusion_level"].get("high", 0)
    occlusion_ratio = high_occlusion / total
    if occlusion_ratio > 0.2:
        recommendations["essential"].append({
            "type": "CoarseDropout, Cutout",
            "reason": f"가림 많음 ({occlusion_ratio*100:.1f}%), 가림 증강 필수"
        })
    elif occlusion_ratio > 0.05:
        recommendations["recommended"].append({
            "type": "CoarseDropout",
            "reason": f"일부 가림 ({occlusion_ratio*100:.1f}%)"
        })
    
    # 10. 원근 변환
    recommendations["optional"].append({
        "type": "Perspective",
        "reason": "스캔 각도 다양화 (선택적)"
    })
    
    return recommendations


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """데이터셋 분석 메인 함수
    
    사용 예시:
        # 테스트 데이터 분석 (전체)
        python scripts/analyze_test_dataset.py
        
        # 테스트 데이터 샘플링 분석 (500개)
        python scripts/analyze_test_dataset.py +sample_size=500
        
        # 학습 데이터 분석
        python scripts/analyze_test_dataset.py +dataset_type=train
        
        # 상세 분석 비활성화 (빠른 분석)
        python scripts/analyze_test_dataset.py +detailed_analysis=false
    """
    from omegaconf import OmegaConf
    import json
    import pandas as pd
    
    # 분석 옵션
    dataset_type = OmegaConf.select(cfg, 'dataset_type', default='test')
    sample_size = OmegaConf.select(cfg, 'sample_size', default=None)
    detailed_analysis = OmegaConf.select(cfg, 'detailed_analysis', default=True)
    
    print("=" * 80)
    print("📊 데이터셋 상세 분석 시작 - 데이터 증강 전략 수립용")
    print("=" * 80)
    print(f"데이터셋: {dataset_type}")
    print(f"샘플 크기: {sample_size if sample_size else '전체'}")
    print(f"상세 분석: {'ON' if detailed_analysis else 'OFF'}")
    print()
    
    # 이미지 경로 가져오기
    data_root = Path(cfg.data.root_path)
    
    if dataset_type == 'test':
        csv_path = data_root / cfg.data.get('sample_submission_csv', cfg.data.get('test_csv'))
        image_dir = data_root / cfg.data.get('test_image_dir', 'test/')
    else:
        csv_path = data_root / cfg.data.train_csv
        image_dir = data_root / cfg.data.get('train_image_dir', 'train/')
    
    # CSV 로드
    df = pd.read_csv(csv_path)
    print(f"📂 CSV 로드: {csv_path}")
    print(f"   총 {len(df)}개 이미지")
    
    # 샘플링
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"   → {sample_size}개 샘플링")
    
    print()
    
    # 기본 통계 분석
    print("=" * 80)
    print("1️⃣  기본 통계 분석")
    print("=" * 80)
    
    # DataModule을 통한 기본 분석
    data_module = DocumentImageDataModule(
        data_root=cfg.data.root_path,
        train_csv=cfg.data.train_csv,
        test_csv=cfg.data.get('sample_submission_csv', cfg.data.get('test_csv')),
        train_image_dir=cfg.data.get('train_image_dir', 'train/'),
        test_image_dir=cfg.data.get('test_image_dir', 'test/'),
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        train_val_split=cfg.data.train_val_split,
        normalization=cfg.data.normalization,
        augmentation=cfg.data.augmentation,
        seed=cfg.get('seed', 42),
    )
    
    if dataset_type in ['train', 'val']:
        data_module.setup(stage='fit')
    else:
        data_module.setup(stage='test')
    
    info = data_module.get_dataset_info()
    print(f"이미지 크기: {info['img_size']}x{info['img_size']}")
    print(f"배치 크기: {info['batch_size']}")
    
    if 'test' in info:
        print(f"테스트 데이터: {info['test']['size']:,}개")
    
    print()
    
    # 상세 이미지 특성 분석
    if detailed_analysis:
        print("=" * 80)
        print("2️⃣  상세 이미지 특성 분석")
        print("=" * 80)
        print("분석 항목:")
        print("  ✓ 방향성 (좌우/상하 반전, 회전)")
        print("  ✓ Ink/Paper 대비")
        print("  ✓ 그림자 존재")
        print("  ✓ 색상 유형 (흑백/컬러)")
        print("  ✓ 스캔 노이즈 수준")
        print("  ✓ 선명도 (Blur)")
        print("  ✓ 구김/왜곡 (Distortion)")
        print("  ✓ 일부 가림 (Occlusion)")
        print()
        
        # 이미지별 특성 분석
        characteristics_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="이미지 분석 중"):
            image_filename = row.get('ID', row.get('image_id', None))
            if image_filename is None:
                continue
                
            image_path = image_dir / image_filename
            
            if not image_path.exists():
                continue
                
            char = analyze_image_characteristics(str(image_path))
            if char:
                characteristics_list.append(char)
        
        print(f"\n✅ {len(characteristics_list)}개 이미지 분석 완료")
        print()
        
        # 특성 요약
        print("=" * 80)
        print("3️⃣  데이터셋 특성 요약")
        print("=" * 80)
        
        summary = summarize_characteristics(characteristics_list)
        
        print(f"\n📊 방향성 분석:")
        for orient, count in sorted(summary["orientation"].items(), key=lambda x: -x[1]):
            print(f"   {orient:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n🎨 색상 유형:")
        for color, count in sorted(summary["color_type"].items(), key=lambda x: -x[1]):
            print(f"   {color:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n📄 종이 색상:")
        for paper, count in sorted(summary["paper_color"].items(), key=lambda x: -x[1]):
            print(f"   {paper:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n📡 노이즈 수준:")
        for noise, count in sorted(summary["noise_level"].items(), key=lambda x: -x[1]):
            print(f"   {noise:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n🔍 선명도:")
        for sharp, count in sorted(summary["sharpness"].items(), key=lambda x: -x[1]):
            print(f"   {sharp:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n📐 구김/왜곡:")
        for dist, count in sorted(summary["distortion_level"].items(), key=lambda x: -x[1]):
            print(f"   {dist:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n🚫 일부 가림:")
        for occl, count in sorted(summary["occlusion_level"].items(), key=lambda x: -x[1]):
            print(f"   {occl:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n🌓 그림자:")
        for shadow, count in summary["shadow_detected"].items():
            print(f"   {shadow:25s}: {count:4d}개 ({count/summary['total_images']*100:5.1f}%)")
        
        print(f"\n📈 통계 요약:")
        for metric, values in summary["statistics"].items():
            print(f"   {metric:25s}: mean={values['mean']:.3f}, std={values['std']:.3f}")
        
        print()
        
        # 데이터 증강 전략 제안
        print("=" * 80)
        print("4️⃣  데이터 증강 전략 제안")
        print("=" * 80)
        print()
        
        recommendations = generate_augmentation_recommendations(summary)
        
        if recommendations["essential"]:
            print("🔴 필수 증강 (Essential):")
            for i, rec in enumerate(recommendations["essential"], 1):
                print(f"   {i}. {rec['type']}")
                print(f"      └─ {rec['reason']}")
        
        if recommendations["recommended"]:
            print("\n🟡 권장 증강 (Recommended):")
            for i, rec in enumerate(recommendations["recommended"], 1):
                print(f"   {i}. {rec['type']}")
                print(f"      └─ {rec['reason']}")
        
        if recommendations["optional"]:
            print("\n🟢 선택적 증강 (Optional):")
            for i, rec in enumerate(recommendations["optional"], 1):
                print(f"   {i}. {rec['type']}")
                print(f"      └─ {rec['reason']}")
        
        if recommendations["not_needed"]:
            print("\n⚪ 불필요한 증강 (Not Needed):")
            for i, rec in enumerate(recommendations["not_needed"], 1):
                print(f"   {i}. {rec['type']}")
                print(f"      └─ {rec['reason']}")
        
        print()
        
        # 결과 저장
        output_dir = Path("analysis")
        output_dir.mkdir(exist_ok=True)
        
        # 상세 분석 결과
        detailed_output = {
            "dataset_type": dataset_type,
            "total_images_analyzed": len(characteristics_list),
            "summary": summary,
            "recommendations": recommendations,
            "all_characteristics": characteristics_list
        }
        
        output_path = output_dir / f"{dataset_type}_detailed_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 상세 분석 결과 저장: {output_path}")
        
        # 추천 사항만 별도 저장
        rec_output_path = output_dir / f"{dataset_type}_augmentation_recommendations.json"
        with open(rec_output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        print(f"💾 증강 전략 제안 저장: {rec_output_path}")
    
    print()
    print("=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
