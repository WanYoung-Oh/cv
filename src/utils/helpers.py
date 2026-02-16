"""
유틸리티 함수
"""

import json
from pathlib import Path
from typing import Dict, Any


def save_json(data: Dict[str, Any], save_path: str) -> None:
    """데이터를 JSON으로 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
