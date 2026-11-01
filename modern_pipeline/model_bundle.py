"""Verify the pinned classroom model bundle without importing web dependencies."""
import hashlib
import json
from pathlib import Path

from .bootstrap import PROJECT_ROOT


def validate_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((PROJECT_ROOT / 'docs/final/manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest.items():
        path = folder / name
        if not path.is_file():
            raise ValueError('缺少模型文件，请先运行 python download_final_models.py：' + name)
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('模型校验不一致：' + name)
    return json.loads((folder / 'model_config.json').read_text(encoding='utf-8'))
