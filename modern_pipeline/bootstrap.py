"""Select the isolated modern YOLO runtime while retaining the d2l CUDA stack."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = Path(os.environ.get('SRTP_MODERN_HOME', r'D:\SRTP_modern_20261027')).resolve()

def configure():
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ARTIFACT_ROOT / 'runtime' / 'settings'))
    vendor = ARTIFACT_ROOT / 'runtime' / 'site-packages'
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
