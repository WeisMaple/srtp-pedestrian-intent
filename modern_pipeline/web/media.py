"""Video validation and browser-compatible exports, without shell interpolation."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2

from modern_pipeline.bootstrap import PROJECT_ROOT
from modern_pipeline.cli import PredictionCancelled


def validate_bundle(folder):
    manifest = json.loads((PROJECT_ROOT / 'docs/final/manifest.json').read_text())
    for name, record in manifest.items():
        path = folder / name
        if not path.is_file():
            raise ValueError('缺少模型文件，请先运行 python download_final_models.py：' + name)
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('模型校验不一致：' + name)
    return json.loads((folder / 'model_config.json').read_text(encoding='utf-8'))


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, image = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not ok or image is None or not math.isfinite(fps) or fps <= 0 or frames < 1:
            raise ValueError('无法读取视频，请选择有效的 MP4、MOV 或 AVI 文件。')
        height, width = image.shape[:2]
        return {'width': width, 'height': height, 'fps': fps, 'frame_count': frames,
                'duration': frames / fps, 'bytes': path.stat().st_size}, image
    finally:
        cap.release()


def compatible_source(path):
    probe = shutil.which('ffprobe')
    if not probe or path.suffix.lower() not in ('.mp4', '.mov'):
        return False
    result = subprocess.run([probe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=codec_name,pix_fmt', '-of', 'json', str(path)],
                            capture_output=True, timeout=15)
    try:
        stream = json.loads(result.stdout)['streams'][0]
        return stream['codec_name'] == 'h264' and stream.get('pix_fmt') == 'yuv420p'
    except (ValueError, KeyError, IndexError):
        return False


def transcode_video(source, target, cancel=None):
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('缺少 ffmpeg，无法生成浏览器可播放的结果视频。')
    command = [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(source),
               '-map', '0:v:0', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(target)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise PredictionCancelled('Analysis cancelled while encoding')
                time.sleep(0.05)
            if process.returncode:
                errors.seek(0)
                raise RuntimeError('视频转码失败：' + errors.read(3000).decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
