"""One inference worker, persistent task snapshots and bounded visualization traces."""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np

from modern_pipeline.bootstrap import PROJECT_ROOT
from modern_pipeline.cli import PredictionCancelled, arguments, run
from modern_pipeline.live_trace import LiveBuffer
from .media import compatible_source, probe_video, transcode_video, validate_bundle

BUSY = {'loading', 'running', 'encoding'}
STAGES = ['读取视频帧', '提取 COCO17 骨架', '分配并跟踪行人 ID',
          '积累行人骨架历史', '预测未来骨架', '更新意图分类结果']


class JobBusy(ValueError):
    pass


@dataclass
class Job:
    directory: Path
    state: dict
    cancel: threading.Event = field(default_factory=threading.Event)
    traces: deque = field(default_factory=lambda: deque(maxlen=60))
    last_preview: float = -1
    last_trace: float = -1
    seen_stages: set = field(default_factory=set)
    last_image: object = None
    last_trace_event: object = None
    live: object = None

    def __post_init__(self):
        self.live = LiveBuffer(self.state['id'])


def visual_data(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 3).tolist()
    if isinstance(value, dict):
        return {k: visual_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [visual_data(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class JobManager:
    def __init__(self, root, model_dir=None, run_pipeline=run):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir or PROJECT_ROOT / 'weights/final')
        self.run_pipeline = run_pipeline
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-analysis')
        self.jobs = {}
        self.active = None
        self.network_active = False
        self.network_cache = OrderedDict()
        self.network_predictor = None
        self.frontend_cache = OrderedDict()
        self.frontend_predictor = None
        self.config, self.model_error = None, None
        try:
            self.config = validate_bundle(self.model_dir)
        except (OSError, ValueError) as exc:
            self.model_error = str(exc)
        self._recover()

    def _recover(self):
        for path in self.root.glob('*/state.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if path.parent.name != state['id']:
                    continue
                job = Job(path.parent, state)
                trace_file = path.parent / 'trace.json'
                if trace_file.is_file():
                    job.traces.extend(json.loads(trace_file.read_text())['samples'][-60:])
                live_file = path.parent / 'live.json'
                if live_file.is_file():
                    job.live.restore(live_file)
                if state['status'] in BUSY:
                    state.update(status='failed', error='服务重启，先前的分析已中断。', outputs={})
                    job.live.finish('failed')
                    job.live.save(live_file, 'failed')
                    self.persist(job)
                self.jobs[state['id']] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def health(self):
        return {'models_ready': self.config is not None, 'encoder_ready': bool(shutil.which('ffmpeg')),
                'model_name': '行人意图预测 · v' + self.config.get('version', '1.0.0') if self.config else '模型未就绪',
                'device': 'cpu', 'active_job': self.active, 'detail': self.model_error}

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def snapshot(self, job_id):
        with self.lock:
            return deepcopy(self.get(job_id).state)

    def persist(self, job):
        data = json.dumps(job.state, ensure_ascii=False, allow_nan=False)
        temporary = job.directory / 'state.tmp'
        temporary.write_text(data, encoding='utf-8')
        temporary.replace(job.directory / 'state.json')

    def register(self, uploaded, filename):
        metadata, first_frame = probe_video(uploaded)
        identifier = uuid.uuid4().hex
        directory = self.root / identifier
        directory.mkdir()
        source = directory / ('input' + Path(filename).suffix.lower())
        try:
            shutil.move(str(uploaded), source)
            height, width = first_frame.shape[:2]
            poster = cv2.resize(first_frame, (min(1280, width), max(1, round(height * min(1280, width) / width))))
            if not cv2.imwrite(str(directory / 'poster.jpg'), poster):
                raise RuntimeError('无法保存视频预览。')
            browser_file = source.name
            if shutil.which('ffmpeg') and not compatible_source(source):
                browser_file = 'source.mp4'
                transcode_video(source, directory / browser_file)
            state = {'id': identifier, 'filename': filename, 'status': 'ready', 'metadata': metadata,
                     'frame': 0, 'processed_frames': 0, 'progress': 0, 'stage': 0, 'stage_step': 0,
                     'tracks': [], 'logs': [], 'error': None, 'revision': 1, 'preview_version': 0,
                     'trace_count': 0, 'outputs': {}, 'source_file': source.name, 'browser_file': browser_file}
            job = Job(directory, state)
            with self.lock:
                self.jobs[identifier] = job
                self.persist(job)
            return self.snapshot(identifier)
        except Exception:
            shutil.rmtree(directory)
            raise

    def _log(self, job, message, stage):
        job.state['logs'].append({'time': datetime.now().strftime('%H:%M:%S'), 'message': message, 'stage': stage})
        job.state['logs'] = job.state['logs'][-24:]

    def start(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job.state['status'] in BUSY or job.state['status'] == 'completed':
                return self.snapshot(job_id)
            if job.state['status'] != 'ready':
                raise JobBusy('请重新选择视频以开始新的分析。')
            if self.active is not None:
                raise JobBusy('已有视频正在分析，请等待完成或先停止。')
            if self.network_active:
                raise JobBusy('正在读取神经网络计算，请稍后再开始视频分析。')
            if self.config is None:
                raise RuntimeError(self.model_error or '模型未就绪。')
            if not shutil.which('ffmpeg'):
                raise RuntimeError('请先安装 ffmpeg，以便保存可播放的结果视频。')
            self.active = job_id
            job.state.update(status='loading', error=None)
            job.state['revision'] += 1
            self._log(job, '正在加载本地模型', 0)
            self.persist(job)
            self.executor.submit(self._execute, job)
            return self.snapshot(job_id)

    def cancel_job(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job.state['status'] in BUSY:
                job.cancel.set()
                self._log(job, '已请求停止，等待当前计算结束', job.state['stage'])
            elif job.state['status'] == 'ready':
                job.state['status'] = 'cancelled'
                job.live.finish('cancelled')
                job.live.save(job.directory / 'live.json', 'cancelled')
            job.state['revision'] += 1
            self.persist(job)
            return self.snapshot(job_id)

    def _observe(self, job, event):
        with self.lock:
            state = job.state
            kind = event['kind']
            if kind == 'computation':
                job.live.append(event)
                return
            state['status'] = 'running'
            if kind in ('stage', 'forecast_step'):
                stage = event['stage']
                state.update(stage=stage, stage_step=event.get('step', 0))
                if stage not in job.seen_stages:
                    job.seen_stages.add(stage)
                    self._log(job, STAGES[stage], stage)
            elif kind == 'frame':
                state.update(frame=event['frame'], processed_frames=event['frame'] + 1,
                             tracks=visual_data(event['tracks']))
                job.last_image = event['image']
                state['progress'] = min(0.99, state['processed_frames'] / state['metadata']['frame_count'])
                now = time.monotonic()
                if now - job.last_preview >= 0.25:
                    self._write_preview(job)
                    job.last_preview = now
                trace = event.get('trace', [])
                if trace:
                    job.last_trace_event = {'frame': event['frame'], 'time_seconds': event['time_seconds'],
                                            'width': state['metadata']['width'], 'height': state['metadata']['height'],
                                            'tracks': trace[:3]}
                first_prediction = any(t.get('future') is not None for t in trace) and not any(
                    t.get('future') is not None for sample in job.traces for t in sample['tracks'])
                if trace and (event['time_seconds'] - job.last_trace >= 1 or first_prediction):
                    job.traces.append(visual_data(job.last_trace_event))
                    state['trace_count'] += 1
                    job.last_trace = event['time_seconds']
                self.persist(job)
            state['revision'] += 1

    def _write_preview(self, job):
        image = job.last_image
        if image is None:
            return
        height, width = image.shape[:2]
        if width > 1280:
            image = cv2.resize(image, (1280, max(1, round(height * 1280 / width))))
        ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            path = job.directory / 'preview.tmp'
            path.write_bytes(jpeg.tobytes())
            path.replace(job.directory / 'preview.jpg')
            job.state['preview_version'] += 1

    def _execute(self, job):
        try:
            if job.cancel.is_set():
                raise PredictionCancelled('Analysis cancelled')
            config = self.config
            raw = job.directory / 'render.mp4'
            args = ['--input', str(job.directory / job.state['source_file']), '--output', str(raw),
                    '--pose-model', str(self.model_dir / config['pose_model']),
                    '--stgcn', str(self.model_dir / config['stgcn']), '--lstm', str(self.model_dir / config['lstm']),
                    '--stats', str(self.model_dir / config['stats']), '--normalization', config['normalization'],
                    '--behavior-device', 'cpu', '--presentation', '--save-keypoints']
            if config.get('pose_filter'):
                args.append('--pose-filter')
            pipeline_args = arguments(args)
            pipeline_args.live_telemetry = True
            metrics = self.run_pipeline(pipeline_args, on_event=lambda event: self._observe(job, event), cancel=job.cancel)
            if job.cancel.is_set():
                raise PredictionCancelled('Analysis cancelled')
            with self.lock:
                self._write_preview(job)
                if job.last_trace_event and (not job.traces or job.traces[-1]['frame'] != job.last_trace_event['frame']):
                    job.traces.append(visual_data(job.last_trace_event))
                    job.state['trace_count'] += 1
                job.state['status'] = 'encoding'
                self._log(job, '正在封装浏览器可播放的视频和分析数据', 5)
                self.persist(job)
            transcode_video(raw, job.directory / 'result.mp4', job.cancel)
            trace = {'samples': list(job.traces)}
            (job.directory / 'trace.json').write_text(json.dumps(trace, allow_nan=False), encoding='utf-8')
            with zipfile.ZipFile(job.directory / 'results.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
                for source, name in [('result.mp4', 'result.mp4'), ('render.jsonl', 'predictions.jsonl'),
                                     ('render.metrics.json', 'metrics.json'), ('trace.json', 'trace.json')]:
                    archive.write(job.directory / source, name)
            with self.lock:
                if job.cancel.is_set():
                    raise PredictionCancelled('Analysis cancelled')
                base = '/api/jobs/' + job.state['id'] + '/files/'
                job.state.update(status='completed', progress=1, metrics=metrics, stage=5,
                                 outputs={key: base + key for key in ('video', 'data', 'metrics', 'bundle')})
                self._log(job, '分析完成，结果已保存至本地', 5)
        except PredictionCancelled:
            with self.lock:
                job.state.update(status='cancelled', outputs={})
                self._log(job, '分析已停止', job.state['stage'])
        except Exception as exc:
            with self.lock:
                job.state.update(status='failed', error=str(exc), outputs={})
                self._log(job, '分析失败：' + str(exc), job.state['stage'])
        finally:
            with self.lock:
                job.state['revision'] += 1
                job.live.finish(job.state['status'])
                job.live.save(job.directory / 'live.json', job.state['status'])
                self.persist(job)
                self.active = None

    def live_events(self, job_id, after=None):
        with self.lock:
            job = self.get(job_id)
            return job.live.read(after, job.state['status']), job.state['status']

    def trace(self, job_id):
        with self.lock:
            return {'samples': deepcopy(list(self.get(job_id).traces))}

    def network(self, job_id, frame, track_id):
        """Inspect one recorded selection; serialize against video inference.

        A maximum of eight responses (each capped at 5 MiB serialized) is kept.
        The lock guards worker admission, not expensive model computation.
        """
        from modern_pipeline.behavior import BehaviorPredictor
        from .network import inspect_model, load_reference

        key = (job_id, frame, track_id)
        with self.lock:
            job = self.get(job_id)
            if self.active is not None or self.network_active:
                raise JobBusy('视频或网络计算正在运行，请等待完成后查看神经计算。')
            sample = next((row for row in job.traces if row['frame'] == frame), None)
            if sample is None:
                raise KeyError('未保存这个帧的过程记录。')
            track = next((row for row in sample['tracks'] if row['track_id'] == track_id), None)
            if track is None:
                raise KeyError('这个过程记录中没有所选行人。')
            history = np.asarray(track.get('history', []), dtype=np.float32)
            if history.shape != (50, 17, 3) or not np.isfinite(history).all():
                raise ValueError('所选记录需要完整、有效的 50 帧骨架历史。')
            if key in self.network_cache:
                self.network_cache.move_to_end(key)
                return deepcopy(self.network_cache[key])
            if self.config is None:
                raise RuntimeError(self.model_error or '模型未就绪。')
            recorded_probabilities = deepcopy(track.get('probabilities'))
            self.network_active = True
        try:
            config = self.config
            if self.network_predictor is None:
                import torch
                torch.set_num_threads(2)
                self.network_predictor = BehaviorPredictor(
                    self.model_dir / config['stgcn'], self.model_dir / config['lstm'],
                    'cpu', config['normalization'], self.model_dir / config['stats'])
            reference, reference_source = None, {}
            if config['normalization'] == 'legacy_pair':
                reference, reference_source = load_reference(job.directory / 'render.jsonl')
            result = inspect_model(self.network_predictor, history, reference)
            result['normalization']['stgcn'].update(reference_source)
            result['source'] = {
                'kind': 'recomputed_from_saved_history', 'history_decimals': 3,
                'job_id': job_id, 'frame': frame, 'track_id': track_id, 'device': 'cpu',
                'model_version': config.get('version'), 'recorded_probabilities': recorded_probabilities,
                'note': '使用保存时保留三位小数的骨架历史和同一组已校验权重重新计算；不是原视频推理时的未舍入中间激活。'}
            serialized = json.dumps(result, allow_nan=False, ensure_ascii=False)
            if len(serialized.encode('utf-8')) > 5 * 1024 * 1024:
                raise RuntimeError('神经计算记录超过大小限制。')
            with self.lock:
                self.network_cache[key] = result
                while len(self.network_cache) > 8:
                    self.network_cache.popitem(last=False)
            return deepcopy(result)
        finally:
            with self.lock:
                self.network_active = False

    def frontend_network(self, job_id, frame):
        """Replay one source frame; share admission with all inference work."""
        from .frontend_network import FrontendInspector, read_source_frame

        key = (job_id, frame)
        with self.lock:
            job = self.get(job_id)
            if self.active is not None or self.network_active:
                raise JobBusy('视频或网络计算正在运行，请等待完成后查看神经计算。')
            if not 0 <= frame < job.state['metadata']['frame_count']:
                raise ValueError('所选帧超出原始视频范围。')
            if key in self.frontend_cache:
                self.frontend_cache.move_to_end(key)
                return deepcopy(self.frontend_cache[key])
            if self.config is None:
                raise RuntimeError(self.model_error or '模型未就绪。')
            self.network_active = True
        try:
            image = read_source_frame(job.directory / job.state['source_file'], frame)
            if self.frontend_predictor is None:
                self.frontend_predictor = FrontendInspector(self.model_dir / self.config['pose_model'])
            result = self.frontend_predictor.replay(image)
            result['source'] = {
                'kind': 'recomputed_from_source_frame', 'job_id': job_id, 'frame': frame,
                'time_seconds': frame / job.state['metadata']['fps'], 'device': 'cpu',
                'model_version': self.config.get('version'), 'threads': 2,
                'tracking_replayed': False,
                'note': '从原始视频所选帧以同一组权重重新计算；保留独立 Conv、BatchNorm、SiLU 以显示计算，'
                        '与融合推理可能存在浮点舍入差异。特征图来自本次实际激活；未重放 ByteTrack，检测框不含轨迹 ID。'}
            if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 3 * 1024 * 1024:
                raise RuntimeError('YOLO 计算记录超过大小限制。')
            with self.lock:
                self.frontend_cache[key] = result
                while len(self.frontend_cache) > 3:
                    self.frontend_cache.popitem(last=False)
            return deepcopy(result)
        finally:
            with self.lock:
                self.network_active = False

    def file(self, job_id, name):
        job = self.get(job_id)
        names = {'source': job.state['browser_file'], 'poster': 'poster.jpg'}
        if job.state['status'] == 'completed':
            names.update(video='result.mp4', data='render.jsonl', metrics='render.metrics.json', bundle='results.zip')
        path = job.directory / names[name]
        if not path.is_file():
            raise KeyError(name)
        return path

    def close(self):
        with self.lock:
            if self.active is not None:
                self.jobs[self.active].cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
