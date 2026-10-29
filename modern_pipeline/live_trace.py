"""Bounded observation of the *running* inference; never performs a model forward.

Hooks copy only small selected tensor slices. Timing is host wall time between
module pre/post hooks (including nested observer overhead), not a GPU profiler.
Warm-up forwards are excluded, and tensor batches retain their real track order.
"""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import time


TERMINAL = {'completed', 'failed', 'cancelled'}
# Actual YOLO plus three people's complete representative sums exceed 512 KiB
# (frame-60 audit dropped two classifier records). Keep each event at 64 KiB
# and the ring at 4 MiB; 768 KiB admits the measured ~535 KiB complete frame.
SAMPLING = {'max_frames_per_second': 4, 'max_tracks': 3,
            'yolo_channels': 4, 'yolo_grid': [8, 8],
            'lstm_hidden_units': 16, 'lstm_layers': 10,
            'stgcn_channels': 4, 'stgcn_times': 1, 'stgcn_joints': 17,
            'ring_max_events': 512, 'ring_max_bytes': 4 * 1024 * 1024,
            'frame_max_events': 128, 'frame_max_bytes': 768 * 1024,
            'arithmetic': 'complete representative outputs; first YOLO convolution, first decoder round, first graph block; up to 3 tracks',
            'source': 'live_forward_hooks', 'timing': 'host_wall_including_nested_observers',
            'note': 'Actual sampled values; unsampled frames and expired replay events are counted. No inference replay.'}


def sample_tensor(value, axes, limits, first_axes=()):
    """Slice before copying to CPU; return source indices and original shape."""
    import torch
    if not isinstance(value, torch.Tensor):
        return None
    result = value.detach()
    original = list(result.shape)
    indices = []
    for axis, (size, limit) in enumerate(zip(original, limits)):
        count = min(size, limit)
        if axis in first_axes or count == size:
            chosen = list(range(count))
        elif count == 1:
            chosen = [size - 1]
        else:
            chosen = [round(i * (size - 1) / (count - 1)) for i in range(count)]
        indices.append(chosen)
        result = result.index_select(axis, torch.tensor(chosen, device=result.device))
    # The nested Python values own their data even when a later inplace ReLU runs.
    array = result.float().cpu().numpy()
    import numpy as np
    finite = np.isfinite(array)
    rounded = np.round(np.where(finite, array, 0).astype(np.float64), 5).astype(object)
    rounded[~finite] = None
    return {'shape': original, 'sample_shape': list(result.shape), 'axes': list(axes),
            'indices': indices, 'values': rounded.tolist(),
            'nonfinite_count': int((~finite).sum())}


def first_tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


class LiveTrace:
    def __init__(self, emit, interval=0.25):
        from .live_arithmetic import LiveArithmetic
        self.emit = emit
        self.arithmetic = LiveArithmetic(self)
        self.interval = max(0.25, interval) if interval else 0
        self.last_capture = float('-inf')
        self.active = False
        self.frame = -1
        self.time_seconds = 0
        self.track_ids = []
        self.batch_size = 0
        self.round = 0
        self.handles = []
        self.yolo_handles = []
        self.yolo_model = None
        self.yolo_callback = None
        self.yolo_module = None
        self.predictor = None
        self.dropped = {'capture_frames': 0, 'oversized_events': 0, 'budget_events': 0}
        self.frame_started = 0
        self.frame_events = self.frame_bytes = 0

    def begin_frame(self, frame, time_seconds):
        now = time.perf_counter()
        self.frame, self.time_seconds = int(frame), float(time_seconds)
        self.track_ids, self.batch_size, self.round = [], 0, 0
        self.arithmetic.reset()
        self.active = now - self.last_capture >= self.interval
        if not self.active:
            self.dropped['capture_frames'] += 1
            return False
        self.last_capture = now
        self.frame_started = now
        self.frame_events = self.frame_bytes = 0
        self._publish('pipeline', 'pipeline.frame', 1, {}, None, phase='start',
                      dropped=dict(self.dropped), sampling=SAMPLING)
        return True

    def begin_behavior(self, track_ids, histories=None):
        self.track_ids = [int(x) for x in track_ids[:3]]
        self.batch_size = len(track_ids)
        self.round = 0
        self.arithmetic.begin_behavior(histories)
        if self.active and histories is not None:
            import torch
            self._publish('pipeline', 'pipeline.history', 3, {
                'history': sample_tensor(torch.as_tensor(histories), ['track', 'time', 'joint', 'coordinate'],
                                         [3, 10, 17, 3], (0, 2, 3))}, None,
                          track_ids=list(self.track_ids), coordinate_system='source_video_pixels_xy_confidence')

    def forecast_step(self, event):
        if self.active and event.get('kind') == 'forecast_step':
            import torch
            step = event['step']
            self.arithmetic.denormalization(event)
            self._publish('lstm', 'lstm.%d.forecast' % step, 4, {
                'future': sample_tensor(torch.as_tensor(event['poses']), ['track', 'time', 'joint', 'coordinate'],
                                        [3, 10, 17, 3], (0, 1, 2, 3))}, None,
                          step=step, horizon_start=(step - 1) * 10 + 1,
                          coordinate_system='source_video_pixels_xy_confidence',
                          timing_note='Actual denormalized output of completed forecast round.')

    def end_frame(self, tracks):
        if self.active:
            self.arithmetic.finish(tracks)
            self._publish('pipeline', 'pipeline.output', 7, {},
                          (time.perf_counter() - self.frame_started) * 1000,
                          tracks=tracks[:3], phase='complete', dropped=dict(self.dropped))
        self.active = False

    def _publish(self, model, node, stage, tensors, duration, **extra):
        if not self.active:
            return
        event = {'kind': 'computation', 'frame': self.frame, 'time_seconds': self.time_seconds,
                 'model': model, 'node': node, 'stage': stage,
                 'track_ids': list(self.track_ids) if model in ('lstm', 'stgcn') else [],
                 'batch_size': self.batch_size if model in ('lstm', 'stgcn') else 1,
                 'captured_at_ms': round(time.perf_counter() * 1000, 3),
                 'duration_ms': None if duration is None else round(duration, 3),
                 'timing': 'host_wall_including_nested_observers', 'tensors': tensors}
        event.update(extra)
        size = len(json.dumps(event, allow_nan=False, separators=(',', ':')).encode('utf-8'))
        if size > 64 * 1024:
            self.dropped['oversized_events'] += 1
            return
        if self.frame_events >= SAMPLING['frame_max_events'] or self.frame_bytes + size > SAMPLING['frame_max_bytes']:
            self.dropped['budget_events'] += 1
            return
        self.frame_events += 1
        self.frame_bytes += size
        self.emit(event)

    def _hook(self, module, callback, guard=None, handles=None):
        started = []
        target = self.handles if handles is None else handles

        def before(_module, _inputs):
            if self.active and (guard is None or guard()):
                started.append(time.perf_counter())
            else:
                started.append(None)

        def after(_module, inputs, output):
            timestamp = started.pop() if started else None
            if timestamp is not None and self.active:
                duration = (time.perf_counter() - timestamp) * 1000
                callback(inputs, output, duration)

        target.append(module.register_forward_pre_hook(before))
        target.append(module.register_forward_hook(after))

    def attach_yolo(self, model):
        """on_predict_start runs after backend setup/fusion, before warm-up."""
        self.yolo_model = model

        def attach(predictor):
            self.predictor = predictor
            network = predictor.model.model
            if network is self.yolo_module:
                return
            for handle in self.yolo_handles:
                handle.remove()
            self.yolo_handles.clear()
            self.yolo_module = network
            fused = bool(getattr(network, 'is_fused', lambda: False)())
            self.arithmetic.attach_yolo(network, fused, guard=lambda: bool(self.predictor.done_warmup))
            for index, module in enumerate(network.model):
                def capture(inputs, output, duration, index=index, module=module):
                    tensor = first_tensor(output)
                    if tensor is None:
                        return
                    if tensor.ndim == 4:
                        sampled = sample_tensor(tensor, ['batch', 'channel', 'y', 'x'], [1, 4, 8, 8], (0, 1))
                    elif tensor.ndim == 3:
                        sampled = sample_tensor(tensor, ['batch', 'row', 'feature'], [1, 8, 16], (0,))
                    else:
                        sampled = sample_tensor(tensor, ['axis%d' % i for i in range(tensor.ndim)],
                                                [4] * tensor.ndim, tuple(range(tensor.ndim)))
                    self._publish('yolo', 'yolo.%d' % index, 1, {'output': sampled}, duration,
                                  module_type=type(module).__name__, from_nodes=getattr(module, 'f', -1),
                                  fused=fused, warmup=False,
                                  calculations=self.arithmetic.yolo_output(tensor) if index == 0 else [])
                self._hook(module, capture, guard=lambda: bool(self.predictor.done_warmup),
                           handles=self.yolo_handles)

        self.yolo_callback = attach
        model.add_callback('on_predict_start', attach)

    def attach_behavior(self, predictor):
        if predictor.lstm is not None:
            def recurrent(inputs, output, duration):
                self.round += 1
                sequence, (hidden, cell) = output
                if self.round == 1:
                    self.arithmetic.normalization(inputs[0])
                self._publish('lstm', 'lstm.%d' % self.round, 4, {
                    'input': sample_tensor(inputs[0], ['track', 'time', 'feature'], [3, 4, 12], (0,)),
                    'sequence': sample_tensor(sequence, ['track', 'time', 'hidden'], [3, 4, 16], (0, 2)),
                    'hidden': sample_tensor(hidden, ['layer', 'track', 'hidden'], [10, 3, 16], (0, 1, 2)),
                    'cell': sample_tensor(cell, ['layer', 'track', 'hidden'], [10, 3, 16], (0, 1, 2)),
                }, duration, step=self.round, module_type='LSTM')
            self._hook(predictor.lstm.lstm, recurrent)
        for index, block in enumerate(predictor.stgcn.st_gcn_networks):
            for name, module in (('spatial', block.gcn), ('temporal', block.tcn), ('output', block)):
                def capture(inputs, output, duration, index=index, name=name):
                    tensor = first_tensor(output)
                    self._publish('stgcn', 'stgcn.%d.%s' % (index, name), 5, {
                        'output': sample_tensor(tensor, ['track', 'channel', 'time', 'joint'],
                                                [3, 4, 1, 17], (0, 1))}, duration,
                                  block=index, operation=name)
                self._hook(module, capture)

        def pool(_module, inputs):
            if self.active:
                self.arithmetic.pool(inputs[0])
                self._publish('stgcn', 'stgcn.pool', 5, {
                    'output': sample_tensor(inputs[0], ['track', 'channel', 'y', 'x'], [3, 16, 1, 1], (0, 1))},
                    None, operation='global_pool', timing_note='Captured at classifier input; functional pool has no module duration.')
        self.handles.append(predictor.stgcn.fcn.register_forward_pre_hook(pool))

        def logits(inputs, output, duration):
            self.arithmetic.classifier(predictor.stgcn.fcn, inputs, output)
            self._publish('stgcn', 'stgcn.logits', 5, {
                'output': sample_tensor(output.reshape(output.shape[0], -1), ['track', 'class'], [3, 3], (0, 1))}, duration,
                          operation='classifier_logits')
        self._hook(predictor.stgcn.fcn, logits)
        self.arithmetic.attach_behavior(predictor)

    def close(self):
        self.active = False
        self.arithmetic.reset()
        self.arithmetic.predictor = None
        for handle in self.handles + self.yolo_handles:
            handle.remove()
        self.handles.clear()
        self.yolo_handles.clear()
        if self.yolo_model is not None and self.yolo_callback is not None:
            callbacks = self.yolo_model.callbacks.get('on_predict_start', [])
            if self.yolo_callback in callbacks:
                callbacks.remove(self.yolo_callback)
        self.yolo_model = self.predictor = self.yolo_module = self.yolo_callback = None


class LiveBuffer:
    """Access under the JobManager lock. No client owns inference resources."""
    def __init__(self, job_id, max_events=512, max_bytes=4 * 1024 * 1024):
        self.job_id = job_id
        self.max_events, self.max_bytes = max_events, max_bytes
        self.events = deque()
        self.bytes = 0
        self.seq = 0
        self.nodes = {}
        self.frame = -1
        self.time_seconds = 0
        self.tracks = []
        self.dropped = {'capture_frames': 0, 'ring_events': 0, 'oversized_events': 0, 'budget_events': 0}
        self.terminal = None
        self.captured_at_ms = None

    def _store(self, name, data):
        self.seq += 1
        data = dict(data, seq=self.seq, job_id=self.job_id)
        size = len(json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8'))
        self.events.append({'event': name, 'data': data, 'size': size})
        self.bytes += size
        while len(self.events) > self.max_events or self.bytes > self.max_bytes:
            self.bytes -= self.events.popleft()['size']
            self.dropped['ring_events'] += 1
        return data

    def append(self, event):
        event = dict(event)
        event.pop('kind', None)
        if event['frame'] != self.frame:
            self.nodes = {}
            self.tracks = []
            self.frame = event['frame']
            self.time_seconds = event['time_seconds']
        self.dropped.update(event.get('dropped', {}))
        data = self._store('computation', event)
        self.nodes[event['node']] = data
        if 'tracks' in event:
            self.tracks = event['tracks']
        self.captured_at_ms = event.get('captured_at_ms')
        return data

    def snapshot(self, status):
        return deepcopy({'seq': self.seq, 'job_id': self.job_id, 'status': status,
                         'frame': self.frame, 'time_seconds': self.time_seconds,
                         'captured_at_ms': self.captured_at_ms, 'nodes': self.nodes,
                         'tracks': self.tracks, 'dropped': self.dropped, 'sampling': SAMPLING})

    def finish(self, status):
        if self.terminal is None:
            self.terminal = self._store('terminal', {'status': status, 'frame': self.frame,
                                                   'dropped': dict(self.dropped)})

    def read(self, after, status):
        """Return reconnect replay or a coherent reset; sequence IDs never regress."""
        if after is None:
            rows = [{'event': 'snapshot', 'data': self.snapshot(status)}]
        elif after > self.seq or after < (self.events[0]['data']['seq'] - 1 if self.events else self.seq):
            oldest = self.events[0]['data']['seq'] if self.events else self.seq + 1
            rows = [{'event': 'gap', 'data': {'seq': self.seq, 'job_id': self.job_id,
                    'requested_after': after, 'oldest_available': oldest,
                    'dropped_events': max(0, oldest - after - 1), 'reset': True}},
                    {'event': 'snapshot', 'data': self.snapshot(status)}]
        else:
            rows = [{'event': row['event'], 'data': deepcopy(row['data'])}
                    for row in self.events if row['data']['seq'] > after]
        if status in TERMINAL and not any(row['event'] == 'terminal' for row in rows):
            rows.append({'event': 'terminal', 'data': deepcopy(self.terminal or {
                'seq': self.seq, 'job_id': self.job_id, 'status': status, 'frame': self.frame})})
        return rows

    def save(self, path, status):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.snapshot(status), ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def restore(self, path):
        saved = json.loads(Path(path).read_text(encoding='utf-8'))
        if saved.get('job_id') != self.job_id:
            raise ValueError('Live snapshot belongs to another job')
        self.seq = int(saved['seq'])
        self.nodes = saved.get('nodes', {})
        self.frame = saved.get('frame', -1)
        self.time_seconds = saved.get('time_seconds', 0)
        self.tracks = saved.get('tracks', [])
        self.dropped.update(saved.get('dropped', {}))
        self.captured_at_ms = saved.get('captured_at_ms')
        if saved.get('status') in TERMINAL:
            self.terminal = {'seq': self.seq, 'job_id': self.job_id,
                             'status': saved['status'], 'frame': self.frame}
