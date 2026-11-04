"""Manual benchmark for detailed live telemetry; not part of unittest discovery.

Run with a real test video and the final weights:
  python -s modern_pipeline/tests/bench_live_detail.py \
    --input artifacts/demo/demo_input.mp4 --model-dir weights/final \
    --max-frames 75 --repeat 3

The video pass measures detail-off vs detail-on processing and event budgets.
The worst-case pass drives the four measured heavy pairings (blocks 7/9 with
module groups {4, 5, 6, 7} or {20, 21, 22, 23}) with three people: first the
pre-fallback four-module schedule, then production scheduling. It uses the
first real video frame; the three 50-frame histories are the synthetic stress
input from the unit tests. Model forwards are real and nothing is replayed.
"""
import argparse
import contextlib
import io
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modern_pipeline.cli import arguments, run  # noqa: E402


def encoded_sizes(events):
    return [len(json.dumps(event, allow_nan=False, separators=(',', ':')).encode('utf-8'))
            for event in events]


def last_dropped(events):
    for event in reversed(events):
        if 'dropped' in event:
            return dict(event['dropped'])
    return {}


def coverage(events):
    rows = [event for event in events if event.get('node') == 'detail.coverage']
    yolo = Counter(module for row in rows for module in row.get('yolo_modules', []))
    blocks = Counter(row.get('stgcn_block') for row in rows
                     if row.get('stgcn_block') is not None)
    return {
        'detail_frames': len(rows),
        'boundary_frames': sum(bool(row.get('detail_boundary')) for row in rows),
        'yolo_modules_seen': sorted(yolo),
        'yolo_module_counts': {str(module): yolo.get(module, 0) for module in range(24)},
        'yolo_missing_modules': [module for module in range(24) if module not in yolo],
        'yolo_full_rounds': min(yolo.get(module, 0) for module in range(24)),
        'stgcn_blocks_seen': sorted(blocks),
        'stgcn_block_counts': {str(block): blocks.get(block, 0) for block in range(10)},
        'stgcn_missing_blocks': [block for block in range(10) if block not in blocks],
        'stgcn_full_rounds': min(blocks.get(block, 0) for block in range(10)),
    }


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def aggregate_coverage(rows):
    yolo = sorted({module for row in rows for module in row['yolo_modules_seen']})
    blocks = sorted({block for row in rows for block in row['stgcn_blocks_seen']})
    return {
        'runs': len(rows),
        'max_detail_frames': max((row['detail_frames'] for row in rows), default=0),
        'yolo_modules_seen': yolo,
        'yolo_missing_modules': [module for module in range(24) if module not in yolo],
        'yolo_full_rounds_max': max((row['yolo_full_rounds'] for row in rows), default=0),
        'stgcn_blocks_seen': blocks,
        'stgcn_missing_blocks': [block for block in range(10) if block not in blocks],
        'stgcn_full_rounds_max': max((row['stgcn_full_rounds'] for row in rows), default=0),
    }


def measure_video(input_path, model_dir, max_frames, repeats):
    runs = {'off': [], 'on': []}
    coverage_rows = {'off': [], 'on': []}
    metas = {}
    with tempfile.TemporaryDirectory(prefix='live-detail-bench-') as directory:
        work = Path(directory)
        for repeat in range(repeats):
            # Alternate order so a cold cache or thermal drift does not
            # systematically favor one label.
            order = (('off', False), ('on', True))
            if repeat % 2:
                order = tuple(reversed(order))
            for label, detail in order:
                output = work / ('bench_%s_%d.mp4' % (label, repeat))
                argv = ['--input', str(input_path), '--output', str(output),
                        '--pose-model', str(model_dir / 'yolo26s-pose.pt'),
                        '--stgcn', str(model_dir / 'stgcn.pt'),
                        '--lstm', str(model_dir / 'lstm.pt'),
                        '--stats', str(model_dir / 'normalization.pt'),
                        '--normalization', 'legacy_pair',
                        '--device', 'cpu', '--behavior-device', 'cpu', '--presentation',
                        '--save-keypoints', '--max-frames', str(max_frames)]
                a = arguments(argv)
                a.live_telemetry = True
                a.live_detail = detail
                events = []
                started = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    meta = run(a, on_computation=events.append)
                wall = time.perf_counter() - started
                sizes = encoded_sizes(events)
                detail_sizes = [size for event, size in zip(events, sizes)
                                if event.get('detail')]
                runs[label].append({
                    'processing_seconds': round(meta['processing_seconds'], 3),
                    'wall_seconds': round(wall, 3),
                    'processing_fps': round(meta['processing_fps'], 3),
                    'frames': meta['frames'],
                    'events': len(events),
                    'event_bytes': sum(sizes),
                    'detail_events': len(detail_sizes),
                    'detail_bytes': sum(detail_sizes),
                    'max_event_bytes': max(sizes, default=0),
                    'dropped': last_dropped(events),
                })
                coverage_rows[label].append(coverage(events))
                metas[label] = meta
    results = {}
    for label, label_runs in runs.items():
        results[label] = {
            'processing_seconds': round(median([run['processing_seconds']
                                                for run in label_runs]), 3),
            'wall_seconds': round(median([run['wall_seconds'] for run in label_runs]), 3),
            'processing_fps': round(median([run['processing_fps'] for run in label_runs]), 3),
            'frames': label_runs[-1]['frames'],
            'events': label_runs[-1]['events'],
            'event_bytes': label_runs[-1]['event_bytes'],
            'detail_events': label_runs[-1]['detail_events'],
            'detail_bytes': label_runs[-1]['detail_bytes'],
            'max_event_bytes': max(run['max_event_bytes'] for run in label_runs),
            'dropped': label_runs[-1]['dropped'],
            'coverage': coverage_rows[label][-1],
            'coverage_all_runs': aggregate_coverage(coverage_rows[label]),
            'runs': label_runs,
        }
    processing_overhead = ((results['on']['processing_seconds']
                            - results['off']['processing_seconds'])
                           / results['off']['processing_seconds'] * 100)
    wall_overhead = ((results['on']['wall_seconds'] - results['off']['wall_seconds'])
                     / results['off']['wall_seconds'] * 100)
    return {
        'repeats': repeats,
        'results': results,
        'overhead_percent': round(processing_overhead, 2),
        'wall_overhead_percent': round(wall_overhead, 2),
        'environment': {key: metas['on'].get(key) for key in
                        ('device', 'behavior_device', 'torch', 'ultralytics',
                         'input_fps', 'width', 'height', 'frames', 'median_frame_ms',
                         'fps_excluding_first_frame', 'peak_torch_allocated_MiB')},
    }


def pairing_frame(input_path, model_dir, detail_index=15, forced_modules=None):
    """Run real models through one heavy detail pairing.

    forced_modules reproduces the pre-fallback four-module schedule; None uses
    the production scheduler, whose budget fallback should capture two.
    """
    import numpy as np
    import torch
    from modern_pipeline.behavior import BehaviorPredictor
    from modern_pipeline.frontend import YOLOPoseTracker
    from modern_pipeline.live_trace import LiveTrace

    torch.set_num_threads(2)
    cap = cv2.VideoCapture(str(input_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Input video contains no decodable frame')
    frontend = YOLOPoseTracker(model_dir / 'yolo26s-pose.pt', 'cpu')
    behavior = BehaviorPredictor(model_dir / 'stgcn.pt', model_dir / 'lstm.pt', 'cpu')
    rng = np.random.default_rng(17)
    poses = rng.normal(size=(3, 50, 17, 3)).astype(np.float32)
    poses[..., :2] = poses[..., :2] * 20 + 300
    poses[..., 2] = 0.9
    events = []
    trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
    trace.attach_yolo(frontend.model)
    trace.attach_behavior(behavior)
    try:
        # The untraced warm-up installs the fused hooks and module metadata.
        frontend(frame)
        trace.detail_index = detail_index - 1
        trace.begin_frame(detail_index, detail_index / 30.)
        if forced_modules is not None:
            trace.detail_modules = set(forced_modules)
        frontend(frame)
        trace.begin_behavior([8, 2, 91], poses)
        started = time.perf_counter()
        result = behavior.predict(poses, poses[0], on_event=trace.forecast_step)
        behavior_seconds = time.perf_counter() - started
        trace.end_frame([{'track_id': track_id, 'prediction_updated': True,
                          'probabilities': result['probabilities'][index].tolist()}
                         for index, track_id in enumerate([8, 2, 91])])
        details = {'detail_index': trace.detail_index,
                   'block': trace.detail_block,
                   'boundary': trace.detail_boundary,
                   'scheduled_modules': sorted(trace.detail_modules),
                   'dropped': dict(trace.dropped)}
    finally:
        trace.close()
    sizes = encoded_sizes(events)
    details.update({
        'events': len(events),
        'event_bytes': sum(sizes),
        'detail_bytes': sum(size for event, size in zip(events, sizes)
                            if event.get('detail')),
        'max_event_bytes': max(sizes, default=0),
        'classifier_events': sum(1 for event in events
                                 if event.get('node', '').startswith('flow.stgcn.classifier')),
        'behavior_seconds': round(behavior_seconds, 3),
    })
    return details


def main():
    parser = argparse.ArgumentParser(description='Benchmark bounded live detail telemetry')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--max-frames', type=int, default=75)
    parser.add_argument('--repeat', type=int, default=1,
                        help='video passes per detail setting; the report uses 3')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('repeat must be positive')
    if not args.input.is_file():
        parser.error('input video does not exist: %s' % args.input)
    for name in ('yolo26s-pose.pt', 'stgcn.pt', 'lstm.pt', 'normalization.pt'):
        if not (args.model_dir / name).is_file():
            parser.error('model file does not exist: %s' % (args.model_dir / name))
    import torch
    import ultralytics
    video = measure_video(args.input, args.model_dir, args.max_frames, args.repeat)
    pairings = ((15, 7, {4, 5, 6, 7}), (35, 7, {20, 21, 22, 23}),
                (39, 9, {4, 5, 6, 7}), (59, 9, {20, 21, 22, 23}))
    worst_case = {
        'note': 'Real first video frame, real weights, three synthetic 50-frame histories.',
        'pairings': [{
            'detail_index': index,
            'block': block,
            'four_module_baseline': pairing_frame(args.input, args.model_dir, index,
                                                  forced_modules=modules),
            'production_fallback': pairing_frame(args.input, args.model_dir, index),
        } for index, block, modules in pairings],
    }
    fallback_clean = all(
        not item['production_fallback']['dropped'].get('budget_events')
        and not item['production_fallback']['dropped'].get('detail_budget')
        for item in worst_case['pairings'])
    output = {
        'command': ' '.join([sys.executable, '-s', sys.argv[0]] + sys.argv[1:]),
        'environment': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'python': sys.version.split()[0],
            'executable': sys.executable,
            'torch': torch.__version__,
            'ultralytics': ultralytics.__version__,
            'opencv': cv2.__version__,
            'cpu_count': os.cpu_count(),
        },
        'video': video,
        'worst_case': worst_case,
        'overhead_gate': ('keep 1 fps / 4 modules / 1 block'
                          if video['overhead_percent'] <= 15
                          else 'fallback: 0.5 fps / 2 modules'),
        'budget_fallback': 'applied and clean' if fallback_clean else 'still dropping',
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
