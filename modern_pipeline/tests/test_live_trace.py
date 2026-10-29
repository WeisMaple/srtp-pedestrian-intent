"""Live telemetry is bounded, frame-aligned and observes production forwards."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_trace import LiveTrace, LiveBuffer, SAMPLING, sample_tensor
from st_gcn import Model, SkeletonLSTM


class LiveTraceTests(unittest.TestCase):
    def test_tensor_sampler_preserves_original_dimensions_and_bounded_real_values(self):
        values = torch.arange(2 * 8 * 11 * 17).reshape(2, 8, 11, 17).float()
        result = sample_tensor(values, ['track', 'channel', 'time', 'joint'], [2, 4, 1, 17], first_axes=(0, 1))
        self.assertEqual(result['shape'], [2, 8, 11, 17])
        self.assertEqual(result['sample_shape'], [2, 4, 1, 17])
        indexes = result['indices']
        self.assertEqual(indexes[1], [0, 1, 2, 3])
        for b in range(2):
            self.assertEqual(result['values'][b][0][0][0], float(values[b, 0, indexes[2][0], 0]))
        self.assertLessEqual(len(json.dumps(result)), 5000)

    def test_real_modules_emit_five_lstm_rounds_and_ten_blocks_without_changing_math(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        predictor = BehaviorPredictor.__new__(BehaviorPredictor)
        predictor.device = torch.device('cpu')
        predictor.normalization = 'legacy_pair'
        predictor.lstm = SkeletonLSTM().eval()
        predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        predictor.stats = {}
        poses = np.random.default_rng(11).normal(size=(4, 50, 17, 3)).astype(np.float32)
        baseline = predictor.predict(poses, poses[0])
        events = []
        trace = LiveTrace(events.append, interval=0)
        trace.attach_behavior(predictor)
        trace.begin_frame(55, 5.5)
        trace.begin_behavior([13, 7, 91, 4])
        result = predictor.predict(poses, poses[0])
        trace.close()
        np.testing.assert_array_equal(result['probabilities'], baseline['probabilities'])
        np.testing.assert_array_equal(result['future_poses'], baseline['future_poses'])
        rounds = [e for e in events if e['node'] in {'lstm.%d' % i for i in range(1, 6)}]
        self.assertEqual([e['node'] for e in rounds], ['lstm.%d' % i for i in range(1, 6)])
        self.assertEqual(rounds[-1]['track_ids'], [13, 7, 91])
        self.assertEqual(rounds[-1]['tensors']['hidden']['shape'], [10, 4, 256])
        self.assertEqual(rounds[-1]['tensors']['hidden']['sample_shape'], [10, 3, 16])
        blocks = [e for e in events if e['node'].endswith('.output') and e['model'] == 'stgcn']
        self.assertEqual(len(blocks), 10)
        self.assertTrue(all(e['frame'] == 55 for e in events))
        self.assertTrue(all(e['duration_ms'] is None or e['duration_ms'] >= 0 for e in events))
        self.assertFalse(predictor.lstm.lstm._forward_hooks)
        self.assertFalse(predictor.stgcn._forward_hooks)
        self.assertLess(sum(len(json.dumps(e, separators=(',', ':')).encode()) for e in events),
                        SAMPLING['frame_max_bytes'])

    def test_yolo_hooks_attach_after_fusion_exclude_warmup_and_remove_on_error(self):
        class Yolo:
            def __init__(self):
                self.callbacks = {'on_predict_start': []}

            def add_callback(self, name, callback):
                self.callbacks[name].append(callback)

        events = []
        trace = LiveTrace(events.append, interval=0)
        model = Yolo()
        trace.attach_yolo(model)
        fused = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Conv2d(3, 6, 3)), is_fused=lambda: True)
        predictor = SimpleNamespace(model=SimpleNamespace(model=fused), done_warmup=False)
        trace.begin_frame(0, 0)
        callback = model.callbacks['on_predict_start'][0]
        callback(predictor)
        callback(predictor)  # Subsequent calls must not add duplicate hooks.
        inputs = torch.randn(1, 3, 16, 16)
        fused.model(inputs)  # Real forward used for warm-up, must not be a frame activation.
        self.assertFalse(any(event['model'] == 'yolo' for event in events))
        predictor.done_warmup = True
        expected = fused.model(inputs)
        captured = [event for event in events if event['model'] == 'yolo']
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]['fused'])
        self.assertEqual(captured[0]['tensors']['output']['shape'], [1, 6, 14, 14])
        try:
            raise RuntimeError('simulated cancellation or inference failure')
        except RuntimeError:
            pass
        finally:
            trace.close()
        torch.testing.assert_close(fused.model(inputs), expected)
        self.assertFalse(fused.model[0]._forward_hooks)
        self.assertFalse(fused.model[0]._forward_pre_hooks)
        self.assertEqual(model.callbacks['on_predict_start'], [])

    def test_per_frame_budget_drops_extra_events_without_unbounded_storage(self):
        events = []
        trace = LiveTrace(events.append, interval=0)
        trace.begin_frame(1, 0.1)
        for number in range(200):
            trace._publish('pipeline', 'test.%d' % number, 1, {}, None)
        self.assertEqual(len(events), 128)
        self.assertEqual(trace.dropped['budget_events'], 73)
        trace.close()

    def test_short_history_and_skipped_frames_never_emit_fake_behavior(self):
        events = []
        trace = LiveTrace(events.append, interval=999)
        self.assertTrue(trace.begin_frame(0, 0))
        trace.end_frame([{'track_id': 1, 'history_length': 1, 'prediction_updated': False}])
        self.assertFalse(trace.begin_frame(1, 0.1))
        trace.end_frame([{'track_id': 1, 'history_length': 2, 'prediction_updated': False}])
        self.assertFalse(any(e['model'] in ('lstm', 'stgcn') for e in events))
        self.assertEqual(trace.dropped['capture_frames'], 1)
        self.assertTrue(all(e['frame'] == 0 for e in events))
        trace.close()


class LiveBufferTests(unittest.TestCase):
    def event(self, frame, node='yolo.0'):
        return {'frame': frame, 'time_seconds': frame / 30, 'model': 'yolo',
                'node': node, 'tensors': {}, 'track_ids': [], 'duration_ms': 1,
                'captured_at_ms': 100 + frame, 'stage': 1}

    def test_bounded_ring_reconnect_gap_and_frame_snapshot(self):
        buffer = LiveBuffer('abc', max_events=3, max_bytes=4000)
        buffer.append(self.event(0))
        buffer.append(self.event(0, 'yolo.1'))
        cursor = buffer.seq
        buffer.append(self.event(1, 'pipeline.frame'))
        buffer.append(self.event(1))
        snapshot = buffer.snapshot('running')
        self.assertNotIn('yolo.1', snapshot['nodes'])
        self.assertEqual(snapshot['frame'], 1)
        self.assertLessEqual(len(buffer.events), 3)
        replay = buffer.read(cursor, 'running')
        self.assertEqual([row['event'] for row in replay], ['computation', 'computation'])
        recovery = buffer.read(0, 'running')
        self.assertEqual([row['event'] for row in recovery], ['gap', 'snapshot'])
        self.assertEqual(recovery[0]['data']['dropped_events'], 1)
        self.assertEqual(buffer.read(None, 'running')[0]['event'], 'snapshot')

    def test_saved_snapshot_restores_without_claiming_replay_events(self):
        buffer = LiveBuffer('abc')
        buffer.append(self.event(12))
        buffer.finish('completed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'live.json'
            buffer.save(path, 'completed')
            restored = LiveBuffer('abc')
            restored.restore(path)
            self.assertEqual(restored.snapshot('completed')['nodes']['yolo.0']['frame'], 12)
            rows = restored.read(0, 'completed')
            self.assertEqual([r['event'] for r in rows], ['gap', 'snapshot', 'terminal'])
            self.assertEqual(rows[-1]['data']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
