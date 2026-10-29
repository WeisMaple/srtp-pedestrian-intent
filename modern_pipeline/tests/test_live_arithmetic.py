"""Representative sums use complete operands from one production forward."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_trace import LiveTrace, SAMPLING
from st_gcn import Model, SkeletonLSTM


class LiveArithmeticTests(unittest.TestCase):
    def predictor(self, normalization='legacy_pair'):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        predictor = BehaviorPredictor.__new__(BehaviorPredictor)
        predictor.device = torch.device('cpu')
        predictor.normalization = normalization
        predictor.lstm = SkeletonLSTM(num_layers=2).eval()
        predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        predictor.stats = {'lstm_mean': torch.ones(17, 3), 'lstm_std': torch.full((17, 3), 2.),
                           'stgcn_mean': torch.zeros(1, 3, 50, 17, 1),
                           'stgcn_std': torch.ones(1, 3, 50, 17, 1)}
        # A connected zero learned coefficient must remain connected.
        with torch.no_grad():
            predictor.lstm.fc.weight[0, 0] = 0
            predictor.stgcn.edge_importance[0].mul_(1.7)
            predictor.stgcn.edge_importance[0][0, 0, 0] = 0.
        return predictor

    def capture(self, predictor, poses):
        events = []
        trace = LiveTrace(events.append, interval=0)
        trace.attach_behavior(predictor)
        trace.begin_frame(88, 2.9)
        trace.begin_behavior([8, 2, 91, 17][:len(poses)], poses)
        try:
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
            trace.end_frame([{'track_id': track_id, 'prediction_updated': True,
                              'probabilities': result['probabilities'][index].tolist()}
                             for index, track_id in enumerate([8, 2, 91][:len(poses)])])
        finally:
            trace.close()
        return result, events, trace

    def assert_sum(self, calculation):
        if calculation['stage'] == 'spatial':
            graph = calculation['metadata']['graph_slice']
            self.assertEqual(graph['joint_count'], 17)
            self.assertEqual(len(graph['outputs']), 17)
            self.assertEqual(graph['timestep'], calculation['metadata']['target_indices'][1])
            self.assertEqual(graph['channel'], 0)
            # Validate all receivers against independently captured PyTorch output.
            for receiver, actual in enumerate(graph['outputs']):
                expected = math.fsum(calculation['inputs'][k * 17 + source] * coefficient
                                     for k, source, target, coefficient in graph['edges'] if target == receiver)
                self.assertAlmostEqual(expected, actual, delta=max(2e-5, abs(actual) * 2e-5))
        size = len(calculation['inputs'])
        for field in ('weights', 'input_indices', 'connected', 'padding'):
            self.assertEqual(len(calculation[field]), size, (calculation['id'], field))
        total = math.fsum(x * w for x, w in zip(calculation['inputs'], calculation['weights'])) + calculation['bias']
        if calculation['activation'] == 'silu':
            self.assertAlmostEqual(total, calculation['preactivation'], delta=2e-5 * max(1., abs(total)))
            total = total / (1 + math.exp(-total))
        self.assertAlmostEqual(total, calculation['actual'], delta=2e-5 * max(1., abs(total)), msg=calculation['id'])
        self.assertTrue(calculation['metadata']['complete_sum'])

    def test_three_people_complete_sums_preserve_ids_forward_count_and_predictions(self):
        predictor = self.predictor()
        poses = np.random.default_rng(3).normal(size=(4, 50, 17, 3)).astype(np.float32)
        baseline = predictor.predict(poses, poses[0])
        calls = []
        handle = predictor.lstm.register_forward_hook(lambda *unused: calls.append(1))
        result, events, trace = self.capture(predictor, poses)
        handle.remove()
        self.assertEqual(len(calls), 5)
        np.testing.assert_array_equal(result['probabilities'], baseline['probabilities'])
        np.testing.assert_array_equal(result['future_poses'], baseline['future_poses'])
        calculations = [c for e in events for c in e.get('calculations', [])]
        self.assertEqual({c['stage'] for c in calculations},
                         {'normalization', 'decoder', 'denormalization', 'spatial', 'temporal', 'pool', 'classifier'})
        self.assertEqual({c['track_id'] for c in calculations}, {8, 2, 91})
        for calculation in calculations:
            self.assert_sum(calculation)
        counts = {'decoder': 256, 'spatial': 51, 'temporal': 576, 'pool': 221, 'classifier': 256}
        for stage, count in counts.items():
            rows = [c for c in calculations if c['stage'] == stage]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(len(c['inputs']) == count for c in rows))
        decoder = next(c for c in calculations if c['stage'] == 'decoder')
        self.assertEqual(decoder['weights'][0], 0.)
        self.assertTrue(decoder['connected'][0])
        spatial = next(c for c in calculations if c['stage'] == 'spatial')
        self.assertIn(False, spatial['connected'])
        joint = spatial['metadata']['target_indices'][-1]
        self.assertEqual(spatial['connected'], (predictor.stgcn.A[:, :, joint] != 0).flatten().tolist())
        self.assertTrue(any(connected and weight == 0 for connected, weight in zip(spatial['connected'], spatial['weights'])))
        torch.testing.assert_close(torch.tensor(spatial['weights']).reshape(3, 17),
                                   (predictor.stgcn.A * predictor.stgcn.edge_importance[0])[:, :, joint])
        temporal = next(c for c in calculations if c['stage'] == 'temporal')
        self.assertTrue(any(temporal['padding']))
        self.assertTrue(all(temporal['connected']))
        self.assertTrue(all(x == 0 for x, padded in zip(temporal['inputs'], temporal['padding']) if padded))
        denormalized = [c for c in calculations if c['stage'] == 'denormalization']
        self.assertEqual(len(denormalized), 15)
        for row in [c for c in calculations if c['stage'] == 'classifier']:
            index = [8, 2, 91].index(row['track_id'])
            np.testing.assert_allclose(row['metadata']['probabilities'], result['probabilities'][index], atol=1e-7)
            self.assertEqual(row['metadata']['selected_class'], result['class_ids'][index])
            self.assertEqual(row['metadata']['class_labels'], ['irrelevant', 'not crossing', 'crossing'])
        sizes = [len(json.dumps(event, separators=(',', ':'), allow_nan=False).encode()) for event in events]
        self.assertLessEqual(max(sizes), 64 * 1024)
        self.assertLessEqual(sum(sizes), SAMPLING['frame_max_bytes'])
        self.assertEqual(trace.dropped['oversized_events'], 0)
        self.assertEqual(trace.dropped['budget_events'], 0)
        for module in list(predictor.stgcn.modules()) + list(predictor.lstm.modules()):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_saved_statistics_normalization_uses_actual_lstm_input(self):
        predictor = self.predictor('training_stats')
        poses = np.full((1, 50, 17, 3), 7., dtype=np.float32)
        _, events, _ = self.capture(predictor, poses)
        rows = [c for e in events for c in e.get('calculations', [])]
        normalization = next(c for c in rows if c['stage'] == 'normalization')
        self.assertEqual(normalization['actual'], 3.)
        self.assertEqual(normalization['metadata']['mean'], 1.)
        self.assertEqual(normalization['metadata']['std'], 2.)
        for row in rows:
            self.assert_sum(row)

    def test_fused_yolo_includes_all_center_rgb_terms_and_actual_silu(self):
        class Fused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
                self.act = torch.nn.SiLU()

            def forward(self, value):
                return self.act(self.conv(value))

        callbacks = {'on_predict_start': []}
        model = SimpleNamespace(callbacks=callbacks,
                                add_callback=lambda name, callback: callbacks[name].append(callback))
        block = Fused().eval()
        network = SimpleNamespace(model=torch.nn.Sequential(block), is_fused=lambda: True)
        runner = SimpleNamespace(model=SimpleNamespace(model=network), done_warmup=False)
        events = []
        trace = LiveTrace(events.append, interval=0)
        trace.attach_yolo(model)
        trace.begin_frame(1, 0)
        callbacks['on_predict_start'][0](runner)
        value = torch.arange(3 * 12 * 12).reshape(1, 3, 12, 12).float() / 100
        block(value)
        self.assertFalse(any(e.get('calculations') for e in events))
        runner.done_warmup = True
        expected = block(value)
        trace.close()
        rows = [c for e in events for c in e.get('calculations', [])]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['stage'], 'yolo')
        self.assertEqual(len(row['inputs']), 27)
        self.assertEqual(row['activation'], 'silu')
        self.assertEqual(row['actual'], float(expected[0, 0, 3, 3]))
        self.assertEqual(row['metadata']['target_indices'], [0, 3, 3])
        self.assertEqual(row['inputs'], value[0, :, 5:8, 5:8].reshape(-1).tolist())
        self.assertFalse(any(row['padding']))
        self.assert_sum(row)
        for module in block.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / 'weights/final/lstm.pt').is_file(),
                         'local final reproduction weights are absent')
    def test_local_real_weights_keep_full_prediction_parity_and_scalar_reconstruction(self):
        directory = Path(__file__).resolve().parents[2] / 'weights/final'
        torch.set_num_threads(2)
        predictor = BehaviorPredictor(directory / 'stgcn.pt', directory / 'lstm.pt', device='cpu')
        poses = np.random.default_rng(5).normal(size=(3, 50, 17, 3)).astype(np.float32)
        poses[..., :2] = poses[..., :2] * 20 + 300
        poses[..., 2] = 0.9
        baseline = predictor.predict(poses, poses[0])
        result, events, trace = self.capture(predictor, poses)
        np.testing.assert_array_equal(result['probabilities'], baseline['probabilities'])
        np.testing.assert_array_equal(result['future_poses'], baseline['future_poses'])
        rows = [c for event in events for c in event.get('calculations', [])]
        self.assertEqual(len(rows), 33)
        for row in rows:
            self.assert_sum(row)
        self.assertEqual(trace.dropped['budget_events'], 0)
        self.assertEqual(trace.dropped['oversized_events'], 0)

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / 'weights/final/yolo26s-pose.pt').is_file(),
                         'local final frontend weights are absent')
    def test_local_yolo_and_three_people_fit_budget_without_losing_classifier(self):
        from modern_pipeline.frontend import YOLOPoseTracker
        directory = Path(__file__).resolve().parents[2] / 'weights/final'
        predictor = BehaviorPredictor(directory / 'stgcn.pt', directory / 'lstm.pt', device='cpu')
        frontend = YOLOPoseTracker(directory / 'yolo26s-pose.pt', 'cpu')
        rng = np.random.default_rng(17)
        poses = rng.normal(size=(3, 50, 17, 3)).astype(np.float32)
        poses[..., :2] = poses[..., :2] * 20 + 300
        poses[..., 2] = 0.9
        events = []
        trace = LiveTrace(events.append, interval=0)
        trace.attach_behavior(predictor)
        trace.attach_yolo(frontend.model)
        try:
            trace.begin_frame(60, 2.)
            frontend(rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8))
            trace.begin_behavior([8, 2, 91], poses)
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
            trace.end_frame([{'track_id': track_id, 'prediction_updated': True,
                              'probabilities': result['probabilities'][i].tolist()}
                             for i, track_id in enumerate([8, 2, 91])])
        finally:
            trace.close()
        self.assertEqual(trace.dropped['budget_events'], 0)
        self.assertEqual(trace.dropped['oversized_events'], 0)
        rows = [c for event in events for c in event.get('calculations', [])]
        self.assertEqual(len(rows), 34)
        for row in rows:
            self.assert_sum(row)
        self.assertEqual(events[-1]['node'], 'pipeline.output')
        self.assertLessEqual(sum(len(json.dumps(event, separators=(',', ':')).encode()) for event in events),
                             SAMPLING['frame_max_bytes'])


if __name__ == '__main__':
    unittest.main()
