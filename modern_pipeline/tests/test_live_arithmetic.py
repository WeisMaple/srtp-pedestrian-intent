"""Representative sums use complete operands from one production forward."""
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_arithmetic import dense_conv_row
from modern_pipeline.live_trace import LiveTrace, SAMPLING
from modern_pipeline.model_catalog import select_representative_conv
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
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        trace.detail_index = 1  # the next frame is block 0's boundary subframe
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
                self.assertAlmostEqual(expected, actual, delta=max(1e-4, abs(actual) * 1e-4))
        size = len(calculation['inputs'])
        for field in ('weights', 'input_indices', 'connected', 'padding'):
            self.assertEqual(len(calculation[field]), size, (calculation['id'], field))
        total = math.fsum(x * w for x, w in zip(calculation['inputs'], calculation['weights'])) + calculation['bias']
        if calculation['activation'] == 'silu':
            self.assertAlmostEqual(total, calculation['preactivation'], delta=1e-4 * max(1., abs(total)))
            total = total / (1 + math.exp(-total))
        self.assertAlmostEqual(total, calculation['actual'], delta=1e-4 * max(1., abs(total)), msg=calculation['id'])
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
            expected = 6 if stage == 'temporal' else 3
            self.assertEqual(len(rows), expected)
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
        temporal_rows = [c for c in calculations if c['stage'] == 'temporal']
        self.assertTrue(any(any(c['padding']) for c in temporal_rows))
        self.assertTrue(any(not any(c['padding']) for c in temporal_rows))
        self.assertTrue(all(all(c['connected']) for c in temporal_rows))
        self.assertTrue(all(all(x == 0 for x, padded in zip(c['inputs'], c['padding']) if padded)
                            for c in temporal_rows))
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

    def test_dense_batch_norm_rows_expose_reconstruction_metadata(self):
        torch.manual_seed(6)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=True)
                self.bn = torch.nn.BatchNorm2d(4)
                self.act = torch.nn.SiLU()

            def forward(self, value):
                return self.act(self.bn(self.conv(value)))

        block = Block().eval()
        with torch.no_grad():
            block.bn.running_mean.fill_(0.25)
            block.bn.running_var.fill_(1.5)
            block.bn.weight.fill_(1.2)
            block.bn.bias.fill_(-0.4)
        selection = select_representative_conv(block, fused=False)
        captured = {}
        handle = block.conv.register_forward_hook(
            lambda _module, inputs, output: captured.update(
                input=inputs[0].detach(), output=output.detach()))
        value = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            output = block(value)
        handle.remove()
        row = dense_conv_row(0, selection, captured['input'], captured['output'], False)
        self.assertEqual(row['activation'], 'batch_norm_silu')
        bn = row['metadata']['bn']
        self.assertEqual(set(bn), {'gamma', 'beta', 'running_mean', 'running_var', 'epsilon'})
        preactivation = math.fsum(x * w for x, w in zip(row['inputs'], row['weights'])) + row['bias']
        normalized = (bn['gamma'] * (preactivation - bn['running_mean'])
                      / math.sqrt(bn['running_var'] + bn['epsilon']) + bn['beta'])
        activated = normalized / (1 + math.exp(-normalized))
        self.assertAlmostEqual(row['actual'], round(activated, 5), places=5)
        y, x = row['target']['y'], row['target']['x']
        self.assertEqual(row['actual'], round(float(output[0, 0, y, x]), 5))

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
        self.assertEqual({c['stage'] for c in rows},
                         {'normalization', 'decoder', 'denormalization', 'spatial',
                          'temporal', 'pool', 'classifier'})
        self.assertEqual(len([c for c in rows if c['stage'] == 'spatial']), 3)
        self.assertEqual(len([c for c in rows if c['stage'] == 'temporal']), 6)
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
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        trace.attach_yolo(frontend.model)
        image = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
        try:
            # on_predict_start installs the hooks; finish the untraced warm-up
            # first so the detail scheduler knows the top-level module count.
            frontend(image)
            trace.begin_frame(60, 2.)
            frontend(image)
            trace.begin_behavior([8, 2, 91], poses)
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
            trace.end_frame([{'track_id': track_id, 'prediction_updated': True,
                              'probabilities': result['probabilities'][i].tolist()}
                             for i, track_id in enumerate([8, 2, 91])])
        finally:
            trace.close()
        self.assertEqual(trace.dropped['budget_events'], 0)
        self.assertEqual(trace.dropped['oversized_events'], 0)
        self.assertEqual(trace.dropped['detail_budget'], 0)
        rows = [c for event in events for c in event.get('calculations', [])]
        self.assertTrue(rows)
        stages = {row['stage'] for row in rows}
        self.assertIn('yolo', stages)
        self.assertTrue({'normalization', 'decoder', 'denormalization', 'spatial',
                         'temporal', 'pool', 'classifier'} <= stages)
        for row in rows:
            if row['stage'] != 'yolo':
                self.assert_sum(row)
        self.assertTrue(all(row['metadata']['complete_sum'] for row in rows if row['stage'] == 'yolo'))
        self.assertEqual(events[-1]['node'], 'pipeline.output')
        self.assertLessEqual(sum(len(json.dumps(event, separators=(',', ':')).encode()) for event in events),
                             SAMPLING['frame_max_bytes'])


if __name__ == '__main__':
    unittest.main()
