"""Every ST-GCN block exposes one complete spatial and temporal example."""
from collections import defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_trace import LiveTrace
from st_gcn import Model, SkeletonLSTM


class LiveDetailGraphTests(unittest.TestCase):
    def predictor(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        predictor = BehaviorPredictor.__new__(BehaviorPredictor)
        predictor.device = torch.device('cpu')
        predictor.normalization = 'legacy_pair'
        predictor.lstm = SkeletonLSTM(num_layers=2).eval()
        predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        predictor.stats = {}
        return predictor

    def assert_row(self, calculation):
        block = calculation['metadata']['detail_block']
        self.assertIsInstance(calculation['metadata']['detail_frame'], int)
        self.assertIn(block, range(10))
        suffix = '.boundary' if calculation['metadata'].get('boundary') else ''
        self.assertEqual(calculation['id'], 'stgcn.%s.block%d%s.track%d' % (
            calculation['stage'], block, suffix, calculation['track_id']))
        self.assertEqual(calculation['operation'],
                         'graph_aggregation' if calculation['stage'] == 'spatial'
                         else 'temporal_convolution')
        self.assertTrue(calculation['metadata']['complete_sum'])
        size = len(calculation['inputs'])
        self.assertEqual(len(calculation['weights']), size)
        total = math.fsum(x * w for x, w in zip(calculation['inputs'], calculation['weights']))
        total += calculation['bias']
        if calculation['activation'] == 'silu':
            total = total / (1 + math.exp(-total))
        self.assertAlmostEqual(total, calculation['actual'],
                               delta=1e-4 * max(1., abs(total)), msg=calculation['id'])

    def test_three_block_cycles_cover_yolo_modules_and_rotate_deep_boundaries(self):
        predictor = self.predictor()
        rng = np.random.default_rng(23)
        poses = rng.normal(size=(3, 50, 17, 3)).astype(np.float32)
        poses[..., :2] = poses[..., :2] * 20 + 300
        poses[..., 2] = 0.9
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        callbacks = {'on_predict_start': []}
        yolo = SimpleNamespace(callbacks=callbacks,
                               add_callback=lambda name, callback: callbacks[name].append(callback))
        network = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(24)])
        backend = SimpleNamespace(model=network, is_fused=lambda: True)
        runner = SimpleNamespace(model=SimpleNamespace(model=backend), done_warmup=False)
        trace.attach_yolo(yolo)
        callbacks['on_predict_start'][0](runner)
        seen_yolo = set()
        frame_meta = {}
        try:
            for step in range(60):
                trace.begin_frame(100 + step, step)
                detail_index = trace.detail_index
                self.assertEqual(trace.detail_block, ((detail_index - 1) // 2) % 10)
                if trace.detail_boundary:
                    self.assertEqual(trace.detail_modules, set())
                else:
                    self.assertEqual(len(trace.detail_modules), 4)
                seen_yolo.update(trace.detail_modules)
                frame_meta[100 + step] = (detail_index, trace.detail_block,
                                          trace.detail_boundary)
                trace.begin_behavior([8, 2, 91], poses)
                result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
                trace.end_frame([{'track_id': track, 'prediction_updated': True,
                                  'probabilities': probability.tolist()}
                                 for track, probability in zip([8, 2, 91], result['probabilities'])])
        finally:
            trace.close()
        # Three 20-frame block cycles exercise all ten blocks six times each.
        self.assertEqual(len(frame_meta), 60)
        # Six odd detail frames cover the 24 scheduled modules.
        self.assertEqual(seen_yolo, set(range(24)))
        detail_events = [event for event in events
                         if event.get('detail') and 'detail_block' in event]
        self.assertEqual({event['frame'] for event in detail_events}, set(range(100, 160)))
        self.assertEqual({event['detail_block'] for event in detail_events}, set(range(10)))
        detail_rows = [(event['frame'], c) for event in events
                       for c in event.get('calculations', [])
                       if c['stage'] in ('spatial', 'temporal')]
        self.assertTrue(detail_rows)
        for _, row in detail_rows:
            self.assert_row(row)
        by_frame = defaultdict(list)
        for frame, row in detail_rows:
            by_frame[frame].append(row)
        track_order = [8, 2, 91]
        tracks = set(track_order)
        boundary_by_frame = {}
        for step in range(60):
            frame = 100 + step
            detail_index, block, boundary_frame = frame_meta[frame]
            frame_rows = by_frame[frame]
            self.assertEqual({c['metadata']['detail_block'] for c in frame_rows}, {block})
            self.assertEqual({c['track_id'] for c in frame_rows if c['stage'] == 'spatial'}, tracks)
            self.assertEqual({c['track_id'] for c in frame_rows
                              if c['stage'] == 'temporal' and not c['metadata']['boundary']}, tracks)
            boundary_by_frame[frame] = {c['track_id'] for c in frame_rows
                                        if c['stage'] == 'temporal' and c['metadata']['boundary']}
            if boundary_frame:
                expected = (tracks if block < 4
                            else {track_order[(detail_index // 2) % len(track_order)]})
                self.assertEqual(boundary_by_frame[frame], expected)
            else:
                self.assertEqual(boundary_by_frame[frame], set())
        for block in range(10):
            frames = [frame for frame, (_, b, _) in frame_meta.items() if b == block]
            yolo_frames = [frame for frame in frames if not frame_meta[frame][2]]
            boundary_frames = [frame for frame in frames if frame_meta[frame][2]]
            self.assertEqual(len(yolo_frames), 3)
            self.assertEqual(len(boundary_frames), 3)
            # Both of each block's frames carry spatial + middle for all tracks.
            for frame in frames:
                self.assertTrue({c['track_id'] for c in by_frame[frame]
                                 if c['stage'] == 'spatial'} == tracks)
                self.assertTrue({c['track_id'] for c in by_frame[frame]
                                 if c['stage'] == 'temporal'
                                 and not c['metadata']['boundary']} == tracks)
            if block < 4:
                for frame in boundary_frames:
                    self.assertEqual(boundary_by_frame[frame], tracks)
            else:
                seen = set()
                for frame in boundary_frames:
                    self.assertEqual(len(boundary_by_frame[frame]), 1)
                    seen.update(boundary_by_frame[frame])
                self.assertEqual(seen, tracks)
        self.assertEqual(trace.dropped['detail_budget'], 0)
        self.assertEqual(trace.dropped['budget_events'], 0)
        self.assertEqual(trace.dropped['oversized_events'], 0)
        sizes = [len(json.dumps(event, separators=(',', ':'), allow_nan=False).encode())
                 for event in events]
        self.assertLessEqual(max(sizes), 128 * 1024)
        detail_bytes, frame_bytes = defaultdict(int), defaultdict(int)
        for event, size in zip(events, sizes):
            frame_bytes[event['frame']] += size
            if event.get('detail'):
                detail_bytes[event['frame']] += size
        self.assertLessEqual(max(detail_bytes.values()), 512 * 1024)
        self.assertLessEqual(max(frame_bytes.values()), 768 * 1024)
        block9_frames = sorted(frame for frame, (_, b, boundary) in frame_meta.items()
                               if b == 9 and boundary)
        self.assertEqual(block9_frames, [119, 139, 159])
        for frame in block9_frames:
            self.assertLessEqual(detail_bytes[frame], 512 * 1024)
            self.assertLessEqual(frame_bytes[frame], 768 * 1024)

    def test_spatial_row_contains_all_51_terms_and_graph_slice(self):
        predictor = self.predictor()
        poses = np.full((1, 50, 17, 3), 1., dtype=np.float32)
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        trace.detail_index = 1  # the next frame is block 0's boundary subframe
        try:
            trace.begin_frame(1, 0)
            trace.begin_behavior([7], poses)
            predictor.predict(poses, poses[0], on_event=trace.forecast_step)
        finally:
            trace.close()
        spatial = next(c for event in events for c in event.get('calculations', [])
                       if c.get('stage') == 'spatial')
        self.assertEqual(len(spatial['inputs']), 51)
        self.assertEqual(spatial['operation'], 'graph_aggregation')
        self.assertEqual(len(spatial['metadata']['graph_slice']['outputs']), 17)
        self.assertEqual(spatial['metadata']['graph_slice']['joint_count'], 17)
        temporal = [c for event in events for c in event.get('calculations', [])
                    if c.get('stage') == 'temporal']
        self.assertTrue(temporal)
        self.assertTrue(all(c['operation'] == 'temporal_convolution' for c in temporal))
        self.assertTrue(any(any(c['padding']) for c in temporal))
        self.assertTrue(any(not any(c['padding']) for c in temporal))
        self.assertTrue(any(c['metadata']['boundary'] and '.boundary.' in c['id']
                            for c in temporal))
        self.assertTrue(any(not c['metadata']['boundary'] and '.boundary.' not in c['id']
                            for c in temporal))

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[2] / 'weights/final/stgcn.pt').is_file()
        and (Path(__file__).resolve().parents[2] / 'weights/final/lstm.pt').is_file()
        and (Path(__file__).resolve().parents[2] / 'weights/final/yolo26s-pose.pt').is_file(),
        'local final reproduction weights are absent')
    def test_paired_yolo_and_boundary_subframes_fit_budgets(self):
        from modern_pipeline.frontend import YOLOPoseTracker
        directory = Path(__file__).resolve().parents[2] / 'weights/final'
        torch.set_num_threads(2)
        predictor = BehaviorPredictor(directory / 'stgcn.pt', directory / 'lstm.pt', device='cpu')
        frontend = YOLOPoseTracker(directory / 'yolo26s-pose.pt', 'cpu')
        rng = np.random.default_rng(17)
        poses = rng.normal(size=(3, 50, 17, 3)).astype(np.float32)
        poses[..., :2] = poses[..., :2] * 20 + 300
        poses[..., 2] = 0.9
        image = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        trace.attach_yolo(frontend.model)
        # Block 9's two detail frames: d19 is the YOLO subframe with four
        # scheduled modules and no boundary rows; d20 is the boundary subframe
        # with one rotating track and no YOLO details. YOLO inference still runs
        # on both frames, so the even frame carries its layer-output events.
        frames = ((18, 19, {12, 13, 14, 15}), (19, 20, set()))
        try:
            frontend(image)
            for step, (preset, expected_index, modules) in enumerate(frames):
                trace.detail_index = preset
                trace.begin_frame(60 + step, 2. + step)
                self.assertEqual(trace.detail_index, expected_index)
                self.assertEqual(trace.detail_block, 9)
                self.assertEqual(trace.detail_modules, modules)
                self.assertEqual(trace.detail_boundary, expected_index % 2 == 0)
                frontend(image)
                trace.begin_behavior([8, 2, 91], poses)
                result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
                trace.end_frame([{'track_id': track, 'prediction_updated': True,
                                  'probabilities': probability.tolist()}
                                 for track, probability in zip([8, 2, 91], result['probabilities'])])
        finally:
            trace.close()
        self.assertEqual(trace.dropped['detail_budget'], 0)
        self.assertEqual(trace.dropped['budget_events'], 0)
        self.assertEqual(trace.dropped['oversized_events'], 0)
        expected = {0: {'yolo_min': 1, 'boundary': set()},
                    1: {'yolo_min': 0, 'boundary': {2}}}
        for step in range(2):
            frame_events = [event for event in events if event['frame'] == 60 + step]
            rows = [c for event in frame_events for c in event.get('calculations', [])]
            yolo_rows = [c for c in rows if c['stage'] == 'yolo']
            self.assertGreaterEqual(len(yolo_rows), expected[step]['yolo_min'])
            self.assertTrue(all(row['metadata']['complete_sum'] for row in yolo_rows))
            self.assertEqual({c['track_id'] for c in rows if c['stage'] == 'spatial'},
                             {8, 2, 91})
            self.assertEqual({c['track_id'] for c in rows
                              if c['stage'] == 'temporal' and not c['metadata']['boundary']},
                             {8, 2, 91})
            boundary = {c['track_id'] for c in rows
                        if c['stage'] == 'temporal' and c['metadata']['boundary']}
            self.assertEqual(boundary, expected[step]['boundary'])
            for row in rows:
                if row['stage'] in ('spatial', 'temporal'):
                    self.assert_row(row)
            sizes = [len(json.dumps(event, separators=(',', ':'), allow_nan=False).encode())
                     for event in frame_events]
            self.assertLessEqual(max(sizes), 128 * 1024)
            self.assertLessEqual(sum(size for event, size in zip(frame_events, sizes)
                                     if event.get('detail')), 512 * 1024)
            self.assertLessEqual(sum(sizes), 768 * 1024)
            self.assertEqual(frame_events[-1]['node'], 'pipeline.output')
            self.assertEqual({row['metadata']['detail_block'] for row in rows
                              if row['stage'] in ('spatial', 'temporal')}, {9})
        # The even frame supplies block 9's boundary example.
        self.assertEqual(expected[1]['boundary'], {2})

    def test_pool_serializes_nonfinite_contributions_as_null(self):
        predictor = self.predictor()
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        try:
            trace.begin_frame(1, 0)
            trace.begin_behavior([5])
            arithmetic = trace.arithmetic
            arithmetic.pool_values = torch.tensor([[[1.0, float('nan'), float('inf')]]])
            arithmetic.pool(torch.zeros(1, 3, 1, 1))
        finally:
            trace.close()
        row = next(c for event in events for c in event.get('calculations', [])
                   if c['stage'] == 'pool')
        self.assertEqual(row['inputs'], [0.33333, None, None])
        self.assertEqual(row['weights'], [1.0, 1.0, 1.0])
        self.assertEqual(row['actual'], 0.0)
        self.assertEqual(row['metadata']['weight_rounding'], 'operands_scaled_by_divisor')

    def test_non_detail_frame_emits_no_arithmetic_rows(self):
        predictor = self.predictor()
        poses = np.full((1, 50, 17, 3), 1., dtype=np.float32)
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=999)
        trace.attach_behavior(predictor)
        try:
            self.assertTrue(trace.begin_frame(1, 0.0))
            self.assertTrue(trace.detail_frame)
            self.assertTrue(trace.begin_frame(2, 0.1))
            self.assertFalse(trace.detail_frame)
            trace.begin_behavior([7], poses)
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
            trace.end_frame([{'track_id': 7, 'prediction_updated': True,
                              'probabilities': result['probabilities'][0].tolist()}])
        finally:
            trace.close()
        self.assertTrue(any(event['frame'] == 2 and event['model'] == 'stgcn'
                            for event in events))
        self.assertFalse(any(event.get('calculations') for event in events))

    def test_classifier_handles_nonfinite_logits(self):
        predictor = self.predictor()
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        try:
            trace.begin_frame(1, 0)
            trace.begin_behavior([5])
            arithmetic = trace.arithmetic
            inputs = [torch.zeros(1, 256, 1, 1)]
            arithmetic.classifier(predictor.stgcn.fcn, inputs,
                                  torch.tensor([[float('nan'), 0.5, float('-inf')]]))
            row = arithmetic.classifiers[0]
            self.assertEqual(row['metadata']['selected_class'], 1)
            self.assertEqual(row['actual'], 0.5)
            self.assertIsNone(row['metadata']['logits'][0])
            self.assertIsNone(row['metadata']['logits'][2])
            self.assertEqual(len(row['weights']), 256)
            arithmetic.classifier(predictor.stgcn.fcn, inputs,
                                  torch.tensor([[float('nan'), float('inf'), float('-inf')]]))
            row = arithmetic.classifiers[0]
            self.assertIsNone(row['metadata']['selected_class'])
            self.assertIsNone(row['actual'])
            self.assertIsNone(row['bias'])
            self.assertEqual(row['metadata']['target_indices'], [])
            self.assertEqual(len(row['weights']), 256)
            self.assertTrue(all(weight is None for weight in row['weights']))
            self.assertEqual(row['metadata']['logits'], [None, None, None])
            self.assertFalse(row['metadata']['complete_sum'])
            self.assertTrue(row['metadata']['nonfinite_logits'])
        finally:
            trace.close()

    def test_detail_off_install_no_behavior_arithmetic_hooks_or_rows(self):
        predictor = self.predictor()
        poses = np.full((1, 50, 17, 3), 1., dtype=np.float32)
        events = []
        trace = LiveTrace(events.append, interval=0)
        trace.attach_behavior(predictor)
        try:
            self.assertFalse(predictor.lstm.fc._forward_hooks)
            self.assertFalse(predictor.lstm.fc._forward_pre_hooks)
            for block in predictor.stgcn.st_gcn_networks:
                self.assertFalse(block.gcn.conv._forward_hooks)
                self.assertFalse(block.gcn.conv._forward_pre_hooks)
                self.assertFalse(block.tcn[2]._forward_hooks)
                self.assertFalse(block.tcn[2]._forward_pre_hooks)
            # Only LiveTrace's own layer-output hook remains on the final block.
            self.assertEqual(len(predictor.stgcn.st_gcn_networks[-1]._forward_hooks), 1)
            trace.begin_frame(1, 0)
            trace.begin_behavior([7], poses)
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
            trace.end_frame([{'track_id': 7, 'prediction_updated': True,
                              'probabilities': result['probabilities'][0].tolist()}])
        finally:
            trace.close()
        self.assertTrue(any(event['node'] == 'stgcn.9.output' for event in events))
        self.assertFalse(any(event.get('calculations') for event in events))
