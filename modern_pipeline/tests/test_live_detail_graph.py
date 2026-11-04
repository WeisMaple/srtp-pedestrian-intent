"""Every ST-GCN block exposes one complete spatial and temporal example."""
from collections import defaultdict
import json
import math
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

    def test_ten_detail_frames_cover_all_blocks_with_complete_sums(self):
        predictor = self.predictor()
        rng = np.random.default_rng(23)
        poses = rng.normal(size=(3, 50, 17, 3)).astype(np.float32)
        poses[..., :2] = poses[..., :2] * 20 + 300
        poses[..., 2] = 0.9
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        try:
            for step in range(10):
                trace.begin_frame(100 + step, step)
                trace.begin_behavior([8, 2, 91], poses)
                result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
                trace.end_frame([{'track_id': track, 'prediction_updated': True,
                                  'probabilities': probability.tolist()}
                                 for track, probability in zip([8, 2, 91], result['probabilities'])])
        finally:
            trace.close()
        detail_events = [event for event in events
                         if event.get('detail') and 'detail_block' in event]
        blocks = {event['detail_block'] for event in detail_events}
        self.assertEqual(blocks, set(range(10)))
        self.assertEqual({event['frame'] for event in detail_events}, set(range(100, 110)))
        rows = [c for event in events for c in event.get('calculations', [])
                if c['stage'] in ('spatial', 'temporal')]
        self.assertTrue(rows)
        for row in rows:
            block = row['metadata']['detail_block']
            self.assertIsInstance(row['metadata']['detail_frame'], int)
            self.assertIn(block, range(10))
            suffix = '.boundary' if row['metadata'].get('boundary') else ''
            self.assertEqual(row['id'], 'stgcn.%s.block%d%s.track%d'
                             % (row['stage'], block, suffix, row['track_id']))
            self.assertEqual(row['operation'],
                             'graph_aggregation' if row['stage'] == 'spatial'
                             else 'temporal_convolution')
            self.assertTrue(row['metadata']['complete_sum'])
            size = len(row['inputs'])
            self.assertEqual(len(row['weights']), size)
            total = math.fsum(x * w for x, w in zip(row['inputs'], row['weights'])) + row['bias']
            if row['activation'] == 'silu':
                total = total / (1 + math.exp(-total))
            self.assertAlmostEqual(total, row['actual'], delta=1e-4 * max(1., abs(total)))
        spatial = {(c['metadata']['detail_block'], c['track_id']) for c in rows
                   if c['stage'] == 'spatial'}
        middle = {(c['metadata']['detail_block'], c['track_id']) for c in rows
                  if c['stage'] == 'temporal' and not c['metadata']['boundary']}
        boundary = {(c['metadata']['detail_block'], c['track_id']) for c in rows
                    if c['stage'] == 'temporal' and c['metadata']['boundary']}
        tracks = {8, 2, 91}
        for block in range(10):
            self.assertEqual({track for b, track in spatial if b == block}, tracks)
            self.assertEqual({track for b, track in middle if b == block}, tracks)
            # 256-channel blocks cap boundary operands at two tracks (4608 terms).
            self.assertEqual({track for b, track in boundary if b == block},
                             tracks if block < 7 else {8, 2})
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
        block9_frames = {event['frame'] for event in detail_events if event['detail_block'] == 9}
        self.assertEqual(len(block9_frames), 1)
        block9_frame = block9_frames.pop()
        self.assertLessEqual(detail_bytes[block9_frame], 512 * 1024)
        self.assertLessEqual(frame_bytes[block9_frame], 768 * 1024)

    def test_spatial_row_contains_all_51_terms_and_graph_slice(self):
        predictor = self.predictor()
        poses = np.full((1, 50, 17, 3), 1., dtype=np.float32)
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
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

    def test_detail_off_installs_no_behavior_arithmetic_hooks(self):
        predictor = self.predictor()
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
        finally:
            trace.close()
