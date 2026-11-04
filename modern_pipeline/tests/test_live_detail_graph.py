"""Every ST-GCN block exposes one complete spatial and temporal example."""
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
        blocks = {event['detail_block'] for event in events if event.get('detail')}
        self.assertEqual(blocks, set(range(10)))
        rows = [c for event in events for c in event.get('calculations', [])
                if c['stage'] in ('spatial', 'temporal')]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row['metadata']['complete_sum'])
            size = len(row['inputs'])
            self.assertEqual(len(row['weights']), size)
            total = math.fsum(x * w for x, w in zip(row['inputs'], row['weights'])) + row['bias']
            if row['activation'] == 'silu':
                total = total / (1 + math.exp(-total))
            self.assertAlmostEqual(total, row['actual'], delta=3e-5 * max(1., abs(total)))

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
        self.assertEqual(len(spatial['metadata']['graph_slice']['outputs']), 17)
        self.assertEqual(spatial['metadata']['graph_slice']['joint_count'], 17)
        temporal = [c for event in events for c in event.get('calculations', [])
                    if c.get('stage') == 'temporal']
        self.assertTrue(any(any(c['padding']) for c in temporal))
        self.assertTrue(any(not any(c['padding']) for c in temporal))
