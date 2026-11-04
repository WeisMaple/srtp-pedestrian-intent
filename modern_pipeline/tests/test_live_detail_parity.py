"""Detail telemetry never changes the production prediction."""
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_trace import LiveTrace
from st_gcn import Model, SkeletonLSTM


class LiveDetailParityTests(unittest.TestCase):
    def test_detail_hooks_leave_probabilities_and_future_poses_unchanged(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        predictor = BehaviorPredictor.__new__(BehaviorPredictor)
        predictor.device = torch.device('cpu')
        predictor.normalization = 'legacy_pair'
        predictor.lstm = SkeletonLSTM(num_layers=2).eval()
        predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        predictor.stats = {}
        poses = np.random.default_rng(31).normal(size=(3, 50, 17, 3)).astype(np.float32)
        baseline = predictor.predict(poses, poses[0])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        try:
            trace.begin_frame(1, 0)
            trace.begin_behavior([8, 2, 91], poses)
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
        finally:
            trace.close()
        np.testing.assert_array_equal(result['probabilities'], baseline['probabilities'])
        np.testing.assert_array_equal(result['future_poses'], baseline['future_poses'])
        np.testing.assert_array_equal(result['class_ids'], baseline['class_ids'])
        # The detail path must actually run; otherwise this parity is vacuous.
        self.assertTrue(any(event.get('detail') for event in events))
