"""Observers expose actual computation without changing prediction records."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from modern_pipeline.behavior import forecast_with_statistics
from modern_pipeline.cli import PredictionCancelled, arguments, run
from modern_pipeline.frontend import PersonPose


class ForecastObserverTests(unittest.TestCase):
    def test_five_observed_blocks_are_actual_outputs_and_do_not_change_forecast(self):
        class NextBlock(torch.nn.Module):
            def forward(self, values):
                return (values[:, -10:] + 1).reshape(len(values), 10, 17, 3)

        poses = torch.zeros(1, 50, 17, 3)
        mean, std = torch.zeros(17, 3), torch.ones(17, 3)
        expected = forecast_with_statistics(NextBlock(), poses, mean, std)
        events = []
        actual = forecast_with_statistics(NextBlock(), poses, mean, std, on_event=events.append)
        torch.testing.assert_close(actual, expected)
        self.assertEqual([e['step'] for e in events], [1, 2, 3, 4, 5])
        self.assertEqual([float(e['poses'][0, 0, 0, 0]) for e in events], [1, 2, 3, 4, 5])
        self.assertEqual(actual.shape, (1, 50, 17, 3))


class CliObserverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'input.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (160, 120))
        self.assertTrue(writer.isOpened())
        for _ in range(54):
            writer.write(np.zeros((120, 160, 3), np.uint8))
        writer.release()

    def execute(self, name, on_event=None, cancel=None):
        class Front:
            def __init__(self, *args):
                pass

            def __call__(self, frame):
                pose = np.tile([80, 90, 0.9], (17, 1)).astype(np.float32)
                return [PersonPose(7, np.array([60, 10, 100, 110]), 0.9, pose)]

        class Behavior:
            def __init__(self, *args):
                self.device = torch.device('cpu')

            def predict(self, histories, reference, on_event=None):
                return {'class_ids': np.full(len(histories), 2),
                        'probabilities': np.tile([0.1, 0.2, 0.7], (len(histories), 1)),
                        'future_poses': histories.copy() + np.array([1, 0, 0], np.float32)}

        output = self.root / (name + '.mp4')
        args = arguments(['--input', str(self.source), '--output', str(output),
                          '--no-video', '--device', 'cpu', '--save-keypoints'])
        with patch('modern_pipeline.frontend.YOLOPoseTracker', Front), \
                patch('modern_pipeline.behavior.BehaviorPredictor', Behavior), \
                contextlib.redirect_stdout(io.StringIO()):
            run(args, on_event=on_event, cancel=cancel)
        return [json.loads(line) for line in output.with_suffix('.jsonl').read_text().splitlines()]

    def test_cli_records_match_and_trace_keeps_history_future_and_ids(self):
        plain = self.execute('plain')
        events = []
        observed = self.execute('observed', events.append)
        self.assertEqual(plain, observed)
        frames = [e for e in events if e['kind'] == 'frame']
        self.assertEqual(len(frames), 54)
        trace = frames[-1]['trace'][0]
        self.assertEqual(trace['track_id'], 7)
        self.assertEqual(np.asarray(trace['history']).shape, (50, 17, 3))
        self.assertEqual(np.asarray(trace['future']).shape, (50, 17, 3))
        self.assertEqual(float(trace['future'][0, 0, 0]), 81)

    def test_cancellation_does_not_mark_partial_job_complete(self):
        cancelled = threading.Event()

        def observe(event):
            if event['kind'] == 'frame':
                cancelled.set()

        with self.assertRaises(PredictionCancelled):
            self.execute('cancelled', observe, cancelled)
        metrics = json.loads((self.root / 'cancelled.metrics.json').read_text())
        self.assertEqual(metrics['status'], 'cancelled')
        self.assertEqual(metrics['frames'], 1)


if __name__ == '__main__':
    unittest.main()
