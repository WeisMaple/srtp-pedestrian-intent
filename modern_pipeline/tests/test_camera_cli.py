"""CLI validation and live-display expiry checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from modern_pipeline.camera_cli import arguments, expire_display, run, validate_artifacts


class CameraCliTests(unittest.TestCase):
    def test_rejects_invalid_timing_and_headless_without_road(self):
        for args in (['--horizon', 'nan'], ['--model-fps', '0'], ['--max-gap', '-1'],
                     ['--model-interval', 'inf'], ['--camera', '-1'], ['--no-display']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    arguments(args)
                self.assertEqual(raised.exception.code, 2)

    def test_existing_log_is_refused_and_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'pose.pt'
            weights.write_bytes(b'fixture')
            log = Path(directory) / 'demo.jsonl'
            log.write_text('keep me', encoding='utf-8')
            args = arguments(['--pose-model', str(weights), '--log', str(log)])
            with self.assertRaises(FileExistsError):
                validate_artifacts(args)
            self.assertEqual(log.read_text(encoding='utf-8'), 'keep me')

    def test_behavior_weights_required_only_for_explicit_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'pose.pt'
            weights.write_bytes(b'fixture')
            args = arguments(['--pose-model', str(weights)])
            validate_artifacts(args)
            args.behavior = True
            args.stgcn = Path(directory) / 'absent.pt'
            with self.assertRaises(FileNotFoundError):
                validate_artifacts(args)

    def test_slow_inference_cannot_display_outdated_forecast_as_current(self):
        rows = [{'baseline': {'state': 'entering', 'eta_seconds': 0.2, 'forecast': [0.5, 0.5]},
                 'model': {'state': 'valid', 'class_id': 2, 'score': 0.9, 'age_seconds': 0.1}}]
        expire_display(rows, frame_age=0.8, ttl=0.6)
        self.assertEqual(rows[0]['baseline']['state'], 'expired')
        self.assertIsNone(rows[0]['baseline']['forecast'])
        self.assertIsNone(rows[0]['baseline']['eta_seconds'])
        self.assertEqual(rows[0]['model']['state'], 'expired')
        self.assertIsNone(rows[0]['model']['class_id'])
        self.assertAlmostEqual(rows[0]['model']['age_seconds'], 0.9)

    def run_camera_fixture(self, directory, display, camera_open=True, slow_second=False,
                           stall_second=False, display_failure=False):
        """Only the hardware and heavyweight detector are replaced."""
        class Camera:
            released = False
            reads = 0

            def isOpened(self):
                return camera_open

            def set(self, prop, value):
                return True

            def read(self):
                self.reads += 1
                if stall_second and self.reads == 2:
                    busy.set()
                    time.sleep(0.18)
                    busy.clear()
                time.sleep(0.01)
                return True, np.full((240, 320, 3), 80, dtype=np.uint8)

            def release(self):
                self.released = True

        class Frontend:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def __call__(self, frame):
                self.calls += 1
                if slow_second and self.calls == 2:
                    busy.set()
                    time.sleep(0.18)
                    busy.clear()
                return [SimpleNamespace(track_id=7, bbox=np.array([130, 30, 190, 210]),
                                        pose=np.tile([160, 200, 0.9], (17, 1)).astype(np.float32))]

        weights = Path(directory) / 'pose.pt'
        weights.write_bytes(b'fixture')
        log = Path(directory) / 'session.jsonl'
        args = ['--pose-model', str(weights), '--log', str(log), '--device', 'cpu',
                '--road', '0.4,0.2', '0.6,0.2', '0.6,0.9', '0.4,0.9',
                '--max-frames', '3', '--save-keypoints']
        if not display:
            args.append('--no-display')
        if slow_second or stall_second:
            args.extend(['--prediction-ttl', '0.05'])
        camera = Camera()
        busy = threading.Event()
        self.displayed_while_busy = []
        from modern_pipeline.camera_cli import render_frame

        def render(*render_args, **kwargs):
            if busy.is_set() and render_args[2]:
                self.displayed_while_busy.append(render_args[2][0]['baseline']['state'])
            return render_frame(*render_args, **kwargs)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('cv2.VideoCapture', return_value=camera))
            stack.enter_context(patch('modern_pipeline.frontend.YOLOPoseTracker', Frontend))
            for name in ('namedWindow', 'setMouseCallback', 'imshow', 'destroyAllWindows'):
                stack.enter_context(patch('cv2.' + name))
            stack.enter_context(patch('modern_pipeline.camera_cli.render_frame', side_effect=render))
            stack.enter_context(patch('cv2.waitKey', return_value=-1 if slow_second or stall_second else ord('q')))
            stack.enter_context(patch('cv2.getWindowProperty', return_value=1))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if display_failure:
                import cv2
                stack.enter_context(patch('cv2.namedWindow', side_effect=cv2.error('display unavailable')))
                stack.enter_context(patch('cv2.destroyAllWindows', side_effect=cv2.error('cleanup unavailable')))
                with self.assertRaisesRegex(RuntimeError, 'display unavailable'):
                    run(arguments(args))
            elif camera_open:
                run(arguments(args))
            else:
                with self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
                    run(arguments(args))
        self.assertTrue(camera.released)
        return [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]

    def test_camera_loop_writes_real_session_results_and_releases_device(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_camera_fixture(directory, display=False)
        self.assertEqual(records[0]['event'], 'session')
        frames = [r for r in records if r['event'] == 'frame']
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[-1]['tracks'][0]['baseline']['state'], 'inside')
        self.assertEqual(len(frames[-1]['tracks'][0]['keypoints']), 17)
        self.assertEqual(records[-1]['status'], 'complete')
        self.assertEqual(records[-1]['processed_frames'], len(frames))

    def test_quit_counts_the_frame_already_displayed_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_camera_fixture(directory, display=True)
        self.assertEqual(len([r for r in records if r['event'] == 'frame']), 1)
        self.assertEqual(records[-1]['processed_frames'], 1)

    def test_camera_open_failure_is_logged_and_device_released(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_camera_fixture(directory, display=False, camera_open=False)
        self.assertEqual(records[-1]['status'], 'failed')
        self.assertEqual(records[-1]['processed_frames'], 0)

    def test_display_expires_while_next_detector_call_is_still_running(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_camera_fixture(directory, display=True, slow_second=True)
        self.assertIn('expired', self.displayed_while_busy)

    def test_display_expires_while_camera_is_not_delivering_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_camera_fixture(directory, display=True, stall_second=True)
        self.assertIn('expired', self.displayed_while_busy)

    def test_gui_cleanup_error_cannot_mask_original_failure_or_skip_end_log(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_camera_fixture(directory, display=True, display_failure=True)
        self.assertEqual(records[-1]['event'], 'end')
        self.assertEqual(records[-1]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
