"""Camera inference continuity and socket resource ownership."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from modern_pipeline.frontend import PersonPose
from modern_pipeline.web.camera import CameraSession
from modern_pipeline.web.app import create_app
from modern_pipeline.web.jobs import JobManager


class Predictor:
    normalization = 'training_stats'

    def __init__(self):
        self.histories = []

    def predict(self, histories, reference):
        self.histories.append(histories.copy())
        return {'class_ids': np.full(len(histories), 2),
                'probabilities': np.tile([0.1, 0.2, 0.7], (len(histories), 1))}


def jpeg(width=160, height=120):
    return cv2.imencode('.jpg', np.zeros((height, width, 3), np.uint8))[1].tobytes()


class CameraSessionTests(unittest.TestCase):
    def setUp(self):
        self.people = [PersonPose(7, np.array([10, 10, 80, 110]), 0.9,
                                  np.tile([30, 40, 0.9], (17, 1)).astype(np.float32))]
        self.predictor = Predictor()
        self.session = CameraSession(lambda frame: self.people, self.predictor)

    def test_fifty_observations_predict_and_return_aligned_pose_and_result(self):
        for index in range(49):
            row = self.session.process(jpeg(), timestamp=index * 0.1)
            self.assertIsNone(row['tracks'][0]['class_id'])
        row = self.session.process(jpeg(), timestamp=4.9)
        self.assertEqual(row['frame'], 49)
        self.assertEqual(row['tracks'][0]['history_length'], 50)
        self.assertEqual(row['tracks'][0]['class_id'], 2)
        self.assertEqual(row['tracks'][0]['keypoints'][0][:2], [30, 40])
        self.assertEqual(self.predictor.histories[0].shape, (1, 50, 17, 3))
        self.assertNotIn('calculations', row)

    def test_invalid_pose_gap_and_changed_resolution_reset_warmup(self):
        for index in range(50):
            self.session.process(jpeg(), timestamp=index * 0.1)
        self.people[0].pose[:, 2] = 0
        row = self.session.process(jpeg(), timestamp=5.0)['tracks'][0]
        self.assertIsNone(row['class_id'])
        self.assertEqual(row['history_length'], 0)
        self.people[0].pose[:, 2] = 0.9
        self.assertEqual(self.session.process(jpeg(), timestamp=5.1)['tracks'][0]['history_length'], 1)
        self.assertEqual(self.session.process(jpeg(), timestamp=8.0)['tracks'][0]['history_length'], 1)
        self.assertEqual(self.session.process(jpeg(320), timestamp=8.1)['tracks'][0]['history_length'], 1)

    def test_empty_frames_drop_results_and_legacy_reference_warms_once(self):
        self.predictor.normalization = 'legacy_pair'
        for index in range(50):
            row = self.session.process(jpeg(), timestamp=index * 0.1)
        self.assertIsNone(row['tracks'][0]['class_id'])
        self.assertEqual(self.session.process(jpeg(), timestamp=5.0)['tracks'][0]['class_id'], 2)
        person = self.people.pop()
        for index in range(3):
            self.assertEqual(self.session.process(jpeg(), timestamp=5.1 + index * 0.1)['tracks'], [])
        self.people.append(person)
        row = self.session.process(jpeg(), timestamp=5.4)['tracks'][0]
        self.assertEqual(row['history_length'], 1)
        self.assertIsNone(row['class_id'])

    def test_invalid_frames_are_rejected_before_tracking(self):
        for payload in (b'', b'invalid', b'x' * (2 * 1024 * 1024 + 1), jpeg(2000, 1200)):
            with self.assertRaises(ValueError):
                self.session.process(payload)
        self.assertEqual(self.session.process(jpeg())['frame'], 0)

    def test_configured_pose_filter_holds_two_frames_then_expires_without_cached_intent(self):
        self.session = CameraSession(lambda frame: self.people, self.predictor, pose_filter=True)
        for index in range(50):
            self.session.process(jpeg(), timestamp=index * 0.1)
        self.people[0].pose[:, 2] = 0
        for timestamp in (5.0, 5.1):
            row = self.session.process(jpeg(), timestamp=timestamp)['tracks'][0]
            self.assertTrue(row['prediction_updated'])
        row = self.session.process(jpeg(), timestamp=5.2)['tracks'][0]
        self.assertFalse(row['pose_accepted'])
        self.assertIsNone(row['class_id'])
        self.assertEqual(row['history_length'], 0)


class CameraSocketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with patch('modern_pipeline.web.jobs.validate_bundle', return_value={'normalization': 'training_stats'}):
            self.manager = JobManager(Path(self.tmp.name))
        self.addCleanup(self.manager.close)
        self.client = TestClient(create_app(self.manager))
        self.addCleanup(self.client.close)
        factory = patch('modern_pipeline.web.camera.CameraSession.from_models',
                        side_effect=lambda *args: CameraSession(lambda frame: [], Predictor()))
        factory.start()
        self.addCleanup(factory.stop)

    def test_socket_excludes_other_inference_and_disconnect_releases_admission(self):
        with self.client.websocket_connect('/api/camera/live') as socket:
            self.assertEqual(socket.receive_json()['type'], 'ready')
            self.assertTrue(self.manager.network_active)
            with self.client.websocket_connect('/api/camera/live') as second:
                self.assertEqual(second.receive_json()['type'], 'error')
            self.assertEqual(self.client.get('/api/model').status_code, 409)
            socket.send_bytes(jpeg())
            row = socket.receive_json()
            self.assertEqual(row['type'], 'result')
            self.assertEqual(row['frame'], 0)
            self.assertEqual(row['tracks'], [])
        self.assertFalse(self.manager.network_active)
        self.assertFalse(self.manager.camera_active)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_invalid_input_and_loading_failure_release_admission(self):
        with self.client.websocket_connect('/api/camera/live') as socket:
            socket.receive_json()
            socket.send_bytes(b'bad jpeg')
            self.assertEqual(socket.receive_json()['type'], 'error')
        self.assertFalse(self.manager.network_active)
        with patch('modern_pipeline.web.camera.CameraSession.from_models', side_effect=RuntimeError('load failed')):
            with self.client.websocket_connect('/api/camera/live') as socket:
                self.assertEqual(socket.receive_json()['type'], 'error')
        self.assertFalse(self.manager.network_active)

    def test_busy_video_and_cross_origin_socket_are_rejected(self):
        self.manager.active = 'video'
        try:
            with self.client.websocket_connect('/api/camera/live') as socket:
                self.assertEqual(socket.receive_json()['type'], 'error')
            self.assertFalse(self.manager.network_active)
        finally:
            self.manager.active = None
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/api/camera/live', headers={'origin': 'https://unrelated.example'}):
                pass

    def test_camera_busy_reason_survives_prior_session_finishing_before_response(self):
        self.manager.network_active = self.manager.camera_active = True
        original = self.manager.open_camera

        def racing_open():
            try:
                return original()
            finally:
                self.manager.close_camera()

        with patch.object(self.manager, 'open_camera', side_effect=racing_open):
            with self.client.websocket_connect('/api/camera/live') as socket:
                self.assertEqual(socket.receive_json()['code'], 'camera_busy')
