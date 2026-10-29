"""Numerical and admission checks for the source-frame YOLO replay."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient
import numpy as np
import torch
from torch import nn

from modern_pipeline.bootstrap import PROJECT_ROOT
from modern_pipeline.web.app import create_app
from modern_pipeline.web.jobs import Job, JobBusy, JobManager


class FrontendNumericalTests(unittest.TestCase):
    def test_feature_map_keeps_native_shape_and_exact_sampled_values(self):
        from modern_pipeline.web.frontend_network import feature_map
        tensor = torch.arange(2 * 20 * 30, dtype=torch.float32).reshape(1, 2, 20, 30)
        sampled = feature_map(tensor, channels=1, side=8)
        self.assertEqual(sampled['shape'], [1, 2, 20, 30])
        self.assertEqual(sampled['channels'], [0])
        self.assertEqual(np.shape(sampled['values']), (1, 8, 8))
        self.assertEqual(sampled['values'][0][0][0], 0)
        self.assertEqual(sampled['values'][0][-1][-1], 599)
        for y, native_y in enumerate(sampled['sample_y']):
            for x, native_x in enumerate(sampled['sample_x']):
                self.assertEqual(sampled['values'][0][y][x], native_y * 30 + native_x)

    def test_convolution_products_include_all_rgb_channels_and_match_bn_silu(self):
        from modern_pipeline.web.frontend_network import convolution_example
        block = nn.Module()
        block.conv = nn.Conv2d(3, 1, 3, stride=2, padding=1, bias=False)
        block.bn = nn.BatchNorm2d(1, eps=0.001)
        block.act = nn.SiLU()
        with torch.no_grad():
            block.conv.weight.fill_(0.5)
            block.bn.weight.fill_(2)
            block.bn.bias.fill_(0.25)
            block.bn.running_mean.fill_(1)
            block.bn.running_var.fill_(4)
        block.eval()
        value = torch.ones(1, 3, 5, 5)
        actual = block.act(block.bn(block.conv(value)))
        example = convolution_example(block, value, actual, position=(1, 1), channel=0)
        self.assertEqual(example['input_origin'], [1, 1])
        self.assertEqual(np.shape(example['products']), (3, 3, 3))
        self.assertEqual(example['sum'], 13.5)
        expected_bn = (13.5 - 1) * 2 / np.sqrt(4.001) + 0.25
        expected = expected_bn / (1 + np.exp(-expected_bn))
        self.assertAlmostEqual(example['bn']['output'], expected_bn, places=5)
        self.assertAlmostEqual(example['activation_output'], expected, places=5)
        self.assertLess(example['validation_max_error'], 1e-5)
        edge = convolution_example(block, value, actual, position=(0, 0), channel=0)
        self.assertEqual(edge['sum'], 6)
        self.assertEqual(edge['input_origin'], [-1, -1])

    def test_nonfinite_features_are_rejected_instead_of_becoming_json_nan(self):
        from modern_pipeline.web.frontend_network import feature_map
        with self.assertRaises(ValueError):
            feature_map(torch.full((1, 3, 5, 5), float('nan')))


class FrontendEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with patch('modern_pipeline.web.jobs.validate_bundle', return_value={
                'pose_model': 'pose.pt', 'version': 'test'}):
            self.manager = JobManager(self.root)
        self.addCleanup(self.manager.close)
        directory = self.root / 'one'
        directory.mkdir()
        self.job = Job(directory, {'id': 'one', 'source_file': 'input.avi', 'status': 'completed',
                                   'metadata': {'frame_count': 3, 'fps': 10}, 'revision': 1})
        self.manager.jobs['one'] = self.job
        writer = cv2.VideoWriter(str(directory / 'input.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for color in (20, 80, 140):
            writer.write(np.full((24, 32, 3), color, dtype=np.uint8))
        writer.release()
        self.client = TestClient(create_app(self.manager))
        self.addCleanup(self.client.close)

    def test_frontend_endpoint_validates_frame_before_running_model(self):
        for frame in (-1, 3):
            response = self.client.get('/api/jobs/one/frontend-network', params={'frame': frame})
            self.assertEqual(response.status_code, 422, response.text)

    def test_source_frame_decode_selects_requested_frame(self):
        from modern_pipeline.web.frontend_network import read_source_frame
        value = read_source_frame(self.job.directory / 'input.avi', 1)
        self.assertEqual(value.shape, (24, 32, 3))
        self.assertAlmostEqual(float(value.mean()), 80, delta=2)
        with self.assertRaises(ValueError):
            read_source_frame(self.job.directory / 'input.avi', 9)

    def test_frontend_admission_shares_inference_lock(self):
        self.manager.active = 'another-job'
        self.addCleanup(setattr, self.manager, 'active', None)
        response = self.client.get('/api/jobs/one/frontend-network?frame=0')
        self.assertEqual(response.status_code, 409)
        self.manager.active = None
        self.manager.network_active = True
        response = self.client.get('/api/jobs/one/frontend-network?frame=0')
        self.assertEqual(response.status_code, 409)

    def test_failed_replay_releases_admission_and_bounded_cache_returns_copies(self):
        class Inspector:
            def replay(self, image):
                return {'input': {'mean': float(image.mean())}}
        self.manager.frontend_predictor = Inspector()
        first = self.manager.frontend_network('one', 1)
        self.assertAlmostEqual(first['input']['mean'], 80, delta=2)
        self.assertEqual(first['source']['kind'], 'recomputed_from_source_frame')
        first['input']['mean'] = -1
        self.assertGreater(self.manager.frontend_network('one', 1)['input']['mean'], 0)
        with patch.object(self.manager.frontend_predictor, 'replay', side_effect=RuntimeError('failed replay')):
            with self.assertRaises(RuntimeError):
                self.manager.frontend_network('one', 2)
        self.assertFalse(self.manager.network_active)
        self.job.state['metadata']['frame_count'] = 12
        with patch('modern_pipeline.web.frontend_network.read_source_frame', return_value=np.zeros((24, 32, 3), np.uint8)):
            for frame in range(8):
                self.manager.frontend_network('one', frame)
        self.assertLessEqual(len(self.manager.frontend_cache), 3)


@unittest.skipUnless((PROJECT_ROOT / 'weights/final/yolo26s-pose.pt').is_file(), 'Final YOLO weights absent')
class FrontendRealWeightTests(unittest.TestCase):
    def test_real_replay_has_native_multiscale_features_and_bounded_payload(self):
        from modern_pipeline.web.frontend_network import FrontendInspector
        inspector = FrontendInspector(PROJECT_ROOT / 'weights/final/yolo26s-pose.pt')
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, :, 0] = np.arange(160, dtype=np.uint8)
        frame[:, :, 2] = 180
        result = inspector.replay(frame)
        self.assertEqual(result['input']['tensor_shape'], [1, 3, 480, 640])
        self.assertEqual(result['layers'][0]['shape'], [1, 32, 240, 320])
        self.assertEqual({head['stride'] for head in result['heads']}, {8, 16, 32})
        self.assertEqual({head['branch'] for head in result['heads']}, {'box', 'class', 'keypoints'})
        self.assertLess(result['first_conv']['example']['validation_max_error'], 1e-4)
        self.assertFalse(result['postprocess']['end2end'])
        payload = json.dumps(result, allow_nan=False).encode()
        self.assertLess(len(payload), 3 * 1024 * 1024)
        image = cv2.imdecode(np.frombuffer(base64.b64decode(result['input']['image_data_url'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, frame.shape)
        self.assertGreater(float(image[:, :, 2].mean()), float(image[:, :, 0].mean()))
        self.assertEqual(len(inspector.model.model[0]._forward_hooks), 0)

    def test_replay_matches_normal_production_detection_head_and_results(self):
        from modern_pipeline.web.frontend_network import FrontendInspector
        from modern_pipeline.bootstrap import configure
        configure()
        from ultralytics import YOLO
        from ultralytics.utils import ASSETS
        frame = cv2.imread(str(ASSETS / 'bus.jpg'))
        if frame is None:
            self.skipTest('Ultralytics bundled bus fixture unavailable')
        weights = PROJECT_ROOT / 'weights/final/yolo26s-pose.pt'
        inspector = FrontendInspector(weights)
        replay = inspector.replay(frame)
        production = YOLO(str(weights), task='pose')
        actual = production.predict(frame, imgsz=640, conf=0.1, device='cpu', quantize=32,
                                    classes=[0], verbose=False, save=False, max_det=100)[0]
        self.assertEqual(replay['postprocess']['end2end'], production.predictor.model.end2end)
        self.assertGreater(len(actual.boxes), 0)
        self.assertEqual(len(replay['detections']), len(actual.boxes))
        np.testing.assert_allclose([row['bbox'] for row in replay['detections']], actual.boxes.xyxy.numpy(), atol=0.01, rtol=0)
        np.testing.assert_allclose([row['confidence'] for row in replay['detections']], actual.boxes.conf.numpy(), atol=1e-5, rtol=0)
        np.testing.assert_allclose([row['keypoints'] for row in replay['detections']], actual.keypoints.data.numpy(), atol=0.01, rtol=0)


if __name__ == '__main__':
    unittest.main()
