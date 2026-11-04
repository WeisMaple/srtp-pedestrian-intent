"""The on-demand record endpoint validates inputs and serializes results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import torch
from torch import nn

from modern_pipeline.bootstrap import PROJECT_ROOT
from modern_pipeline.web.app import create_app
from modern_pipeline.web.frontend_network import FrontendInspector
from modern_pipeline.web.jobs import Job, JobBusy, JobManager


class Manager:
    def __init__(self):
        self.calls = []

    def yolo_conv(self, job_id, frame, module):
        self.calls.append((job_id, frame, module))
        return {'module': module, 'frame': frame, 'operation': 'dense_conv'}

    def snapshot(self, job_id):
        return {'id': job_id, 'status': 'completed'}


class ErrorManager:
    def snapshot(self, job_id):
        return {'id': job_id, 'status': 'completed'}

    def yolo_conv(self, job_id, frame, module):
        error = {1: JobBusy('busy'), 2: KeyError('missing'), 3: ValueError('bad')}.get(module)
        raise error or RuntimeError('unavailable')


class YoloConvEndpointTests(unittest.TestCase):
    def test_route_passes_query_and_returns_dense_row(self):
        manager = Manager()
        client = TestClient(create_app(manager=manager))
        response = client.get('/api/jobs/abc/yolo-conv?frame=3&module=7')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['module'], 7)
        self.assertEqual(manager.calls, [('abc', 3, 7)])

    def test_route_rejects_negative_module(self):
        client = TestClient(create_app(manager=Manager()))
        response = client.get('/api/jobs/abc/yolo-conv?frame=3&module=-1')
        self.assertEqual(response.status_code, 422)

    def test_route_maps_manager_errors(self):
        client = TestClient(create_app(manager=ErrorManager()))
        for module, status in ((1, 409), (2, 404), (3, 422), (4, 503)):
            response = client.get('/api/jobs/abc/yolo-conv?frame=3&module=' + str(module))
            self.assertEqual(response.status_code, status, response.text)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.SiLU()

    def forward(self, value):
        return self.act(self.bn(self.conv(value)))


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([Block(), nn.Identity()])

    def forward(self, value):
        for module in self.model:
            value = module(value)
        return value


class Predictor:
    def __init__(self):
        self.batch = None

    def preprocess(self, images):
        return torch.ones(1, 3, 16, 16)


def fake_inspector():
    inspector = object.__new__(FrontendInspector)
    inspector.model = Network().eval()
    inspector.predictor = Predictor()
    return inspector


class ModuleConvTests(unittest.TestCase):
    def test_module_conv_returns_unfused_row_and_cleans_up(self):
        inspector = fake_inspector()
        block = inspector.model.model[0]
        with torch.no_grad():
            block.bn.running_mean.fill_(0.5)
        row = inspector.module_conv(np.zeros((24, 32, 3), np.uint8), 0)
        self.assertEqual(row['stage'], 'yolo')
        self.assertEqual(row['module'], 0)
        self.assertEqual(row['operation'], 'dense_conv')
        self.assertFalse(row['fused'])
        self.assertEqual(row['activation'], 'batch_norm_silu')
        self.assertEqual(row['term_shape'], [3, 3, 3])
        self.assertEqual(len(row['inputs']), 27)
        self.assertFalse(row['metadata']['fused'])
        self.assertIsNone(inspector.predictor.batch)
        self.assertEqual(len(block.conv._forward_hooks), 0)
        json.dumps(row, allow_nan=False)

    def test_module_conv_rejects_module_without_conv(self):
        with self.assertRaises(ValueError):
            fake_inspector().module_conv(np.zeros((24, 32, 3), np.uint8), 1)

    def test_module_conv_rejects_out_of_range_module(self):
        for index in (-1, 2):
            with self.assertRaises(ValueError):
                fake_inspector().module_conv(np.zeros((24, 32, 3), np.uint8), index)

    def test_module_conv_rejects_non_uint8_bgr_image(self):
        inspector = fake_inspector()
        with self.assertRaises(ValueError):
            inspector.module_conv(np.zeros((24, 32), np.uint8), 0)
        self.assertIsNone(inspector.predictor.batch)

    def test_module_conv_cleans_up_after_forward_failure(self):
        inspector = fake_inspector()
        block = inspector.model.model[0]
        with patch.object(inspector.model, 'forward', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                inspector.module_conv(np.zeros((24, 32, 3), np.uint8), 0)
        self.assertIsNone(inspector.predictor.batch)
        self.assertEqual(len(block.conv._forward_hooks), 0)


class Inspector:
    def __init__(self, calls):
        self.calls = calls

    def module_conv(self, image, module):
        self.calls.append((module, float(image.mean())))
        return {'module': module, 'operation': 'dense_conv', 'payload': [1.0] * 10}


class YoloConvManagerTests(unittest.TestCase):
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
        job = Job(directory, {'id': 'one', 'source_file': 'input.avi', 'status': 'completed',
                              'metadata': {'frame_count': 3, 'fps': 10}, 'revision': 1})
        self.manager.jobs['one'] = job
        self.calls = []
        self.manager.frontend_predictor = Inspector(self.calls)

    def test_yolo_conv_rejects_out_of_range_frame_and_module(self):
        with self.assertRaises(ValueError):
            self.manager.yolo_conv('one', 3, 0)
        with self.assertRaises(ValueError):
            self.manager.yolo_conv('one', 0, 24)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.manager.network_active)

    def test_yolo_conv_shares_inference_admission(self):
        self.manager.active = 'another-job'
        self.addCleanup(setattr, self.manager, 'active', None)
        with self.assertRaises(JobBusy):
            self.manager.yolo_conv('one', 0, 0)
        self.manager.active = None
        self.manager.network_active = True
        self.addCleanup(setattr, self.manager, 'network_active', False)
        with self.assertRaises(JobBusy):
            self.manager.yolo_conv('one', 0, 0)
        self.assertEqual(self.calls, [])

    def test_cache_is_bounded_copies_and_admission_is_released(self):
        self.manager.jobs['one'].state['metadata']['frame_count'] = 20
        with patch('modern_pipeline.web.frontend_network.read_source_frame',
                   return_value=np.zeros((24, 32, 3), np.uint8)):
            first = self.manager.yolo_conv('one', 1, 0)
            self.assertEqual(first['source']['kind'], 'recomputed_module_conv')
            self.assertEqual(first['source']['frame'], 1)
            self.assertEqual(first['source']['module'], 0)
            first['payload'][0] = -1
            self.assertEqual(self.manager.yolo_conv('one', 1, 0)['payload'][0], 1.0)
            self.assertEqual(len(self.calls), 1)
            self.assertFalse(self.manager.network_active)
            for frame in range(20):
                for module in range(3):
                    self.manager.yolo_conv('one', frame, module)
        self.assertLessEqual(len(self.manager.frontend_cache), 8)
        self.assertNotIn(('one', 1, 0), self.manager.frontend_cache)

    def test_oversized_response_is_rejected_and_admission_is_released(self):
        class Oversized(Inspector):
            def module_conv(self, image, module):
                return {'module': module, 'payload': 'x' * (2 * 1024 * 1024)}
        self.manager.frontend_predictor = Oversized(self.calls)
        with patch('modern_pipeline.web.frontend_network.read_source_frame',
                   return_value=np.zeros((24, 32, 3), np.uint8)):
            with self.assertRaises(RuntimeError):
                self.manager.yolo_conv('one', 0, 0)
        self.assertFalse(self.manager.network_active)
        self.assertEqual(len(self.manager.frontend_cache), 0)

    def test_failed_module_conv_releases_admission_and_is_not_cached(self):
        with patch.object(self.manager.frontend_predictor, 'module_conv', side_effect=RuntimeError('boom')):
            with patch('modern_pipeline.web.frontend_network.read_source_frame',
                       return_value=np.zeros((24, 32, 3), np.uint8)):
                with self.assertRaises(RuntimeError):
                    self.manager.yolo_conv('one', 0, 0)
        self.assertFalse(self.manager.network_active)
        self.assertEqual(len(self.manager.frontend_cache), 0)


@unittest.skipUnless((PROJECT_ROOT / 'weights/final/yolo26s-pose.pt').is_file(), 'Final YOLO weights absent')
class ModuleConvRealWeightTests(unittest.TestCase):
    def test_stem_and_pose_head_produce_unfused_rows(self):
        inspector = FrontendInspector(PROJECT_ROOT / 'weights/final/yolo26s-pose.pt')
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, :, 0] = np.arange(160, dtype=np.uint8)
        for index in (0, 23):
            row = inspector.module_conv(frame, index)
            self.assertEqual(row['module'], index)
            self.assertEqual(row['operation'], 'dense_conv')
            self.assertFalse(row['fused'])
            self.assertEqual(len(row['inputs']), row['metadata']['term_count'])
            self.assertLessEqual(row['metadata']['term_count'], 4096)
            json.dumps(row, allow_nan=False)
        self.assertIsNone(inspector.predictor.batch)
        self.assertEqual(len(inspector.model.model[0]._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
