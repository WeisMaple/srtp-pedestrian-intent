"""Numerical and HTTP boundaries for on-demand neural computation inspection."""
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.web.app import create_app
from modern_pipeline.web.jobs import Job, JobManager
from st_gcn import Model, SkeletonLSTM


def fixture_predictor():
    torch.manual_seed(37)
    predictor = BehaviorPredictor.__new__(BehaviorPredictor)
    predictor.device = torch.device('cpu')
    predictor.normalization = 'legacy_pair'
    predictor.stats = {}
    predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
    predictor.lstm = SkeletonLSTM(hidden_size=8, num_layers=2).eval()
    return predictor


def poses(count=50):
    value = np.random.RandomState(9).uniform(0.2, 1, (count, 17, 3)).astype(np.float32)
    value[:, :, :2] *= 200
    return value


class NetworkNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def network_module(self):
        spec = importlib.util.find_spec('modern_pipeline.web.network')
        self.assertIsNotNone(spec, 'The real neural inspection module must exist')
        return importlib.import_module('modern_pipeline.web.network')

    def test_gate_reconstruction_matches_pytorch_recurrence(self):
        network = self.network_module()
        torch.manual_seed(17)
        recurrent = torch.nn.LSTM(5, 7, 1, batch_first=True).eval()
        sequence = torch.randn(1, 50, 5)
        with torch.inference_mode():
            expected, (hidden, cell) = recurrent(sequence)
        gates = network.inspect_lstm_gates(recurrent, sequence)
        np.testing.assert_allclose(gates['hidden_final'], hidden[0, 0].numpy(), atol=2e-7)
        np.testing.assert_allclose(gates['cell_final'], cell[0, 0].numpy(), atol=2e-7)
        np.testing.assert_allclose([s['h'] for s in gates['steps']], expected[0, :, 0].numpy(), atol=2e-7)
        for step in gates['steps']:
            self.assertIn('input_values', step)
            self.assertIn('previous_hidden', step)
            np.testing.assert_array_equal(step['input_values'], sequence[0, step['t'] - 1].numpy())
            previous = np.zeros(7) if step['t'] == 1 else expected[0, step['t'] - 2].numpy()
            np.testing.assert_allclose(step['previous_hidden'], previous, atol=2e-7)
            np.testing.assert_allclose(np.array(gates['input_weights']) @ step['input_values'],
                                       step['input_dot'], atol=2e-6)
            np.testing.assert_allclose(np.array(gates['recurrent_weights']) @ step['previous_hidden'],
                                       step['recurrent_dot'], atol=2e-6)
            z = np.array(step['input_dot']) + step['recurrent_dot'] + step['bias']
            np.testing.assert_allclose(z, step['preactivation'], atol=2e-6)
            self.assertAlmostEqual(step['i'], 1 / (1 + np.exp(-z[0])), places=6)
            self.assertAlmostEqual(step['g'], np.tanh(z[2]), places=6)
            self.assertAlmostEqual(step['c'], step['f'] * step['c_previous'] + step['i'] * step['g'], places=6)
            self.assertAlmostEqual(step['h'], step['o'] * np.tanh(step['c']), places=6)

    def test_spatial_temporal_residual_and_pool_terms_reconstruct_hooked_values(self):
        network = self.network_module()
        predictor = fixture_predictor()
        with torch.no_grad():
            for importance in predictor.stgcn.edge_importance:
                importance.mul_(0.5 + torch.rand_like(importance))
        captured, handles = {}, []

        def capture(name, tuple_output=False):
            def hook(module, inputs, output):
                value = output[0] if tuple_output else output
                captured[name] = value.detach().clone()
            return hook

        for index, block in enumerate(predictor.stgcn.st_gcn_networks):
            for name, module, tuple_output in (
                    ('projection', block.gcn.conv, False), ('spatial', block.gcn, True),
                    ('conv', block.tcn[2], False), ('temporal', block.tcn, False),
                    ('output', block, True)):
                handles.append(module.register_forward_hook(capture((index, name), tuple_output)))
            if isinstance(block.residual, torch.nn.Module):
                handles.append(block.residual.register_forward_hook(capture((index, 'residual'))))
        try:
            baseline = predictor.predict(poses()[None], poses() + 3)
        finally:
            for handle in handles:
                handle.remove()
        result = network.inspect_model(predictor, poses(), poses() + 3)
        np.testing.assert_array_equal(result['stgcn']['probabilities'], baseline['probabilities'][0])
        for index, block in enumerate(result['stgcn']['blocks']):
            with self.subTest(block=index + 1):
                self.assertTrue('arithmetic' in block, 'Every block needs an actual arithmetic example')
                arithmetic = block['arithmetic']
                spatial, temporal = arithmetic['spatial'], arithmetic['temporal']
                joint, channel = spatial['output_joint'], spatial['output_channel']
                self.assertGreater(np.count_nonzero(spatial['adjacency']), 1)
                products = np.array(spatial['projected_values']) * spatial['adjacency']
                np.testing.assert_allclose(products, spatial['products'], atol=1e-6)
                self.assertAlmostEqual(float(products.sum()), spatial['sum'], places=5)
                self.assertAlmostEqual(float(products.sum()), spatial['actual'], places=5)
                self.assertEqual(spatial['actual'], float(captured[index, 'spatial'][0, channel, spatial['timestep'], joint]))
                projection = spatial['projection']
                projected = np.dot(projection['input_values'], projection['weights']) + projection['bias']
                self.assertAlmostEqual(projected, projection['sum'], places=5)
                self.assertAlmostEqual(projected, projection['actual'], places=5)
                projection_channel = projection['partition'] * block['out_channels'] + channel
                self.assertEqual(projection['actual'], float(captured[index, 'projection'][
                    0, projection_channel, spatial['timestep'], projection['source_joint']]))
                products = np.array(temporal['input_values']) * temporal['weights']
                self.assertEqual(products.shape, (block['out_channels'], 9))
                np.testing.assert_allclose(products, temporal['products'], atol=1e-6)
                self.assertAlmostEqual(float(products.sum()) + temporal['bias'], temporal['sum'], places=5)
                self.assertAlmostEqual(float(products.sum()) + temporal['bias'], temporal['actual'], places=5)
                position = (0, channel, temporal['timestep'], joint)
                self.assertEqual(temporal['actual'], float(captured[index, 'conv'][position]))
                residual = arithmetic['residual']
                self.assertEqual(residual['temporal'], float(captured[index, 'temporal'][position]))
                expected_residual = (0. if index == 0 else float(captured[index, 'residual'][position])
                                     if block['residual'] == 'projection' else
                                     float(captured[index - 1, 'output'][position]))
                self.assertEqual(residual['residual'], expected_residual)
                self.assertAlmostEqual(residual['temporal'] + residual['residual'], residual['sum'], places=5)
                self.assertAlmostEqual(max(residual['sum'], 0), residual['actual'], places=5)
                self.assertEqual(residual['actual'], float(captured[index, 'output'][position]))
        self.assertTrue('pool_arithmetic' in result['stgcn'], 'Global pooling needs its complete reduction')
        pool = result['stgcn']['pool_arithmetic']
        np.testing.assert_array_equal(pool['values'], captured[9, 'output'][0, pool['channel']].numpy())
        self.assertEqual(pool['count'], 13 * 17)
        self.assertAlmostEqual(np.sum(pool['values']), pool['sum'], places=6)
        self.assertAlmostEqual(np.sum(pool['values']) / pool['count'], pool['actual'], places=6)
        self.assertEqual(pool['actual'], result['stgcn']['pool'][pool['channel']])
        self.assertLess(len(json.dumps(result, allow_nan=False).encode()), 5 * 1024 * 1024)

    def test_decoder_and_input_batch_norm_arithmetic_match_production_outputs(self):
        network = self.network_module()
        predictor = fixture_predictor()
        with torch.no_grad():
            predictor.stgcn.data_bn.running_mean[0] = 1.25
            predictor.stgcn.data_bn.running_var[0] = 2.5
            predictor.stgcn.data_bn.weight[0] = -0.75
            predictor.stgcn.data_bn.bias[0] = 0.4
        decoder_inputs, decoder_outputs = [], []
        normalized = {}

        def capture_decoder(module, inputs, output):
            decoder_inputs.append(inputs[0][0].clone())
            decoder_outputs.append(float(output[0, 0]))

        def capture_normalization(module, inputs, output):
            normalized['input'] = inputs[0].clone()
            normalized['output'] = output.clone()

        handles = [predictor.lstm.fc.register_forward_hook(capture_decoder),
                   predictor.stgcn.data_bn.register_forward_hook(capture_normalization)]
        try:
            predictor.predict(poses()[None], poses() + 3)
        finally:
            for handle in handles:
                handle.remove()
        result = network.inspect_model(predictor, poses(), poses() + 3)
        for index, round_info in enumerate(result['lstm']['rounds']):
            self.assertTrue('decoder_example' in round_info, 'The real forecast decoder needs an arithmetic example')
            decoder = round_info['decoder_example']
            np.testing.assert_array_equal(decoder['input_values'], decoder_inputs[index].numpy())
            np.testing.assert_array_equal(decoder['weights'], predictor.lstm.fc.weight[0].detach().numpy())
            actual = np.dot(decoder['input_values'], decoder['weights']) + decoder['bias']
            self.assertAlmostEqual(actual, decoder['sum'], places=6)
            self.assertAlmostEqual(actual, decoder_outputs[index], places=6)
            self.assertEqual(decoder['actual'], decoder_outputs[index])
            self.assertEqual(decoder['actual'], round_info['forecast']['values'][0][0])
        self.assertTrue('data_bn_example' in result['stgcn'], 'Input batch normalization must expose its trained affine math')
        bn = result['stgcn']['data_bn_example']
        position = (0, bn['feature'], bn['timestep'])
        self.assertEqual(bn['input_value'], float(normalized['input'][position]))
        expected = (bn['input_value'] - bn['running_mean']) / np.sqrt(bn['running_variance'] + bn['epsilon'])
        expected = expected * bn['weight'] + bn['bias']
        self.assertAlmostEqual(expected, bn['sum'], places=6)
        self.assertAlmostEqual(expected, float(normalized['output'][position]), places=6)
        self.assertEqual(bn['actual'], float(normalized['output'][position]))

    def test_real_hooks_preserve_prediction_and_affine_classifier_math(self):
        network = self.network_module()
        predictor = fixture_predictor()
        history, reference = poses(), poses() + 3
        captured = {}

        def capture_spatial(module, inputs, output):
            captured['spatial'] = output[0].clone()

        def capture_recurrent(module, inputs, output):
            captured.setdefault('hidden', output[1][0].clone())

        spatial_handle = predictor.stgcn.st_gcn_networks[0].gcn.register_forward_hook(capture_spatial)
        recurrent_handle = predictor.lstm.lstm.register_forward_hook(capture_recurrent)
        try:
            baseline = predictor.predict(history[None], reference)
        finally:
            spatial_handle.remove()
            recurrent_handle.remove()
        result = network.inspect_model(predictor, history, reference)
        np.testing.assert_allclose(result['stgcn']['probabilities'], baseline['probabilities'][0], atol=1e-7)
        np.testing.assert_allclose(result['future']['values'], baseline['future_poses'][0], atol=1e-6)
        logits = np.array(result['stgcn']['head_weights']) @ result['stgcn']['pool'] + result['stgcn']['head_bias']
        np.testing.assert_allclose(logits, result['stgcn']['logits'], atol=1e-6)
        self.assertEqual(len(result['lstm']['rounds']), 5)
        self.assertEqual(len(result['lstm']['rounds'][0]['layers']), 2)
        self.assertEqual(len(result['lstm']['gates']['steps']), 50)
        self.assertLess(result['lstm']['gates']['validation_max_error'], 2e-6)
        self.assertEqual(result['stgcn']['blocks'][4]['output_shape'], [1, 128, 25, 17])
        self.assertEqual(result['stgcn']['blocks'][7]['output_shape'], [1, 256, 13, 17])
        self.assertEqual(np.shape(result['stgcn']['blocks'][0]['spatial']['values']), (8, 17))
        np.testing.assert_array_equal(result['stgcn']['blocks'][0]['spatial']['values'],
                                      captured['spatial'][0, :8, 25].numpy())
        np.testing.assert_array_equal(result['lstm']['rounds'][0]['layers'][0]['h'],
                                      captured['hidden'][0, 0].numpy())
        np.testing.assert_array_equal(result['stgcn']['adjacency'], predictor.stgcn.A.numpy())
        self.assertNotIn([5, 6], result['stgcn']['edges'])
        self.assertNotIn([11, 12], result['stgcn']['edges'])
        self.assertLess(len(json.dumps(result, allow_nan=False).encode()), 5 * 1024 * 1024)
        for model in (predictor.lstm, predictor.stgcn):
            for module in model.modules():
                self.assertFalse(module._forward_hooks)
                self.assertFalse(module._forward_pre_hooks)

    def test_training_statistics_path_uses_saved_statistics_and_preserves_prediction(self):
        network = self.network_module()
        predictor = fixture_predictor()
        predictor.normalization = 'training_stats'
        predictor.stats = {'lstm_mean': torch.full((17, 3), 20.),
                           'lstm_std': torch.full((17, 3), 4.),
                           'stgcn_mean': torch.full((1, 3, 50, 17, 1), 10.),
                           'stgcn_std': torch.full((1, 3, 50, 17, 1), 2.)}
        history = poses()
        expected = predictor.predict(history[None])
        result = network.inspect_model(predictor, history)
        np.testing.assert_array_equal(result['lstm']['rounds'][0]['input']['values'],
                                      ((history - 20) / 4).reshape(50, 51))
        np.testing.assert_array_equal(result['stgcn']['input']['values'],
                                      ((expected['future_poses'][0] - 10) / 2).reshape(50, 51))
        np.testing.assert_array_equal(result['stgcn']['probabilities'], expected['probabilities'][0])

    def test_hooks_are_removed_after_prediction_failure(self):
        network = self.network_module()
        predictor = fixture_predictor()
        with patch.object(predictor, 'predict', side_effect=RuntimeError('intentional failure')):
            with self.assertRaisesRegex(RuntimeError, 'intentional failure'):
                network.inspect_model(predictor, poses(), poses())
        for model in (predictor.lstm, predictor.stgcn):
            for module in model.modules():
                self.assertFalse(module._forward_hooks)
                self.assertFalse(module._forward_pre_hooks)

    def test_reference_replays_first_eligible_history_after_gap(self):
        network = self.network_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'render.jsonl'
            rows = []
            for frame in range(65):
                accepted = frame not in (8, 9, 10)
                rows.append({'frame': frame, 'tracks': [
                    {'track_id': 4, 'pose_accepted': accepted, 'keypoints': (poses(1)[0] + frame).tolist()}]})
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            reference, source = network.load_reference(path)
            self.assertEqual(source['reference_frame'], 60)
            self.assertEqual(source['reference_track_id'], 4)
            np.testing.assert_array_equal(reference[0], poses(1)[0] + 11)
            np.testing.assert_array_equal(reference[-1], poses(1)[0] + 60)


class NetworkApiTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = {'normalization': 'legacy_pair', 'pose_model': 'pose.pt',
                  'stgcn': 'g.pt', 'lstm': 'l.pt', 'stats': 'stats.pt'}
        with patch('modern_pipeline.web.jobs.validate_bundle', return_value=config):
            self.manager = JobManager(Path(self.directory.name) / 'jobs')
        self.addCleanup(self.manager.close)
        self.predictor = fixture_predictor()
        self.predictor_patch = patch('modern_pipeline.behavior.BehaviorPredictor', return_value=self.predictor)
        self.predictor_patch.start()
        self.addCleanup(self.predictor_patch.stop)
        job_directory = self.manager.root / 'recorded'
        job_directory.mkdir()
        job = Job(job_directory, {'id': 'recorded', 'status': 'completed'})
        job.traces.append({'frame': 64, 'tracks': [
            {'track_id': 4, 'history': poses().round(3).tolist(), 'probabilities': [.1, .2, .7]}]})
        job.traces.append({'frame': 12, 'tracks': [{'track_id': 4, 'history': poses(13).tolist()}]})
        self.manager.jobs['recorded'] = job
        rows = [{'frame': frame, 'tracks': [{'track_id': 4, 'pose_accepted': True,
                                             'keypoints': poses()[frame].tolist()}]} for frame in range(50)]
        (job_directory / 'render.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.client = TestClient(create_app(self.manager))
        self.addCleanup(self.client.close)

    def test_endpoint_returns_actual_inspection_and_explicit_recomputation_source(self):
        response = self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['source']['kind'], 'recomputed_from_saved_history')
        self.assertEqual(data['source']['history_decimals'], 3)
        self.assertEqual(data['normalization']['stgcn']['reference_frame'], 49)
        self.assertEqual(data['source']['recorded_probabilities'], [.1, .2, .7])
        self.assertAlmostEqual(sum(data['stgcn']['probabilities']), 1, places=6)
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_endpoint_rejects_busy_missing_short_and_invalid_selection(self):
        cases = [('unknown/network?frame=64&track_id=4', 404),
                 ('recorded/network?frame=63&track_id=4', 404),
                 ('recorded/network?frame=64&track_id=7', 404),
                 ('recorded/network?frame=12&track_id=4', 422),
                 ('recorded/network?frame=-1&track_id=4', 422),
                 ('recorded/network?frame=64&track_id=bad', 422)]
        for path, status in cases:
            with self.subTest(path=path):
                self.assertEqual(self.client.get('/api/jobs/' + path).status_code, status)
        self.manager.active = 'recorded'
        self.addCleanup(setattr, self.manager, 'active', None)
        response = self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')
        self.assertEqual(response.status_code, 409)

    def test_cached_inspection_is_bounded_and_reused(self):
        response = self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')
        self.assertEqual(response.status_code, 200, response.text)
        with patch.object(self.predictor, 'predict', side_effect=AssertionError('cache missed')):
            again = self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(again.json()['stgcn']['logits'], response.json()['stgcn']['logits'])
        job = self.manager.jobs['recorded']
        for frame in range(100, 109):
            job.traces.append({'frame': frame, 'tracks': [dict(job.traces[0]['tracks'][0])]})
            self.assertEqual(self.client.get(f'/api/jobs/recorded/network?frame={frame}&track_id=4').status_code, 200)
        with patch.object(self.predictor, 'predict', side_effect=RuntimeError('old entry evicted')):
            response = self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')
        self.assertEqual(response.status_code, 503)

    def test_concurrent_network_request_and_video_start_are_rejected(self):
        started, release = threading.Event(), threading.Event()
        original = self.predictor.predict
        responses = []

        def delayed(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise RuntimeError('Test inspection timed out')
            return original(*args, **kwargs)

        with patch.object(self.predictor, 'predict', side_effect=delayed):
            thread = threading.Thread(target=lambda: responses.append(
                self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')))
            thread.start()
            try:
                self.assertTrue(started.wait(3))
                self.assertEqual(self.client.get('/api/jobs/recorded/network?frame=64&track_id=4').status_code, 409)
                ready = Job(self.manager.root / 'ready', {'id': 'ready', 'status': 'ready'})
                self.manager.jobs['ready'] = ready
                self.assertEqual(self.client.post('/api/jobs/ready/start').status_code, 409)
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(responses[0].status_code, 200, responses[0].text)

    def test_failed_reference_read_releases_worker_for_retry(self):
        path = self.manager.jobs['recorded'].directory / 'render.jsonl'
        original = path.read_text()
        path.write_text('')
        self.assertEqual(self.client.get('/api/jobs/recorded/network?frame=64&track_id=4').status_code, 422)
        path.write_text(original)
        response = self.client.get('/api/jobs/recorded/network?frame=64&track_id=4')
        self.assertEqual(response.status_code, 200, response.text)


if __name__ == '__main__':
    unittest.main()
