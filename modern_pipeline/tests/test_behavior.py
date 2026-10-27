"""Real-weight behavior parity tests; run with d2l Python -s -m unittest."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import pathlib
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modern_pipeline.behavior import BehaviorPredictor

ART = pathlib.Path(r'D:\SRTP_reproduction_20261027')
GNN = ART / 'models/stgcn/final_state_dict.pt'
LSTM = ART / 'models/lstm_wd0/final_state_dict.pt'


@unittest.skipUnless(GNN.is_file() and LSTM.is_file(), 'Reproduction weights unavailable')
class BehaviorParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.pred = BehaviorPredictor(GNN, LSTM, cls.device)
        cls.classify = BehaviorPredictor(GNN, device=cls.device)
        rng = np.random.default_rng(2026)
        cls.histories = rng.normal(size=(2, 50, 17, 3)).astype(np.float32)
        cls.histories[..., 0] = 700 + cls.histories[..., 0] * 20
        cls.histories[..., 1] = 400 + cls.histories[..., 1] * 30
        cls.histories[..., 2] = .6 + cls.histories[..., 2] * .1
        cls.reference = cls.histories[0].copy() + np.array([7, -11, .03], np.float32)

    def oracle(self, history, forecast):
        # Recovered predict.data_pro formulas, without importing TensorFlow.
        poses = history.copy()
        with torch.inference_mode(), torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            if forecast:
                mean = np.mean(poses, axis=0)
                std = np.std(poses, axis=0) + 1e-8
                series = (poses - mean) / std
                for _ in range(5):
                    x = torch.tensor(series[None], dtype=torch.float32,
                                     device=self.device).reshape(1, 50, 51)
                    new = self.pred.lstm(x)[0].cpu().numpy()
                    series = np.concatenate((series[10:], new), axis=0)
                poses = series * std + mean
            pair = np.stack((self.reference[None], poses[None]), axis=0)
            x = torch.from_numpy(pair.astype(np.float32)).to(self.device).permute(0, 4, 2, 3, 1)
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            std[std == 0] = 1e-6
            probabilities = self.pred.stgcn((x - mean) / std).softmax(1)[1].cpu().numpy()
        return poses, probabilities

    def test_legacy_forecast_and_classifier_parity(self):
        result = self.pred.predict(self.histories, self.reference)
        for i, history in enumerate(self.histories):
            poses, probabilities = self.oracle(history, True)
            np.testing.assert_allclose(result['future_poses'][i], poses, atol=2e-3, rtol=1e-5)
            np.testing.assert_allclose(result['probabilities'][i], probabilities, atol=2e-4, rtol=1e-4)
        self.assertEqual(result['horizon_frames'], 50)

    def test_classification_only_parity(self):
        result = self.classify.predict(self.histories, self.reference)
        for i, history in enumerate(self.histories):
            _, probabilities = self.oracle(history, False)
            np.testing.assert_allclose(result['probabilities'][i], probabilities, atol=2e-5, rtol=1e-4)
        np.testing.assert_array_equal(result['future_poses'], self.histories)
        self.assertEqual(result['horizon_frames'], 0)

    def test_batch_matches_single(self):
        batched = self.pred.predict(self.histories, self.reference)
        singles = [self.pred.predict(h[None], self.reference) for h in self.histories]
        np.testing.assert_allclose(batched['future_poses'], np.concatenate([s['future_poses'] for s in singles]), atol=2e-3, rtol=1e-5)
        np.testing.assert_allclose(batched['probabilities'], np.concatenate([s['probabilities'] for s in singles]), atol=2e-4, rtol=1e-4)
        np.testing.assert_array_equal(batched['class_ids'], np.concatenate([s['class_ids'] for s in singles]))

    def test_training_stats_shapes_and_zero_std(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'stats.pt'
            stats = {'stgcn_mean': torch.zeros(1, 3, 50, 17, 1),
                     'stgcn_std': torch.ones(1, 3, 50, 17, 1),
                     'lstm_mean': torch.zeros(17, 3), 'lstm_std': torch.ones(17, 3)}
            torch.save(stats, path)
            model = BehaviorPredictor(GNN, LSTM, self.device, 'training_stats', path)
            result = model.predict(self.histories)
            self.assertEqual(result['probabilities'].shape, (2, 3))
            self.assertTrue(np.isfinite(result['future_poses']).all())
            stats['stgcn_std'].zero_()
            torch.save(stats, path)
            model = BehaviorPredictor(GNN, device=self.device, normalization='training_stats', stats_path=path)
            self.assertTrue(torch.all(model.stats['stgcn_std'] >= model.EPSILON))
            stats['stgcn_mean'] = torch.zeros(3)
            torch.save(stats, path)
            with self.assertRaisesRegex(ValueError, 'shape'):
                BehaviorPredictor(GNN, device=self.device, normalization='training_stats', stats_path=path)

    def test_validation_empty_and_no_tensorflow(self):
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.pred.predict(self.histories)
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.pred.predict(self.histories[:, :-1], self.reference)
        bad = self.histories.copy(); bad[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.pred.predict(bad, self.reference)
        empty = self.pred.predict(self.histories[:0], self.reference)
        self.assertEqual(empty['probabilities'].shape, (0, 3))
        self.assertFalse(any(k == 'tensorflow' or k.startswith('tensorflow.') for k in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)


