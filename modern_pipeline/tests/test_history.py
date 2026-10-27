"""Temporal integrity checks for tracked pose histories (no models required)."""
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modern_pipeline.history import HistoryStore


def pose(value=1.0, dtype=np.float32):
    result = np.full((17, 3), value, dtype=dtype)
    result[:, 2] = 0.9
    return result


class HistoryStoreTests(unittest.TestCase):
    def assertStateUnchanged(self, state, expected):
        self.assertEqual(state.last_seen, expected['last_seen'])
        self.assertEqual(state.last_prediction, expected['last_prediction'])
        self.assertEqual(state.prediction, expected['prediction'])
        np.testing.assert_array_equal(state.as_array(), expected['poses'])

    @staticmethod
    def snapshot(state):
        return {
            'last_seen': state.last_seen,
            'last_prediction': state.last_prediction,
            'prediction': state.prediction,
            'poses': state.as_array().copy(),
        }

    def test_initial_state_and_default_fifty_observation_warmup(self):
        store = HistoryStore()
        state = store.update(7, 0, pose())
        self.assertIs(store.states[7], state)
        self.assertEqual(state.last_seen, 0)
        self.assertIsNone(state.prediction)
        self.assertEqual(state.last_prediction, -1)
        self.assertFalse(state.ready)
        self.assertEqual(state.poses.maxlen, 50)
        for frame in range(1, 49):
            store.update(7, frame, pose(frame))
        self.assertEqual(len(state.poses), 49)
        self.assertFalse(state.ready)
        store.update(7, 49, pose(49))
        self.assertEqual(len(state.poses), 50)
        self.assertTrue(state.ready)
        self.assertEqual(state.as_array().shape, (50, 17, 3))

    def test_bounded_history_keeps_newest_observations(self):
        store = HistoryStore(window=3)
        for frame in range(5):
            state = store.update(2, frame, pose(frame))
        self.assertTrue(state.ready)
        self.assertEqual(len(state.poses), 3)
        np.testing.assert_array_equal(state.as_array()[:, 0, 0], [2, 3, 4])
        self.assertEqual(state.last_seen, 4)

    def test_pose_is_copied_and_stacked_as_float32(self):
        store = HistoryStore(window=2)
        incoming = pose(7.0, dtype=np.float64)
        state = store.update(1, 1, incoming)
        incoming[:] = 999.0
        self.assertEqual(state.as_array().dtype, np.float32)
        self.assertEqual(float(state.as_array()[0, 0, 0]), 7.0)
        self.assertAlmostEqual(float(state.as_array()[0, 0, 2]), 0.9, places=6)

    def test_gap_at_threshold_preserves_history_and_prediction(self):
        store = HistoryStore(window=3, max_gap=2)
        state = store.update(3, 10, pose(10))
        state.prediction = {'class_id': 2}
        state.last_prediction = 10
        state = store.update(3, 12, pose(12))
        self.assertEqual(len(state.poses), 2)
        self.assertEqual(state.prediction, {'class_id': 2})
        self.assertEqual(state.last_prediction, 10)
        np.testing.assert_array_equal(state.as_array()[:, 0, 0], [10, 12])

    def test_large_gap_resets_history_and_cached_prediction(self):
        store = HistoryStore(window=3, max_gap=2)
        state = store.update(3, 10, pose(10))
        store.update(3, 11, pose(11))
        state.prediction = {'class_id': 1}
        state.last_prediction = 11
        state = store.update(3, 14, pose(14))
        self.assertEqual(len(state.poses), 1)
        self.assertFalse(state.ready)
        self.assertIsNone(state.prediction)
        self.assertEqual(state.last_prediction, -1)
        self.assertEqual(state.last_seen, 14)
        self.assertEqual(float(state.as_array()[0, 0, 0]), 14.0)

    def test_repeated_or_reversed_frame_is_rejected_atomically(self):
        store = HistoryStore()
        state = store.update(9, 10, pose(10))
        state.prediction = {'class_id': 2}
        state.last_prediction = 10
        expected = self.snapshot(state)
        for bad_frame in (10, 9, 0):
            with self.subTest(frame=bad_frame):
                with self.assertRaises(ValueError):
                    store.update(9, bad_frame, pose(99))
                self.assertStateUnchanged(store.states[9], expected)

    def test_invalid_poses_do_not_mutate_existing_track(self):
        store = HistoryStore()
        state = store.update(9, 10, pose(10))
        state.prediction = {'class_id': 0}
        state.last_prediction = 10
        expected = self.snapshot(state)
        bad_poses = [np.zeros((17, 2)), np.zeros((1, 17, 3)), np.zeros((16, 3))]
        for value in (np.nan, np.inf, -np.inf):
            bad = pose()
            bad[8, 0] = value
            bad_poses.append(bad)
        for index, bad in enumerate(bad_poses):
            with self.subTest(case=index):
                # The gap would otherwise reset history, so validate before mutation.
                with self.assertRaises(ValueError):
                    store.update(9, 100, bad)
                self.assertStateUnchanged(store.states[9], expected)

    def test_invalid_pose_does_not_create_track(self):
        store = HistoryStore()
        with self.assertRaises(ValueError):
            store.update(123, 1, np.full((17, 3), np.nan))
        self.assertNotIn(123, store.states)

    def test_tracks_keep_independent_histories(self):
        store = HistoryStore(window=3)
        first = store.update(1, 10, pose(10))
        second = store.update(2, 5, pose(50))
        store.update(1, 11, pose(11))
        store.update(2, 6, pose(60))
        self.assertEqual(first.last_seen, 11)
        self.assertEqual(second.last_seen, 6)
        np.testing.assert_array_equal(first.as_array()[:, 0, 0], [10, 11])
        np.testing.assert_array_equal(second.as_array()[:, 0, 0], [50, 60])

    def test_expiry_strict_boundary_uses_last_observation(self):
        store = HistoryStore(ttl=30)
        old = store.update(1, 10, pose())
        recent = store.update(2, 20, pose())
        old.last_prediction = 1000  # Prediction timestamp must not prolong lifetime.
        recent.last_prediction = -1
        store.expire(40)
        self.assertIn(1, store.states)  # 40 - 10 == 30: still retained.
        self.assertIn(2, store.states)
        store.expire(41)
        self.assertNotIn(1, store.states)
        self.assertIn(2, store.states)
        store.expire(50)
        self.assertIn(2, store.states)
        store.expire(51)
        self.assertNotIn(2, store.states)
        store.expire(100)  # Empty expiry is harmless.
        self.assertEqual(store.states, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
