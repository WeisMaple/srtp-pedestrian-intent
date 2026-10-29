"""Temporal and geometric contracts for the live-camera prototype."""
import unittest
from types import SimpleNamespace

import numpy as np

from modern_pipeline.camera_demo import DemoSession, RoadZone, TimedPoseWindow


def person(track_id=7, x=0.2, y=0.6, confidence=0.9):
    pose = np.tile([x * 1000, y * 1000, confidence], (17, 1)).astype(np.float32)
    return SimpleNamespace(track_id=track_id, pose=pose,
                           bbox=np.array([x * 1000 - 20, 100, x * 1000 + 20, y * 1000]),
                           confidence=0.9)


def road():
    return RoadZone([(0.4, 0.2), (0.6, 0.2), (0.6, 0.9), (0.4, 0.9)])


class RoadTests(unittest.TestCase):
    def test_crossing_detected_even_if_forecast_endpoint_passes_the_road(self):
        self.assertAlmostEqual(road().entry_time((0.2, 0.5), (1.0, 0.0), 1.0), 0.2)

    def test_parallel_away_and_stationary_outside_do_not_enter(self):
        for velocity in ((0, 0.5), (-0.2, 0), (0, 0)):
            self.assertIsNone(road().entry_time((0.2, 0.5), velocity, 1.0))

    def test_inside_and_boundary_are_already_in_road(self):
        self.assertEqual(road().entry_time((0.5, 0.5), (0, 0), 1), 0)
        self.assertEqual(road().entry_time((0.4, 0.5), (-1, 0), 1), 0)

    def test_clockwise_polygon_has_same_geometry(self):
        reverse = RoadZone(list(reversed(road().points)))
        self.assertAlmostEqual(reverse.entry_time((0.2, 0.5), (0.4, 0), 1), 0.5)

    def test_degenerate_crossed_nonfinite_and_offscreen_polygons_rejected(self):
        for points in ([(0, 0)] * 4,
                       [(0, 0), (1, 1), (0, 1), (1, 0)],
                       [(0, 0), (1, 0), (0.2, 0.2), (0, 1)],
                       [(0, 0), (1, 0), (1, np.nan), (0, 1)],
                       [(0, 0), (2, 0), (1, 1), (0, 1)]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                RoadZone(points)


class TimedHistoryTests(unittest.TestCase):
    def test_fifty_quick_samples_do_not_pretend_to_cover_training_window(self):
        history = TimedPoseWindow(fps=30, max_gap=0.2)
        for i in range(50):
            history.update(i / 100, person().pose)
        self.assertIsNone(history.resample())

    def test_irregular_delivery_resamples_fifty_poses_over_real_time(self):
        history = TimedPoseWindow(fps=30, max_gap=0.2)
        for t in np.arange(0, 1.81, 0.1):
            history.update(float(t), person(x=0.1 + 0.1 * t).pose)
        series = history.resample()
        self.assertEqual(series.shape, (50, 17, 3))
        self.assertAlmostEqual(float(series[0, 0, 0]), 116.66667, places=3)
        self.assertAlmostEqual(float(series[-1, 0, 0]), 280, places=3)

    def test_gap_restarts_history_and_duplicate_time_is_rejected(self):
        history = TimedPoseWindow(fps=30, max_gap=0.2)
        for i in range(60):
            history.update(i / 30, person().pose)
        self.assertIsNotNone(history.resample())
        self.assertTrue(history.update(3, person().pose))
        self.assertIsNone(history.resample())
        with self.assertRaises(ValueError):
            history.update(3, person().pose)


class DemoSessionTests(unittest.TestCase):
    def test_approach_uses_seconds_and_parallel_motion_does_not_warn(self):
        session = DemoSession(road(), max_gap=0.3)
        for t in (0, 0.25, 0.5):
            rows = session.update([person(x=0.1 + 0.2 * t),
                                   person(track_id=8, x=0.2, y=0.4 + 0.2 * t)], t, (1000, 1000))
        self.assertEqual(rows[0]['baseline']['state'], 'entering')
        self.assertAlmostEqual(rows[0]['baseline']['eta_seconds'], 1.0, places=5)
        self.assertEqual(rows[1]['baseline']['state'], 'outside')

    def test_missing_feet_and_gap_clear_trajectory(self):
        session = DemoSession(road(), max_gap=0.3)
        for t in (0, 0.25, 0.5):
            session.update([person(x=0.1 + 0.2 * t)], t, (1000, 1000))
        hidden = person()
        hidden.pose[15:17, 2] = 0
        row = session.update([hidden], 0.6, (1000, 1000))[0]
        self.assertEqual(row['baseline']['state'], 'feet_missing')
        self.assertEqual(session.update([person()], 0.7, (1000, 1000))[0]['baseline']['state'], 'warming')
        self.assertEqual(session.update([person()], 1.2, (1000, 1000))[0]['baseline']['state'], 'warming')

    def test_people_do_not_share_history_and_absent_tracks_expire(self):
        session = DemoSession(road(), max_gap=0.3)
        for t in (0, 0.25, 0.5):
            session.update([person()], t, (1000, 1000))
        row = session.update([person(track_id=8)], 0.6, (1000, 1000))[0]
        self.assertEqual(row['baseline']['state'], 'warming')
        session.update([], 4, (1000, 1000))
        self.assertEqual(len(session.tracks), 0)

    def test_optional_model_cache_expires_and_invalid_pose_clears_it(self):
        # Replace only expensive inference; exercise the real history and scheduler.
        class Predictor:
            normalization = 'training_stats'

            def predict(self, histories, reference):
                return {'class_ids': np.full(len(histories), 2),
                        'probabilities': np.tile([0.1, 0.2, 0.7], (len(histories), 1))}

        session = DemoSession(road(), predictor=Predictor(), model_interval=1.0,
                              prediction_ttl=0.3)
        for i in range(51):
            row = session.update([person()], i / 30, (1000, 1000))[0]
        self.assertEqual(row['model']['state'], 'valid')
        self.assertEqual(row['model']['class_id'], 2)
        for i in range(51, 64):
            row = session.update([person()], i / 30, (1000, 1000))[0]
        self.assertEqual(row['model']['state'], 'expired')
        self.assertIsNone(row['model']['class_id'])
        row = session.update([person(confidence=0)], 64 / 30, (1000, 1000))[0]
        self.assertEqual(row['model']['state'], 'pose_missing')
        self.assertIsNone(row['model']['class_id'])


if __name__ == '__main__':
    unittest.main()
