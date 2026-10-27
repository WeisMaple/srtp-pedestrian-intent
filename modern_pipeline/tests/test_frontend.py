"""Model-free checks that tracking results preserve person/keypoint alignment."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from modern_pipeline.frontend import YOLOPoseTracker


class FakeTensor:
    def __init__(self, data):
        self.array = np.asarray(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def tracked_result(ids=(42, 7), scores=(0.8, 0.6)):
    boxes = np.array([[10, 20, 100, 200], [300, 40, 400, 250]], dtype=np.float32)
    poses = np.zeros((2, 17, 3), dtype=np.float32)
    poses[0, :, :2] = [55, 88]
    poses[1, :, :2] = [330, 140]
    poses[:, :, 2] = 0.9
    return SimpleNamespace(
        boxes=SimpleNamespace(is_track=True, xyxy=FakeTensor(boxes),
                              id=FakeTensor(ids), conf=FakeTensor(scores)),
        keypoints=SimpleNamespace(data=FakeTensor(poses)),
    )


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.weights = Path(self.directory.name) / 'fake_pose.pt'
        self.weights.write_bytes(b'test fixture; never loaded')
        self.model = Mock()
        self.model.task = 'pose'
        self.yolo = Mock(return_value=self.model)
        self.module_patch = patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=self.yolo)})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def make_frontend(self):
        return YOLOPoseTracker(self.weights, 'cpu', imgsz=640, conf=0.1, tracker='bytetrack.yaml')

    def test_id_bbox_and_pose_keep_result_order(self):
        result = tracked_result()
        self.model.track.return_value = [result]
        people = self.make_frontend()(self.frame)
        self.assertEqual([person.track_id for person in people], [42, 7])
        np.testing.assert_array_equal(people[0].bbox, [10, 20, 100, 200])
        np.testing.assert_array_equal(people[1].bbox, [300, 40, 400, 250])
        np.testing.assert_array_equal(people[0].pose[0, :2], [55, 88])
        np.testing.assert_array_equal(people[1].pose[0, :2], [330, 140])
        self.assertAlmostEqual(people[0].confidence, 0.8)
        self.assertEqual(people[0].pose.shape, (17, 3))
        self.assertEqual(people[0].pose.dtype, np.float32)
        self.assertIs(self.model.track.call_args.args[0], self.frame)
        self.assertTrue(self.model.track.call_args.kwargs['persist'])
        self.assertEqual(self.model.track.call_args.kwargs['tracker'], 'bytetrack.yaml')
        # Full-frame coordinates must remain unchanged: no bbox offset or xy normalization.
        self.assertEqual(float(people[0].pose[0, 0]), 55)

    def test_pose_copy_is_independent_of_sdk_result(self):
        result = tracked_result()
        self.model.track.return_value = [result]
        people = self.make_frontend()(self.frame)
        result.keypoints.data.array[0, 0, 0] = 9999
        self.assertEqual(float(people[0].pose[0, 0]), 55)

    def test_empty_detections_still_advance_tracker_each_frame(self):
        no_boxes = SimpleNamespace(boxes=None, keypoints=None)
        no_track = SimpleNamespace(boxes=SimpleNamespace(is_track=False), keypoints=object())
        result = tracked_result()
        self.model.track.side_effect = [[result], [no_boxes], [no_track], [result]]
        frontend = self.make_frontend()
        self.assertEqual(len(frontend(self.frame)), 2)
        self.assertEqual(frontend(self.frame), [])
        self.assertEqual(frontend(self.frame), [])
        self.assertEqual([p.track_id for p in frontend(self.frame)], [42, 7])
        self.assertEqual(self.model.track.call_count, 4)
        self.assertTrue(all(call.kwargs['persist'] for call in self.model.track.call_args_list))

    def test_missing_pose_is_empty_but_advances_tracker(self):
        result = tracked_result()
        result.keypoints = None
        self.model.track.return_value = [result]
        self.assertEqual(self.make_frontend()(self.frame), [])
        self.assertEqual(self.model.track.call_count, 1)

    def test_mismatched_person_and_pose_counts_raise(self):
        result = tracked_result()
        result.keypoints.data = FakeTensor(np.zeros((1, 17, 3)))
        self.model.track.return_value = [result]
        with self.assertRaisesRegex(RuntimeError, 'aligned'):
            self.make_frontend()(self.frame)

    def test_wrong_keypoint_layout_raises(self):
        result = tracked_result()
        result.keypoints.data = FakeTensor(np.zeros((2, 18, 3)))
        self.model.track.return_value = [result]
        with self.assertRaisesRegex(RuntimeError, 'aligned'):
            self.make_frontend()(self.frame)

    def test_nonfinite_pose_or_box_filters_only_affected_person(self):
        for bad_field in ('pose', 'box'):
            with self.subTest(field=bad_field):
                result = tracked_result()
                if bad_field == 'pose':
                    result.keypoints.data.array[0, 1, 0] = np.nan
                else:
                    result.boxes.xyxy.array[0, 0] = np.inf
                self.model.track.return_value = [result]
                self.assertEqual([p.track_id for p in self.make_frontend()(self.frame)], [7])

    def test_nonfinite_confidence_filters_only_affected_person(self):
        for bad_score in (np.nan, np.inf, -np.inf):
            with self.subTest(score=bad_score):
                result = tracked_result(scores=(bad_score, 0.6))
                self.model.track.return_value = [result]
                self.assertEqual([p.track_id for p in self.make_frontend()(self.frame)], [7])

    def test_mismatched_score_count_raises(self):
        result = tracked_result(scores=(0.8,))
        self.model.track.return_value = [result]
        with self.assertRaisesRegex(RuntimeError, 'aligned'):
            self.make_frontend()(self.frame)

    def test_missing_weights_fail_before_loading_model(self):
        with self.assertRaises(FileNotFoundError):
            YOLOPoseTracker(Path(self.directory.name) / 'absent.pt', 'cpu')
        self.yolo.assert_not_called()

    def test_non_pose_model_is_rejected(self):
        self.model.task = 'detect'
        with self.assertRaises(ValueError):
            self.make_frontend()


if __name__ == '__main__':
    unittest.main(verbosity=2)
