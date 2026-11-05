"""In-memory camera inference; no recording or computation telemetry."""
import time

import cv2
import numpy as np

from modern_pipeline.history import HistoryStore
from modern_pipeline.pose_filter import PoseFilter


class CameraSession:
    def __init__(self, frontend, predictor, pose_filter=False):
        self.frontend = frontend
        self.predictor = predictor
        self.use_filter = pose_filter
        self.frame = 0
        self.dimensions = None
        self.last_timestamp = None
        self._reset_history()

    def _reset_history(self):
        self.store = HistoryStore()
        self.pose_filter = PoseFilter() if self.use_filter else None
        self.reference = None

    @classmethod
    def from_models(cls, model_dir, config):
        from modern_pipeline.bootstrap import configure
        configure()
        import torch
        from modern_pipeline.behavior import BehaviorPredictor
        from modern_pipeline.frontend import YOLOPoseTracker
        torch.set_num_threads(2)
        frontend = YOLOPoseTracker(model_dir / config['pose_model'], 'cpu')
        predictor = BehaviorPredictor(model_dir / config['stgcn'], model_dir / config['lstm'],
                                      'cpu', config['normalization'], model_dir / config['stats'])
        return cls(frontend, predictor, config.get('pose_filter', False))

    def process(self, jpeg, timestamp=None):
        started = time.perf_counter()
        if not jpeg or len(jpeg) > 2 * 1024 * 1024:
            raise ValueError('摄像头帧为空或超过 2 MB。')
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('无法解码摄像头画面。')
        height, width = image.shape[:2]
        if max(width, height) > 1920 or width * height > 1920 * 1080:
            raise ValueError('摄像头帧尺寸超过 1920 × 1080 限制。')
        timestamp = time.monotonic() if timestamp is None else timestamp
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError('摄像头帧时间必须递增。')
        if ((self.dimensions is not None and self.dimensions != (width, height)) or
                (self.last_timestamp is not None and timestamp - self.last_timestamp > 2.0)):
            self._reset_history()
        self.last_timestamp = timestamp
        self.dimensions = (width, height)
        people = self.frontend(image)
        ready, eligible, current = [], [], []
        for person in people:
            pose = person.pose
            if self.pose_filter is not None:
                pose = self.pose_filter.update(person.track_id, self.frame, pose)
            valid = (np.isfinite(pose).all() and pose.shape == (17, 3) and
                     int(((pose[:, 2] >= 0.2) & (pose[:, :2] > 0).all(axis=1)).sum()) >= 5)
            state = None
            if valid:
                state = self.store.update(person.track_id, self.frame, pose)
                if state.ready:
                    first_reference = self.reference is None
                    if first_reference:
                        self.reference = state.as_array().copy()
                    if not (first_reference and self.predictor.normalization == 'legacy_pair'):
                        ready.append(state.as_array())
                        eligible.append(state)
            else:
                # A rejected pose is a discontinuity, never a cached intent.
                self.store.states.pop(person.track_id, None)
            current.append((person, pose, state, bool(valid)))
        if ready:
            prediction = self.predictor.predict(np.stack(ready), self.reference)
            for index, state in enumerate(eligible):
                cls = int(prediction['class_ids'][index])
                probabilities = prediction['probabilities'][index].tolist()
                state.prediction = {'class_id': cls, 'probabilities': probabilities,
                                    'class_probability': float(probabilities[cls])}
                state.last_prediction = self.frame
        tracks = []
        for person, pose, state, accepted in current:
            prediction = state.prediction if state is not None else None
            tracks.append({'track_id': person.track_id, 'bbox': person.bbox.tolist(),
                           'keypoints': pose.tolist(), 'detection_confidence': person.confidence,
                           'pose_accepted': accepted, 'history_length': len(state.poses) if state else 0,
                           'class_id': prediction['class_id'] if prediction else None,
                           'class_probability': prediction['class_probability'] if prediction else None,
                           'probabilities': prediction['probabilities'] if prediction else None,
                           'prediction_updated': state is not None and state.last_prediction == self.frame})
        self.store.expire(self.frame)
        result = {'type': 'result', 'frame': self.frame, 'width': width, 'height': height,
                  'tracks': tracks, 'processing_ms': (time.perf_counter() - started) * 1000}
        self.frame += 1
        return result
