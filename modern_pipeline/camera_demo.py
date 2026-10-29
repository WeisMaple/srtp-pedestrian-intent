"""Timestamped classroom demo logic, separate from the offline research pipeline."""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


class RoadZone:
    """A convex quadrilateral in normalized image coordinates, with inclusive edges."""
    def __init__(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError('Road needs four finite x,y points')
        if np.any(points < 0) or np.any(points > 1):
            raise ValueError('Road coordinates must be between 0 and 1')
        edges = np.roll(points, -1, axis=0) - points
        turns = np.array([cross(edges[i], edges[(i + 1) % 4]) for i in range(4)])
        if not (np.all(turns > 1e-8) or np.all(turns < -1e-8)):
            raise ValueError('Click four corners in order around a convex road; no crossed edges')
        self.points = points.copy()
        self.orientation = 1 if turns[0] > 0 else -1

    def entry_time(self, point, velocity, horizon):
        """Clip the whole constant-velocity segment, including entry before its endpoint."""
        point, velocity = np.asarray(point, dtype=float), np.asarray(velocity, dtype=float)
        if (point.shape != (2,) or velocity.shape != (2,) or
                not np.isfinite(point).all() or not np.isfinite(velocity).all() or
                not math.isfinite(horizon) or horizon <= 0):
            raise ValueError('Point, velocity and positive horizon must be finite')
        enter, leave = 0.0, float(horizon)
        for a, b in zip(self.points, np.roll(self.points, -1, axis=0)):
            edge = b - a
            distance = self.orientation * cross(edge, point - a)
            speed = self.orientation * cross(edge, velocity)
            if abs(speed) < 1e-10:
                if distance < -1e-10:
                    return None
                continue
            boundary = -distance / speed
            if speed > 0:
                enter = max(enter, boundary)
            else:
                leave = min(leave, boundary)
            if enter > leave + 1e-8:
                return None
        return min(float(horizon), max(0.0, enter))


class TimedPoseWindow:
    """Interpolate 50 poses at a declared rate; do not interpret processed frames as time."""
    def __init__(self, fps=30.0, max_gap=0.2):
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('Model FPS and maximum gap must be positive and finite')
        self.fps, self.max_gap = fps, max_gap
        self.span = 49 / fps
        self.samples = deque()

    def update(self, timestamp, pose):
        pose = np.asarray(pose, dtype=np.float32)
        if not math.isfinite(timestamp) or pose.shape != (17, 3) or not np.isfinite(pose).all():
            raise ValueError('Finite timestamp and COCO17 pose required')
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError('Capture timestamps must increase')
        reset = bool(self.samples and timestamp - self.samples[-1][0] > self.max_gap)
        if reset:
            self.samples.clear()
        self.samples.append((timestamp, pose.copy()))
        while len(self.samples) > 2 and self.samples[1][0] <= timestamp - self.span:
            self.samples.popleft()
        return reset

    @property
    def progress(self):
        if not self.samples:
            return 0
        return min(50, int((self.samples[-1][0] - self.samples[0][0]) * self.fps + 1 + 1e-6))

    def resample(self):
        if len(self.samples) < 2:
            return None
        end = self.samples[-1][0]
        if self.samples[0][0] > end - self.span + 1e-7:
            return None
        times = np.array([t - end for t, _ in self.samples])
        poses = np.stack([p for _, p in self.samples])
        grid = np.linspace(-self.span, 0, 50)
        flat = poses.reshape(len(poses), 51)
        result = np.column_stack([np.interp(grid, times, flat[:, k]) for k in range(51)])
        result = result.reshape(50, 17, 3).astype(np.float32)
        # Confidence must not become stronger just because a missing joint was interpolated.
        right = np.searchsorted(times, grid, side='left').clip(0, len(times) - 1)
        left = np.maximum(0, right - 1)
        exact = np.isclose(times[right], grid, atol=1e-8, rtol=0)
        left[exact] = right[exact]
        result[:, :, 2] = np.minimum(poses[left, :, 2], poses[right, :, 2])
        return result


@dataclass
class DemoTrack:
    history: TimedPoseWindow
    last_seen: float
    points: deque = field(default_factory=deque)
    prediction: object = None
    last_prediction: float = -math.inf


class DemoSession:
    """Independent motion baseline and optional original neural-model comparison."""
    def __init__(self, road, predictor=None, horizon=1.0, model_fps=30.0,
                 max_gap=0.2, model_interval=0.2, prediction_ttl=0.6, joint_conf=0.2):
        values = (horizon, model_fps, max_gap, model_interval, prediction_ttl)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Timing settings must be positive and finite')
        if not math.isfinite(joint_conf) or not 0 <= joint_conf <= 1:
            raise ValueError('Joint confidence must be in [0,1]')
        self.road, self.predictor = road, predictor
        self.horizon, self.model_fps, self.max_gap = horizon, model_fps, max_gap
        self.model_interval, self.prediction_ttl = model_interval, prediction_ttl
        self.joint_conf = joint_conf
        self.tracks = {}
        self.reference = None
        self.last_timestamp = -math.inf

    def _baseline(self, state, pose, timestamp, dimensions):
        feet = pose[15:17]
        valid = ((feet[:, 2] >= self.joint_conf) & (feet[:, :2] > 0).all(axis=1)
                 & (feet[:, :2] <= dimensions).all(axis=1))
        result = {'state': 'feet_missing', 'point': None, 'forecast': None,
                  'eta_seconds': None, 'trajectory': []}
        if not valid.all():
            state.points.clear()
            return result
        point = feet[:, :2].mean(axis=0) / dimensions
        state.points.append((timestamp, point))
        while state.points and timestamp - state.points[0][0] > 0.5 + 1e-8:
            state.points.popleft()
        result.update(point=point.tolist(), trajectory=[p.tolist() for _, p in state.points])
        if self.road.entry_time(point, (0, 0), self.horizon) == 0:
            result.update(state='inside', eta_seconds=0.0)
        elif len(state.points) < 3 or timestamp - state.points[0][0] < 0.2 - 1e-8:
            result['state'] = 'warming'
        else:
            times = np.array([t - timestamp for t, _ in state.points])
            coords = np.stack([p for _, p in state.points])
            centered = times - times.mean()
            velocity = (centered[:, None] * (coords - coords.mean(axis=0))).sum(axis=0) / (centered ** 2).sum()
            eta = self.road.entry_time(point, velocity, self.horizon)
            result.update(state='entering' if eta is not None else 'outside',
                          eta_seconds=eta, forecast=(point + velocity * self.horizon).tolist())
        return result

    def update(self, people, timestamp, dimensions):
        dimensions = np.asarray(dimensions, dtype=float)
        if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError('Positive image width and height required')
        if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError('Capture timestamps must increase')
        self.last_timestamp = timestamp
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp - v.last_seen <= 2.0}
        rows, ready, eligible = [], [], []
        for person in people:
            state = self.tracks.get(person.track_id)
            if state is None:
                state = DemoTrack(TimedPoseWindow(self.model_fps, self.max_gap), timestamp)
                self.tracks[person.track_id] = state
            if timestamp - state.last_seen > self.max_gap:
                state.points.clear()
                state.history.samples.clear()
                state.prediction = None
                state.last_prediction = -math.inf
            state.last_seen = timestamp
            pose = np.asarray(person.pose)
            finite = pose.shape == (17, 3) and np.isfinite(pose).all()
            accepted = finite and int(((pose[:, 2] >= self.joint_conf)
                                       & (pose[:, :2] > 0).all(axis=1)).sum()) >= 5
            baseline = {'state': 'feet_missing', 'point': None, 'forecast': None,
                        'eta_seconds': None, 'trajectory': []}
            if accepted:
                if state.history.update(timestamp, pose):
                    state.prediction = None
                    state.last_prediction = -math.inf
                baseline = self._baseline(state, pose, timestamp, dimensions)
                if self.predictor is not None and timestamp - state.last_prediction >= self.model_interval:
                    series = state.history.resample()
                    if series is not None:
                        first_reference = self.reference is None
                        if first_reference:
                            self.reference = series.copy()
                        if not (first_reference and self.predictor.normalization == 'legacy_pair'):
                            ready.append(series)
                            eligible.append(state)
            else:
                state.history.samples.clear()
                state.points.clear()
                state.prediction = None
                state.last_prediction = -math.inf
            rows.append({'track_id': person.track_id, 'bbox': person.bbox.tolist(),
                         'pose_accepted': bool(accepted), 'history_grid': state.history.progress,
                         'baseline': baseline})
        if ready:
            prediction = self.predictor.predict(np.stack(ready), self.reference)
            for i, state in enumerate(eligible):
                cls = int(prediction['class_ids'][i])
                state.prediction = {'class_id': cls,
                                    'score': float(prediction['probabilities'][i][cls])}
                state.last_prediction = timestamp
        for row in rows:
            state = self.tracks[row['track_id']]
            model = {'state': 'disabled', 'class_id': None, 'score': None,
                     'age_seconds': None, 'updated': False}
            if self.predictor is not None:
                model['state'] = 'warming' if row['pose_accepted'] else 'pose_missing'
                if state.prediction is not None:
                    age = timestamp - state.last_prediction
                    model.update(state='expired', age_seconds=age)
                    if age <= self.prediction_ttl:
                        model.update(state='valid', updated=state.last_prediction == timestamp,
                                     **state.prediction)
            row['model'] = model
        return rows
