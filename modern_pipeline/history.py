"""Bounded, continuity-aware pose histories, independent of inference frameworks."""
from collections import deque
from dataclasses import dataclass, field
import numpy as np

@dataclass
class TrackState:
    poses: deque
    last_seen: int = -1
    prediction: object = None
    last_prediction: int = -1

    @property
    def ready(self):
        return len(self.poses) == self.poses.maxlen

    def as_array(self):
        if not self.poses:
            return np.empty((0, 17, 3), dtype=np.float32)
        return np.stack(self.poses).astype(np.float32, copy=False)

class HistoryStore:
    def __init__(self, window=50, max_gap=2, ttl=30):
        if window < 1 or max_gap < 1 or ttl < max_gap:
            raise ValueError('Require window>=1, max_gap>=1, ttl>=max_gap')
        self.window, self.max_gap, self.ttl = window, max_gap, ttl
        self.states = {}

    def update(self, track_id, frame_index, pose):
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (17, 3) or not np.isfinite(pose).all():
            raise ValueError('Pose must have finite shape (17, 3)')
        state = self.states.get(track_id)
        if state is not None and frame_index <= state.last_seen:
            raise ValueError('Frame indices must increase for each track')
        if state is None:
            state = TrackState(deque(maxlen=self.window))
            self.states[track_id] = state
        elif frame_index - state.last_seen > self.max_gap:
            state.poses.clear()
            state.prediction = None
            state.last_prediction = -1
        state.poses.append(pose.copy())
        state.last_seen = frame_index
        return state

    def expire(self, frame_index):
        expired = [i for i, s in self.states.items() if frame_index - s.last_seen > self.ttl]
        for i in expired:
            del self.states[i]
        return expired
