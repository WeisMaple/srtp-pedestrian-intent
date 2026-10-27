"""One PyTorch pose pass supplies aligned person boxes, IDs, and COCO keypoints."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class PersonPose:
    track_id: int
    bbox: np.ndarray
    confidence: float
    pose: np.ndarray

class YOLOPoseTracker:
    def __init__(self, weights, device, imgsz=640, conf=0.1, tracker='bytetrack.yaml', half=False):
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise FileNotFoundError('Pose weights do not exist: ' + str(weights))
        self.model = YOLO(str(weights), task='pose')
        if self.model.task != 'pose':
            raise ValueError('Frontend weights must be a COCO 17-keypoint pose model')
        self.kwargs = dict(persist=True, tracker=tracker, imgsz=imgsz, conf=conf,
                           device=device, quantize=16 if half else 32, classes=[0], verbose=False,
                           save=False, max_det=100)

    def __call__(self, frame_bgr):
        # Always advance the tracker, including frames with no detections.
        result = self.model.track(frame_bgr, **self.kwargs)[0]
        if result.boxes is None or result.keypoints is None or not result.boxes.is_track:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        ids = result.boxes.id.detach().cpu().numpy().astype(np.int64)
        scores = result.boxes.conf.detach().cpu().numpy()
        poses = result.keypoints.data.detach().cpu().numpy()
        if poses.shape != (len(ids), 17, 3) or len(boxes) != len(ids) or len(scores) != len(ids):
            raise RuntimeError('Tracked boxes and COCO keypoints are not aligned')
        people = []
        for tid, box, score, pose in zip(ids, boxes, scores, poses):
            if np.isfinite(pose).all() and np.isfinite(box).all() and np.isfinite(score):
                people.append(PersonPose(int(tid), box.astype(np.float32), float(score),
                                         pose.astype(np.float32, copy=True)))
        return people


