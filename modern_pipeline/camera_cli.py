"""Opt-in camera prototype; the default offline CLI remains unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .bootstrap import ARTIFACT_ROOT, configure


class DemoStopped(Exception):
    """The main UI thread requested an orderly exit while work was pending."""


def arguments(argv=None):
    p = argparse.ArgumentParser(description='Fixed-camera simulated crossing demo (experimental)')
    p.add_argument('--camera', type=int, default=0, help='Local camera index')
    p.add_argument('--backend', choices=['auto', 'dshow', 'msmf', 'avfoundation'], default='auto')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--capture-fps', type=float, default=30, help='Requested rate; timestamps use actual delivery')
    p.add_argument('--pose-model', type=Path, default=ARTIFACT_ROOT / 'weights/yolo26s-pose.pt')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0, or 0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.1)
    p.add_argument('--joint-conf', type=float, default=0.2)
    p.add_argument('--half', action='store_true', help='YOLO FP16 on CUDA only')
    p.add_argument('--road', nargs=4, metavar='X,Y', help='Four ordered corners in normalized image coordinates')
    p.add_argument('--horizon', type=float, default=1.0, help='Trajectory extrapolation horizon in seconds')
    p.add_argument('--max-gap', type=float, default=0.2, help='Maximum observation gap in seconds')
    p.add_argument('--behavior', action='store_true', help='Also run the original LSTM then ST-GCN comparison')
    p.add_argument('--stgcn', type=Path, default=ARTIFACT_ROOT / 'weights/stgcn_reproduced.pt')
    p.add_argument('--lstm', type=Path, default=ARTIFACT_ROOT / 'weights/lstm_reproduced_wd0.pt')
    p.add_argument('--normalization', choices=['legacy_pair', 'training_stats'], default='legacy_pair')
    p.add_argument('--stats', type=Path, default=ARTIFACT_ROOT / 'weights/normalization.pt')
    p.add_argument('--model-fps', type=float, default=30, help='Resampling rate for the 50-pose model window')
    p.add_argument('--model-interval', type=float, default=0.2, help='Minimum model update interval in seconds')
    p.add_argument('--prediction-ttl', type=float, default=0.6, help='Expire predictions older than this many seconds')
    p.add_argument('--log', type=Path, help='New .jsonl file for session metadata, predictions and timing')
    p.add_argument('--save-keypoints', action='store_true', help='Include observed pixel keypoints in --log')
    p.add_argument('--no-display', action='store_true', help='Camera soak test; requires --road')
    p.add_argument('--max-frames', type=int, help='Stop after this many processed frames')
    a = p.parse_args(argv)
    timing = (a.capture_fps, a.horizon, a.max_gap, a.model_fps, a.model_interval, a.prediction_ttl)
    if not all(math.isfinite(v) and v > 0 for v in timing):
        p.error('Timing settings must be positive and finite')
    if a.camera < 0 or a.width < 32 or a.height < 32 or a.imgsz < 32:
        p.error('Require camera >= 0 and width, height, imgsz >= 32')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (a.conf, a.joint_conf)):
        p.error('Confidence values must be in [0,1]')
    if a.max_frames is not None and a.max_frames < 1:
        p.error('max-frames must be positive')
    if a.no_display and not a.road:
        p.error('--no-display requires an explicit --road')
    if a.save_keypoints and not a.log:
        p.error('--save-keypoints requires --log')
    if a.log and a.log.suffix.lower() != '.jsonl':
        p.error('--log must be a new .jsonl file')
    if a.road:
        from .camera_demo import RoadZone
        try:
            a.road = RoadZone([[float(v) for v in point.split(',')] for point in a.road])
        except ValueError as exc:
            p.error(str(exc))
    return a


def validate_artifacts(a):
    required = [a.pose_model]
    if a.behavior:
        required.extend([a.stgcn, a.lstm])
        if a.normalization == 'training_stats':
            required.append(a.stats)
    for path in required:
        if not path.expanduser().is_file():
            raise FileNotFoundError('Local weights/statistics missing: ' + str(path))
    if a.log and a.log.expanduser().exists():
        raise FileExistsError('Refusing to overwrite: ' + str(a.log))


def expire_display(rows, frame_age, ttl):
    """Include inference time in result age, not just time between captured frames."""
    for row in rows:
        if frame_age > ttl:
            row['baseline'].update(state='expired', forecast=None, eta_seconds=None)
        model = row['model']
        if model['age_seconds'] is not None:
            model['age_seconds'] += frame_age
            if model['age_seconds'] > ttl:
                model.update(state='expired', class_id=None, score=None)


def preview_size(frame):
    height, width = frame.shape[:2]
    scale = min(960 / width, 540 / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render_frame(frame, people, rows, road, points, fps, age, a, message=''):
    import cv2
    import numpy as np
    from .cli import EDGES
    width, height = preview_size(frame)
    scene = cv2.resize(frame, (width, height))
    scale = np.array([width / frame.shape[1], height / frame.shape[0]])
    dimensions = np.array([width, height])

    def pixel(point, normalized=False):
        value = np.asarray(point) * (dimensions if normalized else scale)
        return tuple(np.clip(value, [0, 0], dimensions - 1).astype(int))

    corners = road.points if road is not None else points
    if len(corners):
        polygon = np.array([pixel(point, True) for point in corners], dtype=np.int32)
        if road is not None:
            overlay = scene.copy()
            cv2.fillConvexPoly(overlay, polygon, (130, 100, 30))
            scene = cv2.addWeighted(overlay, 0.22, scene, 0.78, 0)
        cv2.polylines(scene, [polygon], len(corners) == 4, (230, 210, 80), 2)
        for i, point in enumerate(polygon):
            cv2.circle(scene, tuple(point), 5, (230, 210, 80), -1)
            cv2.putText(scene, str(i + 1), tuple(point + [7, -7]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    records = {row['track_id']: row for row in rows}
    colors = {'entering': (60, 190, 255), 'inside': (240, 180, 80), 'outside': (100, 210, 120)}
    for person in people:
        row = records.get(person.track_id)
        if row is None:
            continue
        baseline = row['baseline']
        color = colors.get(baseline['state'], (170, 170, 170))
        box = person.bbox
        cv2.rectangle(scene, pixel(box[:2]), pixel(box[2:]), color, 2)
        x, y = pixel(box[:2])
        cv2.putText(scene, 'ID %d | %s' % (person.track_id, baseline['state']),
                    (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        pose = person.pose
        for i, j in EDGES:
            if min(pose[i, 2], pose[j, 2]) >= a.joint_conf:
                cv2.line(scene, pixel(pose[i, :2]), pixel(pose[j, :2]), (240, 210, 60), 1)
        for px, py, confidence in pose:
            if confidence >= a.joint_conf:
                cv2.circle(scene, pixel((px, py)), 2, (80, 240, 240), -1)
        trajectory = baseline['trajectory']
        if len(trajectory) > 1:
            cv2.polylines(scene, [np.array([pixel(p, True) for p in trajectory])], False, color, 2)
        if baseline['forecast'] is not None:
            cv2.arrowedLine(scene, pixel(baseline['point'], True), pixel(baseline['forecast'], True),
                           color, 2, tipLength=0.15)
    canvas = np.full((max(600, height + 60), width + 340, 3), (27, 24, 23), dtype=np.uint8)
    canvas[:height, :width] = scene

    def text(value, x, y, color=(225, 225, 225), size=0.48):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    x = width + 16
    text('CAMERA CROSSING DEMO', x, 30, size=0.62)
    text('Simulated road | experimental', x, 55)
    text('BASELINE: image-velocity extrapolation', x, 87, size=0.43)
    text('Forecast horizon: %.1f s' % a.horizon, x, 110)
    text('MODEL: LSTM > ST-GCN' if a.behavior else 'MODEL: off (use --behavior)', x, 140)
    text('Processing: %.1f FPS' % fps, x, 171)
    text('Frame age: %.0f ms' % (age * 1000), x, 194,
         (60, 190, 255) if age > a.prediction_ttl else (225, 225, 225))
    if road is None:
        text('Click 4 road corners in order.', x, 245)
        text('Corners selected: %d / 4' % len(points), x, 270)
        text('ENTER confirms. R clears.', x, 295)
        if message:
            text(message[:42], x, 330, (60, 190, 255), 0.42)
    elif not rows:
        text('Waiting for a tracked person...', x, 245)
        text('Keep the whole body in view.', x, 270)
    for i, row in enumerate(sorted(rows, key=lambda r: r['track_id'])[:3]):
        y = 235 + i * 103
        baseline, model = row['baseline'], row['model']
        color = colors.get(baseline['state'], (170, 170, 170))
        text('ID %d | baseline: %s' % (row['track_id'], baseline['state']), x, y, color)
        if baseline['state'] == 'entering':
            text('Estimated entry: %.2f s' % baseline['eta_seconds'], x, y + 22)
        if model['state'] == 'valid':
            label = ('irrelevant', 'not crossing', 'crossing')[model['class_id']]
            text('Model: %s | score %.2f' % (label, model['score']), x, y + 44, size=0.43)
            text('Result age: %.2f s' % model['age_seconds'], x, y + 65)
        elif a.behavior:
            text('Model: %s | grid %d/50' % (model['state'], row['history_grid']), x, y + 44, size=0.43)
    text('Q / ESC quit   R redraw road   C clear history', 12, height + 24, size=0.46)
    text('Skeleton = observed. Arrow = baseline extrapolation.', 12, height + 47, size=0.45)
    text('Model score is not calibrated probability.', x, canvas.shape[0] - 28, size=0.4)
    return canvas


def run(a):
    configure()
    validate_artifacts(a)
    import cv2
    import torch
    from .camera_capture import LatestFrameCapture
    from .camera_demo import DemoSession, RoadZone
    from .frontend import YOLOPoseTracker
    requested = a.device
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif requested.isdigit():
        requested = 'cuda:' + requested
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use cpu or an available CUDA device')
    if a.half and device.type != 'cuda':
        raise ValueError('--half requires CUDA')
    torch.set_num_threads(2)
    print('Loading local pose model' + (' and behavior models...' if a.behavior else '...'), flush=True)
    front = YOLOPoseTracker(a.pose_model.expanduser(), str(device), a.imgsz, a.conf, half=a.half)
    predictor = None
    if a.behavior:
        from .behavior import BehaviorPredictor
        predictor = BehaviorPredictor(a.stgcn.expanduser(), a.lstm.expanduser(), str(device),
                                      a.normalization, a.stats.expanduser())
    backends = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW,
                'msmf': cv2.CAP_MSMF, 'avfoundation': cv2.CAP_AVFOUNDATION}
    log = None
    capture = None
    camera = None
    window = 'Pedestrian Intent - Deep Learning Course'
    road, points, session = a.road, [], None
    view_dimensions = [1, 1]
    processed = skipped = sequence = 0
    fps, status, message = 0.0, 'failed', ''
    started = time.monotonic()
    last_processed = started
    previous_shape = None
    last_preview = None
    pending_key = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-inference')

    def write(payload):
        if log is not None:
            log.write(json.dumps(payload, allow_nan=False) + '\n')
            log.flush()

    def click(event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN and road is None and len(points) < 4
                and 0 <= x < view_dimensions[0] and 0 <= y < view_dimensions[1]):
            points.append([x / view_dimensions[0], y / view_dimensions[1]])

    def refresh_while_waiting():
        nonlocal pending_key
        if a.no_display:
            return
        if last_preview is not None:
            old_frame, old_people, old_rows, timestamp = last_preview
            age = time.monotonic() - timestamp
            shown = deepcopy(old_rows)
            expire_display(shown, age, a.prediction_ttl)
            cv2.imshow(window, render_frame(old_frame, old_people, shown, road, points,
                                           fps, age, a, message))
        key = cv2.waitKey(1) & 0xff
        if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            raise DemoStopped()
        if key in (ord('r'), ord('c'), 10, 13):
            pending_key = key

    def work(operation, *args):
        # Only one operation can be pending. GUI calls stay on the main thread.
        future = executor.submit(operation, *args)
        while True:
            try:
                return future.result(timeout=None if a.no_display else 0.03)
            except FutureTimeout:
                if future.done():
                    raise
                refresh_while_waiting()

    try:
        if a.log:
            target = a.log.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            log = target.open('x', encoding='utf-8')
            write({'event': 'session', 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'camera': a.camera, 'backend': a.backend, 'device': str(device),
                   'pose_model': str(a.pose_model.expanduser().resolve()),
                   'behavior_enabled': a.behavior, 'stgcn': str(a.stgcn) if a.behavior else None,
                   'lstm': str(a.lstm) if a.behavior else None, 'normalization': a.normalization,
                   'stats': str(a.stats) if a.behavior and a.normalization == 'training_stats' else None,
                   'requested_capture_fps': a.capture_fps, 'imgsz': a.imgsz,
                   'detection_confidence': a.conf, 'joint_confidence': a.joint_conf,
                   'half_pose': a.half, 'horizon_seconds': a.horizon, 'model_fps': a.model_fps,
                   'model_interval_seconds': a.model_interval, 'max_gap_seconds': a.max_gap,
                   'prediction_ttl_seconds': a.prediction_ttl,
                   'baseline': 'constant_image_velocity', 'label_map': ['irrelevant', 'not crossing', 'crossing'],
                   'note': 'Classroom prototype. Resampled histories; no validated road-intent accuracy.'})
        camera = cv2.VideoCapture(a.camera, backends[a.backend])
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera %d; check index, permissions and other camera apps' % a.camera)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        camera.set(cv2.CAP_PROP_FPS, a.capture_fps)
        capture = LatestFrameCapture(camera)
        if not a.no_display:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, click)
        while a.max_frames is None or processed < a.max_frames:
            packet = work(capture.read, sequence, 2.0)
            skipped += max(0, packet.sequence - sequence - 1)
            sequence = packet.sequence
            frame = packet.frame
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError('Camera did not deliver a BGR image')
            view_dimensions[:] = preview_size(frame)
            dimensions = (frame.shape[1], frame.shape[0])
            if previous_shape != dimensions:
                session = None
                previous_shape = dimensions
                write({'event': 'frame_size', 'width': dimensions[0], 'height': dimensions[1]})
            people, rows = [], []
            if last_preview is None:
                last_preview = (frame, [], [], packet.timestamp)
            if road is not None:
                if session is None:
                    session = DemoSession(road, predictor, a.horizon, a.model_fps, a.max_gap,
                                          a.model_interval, a.prediction_ttl, a.joint_conf)
                    write({'event': 'reset', 'time_seconds': packet.timestamp - started,
                           'road': road.points.tolist()})
                def infer():
                    detected = front(frame)
                    return detected, session.update(detected, packet.timestamp, dimensions)

                people, rows = work(infer)
            # Keep unexpired raw rows for repeated UI refreshes; each refresh applies
            # wall-clock age once, including time spent in the next inference/read.
            last_preview = (frame, people, deepcopy(rows), packet.timestamp)
            age = time.monotonic() - packet.timestamp
            expire_display(rows, age, a.prediction_ttl)
            now = time.monotonic()
            instantaneous = 1 / max(now - last_processed, 1e-6)
            last_processed = now
            fps = instantaneous if fps == 0 else 0.85 * fps + 0.15 * instantaneous
            if a.save_keypoints:
                poses = {p.track_id: p.pose.tolist() for p in people}
                for row in rows:
                    row['keypoints'] = poses[row['track_id']]
            write({'event': 'frame', 'sequence': sequence, 'time_seconds': packet.timestamp - started,
                   'frame_age_ms': age * 1000, 'skipped_capture_frames_total': skipped,
                   'processing_fps': fps, 'calibrating': road is None, 'tracks': rows})
            processed += 1
            if not a.no_display:
                cv2.imshow(window, render_frame(frame, people, rows, road, points, fps, age, a, message))
                polled_key = cv2.waitKey(1) & 0xff
                key = pending_key if pending_key is not None and polled_key not in (27, ord('q')) else polled_key
                pending_key = None
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    status = 'complete'
                    break
                if key == ord('r'):
                    road, points, session, message = None, [], None, ''
                    last_preview = (frame, [], [], packet.timestamp)
                elif key == ord('c'):
                    session = None
                    last_preview = (frame, [], [], packet.timestamp)
                elif key in (10, 13) and road is None:
                    try:
                        road = RoadZone(points)
                        message = ''
                    except ValueError:
                        message = 'Need 4 ordered convex corners. R resets.'
        else:
            status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
    except DemoStopped:
        status = 'complete'
    except cv2.error as exc:
        raise RuntimeError('OpenCV camera/display error: ' + str(exc)) from exc
    finally:
        cleanup_errors = []
        actions = [] if a.no_display else [cv2.destroyAllWindows]
        if capture is not None:
            actions.append(capture.close)
        elif camera is not None:
            actions.append(camera.release)
        for action in actions:
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                write({'event': 'end', 'status': status, 'processed_frames': processed,
                       'skipped_capture_frames': skipped, 'elapsed_seconds': time.monotonic() - started,
                       'cleanup_errors': cleanup_errors})
            finally:
                if log is not None:
                    log.close()
    print('Camera demo %s: %d processed frames, %d skipped capture frames.' % (status, processed, skipped))


def main(argv=None):
    try:
        run(arguments(argv))
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        raise SystemExit('Camera demo error: ' + str(exc))


if __name__ == '__main__':
    main()
