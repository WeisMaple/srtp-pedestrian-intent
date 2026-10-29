"""Local web task lifecycle with real upload/video/export boundaries."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import cv2
from fastapi.testclient import TestClient
import numpy as np

from modern_pipeline.cli import PredictionCancelled
from modern_pipeline.live_trace import LiveBuffer
from modern_pipeline.web.app import create_app
from modern_pipeline.web.jobs import JobManager


class WebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / 'fixture.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
        self.assertTrue(writer.isOpened())
        for i in range(6):
            writer.write(np.full((120, 160, 3), 30 + i * 20, np.uint8))
        writer.release()
        self.bundle_patch = patch('modern_pipeline.web.jobs.validate_bundle', return_value={
            'name': 'Test bundle', 'normalization': 'legacy_pair', 'pose_filter': False,
            'pose_model': 'pose.pt', 'stgcn': 'g.pt', 'lstm': 'l.pt', 'stats': 'stats.pt'})
        self.bundle_patch.start()
        self.addCleanup(self.bundle_patch.stop)
        self.started = threading.Event()
        self.release = threading.Event()

        def runner(args, on_event=None, cancel=None):
            self.started.set()
            while not self.release.wait(0.01):
                if cancel.is_set():
                    raise PredictionCancelled('Cancelled test inference')
            cap = cv2.VideoCapture(str(args.input))
            out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
            count = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                out.write(image)
                on_event({'kind': 'frame', 'frame': count, 'time_seconds': count / 10,
                          'image': image, 'tracks': [], 'trace': []})
                count += 1
            cap.release()
            out.release()
            args.output.with_suffix('.jsonl').write_text('{"frame":0,"tracks":[]}\n')
            args.output.with_suffix('.metrics.json').write_text('{"frames":6}')
            return {'frames': count}

        self.manager = JobManager(self.root / 'jobs', run_pipeline=runner)
        self.addCleanup(self.manager.close)
        self.client = TestClient(create_app(self.manager))
        self.addCleanup(self.client.close)

    def upload(self):
        response = self.client.post('/api/jobs', files={'file': ('fixture.mp4', self.video.read_bytes(), 'video/mp4')})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def wait_status(self, job_id, wanted):
        deadline = time.monotonic() + 10
        state = None
        while time.monotonic() < deadline:
            state = self.client.get('/api/jobs/' + job_id).json()
            if state['status'] in wanted:
                return state
            time.sleep(0.02)
        self.fail('Task did not finish: ' + str(state))

    def test_invalid_empty_and_oversized_uploads_are_rejected(self):
        for name, content in [('bad.exe', b'payload'), ('empty.mp4', b''), ('bad.mp4', b'not video')]:
            response = self.client.post('/api/jobs', files={'file': (name, content)})
            self.assertEqual(response.status_code, 400, response.text)
        small_client = TestClient(create_app(self.manager, max_upload_bytes=8))
        with small_client:
            response = small_client.post('/api/jobs', files={'file': ('big.mp4', b'x' * 9)})
            self.assertEqual(response.status_code, 413)

    def test_upload_probes_real_metadata_and_file_ranges_work(self):
        job = self.upload()
        self.assertEqual(job['metadata']['frame_count'], 6)
        self.assertEqual(job['metadata']['width'], 160)
        self.assertEqual(job['status'], 'ready')
        self.assertEqual(job['tracks'], [])
        response = self.client.get('/api/jobs/' + job['id'] + '/files/source', headers={'Range': 'bytes=0-15'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.content), 16)

    def test_duplicate_start_and_concurrent_job_do_not_duplicate_inference(self):
        one, two = self.upload(), self.upload()
        self.assertEqual(self.client.post('/api/jobs/' + one['id'] + '/start').status_code, 200)
        self.assertTrue(self.started.wait(2))
        again = self.client.post('/api/jobs/' + one['id'] + '/start')
        self.assertEqual(again.status_code, 200)
        response = self.client.post('/api/jobs/' + two['id'] + '/start')
        self.assertEqual(response.status_code, 409)
        self.client.post('/api/jobs/' + one['id'] + '/cancel')
        self.wait_status(one['id'], {'cancelled'})

    def test_cancelled_tasks_do_not_offer_final_downloads(self):
        job = self.upload()
        self.client.post('/api/jobs/' + job['id'] + '/start')
        self.assertTrue(self.started.wait(2))
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        state = self.wait_status(job['id'], {'cancelled'})
        self.assertEqual(state['outputs'], {})
        self.assertEqual(self.client.get('/api/jobs/' + job['id'] + '/files/video').status_code, 404)

    def test_completed_task_exports_playable_video_jsonl_and_zip(self):
        job = self.upload()
        self.release.set()
        self.client.post('/api/jobs/' + job['id'] + '/start')
        state = self.wait_status(job['id'], {'completed', 'failed'})
        self.assertEqual(state['status'], 'completed', state)
        self.assertEqual(state['progress'], 1)
        for name in ('video', 'data', 'bundle', 'metrics'):
            response = self.client.get(state['outputs'][name])
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.content), 0)
        path = self.root / 'download.zip'
        path.write_bytes(self.client.get(state['outputs']['bundle']).content)
        with zipfile.ZipFile(path) as archive:
            self.assertIn('result.mp4', archive.namelist())
            self.assertIn('trace.json', archive.namelist())
        cap = cv2.VideoCapture(str(self.manager.jobs[job['id']].directory / 'result.mp4'))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
        cap.release()
        preview = self.client.get('/api/jobs/' + job['id'] + '/preview')
        image = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(image.mean()), 100, 'Preview should contain the last processed frame')

    def test_live_stream_sends_ordered_events_reconnects_and_restores_final_snapshot(self):
        job_id = self.upload()['id']
        job = self.manager.jobs[job_id]
        for frame in range(2):
            self.manager._observe(job, {'kind': 'computation', 'frame': frame,
                'time_seconds': frame / 10, 'model': 'yolo', 'node': 'yolo.0',
                'stage': 1, 'track_ids': [], 'duration_ms': 0.5,
                'captured_at_ms': 100 + frame, 'tensors': {}})
        job.state['status'] = 'completed'
        job.live.finish('completed')
        job.live.save(job.directory / 'live.json', 'completed')
        self.manager.persist(job)
        url = '/api/jobs/' + job_id + '/live'
        initial = self.client.get(url)
        self.assertEqual(initial.status_code, 200)
        self.assertIn('event: snapshot', initial.text)
        self.assertIn('event: terminal', initial.text)
        replay = self.client.get(url, headers={'Last-Event-ID': '1'})
        self.assertIn('id: 2\nevent: computation', replay.text)
        self.assertIn('id: 3\nevent: terminal', replay.text)
        self.assertNotIn('event: snapshot', replay.text)
        self.assertEqual(self.client.get(url, headers={'Last-Event-ID': 'broken'}).status_code, 400)
        self.assertEqual(self.client.get(url, headers={'Last-Event-ID': '-1'}).status_code, 400)
        recovered = JobManager(self.root / 'jobs')
        try:
            rows, status = recovered.live_events(job_id, 0)
            self.assertEqual(status, 'completed')
            self.assertEqual([row['event'] for row in rows], ['gap', 'snapshot', 'terminal'])
            self.assertEqual(rows[1]['data']['nodes']['yolo.0']['frame'], 1)
        finally:
            recovered.close()

    def test_missing_tasks_and_unlisted_paths_are_not_served(self):
        self.assertEqual(self.client.get('/api/jobs/not-a-task').status_code, 404)
        job = self.upload()
        self.assertEqual(self.client.get('/api/jobs/' + job['id'] + '/files/state.json').status_code, 404)

    def test_restart_retains_metadata_and_marks_running_jobs_interrupted(self):
        job = self.upload()
        self.manager.jobs[job['id']].state['status'] = 'running'
        self.manager.persist(self.manager.jobs[job['id']])
        recovered = JobManager(self.root / 'jobs')
        try:
            state = recovered.snapshot(job['id'])
            self.assertEqual(state['status'], 'failed')
            self.assertIn('中断', state['error'])
        finally:
            recovered.close()

    def test_track_state_is_current_even_when_jpeg_update_is_throttled(self):
        job_id = self.upload()['id']
        job = self.manager.jobs[job_id]
        image = np.zeros((120, 160, 3), np.uint8)
        first = {'track_id': 1, 'bbox': [1, 1, 50, 110], 'keypoints': [], 'history_length': 49,
                 'pose_accepted': True, 'class_id': None, 'class_probability': None,
                 'probabilities': None, 'prediction_updated': False}
        second = dict(first, track_id=2, class_id=2, class_probability=0.7,
                      probabilities=[0.1, 0.2, 0.7], history_length=50, prediction_updated=True)
        with patch('modern_pipeline.web.jobs.time.monotonic', side_effect=[1.0, 1.1]):
            self.manager._observe(job, {'kind': 'frame', 'frame': 0, 'time_seconds': 0,
                                       'image': image, 'tracks': [first], 'trace': []})
            self.manager._observe(job, {'kind': 'frame', 'frame': 1, 'time_seconds': 0.1,
                                       'image': image, 'tracks': [second], 'trace': []})
        state = self.manager.snapshot(job_id)
        self.assertEqual(state['frame'], 1)
        self.assertEqual(state['tracks'][0]['track_id'], 2)
        self.assertEqual(state['tracks'][0]['class_id'], 2)

    def test_final_trace_includes_first_person_forecast_after_sampling_interval(self):
        original = self.manager.run_pipeline

        def runner(args, on_event=None, cancel=None):
            def observe(event):
                if event['kind'] == 'frame':
                    poses = np.tile([80, 90, 0.9], (50, 17, 1)).astype(np.float32)
                    event['trace'] = [{'track_id': 1, 'history': poses,
                                       'future': poses if event['frame'] == 5 else None},
                                      {'track_id': 2, 'history': poses, 'future': poses}]
                on_event(event)
            return original(args, on_event=observe, cancel=cancel)

        self.manager.run_pipeline = runner
        job = self.upload()
        self.release.set()
        self.client.post('/api/jobs/' + job['id'] + '/start')
        self.assertEqual(self.wait_status(job['id'], {'completed', 'failed'})['status'], 'completed')
        samples = self.client.get('/api/jobs/' + job['id'] + '/trace').json()['samples']
        self.assertEqual(samples[-1]['frame'], 5)
        self.assertIsNotNone(samples[-1]['tracks'][0]['future'])


class StreamingTerminalTests(unittest.IsolatedAsyncioTestCase):
    class Request:
        headers = {}

        async def is_disconnected(self):
            return False

    class Manager:
        def __init__(self):
            self.status = 'running'
            self.revision = 42
            self.live = LiveBuffer('race')

        def snapshot(self, job_id):
            return {'id': job_id, 'status': self.status, 'revision': self.revision}

        def live_events(self, job_id, after=None):
            return self.live.read(after, self.status), self.status

    def endpoint(self, manager, suffix):
        app = create_app(manager)
        path = '/api/jobs/{job_id}/' + suffix
        return next(route.endpoint for route in app.routes if getattr(route, 'path', None) == path)

    def data(self, chunk):
        return json.loads(next(line[6:] for line in chunk.splitlines() if line.startswith('data: ')))

    async def test_status_stream_sends_terminal_snapshot_even_without_revision_change(self):
        for status in ('completed', 'cancelled', 'failed'):
            with self.subTest(status=status):
                manager = self.Manager()
                response = await self.endpoint(manager, 'events')('race', self.Request())
                iterator = response.body_iterator
                first = self.data(await iterator.__anext__())
                self.assertEqual(first['status'], 'running')
                manager.status = status  # Between status update and the worker's finally block.
                remaining = [self.data(chunk) async for chunk in iterator]
                self.assertEqual(remaining, [{'id': 'race', 'status': status, 'revision': 42}])

    async def test_live_stream_replays_final_frame_then_terminates_before_worker_finish(self):
        for status in ('completed', 'cancelled', 'failed'):
            with self.subTest(status=status):
                manager = self.Manager()
                manager.live.append({'frame': 66, 'time_seconds': 2.2, 'node': 'pipeline.frame'})
                response = await self.endpoint(manager, 'live')('race', self.Request())
                iterator = response.body_iterator
                first = self.data(await iterator.__anext__())
                self.assertEqual(first['status'], 'running')
                manager.live.append({'frame': 74, 'time_seconds': 74 / 30, 'node': 'pipeline.output'})
                manager.status = status
                self.assertIsNone(manager.live.terminal, 'Worker finalization has deliberately not run.')
                remaining = [chunk async for chunk in iterator]
                self.assertEqual([self.data(chunk)['frame'] for chunk in remaining], [74, 74])
                self.assertIn('event: computation', remaining[0])
                self.assertIn('event: terminal', remaining[1])
                self.assertEqual(self.data(remaining[-1])['status'], status)

    async def test_live_stream_terminates_at_same_cursor_before_worker_finish(self):
        for status in ('completed', 'cancelled', 'failed'):
            with self.subTest(status=status):
                manager = self.Manager()
                response = await self.endpoint(manager, 'live')('race', self.Request())
                iterator = response.body_iterator
                first = self.data(await iterator.__anext__())
                manager.status = status
                remaining = [chunk async for chunk in iterator]
                self.assertEqual(len(remaining), 1)
                self.assertIn('event: terminal', remaining[0])
                final = self.data(remaining[0])
                self.assertEqual(final['status'], status)
                self.assertEqual(final['seq'], first['seq'])


class LauncherTests(unittest.TestCase):
    def test_fresh_shell_sets_runtime_root_before_importing_bootstrap(self):
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env.pop('SRTP_MODERN_HOME', None)
        code = """import runpy
from pathlib import Path
from unittest.mock import patch
def check(app, **kwargs):
    from modern_pipeline.bootstrap import ARTIFACT_ROOT
    assert ARTIFACT_ROOT == Path.cwd() / 'artifacts', str(ARTIFACT_ROOT)
with patch('uvicorn.run', check):
    runpy.run_path('run_web_demo.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=str(root), env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
