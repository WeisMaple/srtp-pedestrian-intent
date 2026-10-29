"""Exercise capture buffering without opening a physical camera."""
import threading
import unittest

from modern_pipeline.camera_capture import LatestFrameCapture


class BurstCamera:
    def __init__(self, count):
        self.count = count
        self.index = 0
        self.waiting = threading.Event()
        self.unblock = threading.Event()
        self.released = threading.Event()

    def read(self):
        if self.index < self.count:
            self.index += 1
            return True, self.index
        self.waiting.set()
        self.unblock.wait(2)
        return False, None

    def release(self):
        self.released.set()


class CaptureTests(unittest.TestCase):
    def test_slow_consumer_receives_newest_frame_not_buffered_backlog(self):
        camera = BurstCamera(3)
        capture = LatestFrameCapture(camera, clock=iter([1.0, 1.1, 1.2]).__next__)
        try:
            self.assertTrue(camera.waiting.wait(1))
            packet = capture.read(after=0, timeout=1)
            self.assertEqual(packet.sequence, 3)
            self.assertEqual(packet.frame, 3)
            self.assertEqual(packet.timestamp, 1.2)
        finally:
            camera.unblock.set()
            capture.close()
        self.assertTrue(camera.released.is_set())

    def test_stalled_camera_times_out_instead_of_reusing_old_frame(self):
        camera = BurstCamera(1)
        capture = LatestFrameCapture(camera)
        try:
            self.assertTrue(camera.waiting.wait(1))
            packet = capture.read(timeout=1)
            with self.assertRaises(TimeoutError):
                capture.read(after=packet.sequence, timeout=0.01)
        finally:
            camera.unblock.set()
            capture.close()

    def test_failed_camera_read_releases_device_and_reports_error(self):
        camera = BurstCamera(0)
        camera.unblock.set()
        with LatestFrameCapture(camera) as capture:
            with self.assertRaisesRegex(RuntimeError, 'camera|Camera'):
                capture.read(timeout=1)
        self.assertTrue(camera.released.is_set())


if __name__ == '__main__':
    unittest.main()
