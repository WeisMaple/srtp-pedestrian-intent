"""A single-slot camera buffer: slow inference never queues older frames."""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    timestamp: float
    frame: object


class LatestFrameCapture:
    def __init__(self, camera, clock=time.monotonic):
        self.camera = camera
        self.clock = clock
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.latest = None
        self.error = None
        self.worker = threading.Thread(target=self._capture, name='camera-capture', daemon=True)
        self.worker.start()

    def _capture(self):
        sequence = 0
        try:
            while not self.stopping.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError('Camera read failed; check connection and camera permissions')
                timestamp = self.clock()
                sequence += 1
                with self.condition:
                    self.latest = CapturedFrame(sequence, timestamp, frame)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            self.camera.release()

    def read(self, after=0, timeout=2.0):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: (self.latest is not None and self.latest.sequence > after)
                or self.error is not None or self.stopping.is_set(), timeout)
            if not ready:
                raise TimeoutError('Camera has not delivered a new frame')
            if self.latest is not None and self.latest.sequence > after:
                return self.latest
            if self.error is not None:
                raise RuntimeError('Camera capture stopped: ' + str(self.error)) from self.error
            raise RuntimeError('Camera capture closed')

    def close(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        # The worker owns release(): never release a device concurrently with read().
        # A driver stuck inside read() is a daemon and cannot block process exit.
        self.worker.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
