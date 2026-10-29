# Camera Demo Implementation Plan

> **For agentic workers:** Use the inline execution and test-first checks below for this authorized prototype.

**Goal:** Try a fixed-camera classroom crossing demonstration with an explicitly labelled trajectory baseline and optional existing neural model comparison.

**Architecture:** A capture thread retains only the newest timestamped frame. A separate demo session owns normalized road geometry, per-person motion, and timestamp-resampled pose histories; the OpenCV CLI renders results and optionally saves JSONL for rehearsal analysis.

**Tech Stack:** Existing Python 3.9, NumPy, OpenCV, PyTorch and YOLO26/ByteTrack.

**Spec:** The user's “试试” authorizes a prototype of the preceding fixed-camera proposal. Four ordered clicks define a convex simulated road. Extrapolation is a constant-image-velocity baseline, not learned intent. `--behavior` explicitly enables the existing 50-observation LSTM → ST-GCN comparison; default offline inference remains unchanged. No training, model downloads, or accuracy claims are included.

## Global Constraints

- Preserve `AGENTS.md`, the offline entry point, existing weights and default model behavior.
- Use monotonic capture timestamps, bound frame buffering, and invalidate stale predictions.
- Label the baseline and neural model separately; JAAD class 0 is irrelevant, not uncertainty.
- Require local weights; never silently download or substitute models.
- Save recordings/logs only when requested and refuse existing output paths.

## Review Focus

- Empty/disconnected camera: terminate with a clear error and release resources.
- Fast motion through a narrow road: test intersection along the entire forecast segment.
- Missing feet, occlusion, changed track IDs: no borrowed or stale trajectory prediction.
- Irregular frame delivery: 50 frames cannot silently imply a fixed duration.
- Existing output, invalid polygon, missing weights: fail before capturing or overwriting.

## Tasks

- [x] Test and implement `modern_pipeline/camera_demo.py`: convex road geometry, timestamped history, baseline and optional `BehaviorPredictor` orchestration. Independently derived crossing, parallel-walking, gap and cache-expiry fixtures.
- [x] Test and implement `modern_pipeline/camera_capture.py`: newest-frame delivery, read failure, timeout and shutdown. Camera access is the external test boundary.
- [x] Implement `modern_pipeline/camera_cli.py` and `run_camera_demo.py`: source/device selection, four-click road setup, live annotated preview, optional exclusive JSONL logging, reset/quit controls.
- [x] Document commands and prototype limits in `docs/CAMERA_DEMO.md`, linked from README.
- [x] Run the full unittest suite, CLI checks and synthetic rendering/logging smoke checks. Report missing real-weight and physical-camera validation explicitly.

## Execution Notes

This is a local, uncommitted prototype in the user's working directory. Tests focus on temporal and geometry behavior; hardware inference requires external artifacts. The development environment is an ignored Python 3.9 virtual environment.

Final verification: 51 tests run, 46 passed and 5 real-weight tests skipped. CLI help works without inference dependencies; missing weights fail before opening a camera. Synthetic rendering smoke wrote/decoded 90 MP4 frames and parsed 90 JSONL records. No physical-camera or learned-model accuracy validation was possible without the external artifacts and scene.

Independent review found that a blocking next inference/read could freeze an old label, and that GUI cleanup failure could bypass final logging. Regression tests reproduced both; single-worker inference/read waiting now pumps the main-thread GUI and expires previous results, and cleanup preserves log finalization and the original error. All three added regression cases passed in the final suite.
