# Live Camera Implementation Plan

> Execution: native implementation using executing-plans, then independent review. Preserve all existing uncommitted work; no commit or publication requested.

**Goal:** Add live camera recognition with video, skeletons, tracked IDs and intent, without calculation panels.
**Architecture:** Browser video capture with one outstanding JPEG per WebSocket; session-local inference state, exact-frame result rendering and bounded resource cleanup.
**Tech stack:** Existing React, Canvas2D, FastAPI, YOLO/PyTorch; add Python-3.9-compatible WebSocket server support.
**Spec:** `docs/superpowers/specs/2026-11-05-live-camera-design.md`.

## Constraints and review focus

- Preserve offline inference and the 50-observation LSTM→ST-GCN default.
- No camera images/audio persisted. No computation collection for this route.
- Late permission resolution, page departure and backend failure must release media tracks.
- Never queue camera frames or overlay a previous result on a newer image.
- An empty or invalid pose must not display stale intent. Slow capture gaps reset history.
- Camera, uploaded video and recorded model inspection must not run concurrently.

## Tasks

- [x] Backend: add `modern_pipeline/web/camera.py` with `CameraSession.from_models(model_dir, config)` and `process(jpeg)`; wire `/api/camera/live`, reserve/release methods in `jobs.py`, and WebSocket runtime dependency. First add/run failing unittest cases in `test_web_camera.py`, then implement and rerun.
- [x] Frontend: add a camera controller for capture/transport/cleanup, `LiveCamera.jsx`, dedicated styles, and the navigation route. First add controller lifecycle/backpressure tests; implement exact-frame canvas overlays and explicit camera controls, then run frontend tests.
- [x] Integration: update Vite WebSocket proxy and `docs/WEB_DEMO.md`; run full Python/JS suites and production build. Exercise actual weights through the protocol and native Edge UI; review fixes and record limits.

Validation commands: `.venv/bin/python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v`; `cd web_demo && npm test`; `cd web_demo && npm run build`; `git diff --check`.

## Validation and decisions — 2026-11-05

- Python: 159 tests run, 154 passed, 5 skipped (existing historical Windows weight paths unavailable). Frontend: 106 passed, 0 skipped. Production build passed; existing Three.js bundle advisory remains (963.50 kB before gzip). `git diff --check` passed.
- WebSocket runtime `websockets==15.0.1` installed in the existing Python 3.9 environment and pinned in requirements. Existing local server restarted at `127.0.0.1:8765` with the new interface.
- Real protocol run: `demo_input.mp4`, 75 JPEG frames at the browser's maximum 960-pixel edge, real final weights, CPU, macOS arm64, Python 3.9.25. 103 per-person intent updates; first update at frame 49, ID 2; maximum history 50. 10.585 seconds including connection/model startup, not a camera performance guarantee. JSONL, metrics and annotated MP4 inspected under `/tmp/srtp-camera-qa/demo_input/`; no camera frames saved.
- Separate distant-pedestrian fixture: 200 frames processed successfully, no false results on the empty tail; maximum continuous accepted history 44 after JPEG transport, therefore no intent prediction. Kept the 50-observation requirement rather than fabricating outputs or weakening warm-up.
- Native Edge desktop: actual built-in camera accessed, live preview and aligned annotated canvas rendered; ID and all three intent scores updated, observed about 7–8 fps outside concurrent tests. Verified skeleton toggle, default→explicit built-in camera selection, stop/restart, and leaving the camera page removes the browser's capture indicator. The user then reopened the live view; left that active session under their control.
- UI screenshot showed the entire recognition frame within the panel after constraining desktop height. Mobile layout has responsive CSS but was not visually inspected. No framework error overlay was observed; DevTools console was not inspected because the user was actively interacting with the camera page.
- Independent reviewer identified quick-restart admission contention. Added a bounded generation-aware camera-busy retry and regression test. Follow-up race fixed by capturing the busy code under the manager lock, with a failing-then-passing socket test.
- Ruling: preserve the configured PoseFilter and trained-model input semantics. Two missing observations may use filtered joints, followed by expiration; these frames still run a fresh prediction. Added a regression test verifying expiration clears intent and history.
