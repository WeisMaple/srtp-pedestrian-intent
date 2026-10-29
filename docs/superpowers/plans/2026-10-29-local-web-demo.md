# Local Web Demo Implementation Plan

> **For agentic workers:** Implement the tasks below with test-first backend changes and browser interaction verification. The approved design authorizes direct implementation; no second design approval is needed.

**Goal:** Ship the approved React/Python workbench with actual inference, inspectable computation playback and working result downloads.

**Architecture:** FastAPI owns one background inference job at a time, UUID artifact directories, live snapshots/SSE and exports. Optional observers on the existing CLI/model expose actual progress and bounded history/future samples without changing default predictions. React/Vite renders the two approved screens and drives explanation playback with recorded data.

**Tech Stack:** Python 3.9, existing PyTorch/OpenCV, FastAPI/Uvicorn, system ffmpeg, React/Vite, native SVG/CSS animation.

**Spec:** `docs/superpowers/specs/2026-10-29-local-web-demo-design.md`

## Global Constraints

- Preserve AGENTS.md, the camera prototype and offline default 50-observation LSTM → ST-GCN behavior.
- Use the verified `weights/final` bundle, CPU behavior inference and legacy_pair configuration.
- No synthetic records or scores in product empty states. Class 0 means 其他/irrelevant.
- All artifacts remain under ignored `artifacts/web/jobs`; serve only job-whitelisted files on localhost.
- Follow the two accepted concepts, with dynamic actual data and correct empty/running/completed states.

## Review Focus

- Cancel/error during model calls or encoding must not yield a completed export.
- Concurrent starts and page refresh must not create duplicate inference or lose the active job.
- No-people and insufficient-history videos must remain meaningful without invented predictions.
- Browser playback must work with encoded H.264, including seeking/range requests.
- Animated explanation must remain responsive and preserve observed/future distinction and motion preferences.

## Task 1 — Observation hooks

Files: `modern_pipeline/{behavior,cli}.py`, `modern_pipeline/tests/test_observers.py`.

- [x] Write failing checks for forecast callback order/numeric parity, CLI event/JSONL parity and cancellation.
- [x] Add `on_event=None` to forecasting/predict/run; add `cancel=None` to run. Observer dictionaries use `kind` and zero-based `stage` (0 decode, 1 pose, 2 tracking, 3 history, 4 forecast, 5 classify). Forecast steps carry step 1–5 and actual de-normalized poses. Frame events carry frame/time, rendered BGR, serial tracks and optional trace data. Raise a dedicated `PredictionCancelled` at frame boundaries.
- [x] Check observer-free output remains unchanged, then run the full unittest suite.

## Task 2 — Job/API/export backend

Files: `modern_pipeline/web/{__init__,media,jobs,app}.py`, `run_web_demo.py`, `requirements-web.txt`, `modern_pipeline/tests/test_web.py`.

- [x] Failing tests for invalid upload, start conflict, cancellation, recovery and guarded downloads.
- [x] Implement single-worker `JobManager` with atomic snapshots, bounded logs/traces and 4 Hz preview JPEG updates; a new tracker is created per job. Persist task state and recover interrupted jobs as failed.
- [x] API implements the spec table. Use multipart upload field `file`. SSE emits default `message` events containing the full snapshot, with keepalive comments. File endpoints support ranges using FileResponse. Output MP4 is transcoded via ffmpeg and ZIP contains MP4, JSONL, metrics and trace.
- [x] Verify actual download contents, ffmpeg decode and lifecycle transitions. Provide a same-origin static frontend mount and launcher that binds 127.0.0.1.

## Frontend/Backend contract

`GET /api/health`: `{models_ready, encoder_ready, model_name, device, active_job, detail}`.

Job snapshot (`POST /api/jobs`, `GET /api/jobs/{id}`, start/cancel and each SSE message):

```json
{"id":"uuid","filename":"clip.mp4","status":"ready","metadata":{"width":1280,"height":720,"fps":30,"frame_count":300,"duration":10,"bytes":123},"frame":0,"processed_frames":0,"progress":0,"stage":0,"stage_step":0,"tracks":[],"logs":[],"error":null,"revision":1,"preview_version":0,"trace_count":0,"outputs":{}}
```

Status: ready/loading/running/encoding/completed/failed/cancelled. Logs: `{time:"HH:MM:SS",message,stage}`. Track: `{track_id,bbox,keypoints,history_length,pose_accepted,class_id,class_probability,probabilities,prediction_updated}`; missing prediction fields are null. Metadata duration is seconds. Frame is zero based. Progress is 0–1. `outputs` when completed maps `video`, `data`, `bundle`, `metrics` to relative download URLs.

`GET /api/jobs/{id}/trace`: `{samples:[{frame,time_seconds,width,height,tracks:[...]}]}`. Trace tracks extend the normal record with `history` (N×17×3; N≤50) and `future` (50×17×3 or null). Five LSTM blocks are consecutive chunks of 10 from actual future data. Trace is bounded to ≤60 snapshots and ≤3 people each. Empty `samples` is valid.

Original source: `/api/jobs/{id}/files/source` (browser-compatible preview if needed). Current annotated still: `/api/jobs/{id}/preview?v={preview_version}`. A source poster can be requested at `/api/jobs/{id}/files/poster`. Progress/log time is actual, process replay time is explanatory.

## Task 3 — React workbench and computation playback

Files: `web_demo/package.json`, Vite config, index.html, `src/` components/style/API hooks; root ignore rules allow source but ignore node_modules and dist.

- [x] Build the approved app shell, upload/control rail, original/result player, track inspector, stage strip and process page with local SVG primitives. Use small focused components.
- [x] Wire actual multipart upload/start/SSE/cancel/download and restore last job ID from localStorage. Handle empty/error/zero-track/missing models states without fabricated values.
- [x] Implement real trace selection, history/future skeletons, five forecast blocks, structural graph flow and class bars. Play/pause, stage seek and 0.5×/1×/2× must change real UI state; pause on hidden tab and reduce motion when requested.
- [x] Build with Vite, verify desktop and narrow layout, and no console/overlay errors.

## Task 4 — Integration, review and handoff

- [x] Install isolated dependencies; build frontend; run all Python tests and meaningful frontend checks.
- [x] Browser plugin is absent in this session, so use Playwright for upload → start → live state → completed video → process replay → download; also test cancel and invalid input.
- [x] Use real local model weights and a clearly identified test clip; decode exported MP4 and inspect JSONL/trace. Do not claim accuracy from this.
- [x] Independent review, fix important findings with regression checks, and compare accepted images against browser screenshots with view_image.
- [x] Document setup, launcher, output location and validation limits in `docs/WEB_DEMO.md`; link from README. Keep the localhost server available for user review.

## Execution Record

Approved design selected React/Python. Work is in the existing user workspace to retain their uncommitted camera changes; no automatic commits, stash or branch changes are part of this task.

Completed verification (2026-10-29): all four tasks implemented. Full Python suite: 64 tests, 59 passed and five historical Windows-weight checks skipped. Actual pinned ST-GCN/LSTM observer-on/off outputs matched exactly; all five forecast blocks were emitted. Vite production build passed. Playwright Chromium (Browser plugin unavailable) exercised upload, actual model analysis, H.264 playback, process navigation, play/pause/speed/step seek, ZIP download, refresh recovery, 768px layout and reduced motion; separate browser checks passed cancellation, malformed video and no-people input. No page script errors. A 75-frame clearly labelled synthetic video based on the official Ultralytics bus image was processed with actual models; all 75 exported frames decoded, all 75 JSONL records parsed, the final trace contains frame74 and future poses, and ZIP integrity passed. This is a functional check, not an accuracy evaluation.

Independent review findings were reproduced and fixed: launcher default runtime root was bound too early, and frame/track snapshots could lag the final frame because of JPEG throttling. Regression tests failed before and passed after the fixes. Browser testing additionally reproduced and fixed a React click-event-as-stage error. Final preview, final trace inclusion and visible result-poster behavior were checked. Typography and narrow layout were compared with both accepted concepts.

Local server remains running at http://127.0.0.1:8765 and the actual workbench was opened in a new Edge tab. Built distributions, model files, test videos and runtime artifacts remain excluded from Git.
