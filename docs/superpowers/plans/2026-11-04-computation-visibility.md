# Computation Visibility Implementation Plan

**Goal:** Make the existing six-step computation readable without discovering 3D picking targets.

**Spec:** `docs/superpowers/specs/2026-11-04-blueprint-corridor-design.md` (approved); this is a focused repair of its numerical desk, explanatory states and navigation.

**Architecture:** Keep React/Vite, the existing live stream and inference unchanged. Add a persistent explanation desk with actual stage metrics and an arithmetic inspector. Reuse the ledger and selected track/module/block. No dependencies or generated product data.

## Constraints and review focus

- Preserve the default 50-observation → five LSTM rounds → ST-GCN pipeline.
- Missing values remain missing; old representative arithmetic is labelled as the latest collected sample, not current-frame computation.
- Empty live tracks must not resurrect people from an older job snapshot.
- Pause/follow controls retain their existing snapshot semantics. Manual next-step navigation works while narration is paused.
- Empty jobs, warm-up, unavailable module detail, disconnection and completed jobs must explain their state.
- The explanation and numerical controls remain accessible when WebGL fails and at narrow widths.

## Tasks

- [x] Add regressions for paused next-step navigation, current-frame calculation precedence and selected-track stage metrics.
- [x] Implement stage metrics and a persistent input → operation → output desk for all six steps.
- [x] Expose the current arithmetic, sum, bias, activation, actual output and reconstruction error; add direct term/phase navigation and an explicit detail button.
- [x] Wire stream status/retry, live frame context and a visible workbench process entry; preserve empty live tracks.
- [x] Repair scrolling and responsive layouts; keep the stage visible above the desk.
- [x] Run frontend tests/build and browser checks with actual local weights/video, warm-up, populated arithmetic, pause/follow, step navigation, disconnection, empty state and desktop/mobile widths.

Execution: directly in the current `blueprint-corridor` branch. Browser plugin is unavailable; use the already installed Playwright for rendered checks. Test scripts and screenshots stay in `/tmp`.

## Verification and review

- `npm test`: 75 passed, zero skipped. `npm run build`: passed; the existing Three.js bundle still exceeds Vite's 500 kB advisory threshold.
- Accessibility heuristic: 42 source files scanned, zero warnings after labelling the hidden video file input. Keyboard focus moves into the nonmodal inspector and returns to its trigger on Escape.
- Playwright / Chromium: actual production build at `127.0.0.1:8765`, CPU, Python 3.9.25 / PyTorch 2.2.2, `weights/final`, `artifacts/demo/demo_input.mp4` (75 frames, 1280×720, 30 fps).
- Production task `06061190f7b147bc93677230d0189ae3`: 75 frames, 300 pose observations, 103 classification updates. JSONL contains frames 0–74; saved H.264 video decodes and reports 75 frames at 1280×720. This checks integration, not model accuracy.
- Checked 1920×1080, 1440×900, 1280×800, 768×1024, 390×844 and 320×640; no page horizontal overflow. Verified warm-up, actual YOLO/LSTM/classifier arithmetic, ledger/detail controls, paused manual next, snapshot freeze while backend advances, record recomputation, reduced motion and WebGL loss. A separate aborted-stream check verifies visible reconnect state/retry.
- Visual revision: moved the detail button above the arithmetic, retained the last completed numerical frame to prevent false zero-person flashes, and made the lens/desk scroll and reflow on narrow screens.
- Independent read-only review found two issues, both fixed with failing-then-passing regressions: an explicitly absent person could select old cached arithmetic; arithmetic inspection did not pause the lecture. Existing wildcard lookup remains available to callers that omit a track ID, while the UI passes explicit null for no person.
- Remaining limits: old tasks may not retain LSTM/classifier operands across refresh; missing operands are explained rather than fabricated. No Python inference implementation changed, and the Python full suite / optional real-weight unit tests were not rerun; real inference was exercised through the browser.
