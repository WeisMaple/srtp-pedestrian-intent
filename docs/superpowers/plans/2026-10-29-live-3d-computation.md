# Live 3D Computation Implementation Plan

**Goal:** Display current inference outputs in an interactive 3D computation view.

**Architecture:** Optional inference hooks publish bounded, ordered SSE events. A React hook keeps the newest coherent state; a separately owned Three.js renderer updates sampled geometry without creating React nodes for every cell. The existing numerical views remain accessible.

**Tech Stack:** Python 3.9, PyTorch, FastAPI/SSE, React 19, Vite, Three.js.

**Spec:** ../specs/2026-10-29-live-3d-computation-design.md

## Global Constraints

Preserve 50-observation/five-round behavior and actual prediction values; never label replay as live. No camera, AGENTS.md edits, commits or uploads. Bound live buffers, sampled tensors and draw calls.

## Review Focus

Disconnected browsers resume without old-frame contamination; frozen displays remain stable while inference advances; fewer than 50 observations produce no invented behavior; WebGL loss offers a usable fallback; cancellation removes hooks and terminates streams.

## Tasks

- [x] Backend: optional live hook collector, video pipeline integration, bounded per-job buffer/snapshot, SSE replay/reset and terminal event; Python parity/ordering/cancellation tests plus real weights.
- [x] Renderer: actual sampled tensor volumes and graph/time geometry; orbit, focus, picking, numerical metadata, reduced motion, resize/context loss/disposal.
- [x] Integration: stream reducer/hook with job/frame/sequence isolation and freeze; default 3D view, live status and throughput, 2D numerical fallback; pure reducer tests.
- [x] Validate: build, relevant full Python/frontend tests, real upload/run/browser interactions, desktop/narrow screenshots, evidence video and documentation.

## Verified outcome

96 Python tests: 91 passed, 5 historical-weight skips. 32 frontend tests passed; production build passed with the documented lazy Three.js chunk-size notice. A real 75-frame CPU run preserves baseline JSONL and decoded output pixels. Chromium integration verified live data before completion, freeze/resume, all eight stages, orbit/reset, numerical mode, reload recovery, recorded recomputation, WebGL loss fallback and 768px layout with no script errors. Final actual-data browser job: `56ebc04f1b8d4b0cadc4da1d8e948129`. Screenshot/video artifacts remain ignored.

Additional fixes include terminal SSE delivery even when revision has not advanced, job-keyed record state, frozen terminal messaging, and collision-free sparse tensor placement at native time positions. The Playwright harness uses synchronous UI predicates and explicitly awaits API verification; this version does not repeatedly poll async predicates.
