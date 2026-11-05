# Skeleton Spatiotemporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Show spatial and temporal convolution together on the selected person's real skeleton sequence.
**Architecture:** A pure display adapter binds same-frame arithmetic and poses; a Three.js scene renders skeletons and spatial/temporal edges; one React stage keeps both arithmetic panels visible. Existing production inference remains unchanged.
**Tech Stack:** React 19, Three.js, Vite, node:test, Python unittest.
**Spec:** `docs/superpowers/specs/2026-11-05-skeleton-spatiotemporal-design.md` (approved by user).

## Global Constraints

- Preserve 50 observations → LSTM future 50 → ST-GCN.
- No synthetic poses, no stale-frame arithmetic, no invented graph edges.
- Real-time views consume existing forwards; replay uses existing `/network`.
- Keep both dimensions visible; target 1440×900 and 1920×1080, narrow screens scroll.
- Preserve the current feature branch and local demo runtime; user requested direct implementation.

## Review Focus

- Changing source/person during replay must clear old skeletons and values (Task 1).
- Older cached detailed rows must not be attached to a current skeleton (Task 1).
- Downsampled blocks must label feature time and source receptive range separately (Task 1).
- Padding must never create a duplicate skeleton (Task 1).
- Missing/WebGL-unavailable data must preserve usable arithmetic and controls (Task 2/3).

### Task 1: Same-source display data

**Files:** Create `web_demo/src/domain/spatiotemporal.js`, `web_demo/tests/spatiotemporal.test.js`; modify `web_demo/src/domain/temporal.js`.
**Interfaces:** `spatiotemporalModel({snapshot, result, track, block, boundary, structure})` returns source, history, forecasts, spatial/temporal normalized calculations, edges, temporal slots and receptive-field mapping. `temporalMapping(blocks, block)` returns input stride and radius from real layer descriptors.

- [x] Write tests for same-source pairing, mismatch rejection, actual adjacency, downsampling and padding, and replay metadata.
- [x] Run `node --test tests/spatiotemporal.test.js` and observe failure.
- [x] Implement strict selection from current event envelopes or verified replay; augment replay row metadata with true target/graph/time dimensions.
- [x] Run the targeted tests and existing temporal/replay tests; expect all pass.

### Task 2: Unified skeleton stage

**Files:** Create `web_demo/src/components/SpatiotemporalComputation.jsx`, `web_demo/src/scene/spatiotemporalScene.js`, `web_demo/src/styles/spatiotemporal.css`; modify `web_demo/src/components/Corridor.jsx`.
**Interfaces:** `createSpatiotemporalScene(host, callbacks)` exposes `setData(model)`, `setSelection(selection)`, `reset()`, `dispose()`. React stage consumes the Task 1 model and existing job/state/actions.

- [x] Build one responsive stage with provenance, selectable time slots, spatial neighbors, two arithmetic panels, and inline full terms.
- [x] Render only actual poses and graph edges, with target-joint temporal links and real padding slots; tie picking to existing arithmetic terms.
- [x] Remove the exclusive operation selector for step 5 and redundant large ledger/transport for that step, keeping other steps unchanged.
- [x] Verify scene geometry and interaction model through targeted node tests; run front-end tests and build.

### Task 3: Real-data verification and delivery

**Files:** Update `docs/WEB_DEMO.md` and design status. Temporary screenshots/scripts outside tracked source.

- [x] Inspect the built demo at desktop and narrow viewports, exercise block/window/term/record switches and check console.
- [x] Verify real `/network` arithmetic, pose source and deep-layer mapping with existing final weights and sample job.
- [x] Run appropriate regressions, recording skipped historical-weight tests if the full Python suite is run.
- [x] Obtain a fresh final code review, address important findings, rebuild and leave demo running.

## Execution record

- Baseline: existing frontend tests passed before implementation.
- Ruling: implement in existing `blueprint-corridor` feature checkout for the running local demo, honoring “按此设计直接实现”; no branch publication or model change.
- Browser plugin is not listed; CUA browser acquisition earlier returned “The browser access host is unavailable”. Use available local verification tooling, and report any rendering blocker honestly.

- Task 1: complete. Same-source, stride, padding, real graph and reconnect tests passed; the initial missing-adapter test and reconnect regressions were observed failing before the fixes.
- Task 2: complete. Unified Three.js skeleton scene and simultaneous spatial/temporal arithmetic replaced the exclusive step-5 view; compact controls preserve a single desktop viewport.
- Task 3: complete. Frontend 119/119 passed; Python 160 total, 155 passed, 5 historical Windows reproduction-weight skips; production build and git diff whitespace check passed. Existing Three.js bundle-size warning remains.
- Fresh review found two issues: detail-only live snapshots lost arithmetic; ready views hid live/recomputed provenance. Both fixed. Reconnect server/client regressions passed after failing first; browser checks verified the explicit record label.
- Browser evidence: native Edge plus isolated Playwright Chromium, macOS arm64. 1440×900 and 1920×1080 show both panels and footer in the initial viewport; 390×844 retains both panels in the same scrollable page without horizontal page overflow. Deep block, boundary padding, channel, ledger and person-record switching succeeded.
- Final browser run observed one transient /network HTTP 409 during resource admission; existing retry recovered automatically and all subsequent interactions passed. No JavaScript exceptions. Evidence: /tmp/srtp-spacetime-qa.json and /tmp/srtp-spacetime-*.png.
- Actual final-weight /network replay, CPU: all ten blocks and both windows checked by reviewer; maximum representative reconstruction error approximately 4.63e-7. No production model algorithm was changed.
- Local service restarted to load the provenance fix, listening at http://127.0.0.1:8765.

## Final readability revision

- User feedback replaced the initial Three.js scene with `SpatiotemporalGraph.jsx`: a shared-scale SVG timeline, enlarged spatial neighborhood, named joints, coordinate inspection, and larger arithmetic text. The original `spatiotemporalScene.js` is no longer used or included.
- Added regression coverage for flat pose conversion, wide-pose column overlap, nearest-coordinate picking, low-score point visibility, live-frame zoom retention, and temporal-only layout. Final frontend suite: 125 tests passed.
- Default rendering omits forecast scores at or below 0.2 and their incident edges; an explicit raw-data toggle restores them. The actual model input and arithmetic remain unchanged. The walk-video head regression was checked at source frame 141, track 11, forecast position 25.
- Final user scope is desktop only; the latest focused browser check used 1440×900. Production build passed. The user subsequently requested committing and pushing this work to the current branch.
