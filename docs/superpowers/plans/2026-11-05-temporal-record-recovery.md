# Recover temporal calculations after analysis

Root cause: the completed walking clip has an empty last sampled frame, while valid history survives at frames 141 and 151 (ID 11). The frontend requires a currently visible person and defaults to block 1; limited live details only retain blocks 5 and 6. No more live events will arrive after completion.

- Use the existing recorded-history network inspection API for completed tasks. Select only saved, finite 50×17×3 histories; expose frame/person selection and mark every displayed result as a replay.
- Extend the existing read-only temporal observer with a boundary output, retaining all channels, real padding and independent PyTorch outputs. Do not change model inference.
- Make all ten replayed blocks available, preserve strict identity/block/window selection, clear stale responses and expose loading, no-history and retry states.
- Verify the actual new video, refreshed completed tasks, block/window changes, numerical reconstruction and existing tests. Preserve prior uncommitted changes.

## Live and 3D follow-up

User requests live visible results, automatic switching, and detailed 3D convolution; research references: CNN Explainer (https://arxiv.org/abs/2004.15004), TensorSpace (https://tensorspace.org/html/docs/basicModel.html), Distill (https://distill.pub/2021/understanding-gnns/).

Reading path: source frame/person → available convolution block → input window and weight slices → highlighted product → all-channel sum/bias/output. Default automatic selection skips unavailable blocks; manual selection pauses automatic switching. Completed tasks replay saved histories with explicit provenance.

Renderer contract: one Three.js perspective scene for the temporal stage; time positions run left-to-right, a labelled slice of input channels forms stepped 3D layers, and separate input/weight/product/output objects show computation. No invented tensor cells. At most 4×9×2 cells plus paths and selected values; other channels remain available through selection and the complete numeric ledger. Three.js owns only the scene; React owns labels, controls and exact values. Animation is a teaching clock, not measured inference duration. DPR capped at 1.5, pause when hidden, reduced-motion gets stepped selection and immediate camera changes. WebGL loss leaves the numeric ledger available. Orbit, reset, term selection and camera presets have DOM alternatives. Desktop and portrait use the same values with different framing.

QA: numerical padding/parity; auto/manual state transitions and availability; stale/cross-job responses; live and completed real-video paths; 3D pick/camera/reset; 320–1440 px containment and fallback. No new runtime dependency.

## LSTM follow-up and verification

The user explicitly requested stronger 3D presentation and an LSTM fix. The five progress bars are replaced with a perspective LSTM cell: parallel gates, elevated memory path, actual gate/c/h values on solid cube faces, and phase-linked tube/particle paths. Automatic playback traverses 50 time steps and then enters ST-GCN. Manual camera, phase and time selection pause automatic playback. The detailed gates are the first layer / unit 0 / first round reconstructed using all recurrent hidden units; the other four rounds are not assigned those gate values. Live mode shows actual available inputs/hidden samples and forecast-round completion; full gate values appear only after labelled saved-history replay.

The first visual pass revealed clipped arithmetic below the fixed-height stage, tiny time-index labels, and overly flat geometry. The stage now grows with its content, the active equation precedes the canvas, labels use tightly sized textures, and actual 3D layers use a perspective camera with a bounded sweep. Camera fitting covers occupied cells/labels at the sweep extremes, rather than shrinking into a large empty bounding box.

Validation (macOS, CPU, torch 2.2.2, ultralytics 8.4.190):

- `npm test`: 91 frontend tests passed, 0 skips. Includes live-to-manual selection retention, clock ownership, missing operands, source identity/retry, LSTM memory arithmetic/decoder, and perspective containment at canvas widths 296/366/720/1200 through five sweep positions.
- `npm run build`: passed. Existing bundle-size advisory remains (Three.js bundle).
- `.venv/bin/python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v`: 150 tests, 145 passed and 5 skipped because historical machine-specific weights are absent.
- Actual saved-history API: 10 blocks × middle/boundary windows; response 2,621,414 bytes. Largest convolution reconstruction error < 2.5e-7. All 50 LSTM gate/c/h steps reconstruct within 1.1e-7; actual PyTorch final h/c comparison is 1.91e-6. These are numerical consistency checks, not accuracy measurements.
- New source video: OpenCV official `samples/data/vtest.avi`, trimmed seconds 10–30 to `artifacts/demo/pedestrians_walk_20s.mp4`; 200 frames, 10 fps, 768×576. Browser job `dc6f2483b10f40c78908e496bc4ab3bf` completed all frames, 4 IDs and 19 classification updates. JSONL, metrics and rendered preview inspected; media remains ignored.
- Native Edge visual check observed history 41/50 at frame 131, automatic live block 9 / ID 11 arithmetic at frame 145, then completed replay of frame 151 / ID 11 with all ten blocks and boundary switching. LSTM 3D gate values, light paths, time stepping, decoder and five-round counts render. Manual time selection pauses playback. Full lower arithmetic is reachable by scrolling; no fixed-height cut-off.
- `git diff --check`: passed. Independent review found and verified fixes for auto-selection retention, competing clocks, null-to-zero accumulation, recorded LSTM round counts and mobile camera containment.

Limits: live values are rate-limited samples; replay uses saved 3-decimal skeleton histories and is labelled accordingly. LSTM detailed gates cover one representative layer/unit/round. Narrow-camera containment is tested mathematically; native Edge visual verification used the desktop window. WebGL failure retains DOM values, but context-loss was not forced in this browser check.

## Source-frame replay repair

Follow-up report: “无法回看啊”. Reproduced in the completed job: RecordLens showed source frame 0/module 0 while the main pose view remained on terminal frame 195/module 3. The transport range only changed lecture beats. Neither control selected an actual historical video/pose snapshot.

- Added completed-job replay from the existing per-frame JSONL artifact. The top timeline supports seek, exact frame number, previous/next frame, play/pause, restart, and first detected person. Empty frames remain empty; no live-tail poses are reused.
- Shared the selected source frame between the video, skeletons, IDs, class scores and controlled YOLO RecordLens. Source-stage replay never borrows live arithmetic. Debounce CPU recomputation during scrubbing and defer it while video playback runs; discard stale frame/module responses.
- Network replay chooses the closest saved eligible history for the effective displayed person, then commits both source frame and person. Returning to video retains that frame. LSTM time controls now sit above the 3D scene, with explicit previous/next time-step buttons. Manual selection pauses automatic teaching.
- Separate recorded narration pause from live-stream freeze. Completed narration readiness uses historical data. Decoder acknowledgements gate video advancement so a slow seek cannot leave the overlay perpetually chasing a newer frame.

Validation: 95 frontend tests pass, 0 skips; production build passes with the existing Three.js chunk-size advisory. `git diff --check` passes. Native Edge verified frame 140→151 through RecordLens with matching main video/pose/rail, 151→150 via previous-frame control, continuous playback through frame 199 with automatic stop, 199→140 seek, source 140→network record 141, network record 151→source 151, and LSTM time 50→49→48 with paused, changing gate/c/h values and 3D geometry. Independent review found the identity, recorded pause and slow-decoder cases above; fixes were applied. This follow-up changes only frontend replay; it does not rerun or alter model inference. Replay is available after a job finishes; network details are limited to saved complete histories.
