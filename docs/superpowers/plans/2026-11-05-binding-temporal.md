# Binding and temporal convolution repair

Requested: fix exhibit overflow, expose YOLO skeleton-to-ID binding, and inspect temporal convolution in detail. Extends the approved blueprint design and existing visibility work.

- [x] Fit each 3D content group inside its exhibit; place titles in their own group and frame the selected exhibit responsively. Move navigation/controls outside the canvas.
- [x] Display the actual sampled source frame with staged bounding boxes, COCO17 joints and recorded ByteTrack ID associations. Synchronize video seeking to the numerical frame; identify the display as a replay of recorded associations, not unrecorded tracker cost calculations.
- [x] Select spatial versus temporal convolution, middle versus boundary output, block and person without fallback to another operation.
- [x] Show the complete temporal kernel for each input channel, time indices, zero padding, weights, products, channel sums and final accumulation. Support channel/term inspection and playback of a frozen channel window.
- [x] Verify geometry bounds, operation selection, actual local data, responsive browser behavior and numerical parity. Keep scripts/screenshots outside committed source.

The model, weights and inference remain the existing 50-observation LSTM → ST-GCN pipeline. No fabricated operands or matching costs.

## Validation

- `cd web_demo && npm test`: 79 passed, 0 failed/skipped. Includes exhibit bounds and camera framing, strict temporal selection, complete channel sums and real padding flags.
- `npm run build`: passed; existing Three.js bundle-size advisory remains. Accessibility heuristic: 46 files, 0 warnings. `git diff --check`: passed.
- Production browser checks at 1920, 1440, 1280, 768, 390 and 320 px: no document horizontal overflow, no overlap between header/stage/transport/desk, no runtime errors. Checked source/bbox/joints/ID phases, selected channels, 9-term playback and boundary zero padding. Temporal footer is visible on desktop; narrow kernels scroll within their own container.
- Actual 75-frame, 1280×720, 30 fps clip, task `8e0642479a334ceebfa658a8178d4b38`: completed on CPU, PyTorch 2.2.2 / Ultralytics 8.4.190. Observed 14 distinct live overlay frames aligned with source video time. JSONL: 75 records; render: 75 frames. Artifacts remain ignored under `artifacts/`.
- Stored temporal example: block 6, ID 1, 128 channels × 9 terms. All-term dot −0.150095, bias 0.00908, reconstructed −0.141015, captured −0.14099; the UI exposes the rounding discrepancy (~2.55e−5). No missing terms are silently treated as zero.
- Independent code review found no additional important defects. No backend code changed; full Python/real-weight unittest suites were not rerun.
- Visual revision: replaced edge-on YOLO planes with a contained front-facing module grid, enlarged numerical stage, moved controls out of the canvas, and made time-index/pre- and post-convolution semantics explicit.

Limits: replay shows recorded associations, not uncollected ByteTrack matching costs. Arithmetic covers complete representative outputs, not every output position. Video alignment was validated with constant-frame-rate MP4; variable-frame-rate and unsupported source codecs were not validated (decode failures have an explicit UI state).

Evidence: `/tmp/srtp-binding-temporal-check.cjs`, `/tmp/srtp-binding-temporal-check.json`, `/tmp/srtp-binding-1440.png`, `/tmp/srtp-temporal-boundary.png`, `/tmp/srtp-modules-fit.png`.
