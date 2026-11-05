# Intent explanation implementation plan

> Execution: native implementation, followed by independent code review.

**Goal:** Apply the existing request for detailed, interactive 3D calculations and replay to the intent stage.

**Design:** Keep per-frame original results available in a result view. Default to a computation view using the existing saved-history network response: channel-0 complete time/joint pooling example, all 256 pooled features, three independent weighted classifiers, and stable Softmax. Show source frame/person and separate original versus recomputed scores. Render real 3D matrices, classifier nodes and probability columns with explicit phases, manual seek, class/channel selection and bounded automatic playback. Missing values stay missing. No model or dependency changes.

**Approach:** Extend existing Three.js/React components. Reusing the old three unlabelled bars would not expose computation; adding a new chart library provides no necessary capability. The implementation choice is to reuse the current renderer and read-only API.

- [x] Add tested intent arithmetic adapters: all classes, pooling, stable Softmax, exact frame/person selection, partial live sampling and missing states.
- [x] Integrate completed computation mode with existing recorded history selection while preserving original per-frame result replay.
- [x] Add perspective intent scene and responsive DOM controls, formulas and scores; test fit bounds and playback clock ownership.
- [x] Validate real recorded values, build/tests, native Edge interaction, review fixes and documentation.

Review focus: original/recomputed score confusion; old-person/old-frame arithmetic; missing data becoming zero; auto/manual cursor conflicts; narrow camera and panel containment. Follow the repository's artifact/inference integrity constraints. No commit or publication requested.

## Validation — 2026-11-05

- `cd web_demo && npm test`: 100 passed, 0 failed, 0 skipped. Includes sampled-channel gaps, original/recomputed score isolation, all three classifier sums, extreme/tied Softmax, manual playback ownership, and perspective bounds at widths 296, 366, 720 and 1200 pixels.
- `cd web_demo && npm run build`: passed. Vite still reports the Three.js-containing bundle size advisory (951.26 kB before gzip); no dependencies added.
- `git diff --check`: passed.
- Real saved-history response, frame 151 / ID 11: 221 pooling inputs, 256 pooled channels and 256 products per class. Maximum classifier reconstruction error 1.836e-7; Softmax maximum error 2.868e-8. Original crossing score 78.35%; recomputed score 79.62%, separately labelled.
- Native Edge at `127.0.0.1:8765/#/corridor`: visually checked the full 3D scene, class selection pausing playback, channel 254 after seeking backward from the end, equal accumulation in both formula panels, original frame 151 → empty frame 199 → frame 151 recovery, and clicking the 3D crossing column to enter calculation mode on frame 151 / ID 11. The desktop scene stays within its panel; narrow camera containment is covered mathematically, not by a mobile screenshot.
- Independent review found two issues, both fixed and checked in the UI: original-mode 3D bar clicks now enter detail mode, and both accumulation displays share `flowState` cursor semantics.
- This change is frontend-only. No new inference run or backend test run was needed; model behavior and the default 50-observation pipeline remain unchanged. Generated media and the served build remain ignored.
