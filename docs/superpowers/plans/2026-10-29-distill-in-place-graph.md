# Distill-inspired In-place Graph Explanation

**Goal:** Explain real ST-GCN message passing on the existing perspective 3D skeletons and feature maps, with selectable receiving joints and automatic camera focus.

**Source:** https://distill.pub/2021/gnn-intro/ — study neighborhood selection, linked attributes, and gather/aggregate/update explanations. The implementation remains this project's actual projection → weighted adjacency sum → temporal/residual computation; no generic GNN operations are invented.

**Constraints:** Preserve the original scene, frame/person provenance, manual-camera override, reduced-motion behavior, and independent measured outputs. No second arithmetic diagram, new dependency, camera capture, or AGENTS.md change. Work in the existing authorized workspace without committing unrelated work.

**Design:** Three.js owns one perspective scene. Existing joints carry projected features; signed edge messages connect to existing output cells. DOM annotations are projected onto those objects. A compact in-scene control strip selects the receiving joint, the calculation phase, and neighborhood emphasis. Sequential timing is explanatory, not hardware timing. Gray context preserves orientation. Positive/negative contributions have signs as well as colors; zero and disconnected remain distinct. A keyboard-accessible selector provides the same operation as clicking a joint. Mobile uses the same scene with fewer simultaneous labels, no extra column.

**Data:** Keep the existing 51 complete operands. Add a bounded sparse graph slice to live spatial records: original nonzero topology edges with effective coefficients and the 17 independently captured outputs at channel 0 and one timestep. Recorded inspection can use its already captured adjacency and matching full joint output slice. Old live records without the slice retain their original target. Selecting a target reconstructs all 51 terms against that target's measured output, with no extra inference or mutation.

**Cost / state:** Existing instance counts stay unchanged; at most 51 message paths, normally under 10, and a small selected-neighborhood label set. Rebuild geometry only when the calculation or scene changes. Reuse playback state; selected target is scoped to calculation/frame/person. No URL/persistence changes. Numeric ledger and existing 2D mode remain available.

- [x] Capture and validate the sparse graph slice; test all 17 outputs, zero learned weights, prediction parity, and event budgets.
- [x] Add pure target-selection and neighborhood-summary adapters; test missing slices, direction, partitions, frame isolation, and complete sums.
- [x] Integrate in-scene controls, source/message labels, signed paths, partition sums, and receiver highlighting with the existing camera.
- [x] Run frontend tests/build and appropriate Python checks. Verify actual data, click/keyboard target switching, zoom/manual override, desktop/mobile, and reduced motion in the browser.
- [x] Document the source, scope, validation, and produce reviewable visual evidence.

**Validation:** 63 frontend tests; 101 Python tests (96 passed, 5 historical-weight skips); production build; real-weight browser checks across 17 targets, original 3D click picking, camera modes, recorded mode, mobile, and reduced motion. The 40-second local video is `artifacts/demo/distill_3d_message_passing_demo.mp4`. No extra model forwards were introduced.
