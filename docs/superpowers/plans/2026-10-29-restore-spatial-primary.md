# Restore the Spatial Network as the Primary View

Superseded by `2026-10-29-annotate-original-network.md`: the user rejected the separate arithmetic section and requested annotations on the original objects with automatic zoom.

**Goal:** Restore the user's accepted, rotatable 3D network without losing numerical tracing.

**Design:** The user's correction makes the hierarchy explicit: spatial network first, numerical detail second. The existing perspective camera, volumetric feature lattices and temporal skeletons remain the main scene. A labelled representative arithmetic section shares that scene and cursor; it is an explanatory arrangement of actual operands, not invented edges between sampled network neurons. The flat arithmetic inspector opens below the network and never replaces it.

**Architecture:** `CalculationWorkbench` owns one calculation/cursor and pins the associated scene context. `NeuralScene3D` receives both the scene context and the current arithmetic state. `NeuralScene` retains the network and adds a small 3D operand section with selectable source/weight volumes, value-carrying packets, sum and output. No backend or prediction change.

**Constraints:** Keep AGENTS.md untouched, preserve existing work, no dependencies, camera use, commits or pushes. Missing operands stay missing. A disconnected term must not animate as a contribution. Pause/seek and a selected frame remain deterministic. Reduced motion avoids movement.

- [x] Restore the network on initial load and all stages; keep the full numerical inspector expandable below it.
- [x] Add real operand/weight volumes and numerical flow to the same perspective scene; retain orbit, reset and point inspection.
- [x] Pin scene context with each calculation so old numbers cannot be combined with a newer frame.
- [x] Verify tests/build and browser default view, orbit/reset, all stages, details expansion, numerical states and narrow layout.
