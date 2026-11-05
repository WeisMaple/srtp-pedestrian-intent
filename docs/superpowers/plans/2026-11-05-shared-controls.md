# Shared Browser Controls Implementation Plan

User clarified: unify the appearance and interaction of native controls across the whole app, not browser API wrappers.

**Design choice:** Keep the existing blue/cyan visual language. Add a shared controls layer for buttons, a portal-rendered select-only combobox, styled native range/checkbox/number semantics, progress, disclosures, file-picker entry and a custom video player. Keep inference and camera behavior intact. No new dependency. Browser/OS permission and file dialogs remain platform-owned.

**Execution:** Native implementation, then independent review. Preserve existing uncommitted work; no commit requested.

- [x] Add keyboard/bounds tests; implement shared controls and menu positioning without clipping inside scroll panels.
- [x] Replace all app control instances, including camera controls, model/module selectors, replay sliders/numbers, disclosures and native video control bars.
- [x] Verify tests/build; native Edge check dropdown keyboard/Escape/click, numeric/range controls, video play/seek/speed/fullscreen and camera lifecycle. Review fixes and document limitations.

Review focus: keyboard events leaking into the corridor shortcuts; open menus clipped by panels or fullscreen; selected values committing on mere hover; form controls losing labels/disabled behavior; source changes or failed play promises leaving misleading video state; camera or replay state being changed by a purely visual refactor.

Reference behavior: [WAI select-only combobox](https://www.w3.org/WAI/ARIA/apg/patterns/combobox/examples/combobox-select-only/). Tests: `cd web_demo && npm test`; build: `npm run build`; `git diff --check`.

## Verification — 2026-11-05

- Frontend: 111 tests passed, 0 skipped. Vite production build passed (105 modules, 973.00 kB JS before gzip); existing bundle-size advisory remains. `git diff --check` passed. This controls change does not modify Python inference.
- Native Edge desktop: custom camera menu supports arrow/Enter selection; Escape cancels; 24-option YOLO menu scrolls to End within its viewport. Replay source input and stepper changed frame 141→142 with aligned video and scores. LSTM slider selected time step 26 and paused auto playback; Space on its disclosure expanded actual weighted terms without triggering corridor playback.
- Result video: play/pause, seeking to 10.1 s, 2× and 0.5× selection, Tab confirmation, fullscreen entry/exit all verified. The menu remains visible inside fullscreen and flips above the bottom controls.
- Actual camera: custom start/stop and skeleton checkbox work. Reconfirming the current device keeps the running frame counter advancing (35→69), preserving session history; stopped the test session at completion.
- Independent review found redundant select change callbacks and stale asynchronous playback errors. Fixed both and added regression coverage for equality, source replacement, pause, unmount and real playback failure. Follow-up review found no blockers.
- Raw videos in camera and synchronized frame exhibits intentionally retain their externally controlled media elements, with no browser-native control bar. Native form controls now occur only inside the shared UI implementation. OS file dialogs and camera permission prompts remain native. Mobile/touch and other browser engines have not been manually inspected; responsive layouts and nonblocking touch scrolling are implemented.
