# Live camera display

User request: add a camera-driven live video view without computation diagrams.

Implementation choice: a fourth navigation destination, “现场视频”, with camera selection, explicit start/stop, a large annotated recognition frame, a small continuous camera preview, and per-person intent results. Capture video only, never microphone audio. Stop capture on navigation, tab hiding, disconnect or user stop. Frames stay in memory and are not exported or recorded.

Use browser `getUserMedia` so the chosen device belongs to the viewer. Stream one JPEG at a time over a same-origin WebSocket; wait for each result before capturing the next frame. Overlay results on the exact captured frame, and show actual processing rate/latency. The continuous preview stays visibly separate from the recognition frame. Permission denial, unavailable devices, busy inference and disconnection have recoverable messages.

Reuse YOLOPoseTracker, HistoryStore, optional configured PoseFilter, and BehaviorPredictor with existing final weights on CPU. Keep 50 accepted observations before the existing LSTM→ST-GCN prediction, with no computation telemetry. Reset invalid/discontinuous history and resolution changes. Each connection owns fresh tracking/history and holds the existing inference admission lock; disconnect releases it. Uploaded videos and historical calculations stay available after stopping the camera.

Alternatives considered: server OpenCV capture cannot select the browser's device; repeated uploaded clips reset identity and add encoding delay. A persistent browser stream best fits this feature. These are implementation choices, not user-specified protocol preferences.

Validation: unit and API tests for warm-up, gaps, empty frames, contention and cleanup; frontend tests for one outstanding frame, late permission results, stop and disconnection; real-weight video frames through the camera protocol; native browser start/stop and permission flow where hardware permits. Camera uses secure contexts (localhost qualifies), per [MDN getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia); explicit track cleanup follows [MDN stop](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/stop).
