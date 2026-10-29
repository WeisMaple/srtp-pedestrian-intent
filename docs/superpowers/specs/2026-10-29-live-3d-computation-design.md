# Live Three-Dimensional Computation

The user requests live computation and three-dimensional animation, continuing the approved local React/Python workbench. Preserve the single-page video, skeleton/ID selection, predictions and downloads. Implementation choices below are the builder's decisions within that request.

## Approach

Use Three.js WebGL geometry with orbit/pan/zoom and raycasting. CSS perspective cannot represent selectable volume or spatial graph operations; a separate Blender movie cannot respond to current inference. The existing numerical inspector remains available for detailed arithmetic.

Use optional hooks in the actual video inference forward pass, enabled for the web runner. Do not launch a second model pass to manufacture live activity. Each event identifies its job, frame, timestamp, model/layer, eligible person IDs, native tensor shape and bounded actual samples. Browser motion interpolates presentation only; hardware operations finish at the captured timestamps.

## User experience

- Default calculation view is a spatial network. Overview leads into YOLO feature planes, history/time stacks, recurrent memory volumes, spatial skeleton edges, temporal windows and output scores.
- A live event stream updates layers while the video is being analysed. Show current frame and observed throughput. Insufficient history remains a warm-up state.
- Freeze pins a computation snapshot for orbiting and inspection while inference continues; resume jumps to newest data rather than replaying a backlog.
- Post-analysis explanation is explicitly a recorded/recomputed inspection. Never present elapsed animation time as model execution time.
- A selected voxel exposes actual value, tensor index/shape and data origin. The skeleton depth axis represents time or feature layout, not inferred 3D body coordinates.

## Constraints

Keep Python 3.9, default 50 observations and five 10-frame forecasts. Keep outputs numerically equivalent, no TensorFlow, no camera activation. Bounded telemetry and GPU geometry; cleanup hooks, streams and WebGL resources. Retain AGENTS.md unchanged. No commits or publication requested.

## Verification

Test actual-forward parity, event ordering/frame provenance, backpressure and reconnect, cancellation and recovery. Browser: upload/start -> live layer updates before completion -> freeze -> inspect/orbit -> resume -> terminal classification; verify errors, WebGL fallback and narrow layout. Record real measured timings instead of promising video-rate throughput.
