# Annotate the Original 3D Network

**Goal:** Put actual numerical flow on the original 3D objects and automatically zoom to the current step. The user explicitly rejected an extra row of arithmetic objects.

**Design:** Remove the separate arithmetic section. Bind each calculation to the existing layer, joint, gate, feature voxel or classification bar. Exact native-index matches may highlight a cell; unsampled inputs are labelled at the owning layer's boundary, never assigned to a different sampled neuron. Draw contributions between these original anchors, attach coefficients to the connecting path, and attach accumulation/output labels to the original destination.

**Camera:** Follow each operator automatically by default, with a smooth framing transition. Terms within an operator do not cause a moving camera. Direct manipulation suspends camera following until resumed. Reduced motion snaps to the same framing.

**Scope:** Frontend only. Preserve AGENTS.md, existing predictions and uncommitted work. No new dependencies, commits, camera capture or pushes. One main 3D canvas; optional numeric ledger contains no second diagram.

- [x] Add tested native-index binding and data adapters for existing scene objects.
- [x] Replace separate arithmetic objects with attached labels, original-object highlighting and contribution paths.
- [x] Add automatic current-step framing, manual suspension and resume controls.
- [x] Verify numerical states, seven stages, automatic progress, manual orbit, reduced motion and narrow layouts in the browser; update documentation.
