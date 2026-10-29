# Explicit Numerical Dataflow Implementation Plan

**Goal:** Make the accepted 3D workbench explain which operands contribute to a chosen result, their exact data flow, running accumulation and output.

**Architecture:** Keep the live network overview. Add actual-forward representative calculations to telemetry and normalize both live and saved numerical records to a common calculation model. A 3D operand theatre and its readable ledger share one deterministic cursor. Pinning a calculation is separate from inference progress.

**Tech Stack:** Existing React, Three.js, Python 3.9 / PyTorch. No new dependencies.

## Design decisions

The user's approved 3D presentation remains. Merely increasing glow/duration would not expose operands; replacing it with a flat diagram would discard the accepted presentation. Add an integrated spatial numerical inspector with explicit states, defaulting to data flow when data is available. These implementation choices are the builder's decisions within the requested revision, not additional user approvals.

Show a complete sum for one explicitly named representative output, not every network scalar. Distinguish current input, accumulated terms, pending terms, structurally disconnected terms (original adjacency zero), connected zero contributions and unrecorded values. A learned coefficient of zero does not remove an existing structural connection. Never label unsampled values unused. All terms count, including those outside the visible page.

## Shared contract

A calculation has `id, stage, title, operation, track_id, target, inputs[], weights[], bias, actual, activation, formula, input_indices[], connected[], padding[], metadata`. Arrays are aligned and retain every term of the representative sum. `activation` defaults to identity. `actual` is independently observed. Additional `preactivation` is permitted. Live events carry `calculations: [...]` alongside their existing provenance; absent fields never become invented zero inputs. No inference re-execution for live arithmetic.

## Tasks

- [x] Backend: bounded representative actual-forward convolution, normalization, decoder, denormalization, graph aggregation, temporal convolution, pooling, classifier and softmax context. Preserve prediction parity and cleanup; tests verify full sum, masks and budgets.
- [x] Frontend model: live and saved calculation adapters, deterministic term cursor, semantic states, sums/completeness tests.
- [x] 3D flow: sizeable labelled operand/weight/product/accumulator/output nodes, directional movement carrying actual values, selectable membership cells, shared cursor and readable formula/ledger. Orbit/reset plus reduced motion.
- [x] Integration: accepted network structure remains available; introduce explicit data-flow toggle, live source and pin/resume, stage/ID controls; empty and partially sampled states remain honest.
- [x] Verification: tests/build, real-weight parity, browser interactions, numeric assertions, desktop/narrow screenshots and updated documentation.

## Constraints

Do not modify AGENTS.md. Preserve existing uncommitted work. No camera, commits or pushes. Do not imply milliseconds of actual compute equal pedagogical playback duration. No fabricated neural values or exact unrecorded dependencies. User already selected React/Python, all stages and 3D; this revision implements their specific feedback without restarting architecture approval.
