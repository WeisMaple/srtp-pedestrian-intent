import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import TestRenderer from 'react-test-renderer';
import { advanceNarration, corridorReducer, createCorridorState, useCorridor } from '../src/app/useCorridor.js';

const { act } = TestRenderer;

test('selecting a step pauses auto narration and resets the beat', () => {
    const state = createCorridorState();
    const next = corridorReducer(state, { type: 'step', step: 4 });
    assert.equal(next.step, 4);
    assert.equal(next.beat, 0);
    assert.equal(next.paused, true);
});

test('narration advances only when the step is ready', () => {
    const state = { ...createCorridorState(), step: 2, ready: false };
    assert.equal(advanceNarration(state, { snapshot: {}, job: {} }).beat, 0);
    const ready = { ...state, ready: true };
    assert.equal(advanceNarration(ready, { snapshot: {}, job: {} }).beat, 1);
});

test('pause freezes all cursor actions that advance time', () => {
    const state = corridorReducer(createCorridorState(), { type: 'pause' });
    assert.equal(state.paused, true);
    assert.equal(state.frozen, true);
    assert.equal(corridorReducer(state, { type: 'next' }).beat, state.beat);
});

test('step navigation pauses narration without freezing live values', () => {
    const state = corridorReducer(createCorridorState(), { type: 'step', step: 5 });
    assert.equal(state.paused, true);
    assert.equal(state.frozen, false);
});

test('follow latest clears the freeze and resumes live values', () => {
    const paused = corridorReducer(createCorridorState(), { type: 'pause' });
    const resumed = corridorReducer(paused, { type: 'play' });
    assert.equal(resumed.paused, false);
    assert.equal(resumed.following, true);
    assert.equal(resumed.frozen, false);
});

test('camera interaction pauses narration without resuming a frozen snapshot', () => {
    const frozen = corridorReducer(createCorridorState(), { type: 'pause' });
    const camera = corridorReducer(frozen, { type: 'camera' });
    assert.equal(camera.paused, true);
    assert.equal(camera.frozen, true);
    assert.equal(camera.following, false);
});

test('manual beat stepping pauses narration and clamps within the step', () => {
    const state = { ...createCorridorState(), step: 2, beat: 1 };
    assert.equal(corridorReducer(state, { type: 'beat-step', delta: 1 }).beat, 2);
    assert.equal(corridorReducer(state, { type: 'beat-step', delta: -1 }).beat, 0);
    const clamped = corridorReducer({ ...state, beat: 0 }, { type: 'beat-step', delta: -1 });
    assert.equal(clamped.beat, 0);
    assert.equal(clamped.paused, true);
});

test('narration stops at the final beat of step six instead of looping', () => {
    const lastBeat = { ...createCorridorState(), step: 6, beat: 1, ready: true };
    const final = advanceNarration(lastBeat, {});
    assert.equal(final.step, 6);
    assert.equal(final.beat, 2);
    assert.equal(final.paused, false);
    const stopped = advanceNarration(final, {});
    assert.equal(stopped.step, 6);
    assert.equal(stopped.beat, 2);
    assert.equal(stopped.paused, true);
    assert.equal(stopped.following, false);
    assert.equal(advanceNarration(stopped, {}), stopped);
});

test('auto narration advances while snapshot object identity changes', async () => {
    const timers = new Map();
    let nextTimer = 0;
    globalThis.window = {
        setTimeout(callback, delay) { nextTimer += 1; timers.set(nextTimer, callback); return nextTimer; },
        clearTimeout(timer) { timers.delete(timer); },
    };
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const job = { metadata: { frame_count: 75 } };
    const live = { frame: 1, tracks: [{ track_id: 1, keypoints: [[1, 2, 0.9]] }], nodes: {} };
    let api;
    function Probe(props) { api = useCorridor(props); return null; }
    let renderer;
    try {
        await act(async () => {
            renderer = TestRenderer.create(React.createElement(Probe, { job, live, trackId: 1 }));
        });
        const firstTimer = [...timers.keys()].at(-1);
        await act(async () => { timers.get(firstTimer)(); });
        assert.equal(api.state.beat, 1);
        const secondTimer = [...timers.keys()].at(-1);
        // Every live publish replaces the snapshot object without changing the
        // beat; the pending timer must survive that identity change.
        await act(async () => {
            renderer.update(React.createElement(Probe, {
                job: { ...job }, live: { ...live, frame: 1 }, trackId: 1,
            }));
        });
        assert.ok(timers.has(secondTimer), 'timer was reset by a snapshot identity change');
        await act(async () => { timers.get(secondTimer)(); });
        assert.equal(api.state.beat, 2);
    } finally {
        await act(async () => { renderer?.unmount(); });
        delete globalThis.window;
        delete globalThis.IS_REACT_ACT_ENVIRONMENT;
    }
});
