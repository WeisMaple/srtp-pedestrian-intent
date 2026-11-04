import test from 'node:test';
import assert from 'node:assert/strict';
import { advanceNarration, corridorReducer, createCorridorState } from '../src/app/useCorridor.js';

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
