import test from 'node:test';
import assert from 'node:assert/strict';
import * as temporal from '../src/domain/temporal.js';
import { latestCalculation } from '../src/domain/workbench.js';
import { normalizeCalculation } from '../src/domain/calculation.js';

const history = () => Array.from({ length: 50 }, () => Array.from({ length: 17 }, () => [10, 20, 0.9]));

test('automatic graph playback chooses an available person and block instead of waiting on an uncollected block', () => {
    const raw = (block, track) => ({ id: `${block}.${track}`, stage: 'temporal', track_id: track, inputs: [1], weights: [2], metadata: { block, boundary: false } });
    const live = { tracks: [{ track_id: 18 }, { track_id: 11 }], details: { stgcn: { 4: [raw(4, 11)], 5: [raw(5, 11)] } } };
    const pick = temporal.automaticGraphSelection(live, [], { block: 1, boundary: false, operation: 'temporal' }, 18);
    assert.equal(pick.trackId, 11);
    assert.equal(pick.block, 6);
    const retained = temporal.automaticGraphSelection(live, [], { block: 5, boundary: false, operation: 'temporal' }, 11);
    assert.equal(retained.block, 5);
    assert.equal(temporal.automaticGraphSelection({ ...live, tracks: [] }, [], { block: 1, operation: 'temporal' }, null), null);
});

test('completed records recover eligible people even when the last frame is empty', () => {
    const trace = { samples: [
        { frame: 140, tracks: [{ track_id: 7, history: history().slice(1) }] },
        { frame: 141, time_seconds: 14.1, tracks: [{ track_id: 11, history: history() }] },
        { frame: 151, time_seconds: 15.1, tracks: [{ track_id: 11, history: history() }] },
        { frame: 199, tracks: [] },
    ] };
    const choices = temporal.recordedGraphSelections(trace);
    assert.deepEqual(choices.map(s => [s.frame, s.track.track_id]), [[151, 11], [141, 11]]);
    assert.equal(choices[0].sample.frame, 151);
    trace.samples[2].tracks[0].history[3][2][0] = null;
    assert.deepEqual(temporal.recordedGraphSelections(trace).map(s => s.frame), [141]);
});

test('replayed temporal records preserve the selected frame, full products and boundary flags', () => {
    const arithmetic = { input_values: [[0,2,3],[0,4,5]], weights: [[9,2,3],[8,1,-1]],
        input_timesteps: [-1,0,1], input_length: 4, kernel_size: 3, input_channels: 2,
        timestep: 0, output_channel: 0, output_joint: 6, stride: 1, padding: 1, dilation: 1, bias: 0.5, actual: 12.5 };
    const result = { source: { job_id: 'j', frame: 151, track_id: 11, kind: 'recomputed_from_saved_history' },
        stgcn: { blocks: [{ index: 1, arithmetic: { temporal_boundary: arithmetic } }] } };
    const details = temporal.recordedGraphDetails(result);
    const raw = latestCalculation({ details }, 5, {}, 0, 1, 11, [], { operation: 'temporal', boundary: true });
    assert.ok(raw);
    const row = normalizeCalculation(raw);
    assert.equal(row.metadata.source_frame, 151);
    assert.equal(row.metadata.source_kind, 'recomputed_from_saved_history');
    assert.equal(row.track_id, 11);
    assert.equal(row.reconstructed, 12.5);
    assert.deepEqual(row.terms.map(t => t.padding), [true,false,false,true,false,false]);
    assert.deepEqual(row.terms[4].inputIndex, [1,0,6]);
    assert.equal(latestCalculation({ details }, 5, {}, 0, 1, 18, [], { operation: 'temporal', boundary: true }), null);
    assert.equal(latestCalculation({ details }, 5, {}, 0, 1, 11, [], { operation: 'temporal', boundary: false }), null);
});
