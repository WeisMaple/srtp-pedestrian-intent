import test from 'node:test';
import assert from 'node:assert/strict';
import { lstmDisplayModel, recordedMemoryNodes } from '../src/domain/memory.js';
import { latestCalculation } from '../src/domain/workbench.js';
import { normalizeCalculation } from '../src/domain/calculation.js';
import { processMetrics } from '../src/domain/process.js';

test('LSTM display preserves real gates, recurrent contributions and memory arithmetic', () => {
    const step = { t: 2, f: .5, i: .25, g: -.4, o: .8, c_previous: .6, c: .2,
        h: .8 * Math.tanh(.2), input_values: [1,2], previous_hidden: [0,.3],
        input_dot: [1,2,3,4], recurrent_dot: [.1,.2,.3,.4], bias: [.01,.02,.03,.04] };
    const result = { source: { frame: 151, track_id: 11 }, lstm: {
        gates: { steps: [step], layer: 1, round: 1, unit: 0, validation_max_error: 1e-7 },
        rounds: [{ index: 1, forecast_poses: [[[1,2,.9]]] }],
    } };
    const model = lstmDisplayModel(null, result, 11, 0);
    assert.equal(model.gates, step);
    assert.equal(model.retained, .3);
    assert.equal(model.written, -.1);
    assert.ok(Math.abs(model.reconstructedCell - .2) < 1e-12);
    assert.equal(model.gateTerms[1].z, 2.22);
    assert.equal(model.forecasts.length, 1);
    assert.equal(lstmDisplayModel(null, result, 18, 0).gates, null);
    step.i = null;
    assert.equal(lstmDisplayModel(null, result, 11, 0).reconstructedCell, null);
});

test('live LSTM sampling selects the requested person and never invents unobserved gates', () => {
    const snapshot = { nodes: {
        'lstm.1': { track_ids: [7,11], tensors: { hidden: { axes: ['layer','track','hidden'], values: [[[1,2],[3,4]]] } } },
        'lstm.1.forecast': { track_ids: [7,11], tensors: { future: { axes: ['track','time','joint','coordinate'], values: [[[[5,6,.9]]],[[[8,9,.8]]]] } } },
    } };
    const model = lstmDisplayModel(snapshot, null, 11, 0);
    assert.equal(model.gates, null);
    assert.deepEqual(model.hidden, [3,4]);
    assert.deepEqual(model.forecasts[0].poses, [[[8,9,.8]]]);
    assert.deepEqual(lstmDisplayModel(snapshot, null, 99, 0).hidden, []);
});

test('saved LSTM forecasts and decoder populate the computation desk for the same source person', () => {
    const result = { source: { job_id:'j', frame:151, track_id:11, kind:'recomputed_from_saved_history' }, lstm: { rounds: [
        { index:1, forecast_poses: [[[1,2,.9]]], decoder_example: { input_values:[1,2], weights:[3,4], bias:.5, actual:11.5 } },
    ] } };
    const live = { nodes: recordedMemoryNodes(result), tracks:[{track_id:11}] };
    const calculation = normalizeCalculation(latestCalculation(live,4,{},0,1,11));
    assert.equal(calculation.reconstructed,11.5);
    assert.equal(calculation.metadata.source_frame,151);
    assert.equal(latestCalculation(live,4,{},0,1,99),null);
    assert.equal(processMetrics(4,{live,track:{track_id:11},calculation}).output,'本采样帧已记录 1 / 5 轮');
});
