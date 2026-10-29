import test from 'node:test';
import assert from 'node:assert/strict';
import { collectCalculations, createCalculation, flowState, validateCalculation } from '../src/calculationFlow.js';

const basic = (extra = {}) => ({ id: 'sum', stage: 'decoder', title: 'sum', operation: 'linear',
    inputs: [2, 3, 4], weights: [.5, -1, 2], bias: .25, actual: 6.25, ...extra });

test('the full vector contributes, including all 256 dimensions beyond a visible page', () => {
    const calculation = createCalculation(basic({ inputs: Array.from({ length: 256 }, (_, i) => i + 1),
        weights: Array(256).fill(.5), bias: -.75, actual: 16447.2501 }));
    assert.equal(calculation.terms.length, 256);
    assert.equal(calculation.prefixes[64], 1040);
    assert.equal(calculation.dot, 16448);
    assert.equal(calculation.reconstructed, 16447.25);
    assert.ok(Math.abs(calculation.absoluteError - .0001) < 1e-10);
    assert.equal(calculation.actual, 16447.2501);
});

test('seeking is deterministic, products arrive before accumulation and bias follows every term', () => {
    const calculation = createCalculation(basic());
    assert.equal(flowState(calculation, 1.7).partialSum, 1);
    assert.equal(flowState(calculation, 1.9).partialSum, -2);
    assert.equal(flowState(calculation, 3.5).partialSum, 6);
    assert.equal(flowState(calculation, 4).partialSum, 6.25);
    const earlier = flowState(calculation, .4);
    flowState(calculation, 5);
    assert.deepEqual(flowState(calculation, .4), earlier);
    assert.equal(earlier.phaseName, 'product');
    assert.equal(earlier.product, 1);
    assert.equal(flowState(calculation, 5).finalValue, 6.25);
    assert.equal(flowState(calculation, 5).done, true);
    assert.equal(flowState(calculation, -100).completedCount, 0);
    assert.equal(flowState(calculation, Infinity).completedCount, 0);
});

test('missing observations never become zeros; graph disconnection, zero weight and padding remain distinct', () => {
    const calculation = createCalculation(basic({ inputs: [2, 3, 0, null, 4], weights: [0, 0, 7, 2, 1],
        connected: [false, true, true, true, true], padding: [false, false, true, false, false] }));
    assert.deepEqual(calculation.terms.map(term => term.membership),
        ['disconnected', 'zeroContribution', 'padding', 'unrecorded', 'connected']);
    assert.equal(calculation.terms[1].connected, true);
    assert.equal(calculation.terms[3].product, null);
    assert.equal(calculation.reconstructed, null);
    assert.equal(calculation.complete, false);
    assert.equal(flowState(calculation, 7).partialSum, null);
    assert.deepEqual(calculation.counts, { connected: 3, disconnected: 1, zeroContribution: 1, padding: 1, unrecorded: 1, total: 5 });
});

test('nonfinite values, vector misalignment and inconsistent connectivity are rejected', () => {
    for (const extra of [{ inputs: [] }, { weights: [1] }, { inputs: [NaN, 1, 2] },
        { weights: [Infinity, 1, 1] }, { bias: NaN }, { actual: Infinity },
        { connected: [true] }, { padding: [false] }, { input_indices: ['only one'] },
        { connected: [false, true, true] }, { inputs: [1e308, 1, 1], weights: [1e308, 1, 1] }]) {
        assert.equal(validateCalculation(basic(extra)).valid, false);
        assert.equal(createCalculation(basic(extra)), null);
    }
});

test('activation is applied after bias without replacing independently observed actual output', () => {
    const calculation = createCalculation(basic({ inputs: [-2], weights: [3], bias: 1,
        activation: 'relu', actual: .00001 }));
    assert.equal(calculation.preactivationValue, -5);
    assert.equal(calculation.reconstructed, 0);
    assert.equal(calculation.absoluteError, .00001);
    assert.equal(flowState(calculation, 3).finalValue, 0);
});

test('live calculations use only this frame and selected track, with YOLO shared across tracks', () => {
    const calc = (id, track_id, stage = 'decoder') => basic({ id, track_id, stage });
    const snapshot = { job_id: 'live-job', frame: 42, nodes: {
        old: { frame: 41, seq: 9, calculations: [calc('old', 7)] },
        first: { frame: 42, seq: 10, calculations: [calc('same', 7), calc('other', 8), calc('shared', null, 'yolo')] },
        latest: { frame: 42, seq: 11, calculations: [calc('same', 7)] },
        foreign: { job_id: 'other-job', frame: 42, seq: 12, calculations: [calc('foreign', 7)] },
    } };
    const result = collectCalculations({ snapshot, trackId: '7', frame: 42 });
    assert.deepEqual(result.map(row => row.id), ['shared', 'same']);
    const selected = result.find(row => row.id === 'same');
    assert.deepEqual([selected.source, selected.job_id, selected.frame, selected.track_id, selected.seq], ['live', 'live-job', 42, 7, 11]);
    assert.equal(selected.chapter, 3);
    assert.deepEqual(collectCalculations({ snapshot, trackId: null }).map(row => row.id), ['shared']);
    assert.deepEqual(collectCalculations({ snapshot, trackId: 7, frame: 41 }), []);
});

test('an empty live snapshot cannot fall back to saved arithmetic or another selected person', () => {
    const data = { source: { frame: 2, track_id: 7, job_id: 'saved-job' },
        lstm: { rounds: [{ index: 1, decoder_example: { input_values: [2], weights: [3], bias: 1, actual: 7 } }] } };
    assert.equal(collectCalculations({ data, trackId: 7, frame: 2 }).length, 1);
    assert.deepEqual(collectCalculations({ data, trackId: 8, frame: 2 }), []);
    assert.deepEqual(collectCalculations({ data, trackId: 7, frame: 3 }), []);
    assert.deepEqual(collectCalculations({ snapshot: {}, data, trackId: 7, frame: 2 }), []);
});

test('saved YOLO keeps the whole RGB patch and applies real BatchNorm before SiLU', () => {
    const frontend = { source: { job_id: 'saved-job', frame: 4 }, first_conv: { example: {
        patch: Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => [1, 1, 1])),
        weights: Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => [.1, .1, .1])),
        bias: .3, bn: { gamma: 2, beta: -1, running_mean: 1, running_var: 4, epsilon: 0 },
        actual_output: .7310585786300049, output_channel: 0, position: [2, 2], input_origin: [3, 3],
    } } };
    const [calculation] = collectCalculations({ frontend, trackId: null, frame: 4 });
    assert.equal(calculation.terms.length, 27);
    assert.equal(calculation.preactivationValue, 3);
    assert.equal(calculation.activation, 'batch_norm_silu');
    assert.ok(Math.abs(calculation.reconstructed - .7310585786300049) < 1e-15);
    assert.equal(calculation.source, 'record');
    assert.equal(calculation.track_id, null);
});

test('LSTM gate calculations include every input and recurrent dimension plus cell and hidden updates', () => {
    const data = { source: { frame: 4, track_id: 7 }, lstm: { gates: {
        layer: 1, unit: 0, round: 1, gate_order: ['i', 'f', 'g', 'o'],
        input_weights: Array.from({ length: 4 }, () => Array(51).fill(1)),
        recurrent_weights: Array.from({ length: 4 }, () => Array(256).fill(1)),
        bias: [0, 0, 0, 0], steps: [{ t: 50, input_values: Array(51).fill(1), previous_hidden: Array(256).fill(2),
            i: 1, f: 1, g: 1, o: 1, c_previous: 2, c: 3, h: .9950547536867305,
            preactivation: [563, 563, 563, 563] }],
    } } };
    const result = collectCalculations({ data, trackId: 7, frame: 4 });
    const gates = result.filter(row => row.operation === 'lstm_gate');
    assert.equal(gates.length, 4);
    assert.ok(gates.every(row => row.terms.length === 307 && row.dot === 563));
    assert.equal(gates.find(row => row.target.gate === 'g').activation, 'tanh');
    assert.equal(result.find(row => row.operation === 'lstm_cell').reconstructed, 3);
    assert.equal(result.find(row => row.operation === 'lstm_hidden').reconstructed, .9950547536867305);
    assert.equal(gates[0].metadata.actual_kind, 'reconstructed_gate_validated_against_final_state');
});

test('all saved blocks retain 51 spatial terms, full temporal kernels, residual and pooling arithmetic', () => {
    const blocks = Array.from({ length: 10 }, (_, i) => ({ index: i + 1, input_shape: [1, 3, 50, 17],
        arithmetic: { spatial: { projected_values: Array.from({ length: 3 }, () => Array(17).fill(2)),
            adjacency: Array.from({ length: 3 }, () => Array(17).fill(.5)), actual: 51, output_joint: 5, timestep: 25 },
        temporal: { input_values: Array.from({ length: 64 }, () => Array(9).fill(1)),
            weights: Array.from({ length: 64 }, () => Array(9).fill(.5)), bias: 2, actual: 290,
            input_timesteps: [21,22,23,24,25,26,27,28,29], output_joint: 5, timestep: 25 },
        residual: { temporal: 10, residual: -2, actual: 8, mode: 'identity' } } }));
    const data = { source: { frame: 4, track_id: 7 }, stgcn: { blocks, pool_arithmetic: {
        values: Array.from({ length: 13 }, () => Array(17).fill(3)), count: 221, actual: 3, channel: 0 } } };
    const result = collectCalculations({ data, trackId: 7, frame: 4 });
    assert.equal(result.filter(row => row.stage === 'spatial').length, 10);
    assert.equal(result.filter(row => row.operation === 'temporal_convolution').length, 10);
    assert.equal(result.filter(row => row.operation === 'residual_add').length, 10);
    assert.equal(result.find(row => row.stage === 'spatial').terms.length, 51);
    assert.equal(result.find(row => row.operation === 'temporal_convolution').terms.length, 576);
    const pool = result.find(row => row.stage === 'pool');
    assert.equal(pool.terms.length, 221);
    assert.equal(pool.reconstructed, 3);
});

test('all classifier classes and stable softmax use every pooled channel and all logits', () => {
    const data = { source: { frame: 4, track_id: 7 }, stgcn: {
        pool: Array(256).fill(1), head_weights: [Array(256).fill(0), Array(256).fill(0), Array(256).fill(0)],
        head_bias: [1000, 1001, 1002], logits: [1000, 1001, 1002],
        probabilities: [.09003057317038046, .24472847105479764, .6652409557748218], labels: ['a', 'b', 'c'], class_id: 2,
    } };
    const result = collectCalculations({ data, trackId: 7, frame: 4 });
    assert.equal(result.filter(row => row.stage === 'classifier').length, 3);
    assert.ok(result.filter(row => row.stage === 'classifier').every(row => row.terms.length === 256));
    const softmax = result.find(row => row.stage === 'softmax');
    assert.equal(softmax.terms.length, 3);
    assert.deepEqual(softmax.metadata.logits, [1000, 1001, 1002]);
    assert.equal(softmax.metadata.max_logit, 1002);
    assert.ok(Math.abs(softmax.reconstructed - .6652409557748218) < 1e-15);
    assert.equal(softmax.actual, .6652409557748218);
});

test('normalization and denormalization expose actual source operands and independent outputs', () => {
    const data = { source: { frame: 4, track_id: 7 }, input: { values: [[10]] },
        normalization: { lstm: { mean: [[2]], std: [[4]], formula: '(pose - mean) / std' } },
        lstm: { rounds: [{ index: 1, input: { values: [[2]] }, forecast: { values: [[3]] },
            forecast_poses: [[[14, 0, 0]]] }] } };
    const result = collectCalculations({ data, trackId: 7, frame: 4 });
    const normalized = result.find(row => row.stage === 'normalization');
    assert.deepEqual(normalized.inputs, [10, 2]);
    assert.deepEqual(normalized.weights, [.25, -.25]);
    assert.equal(normalized.reconstructed, 2);
    const denormalized = result.find(row => row.stage === 'denormalization');
    assert.deepEqual(denormalized.inputs, [3, 2]);
    assert.deepEqual(denormalized.weights, [4, 1]);
    assert.equal(denormalized.actual, 14);
    assert.equal(denormalized.reconstructed, 14);
});

test('a zero learned graph coefficient does not erase a structural connection',()=>{
    const adjacency=Array.from({length:3},()=>Array.from({length:17},()=>Array(17).fill(0)));
    adjacency[0][0][0]=1;
    const data={source:{frame:8,track_id:1},stgcn:{adjacency,blocks:[{arithmetic:{spatial:{projected_values:Array.from({length:3},()=>Array(17).fill(2)),adjacency:Array.from({length:3},()=>Array(17).fill(0)),output_joint:0,actual:0}}}]}};
    const calculation=collectCalculations({data,frame:8,trackId:1}).find(c=>c.chapter===5);
    assert.equal(calculation.counts.connected,1);
    assert.equal(calculation.counts.disconnected,50);
    assert.equal(calculation.terms[0].membership,'zeroContribution');
});

test('operand labels explain statistic names and native axes without changing values',()=>{
    const named=createCalculation({id:'n',inputs:[2,3],weights:[1,-1],bias:0,metadata:{input_names:['原坐标','均值']}});
    assert.deepEqual(named.terms.map(t=>t.label),['原坐标','均值']);
    const axes=createCalculation({id:'a',inputs:[7],weights:[1],bias:0,input_indices:[[2,4]],metadata:{input_axes:['channel','joint']}});
    assert.equal(axes.terms[0].label,'通道 2 · 关节 4');
    assert.equal(axes.reconstructed,7);
});
