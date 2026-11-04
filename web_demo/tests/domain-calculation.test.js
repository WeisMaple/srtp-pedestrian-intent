import test from 'node:test';
import assert from 'node:assert/strict';
import { flowState, normalizeCalculation } from '../src/domain/calculation.js';

const dense = {
    id: 'yolo.conv.0', stage: 'yolo', operation: 'dense_conv', module: 0,
    term_axes: ['in_channel', 'kernel_y', 'kernel_x'], term_shape: [1, 1, 2],
    inputs: [2, 3], weights: [4, 5], bias: 1, actual: 11,
    activation: 'identity', fused: true, metadata: { complete_sum: true, detail: true },
};

test('dense convolution expands every term and accumulates completely', () => {
    const calculation = normalizeCalculation(dense);
    assert.equal(calculation.terms.length, 2);
    assert.deepEqual(calculation.terms[0].index, [0, 0, 0]);
    assert.equal(calculation.terms[0].product, 8);
    assert.equal(calculation.terms[1].product, 15);
    assert.equal(calculation.dot, 23);
    assert.equal(calculation.preactivationValue, 24);
    assert.equal(calculation.reconstructed, 24);
    assert.equal(calculation.absoluteError, 13);
});

test('cursor phases cover terms, bias, activation and result', () => {
    const calculation = normalizeCalculation({ ...dense, actual: 24 });
    assert.equal(flowState(calculation, 0).phaseName, 'input');
    assert.equal(flowState(calculation, 0.9).phaseName, 'accumulated');
    assert.equal(flowState(calculation, 2).phaseName, 'bias');
    assert.equal(flowState(calculation, 3).phaseName, 'activation');
    assert.equal(flowState(calculation, 4).phaseName, 'result');
    assert.equal(flowState(calculation, 2).partialSum, 24);
});

test('missing operands stay missing and never become zero', () => {
    const calculation = normalizeCalculation({ ...dense, inputs: [2, null], actual: null });
    assert.equal(calculation.terms[1].recorded, false);
    assert.equal(calculation.terms[1].product, null);
    assert.equal(calculation.terms[1].membership, 'unrecorded');
    assert.equal(calculation.dot, null);
});
