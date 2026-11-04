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

test('record BN metadata reconstructs the dense row output', () => {
    const recorded = {
        ...dense, bias: 0.75, activation: 'batch_norm_silu',
        inputs: [1.25, -0.5], weights: [2, -4],
        metadata: { complete_sum: true, detail: true, bn: {
            gamma: 1.5, beta: -0.25, running_mean: 0.5, running_var: 2, epsilon: 1e-5,
        } },
    };
    const calculation = normalizeCalculation(recorded);
    const preactivation = 1.25 * 2 + -0.5 * -4 + 0.75;
    const normalized = 1.5 * (preactivation - 0.5) / Math.sqrt(2 + 1e-5) - 0.25;
    const expected = normalized / (1 + Math.exp(-normalized));
    recorded.actual = Math.round(expected * 1e5) / 1e5;
    const withActual = normalizeCalculation(recorded);
    assert.ok(Math.abs(withActual.reconstructed - expected) < 1e-5);
    assert.ok(withActual.absoluteError < 1e-4);
    assert.equal(calculation.reconstructed !== null, true);
});
