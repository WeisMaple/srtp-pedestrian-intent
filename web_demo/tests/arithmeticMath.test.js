import test from 'node:test';
import assert from 'node:assert/strict';
import { accumulationFrame, activationValue, displayNumber, weightedTerms } from '../src/components/arithmeticMath.js';

test('all terms contribute to the dot product, including beyond the visible nine-term window', () => {
    const result = weightedTerms(Array.from({ length: 256 }, (_, i) => i + 1), Array(256).fill(.5), -.75);
    assert.equal(result.products.length, 256);
    assert.equal(result.prefixes[9], 22.5);
    assert.equal(result.dot, 16448);
    assert.equal(result.total, 16447.25);
});

test('partial sum has exact completed terms and bias is applied only at the end', () => {
    assert.deepEqual(accumulationFrame(0, 10), { index: 0, phase: 0, completed: 0, biasPhase: 0, outputPhase: 0 });
    const middle = accumulationFrame(.44, 8);
    assert.equal(middle.index, 4);
    assert.equal(middle.completed, 4);
    assert.equal(middle.biasPhase, 0);
    assert.deepEqual(accumulationFrame(1, 10), { index: 9, phase: 1, completed: 10, biasPhase: 1, outputPhase: 1 });
});

test('long vectors linger on the first three actual terms and still accumulate every remaining dimension', () => {
    assert.equal(accumulationFrame(.075, 2304).index, 0);
    assert.equal(accumulationFrame(.225, 2304).index, 1);
    assert.equal(accumulationFrame(.375, 2304).index, 2);
    assert.equal(accumulationFrame(.45, 2304).index, 3);
    assert.equal(accumulationFrame(.88, 2304).completed, 2304);
});

test('the output has a terminal hold before the parent chapter or repeated segment advances', () => {
    assert.equal(accumulationFrame(.94, 256).outputPhase, 0);
    assert.ok(accumulationFrame(.96, 256).outputPhase > 0);
    assert.equal(accumulationFrame(.975, 256).outputPhase, 1);
    assert.equal(accumulationFrame(.995, 256).outputPhase, 1);
    assert.equal(accumulationFrame(.975, 256).biasPhase, 1);
});

test('seeking backwards reconstructs the earlier arithmetic state without history', () => {
    const before = accumulationFrame(.231, 51);
    accumulationFrame(.9, 51);
    assert.deepEqual(accumulationFrame(.231, 51), before);
    assert.equal(accumulationFrame(-3, 51).completed, 0);
    assert.equal(accumulationFrame(8, 51).completed, 51);
});

test('missing, mismatched, invalid and overflowing tensors never become fake zero data', () => {
    for (const args of [[[], []], [[1], []], [[NaN], [1]], [[1], [Infinity]], [[1], [1], NaN], [[1e308], [1e308]]]) {
        assert.equal(weightedTerms(...args), null);
    }
    assert.deepEqual(weightedTerms([0, -2, 3], [4, 0, -.5], .2).prefixes, [0, 0, 0, -1.5]);
});

test('nonlinear transformations use the mathematical function and stay finite on saturated gates', () => {
    assert.equal(activationValue(-1000, 'sigmoid'), 0);
    assert.equal(activationValue(1000, 'sigmoid'), 1);
    assert.equal(activationValue(0, 'sigmoid'), .5);
    assert.equal(activationValue(-1, 'relu'), 0);
    assert.equal(activationValue(.4, 'tanh'), Math.tanh(.4));
    assert.equal(activationValue(null, 'tanh'), null);
    assert.equal(displayNumber(-.00000001), '-1.00e-8');
});
