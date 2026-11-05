import test from 'node:test';
import assert from 'node:assert/strict';
import { intentDisplayModel, recordedIntentNodes, stableSoftmax } from '../src/domain/intent.js';
import { latestCalculation } from '../src/domain/workbench.js';

const source = { job_id: 'a', frame: 151, track_id: 11, kind: 'recomputed_from_saved_history' };
const result = { source, stgcn: { pool: [2, 3], head_weights: [[1, -1], [0, 2], [-1, 0]],
    head_bias: [0.5, -1, 0], logits: [-0.5, 5, -2], probabilities: stableSoftmax([-0.5, 5, -2])?.probabilities,
    pool_arithmetic: { channel: 0, input_shape: [2, 2], values: [[1, 2], [3, 2]], count: 4, sum: 8, mean: 2, actual: 2 } } };
const track = { track_id: 11, probabilities: [0.1, 0.7, 0.2], prediction_updated: false };
const snapshot = () => ({ frame: 151, tracks: [track], nodes: recordedIntentNodes(result) });

test('all intent classes retain complete weighted sums, actual logits and separate original scores', () => {
    const model = intentDisplayModel(snapshot(), result, 11);
    assert.deepEqual(model.classifiers.map(row => row.reconstructed), [-0.5, 5, -2]);
    assert.deepEqual(model.classifiers.map(row => row.absoluteError), [0, 0, 0]);
    assert.equal(model.pooling.count, 4);
    assert.equal(model.winner, 1);
    assert.equal(model.margin, model.scores[1] - model.scores[0]);
    assert.deepEqual(model.originalScores, track.probabilities);
    assert.notDeepEqual(model.scores, model.originalScores);
    const row = latestCalculation(snapshot(), 6, {}, 0, 1, 11, [], { classIndex: 0 });
    assert.equal(row.metadata.selected_class, 0);
    assert.deepEqual(row.weights, [1, -1]);
});

test('stable Softmax handles extreme and tied logits; missing entries are never zeros', () => {
    const values = stableSoftmax([1000, 1001, -1000]);
    assert.ok(values.probabilities.every(Number.isFinite));
    assert.ok(Math.abs(values.probabilities.reduce((a,b) => a+b,0) - 1) < 1e-12);
    assert.equal(stableSoftmax([1, null, 3]), null);
    assert.equal(stableSoftmax([1, Infinity, 3]), null);
    assert.deepEqual(stableSoftmax([2,2,2]).probabilities, [1/3,1/3,1/3]);
    const partial = structuredClone(result); partial.stgcn.pool[0] = null;
    const model = intentDisplayModel({ ...snapshot(), nodes: recordedIntentNodes(partial) }, partial, 11);
    assert.equal(model.classifiers[0].reconstructed, null);
});

test('intent recovery never uses another source frame or person, and original-only replay stays explicit', () => {
    const wrongFrame = intentDisplayModel({ ...snapshot(), frame: 152, nodes: {} }, result, 11);
    assert.equal(wrongFrame.source, null); assert.equal(wrongFrame.pooling, null); assert.deepEqual(wrongFrame.logits, []);
    const wrongPerson = intentDisplayModel(snapshot(), result, 18);
    assert.equal(wrongPerson.source, null); assert.equal(wrongPerson.scores, null);
    const original = intentDisplayModel({ frame: 151, tracks: [track], nodes: {} }, null, 11);
    assert.deepEqual(original.scores, track.probabilities); assert.equal(original.held, true);
    assert.deepEqual(original.classifiers, []); assert.deepEqual(original.pool, []);
    assert.equal(latestCalculation({ ...snapshot(), frame: 152 }, 6, {}, 0, 1, 11,
        Object.values(snapshot().nodes).flatMap(node => node.calculations), { classIndex: 0 }), null);
});

test('live sampled features retain their channel positions and leave unsampled entries empty', () => {
    const nodes = { 'stgcn.pool': { node: 'stgcn.pool', frame: 151, seq: 1, track_ids: [11, 18],
        tensors: { output: { axes: ['track', 'channel'], shape: [2, 256],
            indices: [[0, 1], [0, 127, 255]], values: [[2, 3, 4], [20, 30, 40]] } } } };
    const model = intentDisplayModel({ frame: 151, tracks: [track], nodes }, null, 11);
    assert.equal(model.pool.length, 256);
    assert.deepEqual([model.pool[0], model.pool[127], model.pool[255]], [2, 3, 4]);
    assert.equal(model.pool[1], null);
    assert.equal(model.pool.filter(Number.isFinite).length, 3);
    assert.equal(model.pooling, null);
    assert.equal(model.softmax, null);
    assert.deepEqual(model.classifiers, []);
    const nextFrame = intentDisplayModel({ frame: 152, tracks: [track], nodes }, null, 11);
    assert.deepEqual(nextFrame.pool, []);
    const tied = intentDisplayModel({ frame: 151, tracks: [{ ...track, probabilities: [0.5, 0.5, 0] }] }, null, 11);
    assert.equal(tied.winner, null);
    assert.deepEqual(tied.winners, [0, 1]);
});
