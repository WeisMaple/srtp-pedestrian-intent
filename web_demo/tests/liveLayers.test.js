import test from 'node:test';
import assert from 'node:assert/strict';
import { liveLayerModel } from '../src/domain/liveLayers.js';

const structure = { yoloModules: [{ index: 0 }], stgcnBlocks: [{ index: 1 }] };
const values = track => Array.from({ length: 10 }, (_, time) =>
    Array.from({ length: 17 }, (_, joint) => [track, time, joint]));
const snapshot = {
    frame: 7,
    nodes: {
        'yolo.0': { frame: 7, tensors: { output: { values: [[[[1, 2], [3, 4]]]], shape: [1, 1, 2, 2], indices: [[0], [0], [0, 1], [0, 1]] } } },
        'stgcn.logits': { frame: 7, tensors: { output: { values: [1, 2, 3], shape: [3], indices: [[0, 1, 2]] } } },
        'pipeline.history': { frame: 7, seq: 3, node: 'pipeline.history', track_ids: [4, 2],
            tensors: { history: { axes: ['track', 'time', 'joint', 'coordinate'],
                shape: [2, 50, 17, 3], sample_shape: [2, 10, 17, 3],
                indices: [[0, 1], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                    Array.from({ length: 17 }, (_, index) => index), [0, 1, 2]],
                values: [values(4), values(2)] } } },
        'lstm.1': { frame: 7 },
        'lstm.1.forecast': { frame: 7 },
    },
    tracks: [{ track_id: 2, history_length: 50, probabilities: [0.1, 0.2, 0.7], class_id: 2 }],
    details: { yolo: {}, stgcn: {} },
};

test('live layers select the chosen track and real probabilities', () => {
    const model = liveLayerModel(snapshot, 2, structure);
    assert.equal(model.frame, 7);
    assert.equal(model.intent.probabilities.length, 3);
    assert.equal(model.intent.classId, 2);
    assert.ok(model.features.length >= 1);
});

test('missing probabilities stay null instead of zero', () => {
    const model = liveLayerModel({ ...snapshot, tracks: [{ track_id: 2 }] }, 2, structure);
    assert.equal(model.intent.probabilities, null);
});

test('a live frame without tracks uses the selected job track, while live tracks win', () => {
    const fallback = { track_id: 2, keypoints: [[1, 2, 0.9]], probabilities: [0.1, 0.2, 0.7], class_id: 2 };
    const model = liveLayerModel({ ...snapshot, tracks: [] }, 2, structure, fallback);
    assert.equal(model.track, fallback);
    assert.deepEqual(model.intent.probabilities, [0.1, 0.2, 0.7]);
    assert.equal(liveLayerModel(snapshot, 2, structure, fallback).track, snapshot.tracks[0]);
    assert.equal(liveLayerModel(snapshot, 9, structure, fallback).track, null);
});

test('pipeline.history is selected per track and labelled as a sample', () => {
    const model = liveLayerModel(snapshot, 2, structure);
    assert.equal(model.history.node, 'pipeline.history');
    assert.equal(model.history.track_id, 2);
    assert.equal(model.history.sampled, true);
    assert.deepEqual(model.history.axes, ['time', 'joint', 'coordinate']);
    assert.equal(model.history.poses.length, 10);
    assert.deepEqual(model.history.poses[0][0], [2, 0, 0]);
    assert.match(model.history.sample_note, /不是完整 50 帧/);
    assert.equal(liveLayerModel(snapshot, 9, structure).history, null);
});

test('forecast nodes map to real round numbers without fabricated future data', () => {
    const model = liveLayerModel(snapshot, 2, structure);
    assert.deepEqual(model.memory.rounds, ['lstm.1']);
    assert.deepEqual(model.memory.forecasts, [1]);
    assert.equal('history' in model.memory, false);
    assert.equal('future' in model.memory, false);
});
