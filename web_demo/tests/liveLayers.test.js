import test from 'node:test';
import assert from 'node:assert/strict';
import { liveLayerModel } from '../src/domain/liveLayers.js';

const structure = { yoloModules: [{ index: 0 }], stgcnBlocks: [{ index: 1 }] };
const snapshot = {
    frame: 7,
    nodes: {
        'yolo.0': { frame: 7, tensors: { output: { values: [[[[1, 2], [3, 4]]]], shape: [1, 1, 2, 2], indices: [[0], [0], [0, 1], [0, 1]] } } },
        'stgcn.logits': { frame: 7, tensors: { output: { values: [1, 2, 3], shape: [3], indices: [[0, 1, 2]] } } },
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
