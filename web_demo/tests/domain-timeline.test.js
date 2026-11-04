import test from 'node:test';
import assert from 'node:assert/strict';
import { STEPS, beatsForStep, nextBeat, stepReady } from '../src/domain/timeline.js';

test('six steps and their beats are stable', () => {
    assert.equal(STEPS.length, 6);
    assert.deepEqual(STEPS.map(step => step.title), ['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图']);
    assert.ok(beatsForStep(2).length >= 2);
});

test('step readiness is data-gated', () => {
    const empty = { tracks: [], nodes: {}, details: { yolo: {}, stgcn: {} } };
    assert.equal(stepReady(1, empty, { metadata: {} }), true);
    assert.equal(stepReady(2, empty, { metadata: {} }), false);
    const posed = { tracks: [{ track_id: 1, keypoints: [[1, 2, 3]] }], nodes: {}, details: { yolo: {}, stgcn: {} } };
    assert.equal(stepReady(2, posed, { metadata: {} }), true);
    assert.equal(stepReady(4, posed, { metadata: {} }), false);
    assert.equal(stepReady(6, { ...posed, nodes: { 'stgcn.logits': { frame: 1 } } }, { metadata: {} }), true);
});

test('nextBeat advances and stops at the end', () => {
    assert.equal(nextBeat({ step: 1, beat: 0, paused: false, following: true }, 3), 1);
    assert.equal(nextBeat({ step: 6, beat: 2, paused: false, following: true }, 3), null);
    assert.equal(nextBeat({ step: 1, beat: 0, paused: true, following: true }, 3), null);
});
