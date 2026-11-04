import test from 'node:test';
import assert from 'node:assert/strict';
import { historyPoses, ribbonSignature, trackIsWarm } from '../src/domain/tracks.js';

test('history keeps real poses and missing entries stay visible', () => {
    const track = { history: [[[1, 2, 3]], null, [[4, 5, 6]]], history_length: 3 };
    const poses = historyPoses(track);
    assert.equal(poses.length, 3);
    assert.equal(poses[1], null);
});

test('warm-up is explicit before fifty observations', () => {
    assert.equal(trackIsWarm({ history_length: 49 }), false);
    assert.equal(trackIsWarm({ history_length: 50 }), true);
});

test('ribbon signature changes when the same ID receives new pose data', () => {
    const track = { track_id: 3, history_length: 12, keypoints: [[1, 2, 0.9]] };
    const first = ribbonSignature(track, { frame: 20 });
    const same = ribbonSignature({ ...track }, { frame: 20 });
    const newObservation = ribbonSignature({ ...track, history_length: 13, keypoints: [[2, 3, 0.8]] }, { frame: 21 });
    assert.equal(first, same);
    assert.notEqual(first, newObservation);
    assert.notEqual(ribbonSignature(track, { frame: 21 }), first);
    assert.notEqual(ribbonSignature(track, { frame: 20, poses: [track.keypoints] }), first);
    assert.equal(ribbonSignature(null), 'none');
});
