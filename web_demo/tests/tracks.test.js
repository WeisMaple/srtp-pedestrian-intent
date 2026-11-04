import test from 'node:test';
import assert from 'node:assert/strict';
import { historyGaps, historyPoses, trackIsWarm } from '../src/domain/tracks.js';

test('history keeps real poses and marks missing ones as gaps', () => {
    const track = { history: [[[1, 2, 3]], null, [[4, 5, 6]]], history_length: 3 };
    const poses = historyPoses(track);
    assert.equal(poses.length, 3);
    assert.equal(poses[1], null);
    assert.deepEqual(historyGaps(track), [1]);
});

test('warm-up is explicit before fifty observations', () => {
    assert.equal(trackIsWarm({ history_length: 49 }), false);
    assert.equal(trackIsWarm({ history_length: 50 }), true);
});
