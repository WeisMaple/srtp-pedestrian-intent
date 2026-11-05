import test from 'node:test';
import assert from 'node:assert/strict';
import { processMetrics, visibleTracks } from '../src/domain/process.js';

test('a live empty frame never falls back to pedestrians from an older job frame', () => {
    const old = [{ track_id: 9 }];
    assert.deepEqual(visibleTracks({ tracks: [] }, { tracks: old }), []);
    assert.deepEqual(visibleTracks(null, { tracks: old }), old);
});

test('warm-up uses the selected pedestrian and counts observations rather than source frames', () => {
    const result = processMetrics(4, { job: { processed_frames: 200 }, track: { track_id: 4, history_length: 23 } });
    assert.equal(result.input, '23 / 50 次有效观测');
    assert.equal(result.remaining, 27);
    assert.equal(result.waiting, true);
});

test('a completed job does not invent arithmetic or prediction output', () => {
    const result = processMetrics(6, { job: { status: 'completed' }, track: { track_id: 4, history_length: 50 } });
    assert.equal(result.output, '尚无分类分数');
    assert.equal(result.waiting, true);
});

test('zero class scores and zero-based source frames are valid recorded values', () => {
    const result = processMetrics(6, { live: { frame: 0 }, track: {
        track_id: 4, class_id: 0, history_length: 50, probabilities: [0, 0.25, 0.75],
    } });
    assert.equal(result.frame, 0);
    assert.deepEqual(result.scores, [0, 0.25, 0.75]);
    assert.equal(result.waiting, false);
});
