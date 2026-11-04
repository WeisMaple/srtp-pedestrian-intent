import test from 'node:test';
import assert from 'node:assert/strict';
import { asPose, normalizePose, poseBounds } from '../src/domain/pose.js';

test('flat 51 values become 17 x 3 points', () => {
    const flat = Array.from({ length: 51 }, (_, index) => index);
    const pose = asPose(flat);
    assert.equal(pose.length, 17);
    assert.deepEqual(pose[1], [3, 4, 5]);
});

test('normalization centers and flips image y', () => {
    const pose = [[0, 0, 1], [10, 0, 1], [0, 10, 1]];
    const result = normalizePose(pose, 10);
    assert.ok(Math.abs(result[0][0] + 5) < 1e-9);
    assert.ok(Math.abs(result[2][1] + 5) < 1e-9);
});

test('bounds ignore low-confidence points', () => {
    const points = [[[0, 0, 0.9], [10, 10, 0.1]]];
    assert.deepEqual(poseBounds(points), { minX: 0, maxX: 0, minY: 0, maxY: 0 });
});
