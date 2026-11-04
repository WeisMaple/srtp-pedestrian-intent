import test from 'node:test';
import assert from 'node:assert/strict';
import { BAYS, bayForStep, cameraForStep } from '../src/scene/layout.js';

test('six bays are finite and ordered left to right', () => {
    assert.equal(BAYS.length, 6);
    assert.deepEqual(BAYS.map(bay => bay.step), [1, 2, 3, 4, 5, 6]);
    for (let index = 1; index < BAYS.length; index += 1) assert.ok(BAYS[index].position[0] > BAYS[index - 1].position[0]);
    assert.ok(BAYS.every(bay => bay.position.every(Number.isFinite)));
});

test('camera keeps a readable distance and looks at the bay', () => {
    const camera = cameraForStep(3);
    const distance = Math.hypot(camera.position[0] - camera.target[0], camera.position[1] - camera.target[1], camera.position[2] - camera.target[2]);
    assert.ok(distance > 8 && distance < 30);
    assert.equal(camera.target[0], bayForStep(3).position[0]);
});

test('unknown steps fall back to the first bay', () => {
    assert.equal(bayForStep(99).step, 1);
});
