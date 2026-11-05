import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { fitInBay } from '../src/scene/fit.js';
import { buildFeatureLayer } from '../src/scene/featureLayer.js';
import { buildRibbonGroup } from '../src/scene/idRibbon.js';
import { buildIntentLayer, updateIntentLayer } from '../src/scene/intentLayer.js';
import { cameraForStep, bayForStep } from '../src/scene/layout.js';

test('24 YOLO planes, 50 history poses and full-height intent bars fit their exhibit', () => {
    const pose = Array.from({length:17}, (_, i) => [i % 2 * 20, i * 10, 0.9]);
    const intent = buildIntentLayer(['a', 'b', 'c']);
    updateIntentLayer(intent, [0, 0, 1]);
    for (const group of [buildFeatureLayer(24).group,
        buildRibbonGroup({ history: Array(50).fill(pose) }), intent.group]) {
        const parent = new THREE.Group(); parent.position.set(17, 0, 3); parent.add(group);
        fitInBay(group);
        const box = new THREE.Box3().setFromObject(group);
        assert.ok(box.min.x >= 14.49 && box.max.x <= 19.51);
        assert.ok(box.min.y >= -0.31 && box.max.y <= 2.51);
        assert.ok(box.min.z >= 1.74 && box.max.z <= 4.26);
        const before = box.clone(); fitInBay(group);
        assert.ok(before.min.distanceTo(new THREE.Box3().setFromObject(group).min) < 1e-6);
    }
});

test('exhibit corners stay in the camera viewport at wide and narrow aspect ratios', () => {
    for (const aspect of [0.6, 1, 2.5, 4]) {
        const pose = cameraForStep(5, { aspect });
        const camera = new THREE.PerspectiveCamera(34, aspect, 0.1, 300);
        camera.position.set(...pose.position); camera.lookAt(...pose.target); camera.updateMatrixWorld();
        const bay = bayForStep(5);
        for (const x of [-2.9, 2.9]) for (const y of [-0.7, 3.5]) for (const z of [-1.6, 1.6]) {
            const projected = new THREE.Vector3(x + bay.position[0], y, z + bay.position[2]).project(camera);
            assert.ok(Math.abs(projected.x) < 0.95 && Math.abs(projected.y) < 0.95);
        }
    }
});
