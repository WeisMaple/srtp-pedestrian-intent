import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { fitPerspectiveCamera } from '../src/scene/perspectiveFit.js';
import { MEMORY_BOUNDS, TEMPORAL_BOUNDS, INTENT_BOUNDS } from '../src/scene/computationBounds.js';

test('LSTM and temporal 3D content stays inside wide and portrait cameras throughout automatic sweep', () => {
    for (const [bounds,target,direction,sweep,height] of [
        [MEMORY_BOUNDS,[-.4,1.2,0],[7,8,16],.18,350],
        [TEMPORAL_BOUNDS,[-.4,.85,-.7],[5,6,16],.14,330],
        [INTENT_BOUNDS,[0,1.3,0],[8,10,22],.1,360],
    ]) for (const width of [296,366,720,1200]) {
        const camera = new THREE.PerspectiveCamera(38,width/height,.1,150);
        fitPerspectiveCamera(camera,bounds,target,direction,sweep);
        const center = new THREE.Vector3(...target), vector = camera.position.clone().sub(center);
        for (const angle of [-sweep,-sweep/2,0,sweep/2,sweep]) {
            camera.position.copy(vector).applyAxisAngle(new THREE.Vector3(0,1,0),angle).add(center);
            camera.lookAt(center); camera.updateMatrixWorld();
            for (const box of bounds) for (const x of [box[0][0],box[1][0]]) for (const y of [box[0][1],box[1][1]]) for (const z of [box[0][2],box[1][2]]) {
                const p = new THREE.Vector3(x,y,z).project(camera);
                assert.ok(Math.abs(p.x)<1 && Math.abs(p.y)<1,`${width}px clipped content at ${p.x},${p.y}`);
            }
        }
    }
});
