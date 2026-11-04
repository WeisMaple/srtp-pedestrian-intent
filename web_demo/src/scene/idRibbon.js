import * as THREE from 'three';
import { historyPoses } from '../domain/tracks.js';
import { buildSkeletonGroup } from './skeletonLayer.js';
import { COLORS, lineMaterial } from './materials.js';

export function buildRibbonGroup(track, { maxFrames = 50, spacing = 0.62 } = {}) {
    const group = new THREE.Group();
    const poses = historyPoses(track).slice(-maxFrames);
    poses.forEach((pose, index) => {
        if (!pose) {
            group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(index * spacing, -0.8, 0), new THREE.Vector3(index * spacing, 0.8, 0)]),
                lineMaterial(COLORS.muted, 0.4)));
            return;
        }
        const skeleton = buildSkeletonGroup(pose, { color: COLORS.focus, size: 1.5 });
        skeleton.position.set(index * spacing, 0, 0); group.add(skeleton);
    });
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -1.3, 0), new THREE.Vector3(Math.max(1, poses.length - 1) * spacing, -1.3, 0)]),
        lineMaterial(COLORS.current, 0.5)));
    return group;
}
