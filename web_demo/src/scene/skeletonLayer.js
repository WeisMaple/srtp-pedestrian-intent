import * as THREE from 'three';
import { COCO_EDGES, normalizePose } from '../domain/pose.js';
import { COLORS, lineMaterial } from './materials.js';

const JOINTS = 17;

export function buildSkeletonGroup(pose, { color = COLORS.focus, size = 2.4, frame = null } = {}) {
    const group = new THREE.Group();
    const jointGeometry = new THREE.BufferGeometry();
    jointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(JOINTS * 3), 3));
    const joints = new THREE.Points(jointGeometry, new THREE.PointsMaterial({
        color, size: 0.09, transparent: true, opacity: 0.95,
    }));
    const boneGeometry = new THREE.BufferGeometry();
    boneGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(COCO_EDGES.length * 6), 3));
    const bones = new THREE.LineSegments(boneGeometry, lineMaterial(color, 0.8));
    group.add(joints, bones);
    updateSkeletonGroup(group, pose, { size, frame });
    return group;
}

export function updateSkeletonGroup(group, pose, { size = 2.4, frame = null } = {}) {
    const points = normalizePose(pose, size, frame);
    const [joints, bones] = group.children;
    if (!points) { joints.visible = false; bones.visible = false; return; }
    joints.visible = true; bones.visible = true;
    const jointArray = joints.geometry.attributes.position.array;
    const boneArray = bones.geometry.attributes.position.array;
    for (let index = 0; index < JOINTS; index += 1) {
        const point = points[index] || [0, 0, 0];
        jointArray.set([point[0], point[1], 0], index * 3);
    }
    COCO_EDGES.forEach(([a, b], index) => {
        const left = points[a] || [0, 0, 0], right = points[b] || [0, 0, 0];
        boneArray.set([left[0], left[1], 0, right[0], right[1], 0], index * 6);
    });
    joints.geometry.attributes.position.needsUpdate = true;
    bones.geometry.attributes.position.needsUpdate = true;
    joints.geometry.computeBoundingSphere(); bones.geometry.computeBoundingSphere();
}
