import * as THREE from 'three';
import { COCO_EDGES, normalizePose } from '../domain/pose.js';
import { COLORS, lineMaterial, pointMaterial } from './materials.js';

export function buildGraphLayer() {
    const group = new THREE.Group();
    const points = new THREE.Points(new THREE.BufferGeometry(), pointMaterial(COLORS.focus, 0.11));
    points.userData.kind = 'joint';
    const bones = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial(COLORS.focus, 0.6));
    group.add(points, bones); return { group, points, bones };
}

export function updateGraphLayer(layer, pose) {
    const points = normalizePose(pose, 4.2);
    if (!points) { layer.points.visible = false; layer.bones.visible = false; return; }
    layer.points.visible = true; layer.bones.visible = true;
    layer.points.geometry.setAttribute('position', new THREE.Float32BufferAttribute(
        points.flatMap(point => [point[0], point[1], 0]), 3));
    const segments = [];
    COCO_EDGES.forEach(([a, b]) => segments.push(points[a][0], points[a][1], 0, points[b][0], points[b][1], 0));
    layer.bones.geometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
    layer.points.geometry.computeBoundingSphere(); layer.bones.geometry.computeBoundingSphere();
}
