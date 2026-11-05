import * as THREE from 'three';
import { COLORS, lineMaterial, pointMaterial } from './materials.js';

export function buildFeatureLayer(count = 24) {
    const group = new THREE.Group();
    const points = [];
    Array.from({ length: count }, (_, index) => {
        const row = Math.floor(index / 6);
        const x = (index % 6 - 2.5) * 0.86, y = 2.4 - row * 0.64, z = row * 0.12;
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.72, 0.46), new THREE.MeshBasicMaterial({
            color: COLORS.focus, transparent: true, opacity: 0.12, side: THREE.DoubleSide }));
        mesh.position.set(x, y, z); mesh.rotation.x = -0.12;
        mesh.userData.module = index; group.add(mesh);
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(0.72, 0.46)),
            lineMaterial(COLORS.focus, 0.6));
        outline.position.copy(mesh.position); outline.rotation.copy(mesh.rotation); group.add(outline); points.push(mesh);
    });
    const dots = new THREE.Points(new THREE.BufferGeometry(), pointMaterial(COLORS.focus, 0.06));
    group.add(dots);
    return { group, points, dots };
}

export function updateFeatureLayer(layer, model) {
    model.features.forEach((feature, index) => {
        const value = feature.output?.values?.[0]?.[0]?.[0]?.[0];
        const strength = Number.isFinite(value) ? Math.min(1, Math.abs(value)) : 0;
        const mesh = layer.points[index];
        if (mesh) mesh.material.opacity = 0.12 + strength * 0.25;
    });
}
