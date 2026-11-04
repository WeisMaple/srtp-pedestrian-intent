import * as THREE from 'three';
import { COLORS, lineMaterial, pointMaterial } from './materials.js';

export function buildFeatureLayer(count = 24) {
    const group = new THREE.Group();
    const points = [];
    Array.from({ length: count }, (_, index) => {
        const x = (index % 8 - 3.5) * 1.5, y = 1.2 + Math.floor(index / 8) * 1.4, z = index * 0.12;
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.7), new THREE.MeshBasicMaterial({
            color: COLORS.focus, transparent: true, opacity: 0.06, side: THREE.DoubleSide }));
        mesh.position.set(x, y, z); mesh.rotation.x = -Math.PI / 2; group.add(mesh);
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1.1, 0.7)),
            lineMaterial(COLORS.focus, 0.35));
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
        if (mesh) mesh.material.opacity = 0.05 + strength * 0.25;
    });
}
