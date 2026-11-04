import * as THREE from 'three';
import { COLORS } from './materials.js';

export function buildIntentLayer(labels) {
    const group = new THREE.Group();
    const bars = labels.map((label, index) => {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7),
            new THREE.MeshBasicMaterial({ color: index === 2 ? COLORS.focus : COLORS.muted, transparent: true, opacity: 0.65 }));
        bar.position.set((index - 1) * 1.4, 0, 0); group.add(bar); return bar;
    });
    return { group, bars };
}

export function updateIntentLayer(layer, probabilities) {
    layer.bars.forEach((bar, index) => {
        const value = Array.isArray(probabilities) ? probabilities[index] : 0;
        const height = Math.max(0.04, (Number.isFinite(value) ? value : 0) * 6);
        bar.scale.y = height; bar.position.y = height / 2 - 2;
        bar.material.opacity = Array.isArray(probabilities) ? 0.35 + 0.65 * Math.min(1, value * 2) : 0.2;
    });
}
