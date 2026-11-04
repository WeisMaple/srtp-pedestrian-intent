import * as THREE from 'three';
import { COLORS } from './materials.js';

export function buildMemoryLayer() {
    const group = new THREE.Group(); const bars = [];
    for (let index = 0; index < 5; index += 1) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.2, 0.6),
            new THREE.MeshBasicMaterial({ color: COLORS.focus, transparent: true, opacity: 0.2 }));
        bar.position.set(index * 1.2, 0, 0); group.add(bar); bars.push(bar);
    }
    return { group, bars };
}

export function updateMemoryLayer(layer, memory) {
    // Recurrent events and denormalized forecast events both mark completed
    // rounds; future poses themselves are not rendered here.
    const completed = Math.max(memory.rounds?.length || 0, memory.forecasts?.length || 0);
    layer.bars.forEach((bar, index) => {
        const active = index < completed;
        bar.material.opacity = active ? 0.55 : 0.14;
        bar.scale.y = active ? 1.4 : 0.8;
    });
}
