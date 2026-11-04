import * as THREE from 'three';

export const COLORS = {
    background: 0x061225, grid: 0x17456b, line: 0xe8f1ff, focus: 0x6fd6ff,
    current: 0xffb454, positive: 0xa8e9ff, negative: 0xff8a7a, muted: 0x5f86ab, danger: 0xff6b6b,
};

export function lineMaterial(color = COLORS.line, opacity = 0.5) {
    return new THREE.LineBasicMaterial({ color, transparent: true, opacity });
}

export function glowMaterial(color = COLORS.focus, opacity = 0.5) {
    return new THREE.MeshBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false });
}

export function pointMaterial(color = COLORS.focus, size = 0.08) {
    return new THREE.PointsMaterial({ color, size, transparent: true, opacity: 0.9, sizeAttenuation: true });
}
