import * as THREE from 'three';

/** Fit rendered geometry in local exhibit coordinates, including child transforms. */
export function fitInBay(group) {
    group.position.set(0, 0, 0); group.scale.setScalar(1);
    group.updateWorldMatrix(true, true);
    const inverse = group.matrixWorld.clone().invert();
    const box = new THREE.Box3();
    group.traverse(child => {
        if (!child.geometry || child.visible === false) return;
        child.geometry.computeBoundingBox();
        if (child.geometry.boundingBox && !child.geometry.boundingBox.isEmpty()) {
            box.union(child.geometry.boundingBox.clone().applyMatrix4(
                new THREE.Matrix4().multiplyMatrices(inverse, child.matrixWorld)));
        }
    });
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const scale = Math.min(1, 5 / (size.x || 1), 2.8 / (size.y || 1), 2.5 / (size.z || 1));
    group.scale.setScalar(scale);
    group.position.set(-center.x * scale, 1.1 - center.y * scale, -center.z * scale);
    group.updateWorldMatrix(true, true);
}
