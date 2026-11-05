import * as THREE from 'three';

/** Fit every content corner through the real horizontal and vertical field of view. */
export function fitPerspectiveCamera(camera, bounds, target, direction, sweep = 0) {
    const center = new THREE.Vector3(...target), base = new THREE.Vector3(...direction).normalize();
    const corners = [];
    const boxes = typeof bounds[0][0] === 'number' ? [bounds] : bounds;
    for (const box of boxes) for (const x of [box[0][0],box[1][0]]) for (const y of [box[0][1],box[1][1]]) for (const z of [box[0][2],box[1][2]]) corners.push(new THREE.Vector3(x,y,z).sub(center));
    const tanY = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / camera.zoom;
    const tanX = tanY * camera.aspect;
    let distance = 1;
    for (const angle of [-sweep,0,sweep]) {
        const direction = base.clone().applyAxisAngle(new THREE.Vector3(0,1,0),angle);
        const right = new THREE.Vector3().crossVectors(camera.up,direction).normalize();
        const up = new THREE.Vector3().crossVectors(direction,right).normalize();
        for (const corner of corners) {
            const depth = corner.dot(direction);
            distance = Math.max(distance,depth + 1,depth + Math.abs(corner.dot(right)) * 1.08 / tanX,
                depth + Math.abs(corner.dot(up)) * 1.08 / tanY);
        }
    }
    camera.position.copy(center).addScaledVector(base,distance);
    camera.lookAt(center); camera.updateProjectionMatrix(); camera.updateMatrixWorld();
    return distance;
}
