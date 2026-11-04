import * as THREE from 'three';

export function createCameraRig(host) {
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 300);
    let motion = null;
    function setAspect(width, height) {
        camera.aspect = Math.max(1, width) / Math.max(1, height);
        camera.updateProjectionMatrix();
    }
    function moveTo(target, position, smooth = true) {
        if (!smooth) { camera.position.set(...position); motion = null; return; }
        motion = { target: new THREE.Vector3(...target), position: new THREE.Vector3(...position) };
    }
    function update() {
        if (!motion) return false;
        camera.position.lerp(motion.position, 0.085);
        if (camera.position.distanceTo(motion.position) < 0.02) {
            camera.position.copy(motion.position); motion = null;
        }
        return true;
    }
    return { camera, setAspect, moveTo, update, lookAt: target => camera.lookAt(...target) };
}
