import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BAYS, cameraForStep } from './layout.js';
import { COLORS, lineMaterial, glowMaterial } from './materials.js';
import { buildRibbonGroup } from './idRibbon.js';
import { ribbonSignature } from '../domain/tracks.js';
import { buildFeatureLayer, updateFeatureLayer } from './featureLayer.js';
import { buildMemoryLayer, updateMemoryLayer } from './memoryLayer.js';
import { buildGraphLayer, updateGraphLayer } from './graphLayer.js';
import { buildIntentLayer, updateIntentLayer } from './intentLayer.js';
import { fitInBay } from './fit.js';

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;

export function createCorridorScene(host, { onPick, onError, onCameraInteract } = {}) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);
    scene.fog = new THREE.Fog(COLORS.background, 60, 150);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.className = 'corridor-canvas';
    renderer.domElement.setAttribute('aria-label', '六步神经网络长廊。拖动旋转，滚轮缩放，点击对象检查计算。');
    renderer.domElement.setAttribute('role', 'img');
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 300);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.minDistance = 4; controls.maxDistance = 80; controls.maxPolarAngle = Math.PI * 0.86;
    controls.enabled = true;
    // A user drag, wheel zoom or click cancels the scripted flight and pauses
    // auto narration; the timeline keeps its current position.
    controls.addEventListener('start', () => { motion = null; onCameraInteract?.(); });
    scene.add(new THREE.HemisphereLight(0xcdeaff, 0x061225, 1.6));
    const sun = new THREE.DirectionalLight(0x9fe8ff, 1.4); sun.position.set(-8, 16, 12); scene.add(sun);
    const floor = new THREE.GridHelper(140, 70, 0x17456b, 0x0c2b47);
    floor.position.y = -3.2; floor.material.transparent = true; floor.material.opacity = 0.4; scene.add(floor);
    const root = new THREE.Group(); scene.add(root);
    const bays = new Map();
    const pickables = [];
    let step = 1, motion = null, disposed = false, frame = 0, lastTime = performance.now();

    function makeLabel(text, position, size = 0.32, color = COLORS.line, compact = false) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = compact ? 128 : 512; canvas.height = 64;
        context.font = '500 32px "SFMono-Regular", Menlo, monospace';
        context.textAlign = 'center'; context.textBaseline = 'middle';
        context.fillStyle = '#' + new THREE.Color(color).getHexString();
        context.fillText(text, canvas.width / 2, 32);
        const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
        sprite.position.set(...position);
        sprite.scale.set(compact ? 0.72 : 5 * size * 2, compact ? 0.32 : 0.62 * size * 2, 1);
        sprite.renderOrder = 4; return sprite;
    }

    function bayGroup(bay) {
        const group = new THREE.Group(); group.position.set(...bay.position); root.add(group);
        const plate = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 3.4), glowMaterial(COLORS.focus, 0.035));
        plate.position.set(0, 1.9, 0); plate.rotation.x = -Math.PI / 2; group.add(plate);
        const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(5.8, 4.2, 3.2)), lineMaterial(COLORS.line, 0.28));
        box.position.set(0, 1.4, 0); group.add(box);
        group.add(makeLabel('0' + bay.step + ' / ' + bay.label, [0, 3.16, 0], 0.55));
        group.userData.bay = bay; pickables.push(box); return group;
    }
    BAYS.forEach(bay => bays.set(bay.step, bayGroup(bay)));

    const feature = buildFeatureLayer(24);
    feature.points.forEach((mesh, index) => {
        const label = makeLabel(`#${index}`, [mesh.position.x, mesh.position.y, mesh.position.z + 0.03], 0.12, COLORS.line, true);
        feature.group.add(label);
    });
    bays.get(2).add(feature.group); fitInBay(feature.group);
    feature.points.forEach(mesh => pickables.push(mesh));
    const memory = buildMemoryLayer();
    bays.get(4).add(memory.group); fitInBay(memory.group);
    const graph = buildGraphLayer();
    bays.get(5).add(graph.group);
    pickables.push(graph.points);
    const intent = buildIntentLayer(['其他', '不过街', '过街']);
    bays.get(6).add(intent.group); fitInBay(intent.group);

    const ribbon = new THREE.Group();
    bays.get(3).add(ribbon);
    let trackSignature = 'none';
    function setTrack(track, { frame = null, poses = null } = {}) {
        // Same ID does not mean the same ribbon: history and pose change on
        // every snapshot. Rebuild only when the bounded content signature moves.
        const signature = ribbonSignature(track, { frame, poses });
        if (signature === trackSignature) return;
        trackSignature = signature;
        while (ribbon.children.length) {
            const child = ribbon.children[ribbon.children.length - 1];
            ribbon.remove(child);
            disposeObject(child);
        }
        if (track) ribbon.add(buildRibbonGroup(track, { poses }));
        fitInBay(ribbon);
    }

    function frameCamera(animate = true) {
        const pose = cameraForStep(step, { aspect: camera.aspect });
        bays.forEach((group, number) => { group.visible = number === step; });
        if (!animate || prefersReducedMotion()) {
            controls.target.set(...pose.target);
            camera.position.set(...pose.position); camera.lookAt(...pose.target); motion = null; return;
        }
        motion = { target: V(...pose.target), position: V(...pose.position) };
    }
    function updateCamera(now = performance.now()) {
        const delta = Math.min(0.1, Math.max(0, (now - lastTime) / 1000));
        lastTime = now;
        if (!motion) return;
        camera.position.lerp(motion.position, 1 - Math.pow(1 - 0.08, delta * 60));
        controls.target.lerp(motion.target, 1 - Math.pow(1 - 0.08, delta * 60));
        if (camera.position.distanceTo(motion.position) < 0.03) {
            camera.position.copy(motion.position); controls.target.copy(motion.target); motion = null;
        }
    }
    function resize() {
        const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
        renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
        frameCamera(false);
    }
    const observer = new ResizeObserver(resize); observer.observe(host); resize();

    function select(event) {
        if (motion) return;
        const rect = renderer.domElement.getBoundingClientRect();
        const raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 0.3;
        raycaster.params.Line.threshold = 0.3;
        raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
            -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
        for (const hit of raycaster.intersectObjects(pickables, false)) {
            let ancestor = hit.object;
            while (ancestor && ancestor.visible) ancestor = ancestor.parent;
            if (ancestor) continue;
            const module = hit.object.userData?.module;
            if (Number.isInteger(module)) {
                if (step === 2) onPick?.({ kind: 'module', module });
                return;
            }
            if (hit.object.userData?.kind === 'joint') {
                if (step === 5 && Number.isInteger(hit.index)) onPick?.({ kind: 'joint', joint: hit.index });
                return;
            }
            const bay = hit.object.parent?.userData?.bay;
            if (bay) { onPick?.({ kind: 'bay', step: bay.step }); return; }
        }
    }
    function contextLost(event) {
        event.preventDefault(); onError?.('三维渲染上下文已中断，数值台仍可使用。');
    }
    renderer.domElement.addEventListener('pointerup', select);
    renderer.domElement.addEventListener('webglcontextlost', contextLost);

    function animate(now) {
        if (disposed) return;
        frame = requestAnimationFrame(animate);
        updateCamera(now);
        if (controls.enabled) controls.update();
        renderer.render(scene, camera);
    }
    frameCamera(false); frame = requestAnimationFrame(animate);

    function disposeObject(object) {
        object.traverse(child => {
            child.geometry?.dispose?.();
            const materials = Array.isArray(child.material) ? child.material : child.material ? [child.material] : [];
            materials.forEach(material => {
                Object.values(material).forEach(value => { if (value?.isTexture) value.dispose(); });
                material.dispose?.();
            });
        });
    }

    return {
        setStep(next) { step = next; frameCamera(true); },
        reset() { motion = null; frameCamera(true); },
        setTrack,
        setLive(model) {
            setTrack(model.track, { frame: model.frame, poses: model.history?.poses || null });
            updateFeatureLayer(feature, model);
            updateMemoryLayer(memory, model.memory);
            fitInBay(memory.group);
            updateGraphLayer(graph, model.track?.keypoints || null);
            fitInBay(graph.group);
            updateIntentLayer(intent, model.intent.probabilities);
            fitInBay(intent.group);
        },
        dispose() {
            disposed = true; cancelAnimationFrame(frame); observer.disconnect();
            renderer.domElement.removeEventListener('pointerup', select);
            renderer.domElement.removeEventListener('webglcontextlost', contextLost);
            controls.dispose(); disposeObject(scene);
            renderer.dispose(); renderer.domElement.remove();
        },
    };
}
