import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { temporalWindow } from '../domain/temporal.js';
import { fitPerspectiveCamera } from './perspectiveFit.js';
import { TEMPORAL_BOUNDS } from './computationBounds.js';

const COLORS = { background: 0x061225, positive: 0x65d5ff, negative: 0xf08783, current: 0xffb23e, line: 0x315573 };
const dispose = group => group.traverse(object => {
    object.geometry?.dispose();
    for (const material of (Array.isArray(object.material) ? object.material : object.material ? [object.material] : [])) {
        material.map?.dispose(); material.dispose();
    }
});

/** Depth is input channel, x is time. Only actual recorded channel/window cells are drawn. */
export function createTemporalScene(host, { onSelect, onInteract, onError } = {}) {
    const scene = new THREE.Scene(); scene.background = new THREE.Color(COLORS.background);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', '三维时间卷积：时间沿横轴排列，输入通道向纵深展开。输入、卷积核、乘积与输出联动。');
    renderer.domElement.dataset.renderReady = 'false';
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 150);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.enablePan = false; controls.enableZoom = true;
    controls.minDistance = 8; controls.maxDistance = 65;
    controls.minPolarAngle = 0.2; controls.maxPolarAngle = Math.PI * 0.48;
    controls.addEventListener('start', () => onInteract?.());
    scene.add(new THREE.HemisphereLight(0xffffff, 0x183354, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(2, 9, 6); scene.add(light);
    let group = new THREE.Group(); scene.add(group);
    let cells = [], highlighted = [], output, marker, line, frame, visible = true, stopped = false;
    let data = null, selectedChannel = 0, currentFlow = null;
    let preset = 'overview';
    let playing = false, orbitStart = 0, orbitBase = new THREE.Vector3();
    const reduced = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    function label(text, x, y, z, width = 2, color = '#b5d5ec') {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d'); ctx.font = '500 52px system-ui';
        canvas.width = Math.ceil(ctx.measureText(text).width + 24); canvas.height = 80;
        ctx.font = '500 52px system-ui'; ctx.fillStyle = color;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, canvas.width / 2, 40);
        const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
        sprite.position.set(x, y, z); sprite.scale.set(Math.min(width, 0.3 * canvas.width / 80), 0.3, 1); group.add(sprite);
    }
    function cell(value, x, y, z, term, kind, padding, max) {
        const color = value < 0 ? COLORS.negative : COLORS.positive;
        const material = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.1,
            transparent: true, opacity: padding ? 0.32 : 0.6 + 0.35 * Math.min(1, Math.abs(value || 0) / max), wireframe: padding });
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.5, 0.52), material);
        mesh.position.set(x, y, z); mesh.userData = { term, kind, base: color, opacity: material.opacity, value };
        group.add(mesh); cells.push(mesh);
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color: padding ? 0x8ea8bc : 0x40708c, transparent: true, opacity: 0.7 }));
        outline.position.copy(mesh.position); group.add(outline);
        return mesh;
    }
    function view(next = preset) {
        preset = next;
        const aspect = Math.max(0.2, host.clientWidth / Math.max(1, host.clientHeight));
        camera.aspect = aspect;
        camera.zoom = next === 'focus' ? 1.2 : 1;
        controls.target.set(-.4, .85, -.7);
        fitPerspectiveCamera(camera,TEMPORAL_BOUNDS,[-.4,.85,-.7],
            next === 'top' ? [0,18,.1] : next === 'focus' ? [2,5,14] : [5,6,16],next === 'overview' ? .14 : 0);
        controls.update();
        orbitBase.copy(camera.position).sub(controls.target); orbitStart = performance.now();
    }
    function resize() {
        renderer.setSize(Math.max(1, host.clientWidth), Math.max(1, host.clientHeight), false); view();
    }
    const observer = new ResizeObserver(resize); observer.observe(host);
    const visibility = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting !== false; }); visibility.observe(host);

    function setData(calculation, channel) {
        data = calculation; selectedChannel = channel;
        scene.remove(group); dispose(group); group = new THREE.Group(); scene.add(group); cells = []; highlighted = [];
        const window = temporalWindow(calculation, channel);
        if (!window) return;
        const channels = Array.from({ length: Math.min(4, window.channels) }, (_, i) => (channel + i) % window.channels);
        const terms = channels.flatMap(c => temporalWindow(calculation, c).terms);
        const maxInput = Math.max(1e-9, ...terms.map(term => Math.abs(term.input || 0)));
        const maxWeight = Math.max(1e-9, ...terms.map(term => Math.abs(term.weight || 0)));
        channels.forEach((c, layer) => {
            const row = temporalWindow(calculation, c);
            const z = -layer * 0.6, y = .3 + layer * .62;
            row.terms.forEach((term, q) => {
                cell(term.input, -6.8 + q * 0.43, y, z, c * window.kernel + q, 'input', term.padding, maxInput);
                cell(term.weight, -1.9 + q * 0.43, y, z, c * window.kernel + q, 'weight', false, maxWeight);
                if (layer === 0) {
                    label(String(term.inputIndex[1]), -6.8 + q * 0.43, -0.33, 0.5, 0.75);
                    label(String(q), -1.9 + q * 0.43, -0.33, 0.5, 0.75);
                }
            });
            label(`c=${c}`, -7.3, y, z, 1);
        });
        label('输入窗口 X', -5, 3.1, -1.8, 3, '#65d5ff');
        label('卷积核 W', -0.15, 3.1, -1.8, 3, '#65d5ff');
        label('x × w', 3.5, 1.4, -0.4, 2.2, '#ffb23e');
        label('Σ + b → y', 6, 1.4, -0.4, 2.5, '#ffb23e');
        label('时间 t →', -5, -1, 1, 2.5);
        label('核位置 q →', -0.15, -1, 1, 2.5);
        const product = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), new THREE.MeshStandardMaterial({ color: COLORS.current, transparent: true, opacity: 0.55 }));
        product.position.set(3.5, 0.15, 0); group.add(product);
        output = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: COLORS.positive, transparent: true, opacity: 0.45 }));
        output.position.set(6, 0.15, 0); group.add(output);
        line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 5 }, () => new THREE.Vector3())), new THREE.LineBasicMaterial({ color: COLORS.current, transparent: true, opacity: 0.9 })); group.add(line);
        marker = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffcf6a })); group.add(marker);
        const grid = new THREE.GridHelper(18, 36, COLORS.line, 0x102e48); grid.position.y = -0.65;
        grid.material.transparent = true; grid.material.opacity = 0.4; group.add(grid);
        setCursor(currentFlow);
    }
    function setCursor(flow) {
        currentFlow = flow;
        if (!flow || !data || !line) return;
        highlighted.forEach(mesh => { mesh.material.color.setHex(mesh.userData.base); mesh.material.emissive.setHex(0); mesh.material.opacity = mesh.userData.opacity; });
        const kernel = data.metadata.kernel_size;
        const index = flow.biasApplied ? selectedChannel * kernel + kernel - 1 : flow.termIndex;
        highlighted = cells.filter(mesh => mesh.userData.term === index);
        highlighted.forEach(mesh => { mesh.material.color.setHex(COLORS.current); mesh.material.emissive.setHex(0x573100); mesh.material.opacity = 1; });
        const input = highlighted.find(mesh => mesh.userData.kind === 'input'), weight = highlighted.find(mesh => mesh.userData.kind === 'weight');
        if (!input || !weight) return;
        const points = [input.position, new THREE.Vector3(input.position.x, input.position.y + .7, input.position.z), weight.position,
            new THREE.Vector3(3.5, 0.15, 0), output.position];
        const positions = line.geometry.attributes.position;
        points.forEach((p, i) => positions.setXYZ(i, p.x, p.y, p.z)); positions.needsUpdate = true; line.geometry.computeBoundingSphere();
        const pathPosition = flow.biasApplied ? 4 : Math.min(3.99, flow.phase * 4);
        const segment = Math.min(3, Math.floor(pathPosition));
        marker.position.lerpVectors(points[segment], points[segment + 1], pathPosition - segment);
        output.material.color.setHex(flow.biasApplied ? COLORS.current : COLORS.positive);
        output.material.opacity = flow.biasApplied ? 1 : 0.45;
    }
    let pointerStart;
    const start = event => { pointerStart = [event.clientX, event.clientY]; };
    const pick = event => {
        if (!pointerStart || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) return;
        const rect = renderer.domElement.getBoundingClientRect();
        const ray = new THREE.Raycaster(); ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, 1 - (event.clientY - rect.top) / rect.height * 2), camera);
        const hit = ray.intersectObjects(cells, false)[0]; if (hit) onSelect?.(hit.object.userData.term + 0.5);
    };
    const lost = event => { event.preventDefault(); onError?.('三维渲染中断，仍可使用下方逐项数值。'); };
    renderer.domElement.addEventListener('pointerdown', start); renderer.domElement.addEventListener('pointerup', pick);
    renderer.domElement.addEventListener('webglcontextlost', lost);
    function render(now = performance.now()) {
        if (stopped) return;
        frame = requestAnimationFrame(render);
        if (!visible || document.hidden) return;
        if (playing && !reduced && preset === 'overview') camera.position.copy(orbitBase).applyAxisAngle(new THREE.Vector3(0,1,0), .14 * Math.sin((now - orbitStart) / 5000)).add(controls.target);
        controls.update(); renderer.render(scene, camera); renderer.domElement.dataset.renderReady = 'true';
    }
    resize(); render();
    return { setData, setCursor, view, setPlayback(value) {
        if (value && !playing) view('overview');
        playing = value;
    }, dispose() {
        stopped = true; cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect();
        renderer.domElement.removeEventListener('pointerdown', start); renderer.domElement.removeEventListener('pointerup', pick);
        renderer.domElement.removeEventListener('webglcontextlost', lost); controls.dispose(); dispose(scene); renderer.dispose(); renderer.domElement.remove();
    } };
}
