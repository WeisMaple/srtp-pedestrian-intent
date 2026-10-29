import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLOR = {
    background: 0x08131a,
    surface: 0x14232b,
    border: 0x29454f,
    pending: 0x6b9dbd,
    current: 0xf0b46b,
    accumulated: 0x69d1bb,
    unrecorded: 0x5f6c74,
    disconnected: 0x31414a,
};
const vector = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const POINTS = {
    grid: vector(-8.05, .3, -.35),
    gridTitle: vector(-8.05, 3.84, -.35),
    input: vector(-3.65, 1.88, .8),
    weight: vector(-3.65, -1.35, -.3),
    product: vector(-.1, .48, 1.15),
    sum: vector(3.55, -.66, .35),
    bias: vector(3.55, -2.96, -.45),
    activation: vector(5.3, 2.7, -.4),
    output: vector(7.65, .38, .85),
    destination: vector(7.65, -2.02, .85),
    packet: vector(),
};

function slotPosition(slot) {
    return vector(-9.91 + slot % 4 * 1.24, 2.7 - Math.floor(slot / 4) * .69, -.24);
}

/** Retained numerical scene. Only a cursor supplied by the owner moves data. */
export function createCalculationScene(host, overlay, handlers = {}) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLOR.background);
    const camera = new THREE.OrthographicCamera(-11, 11, 5, -5, .1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;
    renderer.domElement.className = 'calculation-three-canvas';
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', '可旋转的三维实际运算。拖动改变视角；输入格、乘法、累加、激活和目标输出由箭头相连。');
    renderer.domElement.tabIndex = 0;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = .12;
    controls.rotateSpeed = .35;
    controls.minAzimuthAngle = -.25;
    controls.maxAzimuthAngle = .25;
    controls.minPolarAngle = 1.2;
    controls.maxPolarAngle = 1.57;
    controls.minZoom = .85;
    controls.maxZoom = 1.18;
    controls.zoomSpeed = .45;
    scene.add(new THREE.HemisphereLight(0xc1e9ec, 0x102330, 2.5));
    const light = new THREE.DirectionalLight(0xf2ddbc, 3.0);
    light.position.set(-6, 9, 10);
    scene.add(light);
    const rim = new THREE.DirectionalLight(0x6bb9cb, 2);
    rim.position.set(8, 4, -4);
    scene.add(rim);

    const geometries = new Set();
    const materials = new Set();
    const paths = {};
    const slabs = {};
    const slots = [];
    const anchors = { ...POINTS };
    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    geometries.add(boxGeometry);
    const beamGeometry = new THREE.CylinderGeometry(.02, .02, 1, 8);
    const arrowGeometry = new THREE.ConeGeometry(.085, .22, 12);
    geometries.add(beamGeometry);
    geometries.add(arrowGeometry);
    let calculation = null;
    let state = null;
    let page = 0;
    let cursor = 0;
    let projectedElements = [];
    let width = 1;
    let height = 1;
    let disposed = false;
    let contextLost = false;
    let request = null;
    let pointerStart = null;
    let reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const clockVector = vector();
    const raycaster = new THREE.Raycaster();

    function material(color, options = {}) {
        const result = new THREE.MeshStandardMaterial({ color, metalness: .22, roughness: .48, ...options });
        materials.add(result);
        return result;
    }

    function slab(key, position, size, color = COLOR.border) {
        const group = new THREE.Group();
        group.position.copy(position);
        const face = new THREE.Mesh(boxGeometry, material(COLOR.surface));
        face.scale.set(size[0], size[1], size[2] || .12);
        group.add(face);
        const frameGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(size[0], size[1], size[2] || .12));
        geometries.add(frameGeometry);
        const frameMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8 });
        materials.add(frameMaterial);
        const frame = new THREE.LineSegments(frameGeometry, frameMaterial);
        group.add(frame);
        const underlay = new THREE.Mesh(boxGeometry, material(0x0b1b24, { transparent: true, opacity: .8 }));
        underlay.scale.set(size[0], size[1], .05);
        underlay.position.set(.09, -.12, -.16);
        group.add(underlay);
        scene.add(group);
        const entry = { group, face, frame, key };
        slabs[key] = entry;
        return entry;
    }

    slab('grid', POINTS.grid, [5.12, 6.25, .15]);
    for (let index = 0; index < 32; index += 1) {
        const position = slotPosition(index);
        anchors[`slot-${index}`] = position.clone().add(vector(0, 0, .12));
        const cell = slab(`slot-${index}`, position, [1.12, .59, .12], COLOR.pending);
        cell.face.userData.slot = index;
        slots.push(cell);
    }
    slab('input', POINTS.input, [2.8, 1.9, .15], COLOR.current);
    slab('weight', POINTS.weight, [2.8, 1.9, .15], COLOR.pending);
    slab('product', POINTS.product, [2.9, 2.1, .26], COLOR.current);
    slab('sum', POINTS.sum, [3.05, 2.05, .32], COLOR.accumulated);
    slab('activation', POINTS.activation, [2.8, 1.28, .15], COLOR.accumulated);
    slab('output', POINTS.output, [3.1, 2.26, .3], COLOR.accumulated);
    slab('bias', POINTS.bias, [2.75, .98, .12], COLOR.border);

    // The floor is a coordinate reference; it carries no invented activations.
    const floor = new THREE.GridHelper(30, 30, 0x19323e, 0x112732);
    floor.position.set(0, -3.8, -2);
    floor.material.transparent = true;
    floor.material.opacity = .35;
    geometries.add(floor.geometry);
    materials.add(floor.material);
    scene.add(floor);

    function path(name, points, color = COLOR.border) {
        const curve = new THREE.CatmullRomCurve3(points);
        const pathGeometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40));
        geometries.add(pathGeometry);
        const pathMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: .6 });
        materials.add(pathMaterial);
        const line = new THREE.Line(pathGeometry, pathMaterial);
        scene.add(line);
        const arrow = new THREE.Mesh(arrowGeometry, material(color, { emissive: color, emissiveIntensity: .15 }));
        arrow.position.copy(curve.getPoint(.87));
        arrow.quaternion.setFromUnitVectors(vector(0, 1, 0), curve.getTangent(.87).normalize());
        scene.add(arrow);
        paths[name] = { curve, line, arrow };
        return paths[name];
    }
    const inputEnd = POINTS.input.clone().add(vector(-1.5, 0, .1));
    path('source', [slotPosition(0), vector(-5.4, 1.8, .9), inputEnd]);
    path('input', [POINTS.input.clone().add(vector(1.5, 0, 0)), vector(-1.7, 1.8, 1.15), POINTS.product.clone().add(vector(-1.55, .35, 0))]);
    path('weight', [POINTS.weight.clone().add(vector(1.5, 0, 0)), vector(-1.6, -1.2, .6), POINTS.product.clone().add(vector(-1.55, -.4, 0))]);
    path('product', [POINTS.product.clone().add(vector(1.55, 0, 0)), vector(1.85, .3, 1.4), POINTS.sum.clone().add(vector(-1.65, .2, 0))]);
    path('bias', [POINTS.bias.clone().add(vector(0, .55, .1)), vector(3.55, -2, .2), POINTS.sum.clone().add(vector(0, -1.1, 0))]);
    path('activation', [POINTS.sum.clone().add(vector(.45, 1.15, 0)), vector(4, 1.6, .6), POINTS.activation.clone().add(vector(-.3, -.72, 0))]);
    path('output', [POINTS.activation.clone().add(vector(1.5, -.1, 0)), vector(7.6, 2.7, .6), POINTS.output.clone().add(vector(0, 1.25, 0))]);

    const packetGeometry = new THREE.OctahedronGeometry(.12, 0);
    geometries.add(packetGeometry);
    const packet = new THREE.Mesh(packetGeometry, material(COLOR.current, { emissive: COLOR.current, emissiveIntensity: .7, roughness: .25 }));
    packet.visible = false;
    scene.add(packet);
    const selection = new THREE.Mesh(boxGeometry, material(COLOR.current, { transparent: true, opacity: .11, depthWrite: false }));
    selection.scale.set(1.23, .7, .08);
    scene.add(selection);

    function setRoute(name, active, completed = false) {
        const route = paths[name];
        const color = active ? COLOR.current : completed ? COLOR.accumulated : COLOR.border;
        route.line.material.color.set(color);
        route.line.material.opacity = active ? 1 : completed ? .66 : .4;
        route.arrow.material.color.set(color);
        route.arrow.material.emissive.set(color);
        route.arrow.material.emissiveIntensity = active ? .7 : .08;
        route.arrow.visible = true;
    }

    function refreshSourcePath(position) {
        const route = paths.source;
        route.curve.points[0].copy(position);
        route.curve.points[1].set(-5.4, position.y * .5 + .9, .95);
        route.line.geometry.setFromPoints(route.curve.getPoints(40));
        route.arrow.position.copy(route.curve.getPoint(.87));
        route.arrow.quaternion.setFromUnitVectors(vector(0, 1, 0), route.curve.getTangent(.87).normalize());
    }

    function setPacket(routeName, progress) {
        packet.visible = true;
        // Under reduced motion, data changes docks without continuous translation.
        const t = reducedMotion ? progress < .5 ? 0 : 1 : THREE.MathUtils.clamp(progress, 0, 1);
        paths[routeName].curve.getPoint(t, packet.position);
        anchors.packet.copy(packet.position).add(vector(0, .43, .35));
    }

    function update(nextCalculation, nextState, options = {}) {
        calculation = nextCalculation;
        state = nextState;
        page = options.page || 0;
        cursor = Number(options.cursor) || 0;
        const terms = calculation?.terms || [];
        const termIndex = state?.termIndex ?? Math.min(terms.length - 1, Math.floor(cursor));
        const phase = Number(state?.phase) || 0;
        const phaseName = state?.phaseName || 'input';
        const selectedSlot = termIndex - page * 32;
        const currentTerm = state?.currentTerm || terms[termIndex];
        const usable = currentTerm && !['disconnected', 'padding', 'unrecorded'].includes(currentTerm.membership);
        slots.forEach((slot, index) => {
            const term = terms[page * 32 + index];
            slot.group.visible = !!term;
            if (!term) return;
            const status = state?.statusByTerm?.[page * 32 + index] || 'pending';
            const selected = index === selectedSlot && termIndex < terms.length;
            const excluded = ['disconnected', 'padding', 'unrecorded'].includes(term.membership);
            const completed = status === 'accumulated';
            const color = excluded ? COLOR[term.membership] || COLOR.unrecorded : selected ? COLOR.current : completed ? COLOR.accumulated : COLOR.pending;
            slot.frame.material.color.set(color);
            slot.frame.material.opacity = excluded ? .4 : selected ? 1 : .65;
            slot.face.material.color.set(excluded ? COLOR.background : selected ? 0x453526 : completed ? 0x183c38 : 0x172d3d);
            slot.face.material.emissive.set(selected && !excluded ? COLOR.current : COLOR.background);
            slot.face.material.emissiveIntensity = selected && !excluded ? .07 : 0;
        });
        selection.visible = selectedSlot >= 0 && selectedSlot < 32 && termIndex < terms.length;
        if (selection.visible) {
            selection.position.copy(slotPosition(selectedSlot)).add(vector(0, 0, .12));
            refreshSourcePath(slotPosition(selectedSlot));
        }
        paths.source.line.visible = selection.visible && !!usable;
        paths.source.arrow.visible = selection.visible && !!usable;
        const afterTerms = cursor >= terms.length;
        setRoute('source', !!usable && phaseName === 'input');
        paths.source.arrow.visible = paths.source.line.visible;
        setRoute('input', !!usable && (phaseName === 'input' || phaseName === 'product'));
        setRoute('weight', !!usable && phaseName === 'product');
        setRoute('product', !!usable && (phaseName === 'accumulate' || phaseName === 'accumulated'), afterTerms);
        setRoute('bias', phaseName === 'bias', phaseName === 'activation' || state?.done);
        setRoute('activation', phaseName === 'activation', !!state?.done);
        setRoute('output', phaseName === 'result' || phaseName === 'activation' && state?.activationPhase >= .5, !!state?.done);
        packet.visible = false;
        if (usable && !afterTerms) {
            if (phase < .18 && selection.visible) setPacket('source', phase / .18);
            else if (phase < .35) setPacket('input', (phase - .18) / .17);
            else if (phase < .65) setPacket('weight', (phase - .35) / .3);
            else if (phase < .85) setPacket('product', (phase - .65) / .2);
        } else if (phaseName === 'bias') setPacket('bias', cursor - terms.length);
        else if (phaseName === 'activation') {
            const outputPhase = cursor - terms.length - 1;
            if (outputPhase < .5) setPacket('activation', outputPhase * 2);
            else setPacket('output', (outputPhase - .5) * 2);
        }
        Object.entries(slabs).forEach(([name, item]) => {
            if (name.startsWith('slot') || name === 'grid') return;
            const active = name === 'input' ? phaseName === 'input' : name === 'weight' || name === 'product' ? phaseName === 'product' : name === 'sum' ? phaseName === 'accumulate' || phaseName === 'accumulated' : name === phaseName;
            item.frame.material.color.set(active ? COLOR.current : ['sum', 'activation', 'output'].includes(name) ? COLOR.accumulated : COLOR.border);
            item.face.material.emissive.set(active ? COLOR.current : COLOR.background);
            item.face.material.emissiveIntensity = active ? .035 : 0;
        });
        projectedElements = Array.from(overlay.querySelectorAll('[data-calc-anchor]')).map(element => ({ element, key: element.dataset.calcAnchor }));
        projectLabels();
        render();
    }

    function projectLabels() {
        camera.updateMatrixWorld();
        projectedElements.forEach(({ element, key }) => {
            const point = anchors[key];
            if (!point) return;
            clockVector.copy(point).project(camera);
            const visible = key !== 'packet' || packet.visible;
            element.style.visibility = visible && clockVector.z >= -1 && clockVector.z <= 1 ? 'visible' : 'hidden';
            element.style.transform = `translate3d(${(clockVector.x * .5 + .5) * width}px,${(-clockVector.y * .5 + .5) * height}px,0) translate(-50%,-50%)`;
            element.style.zIndex = key === 'packet' ? '4' : key.startsWith('slot') ? '1' : '2';
        });
    }

    function render() {
        if (disposed || contextLost) return;
        renderer.render(scene, camera);
    }

    function resize() {
        width = host.clientWidth || 860;
        height = host.clientHeight || 440;
        renderer.setSize(width, height, false);
        const halfWidth = 11.2;
        const halfHeight = halfWidth * height / width;
        camera.left = -halfWidth;
        camera.right = halfWidth;
        camera.top = halfHeight;
        camera.bottom = -halfHeight;
        camera.updateProjectionMatrix();
        overlay.style.setProperty('--calc-grid-cell-width', `${Math.max(40, Math.min(59, width / 22.4 * 1.18))}px`);
        projectLabels();
        render();
    }

    function resetCamera() {
        camera.position.set(0, 5.4, 30);
        camera.zoom = 1;
        controls.target.set(0, 0, 0);
        camera.lookAt(controls.target);
        camera.updateProjectionMatrix();
        controls.update();
        projectLabels();
        render();
    }

    function focusSelected() {
        // A gentle orbit reveals the selected operand's depth while all docks remain visible.
        camera.position.set(-3.4, 5.4, 30);
        camera.zoom = 1;
        controls.target.set(0, 0, 0);
        controls.update();
        projectLabels();
        render();
    }

    function pointerDown(event) {
        pointerStart = { x: event.clientX, y: event.clientY };
    }

    function pointerUp(event) {
        if (!pointerStart || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 6) return;
        pointerStart = null;
        const bounds = renderer.domElement.getBoundingClientRect();
        const pointer = new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects(slots.filter(slot => slot.group.visible).map(slot => slot.face))[0];
        if (hit) handlers.onTermSelect?.(page * 32 + hit.object.userData.slot);
    }

    function lost(event) {
        event.preventDefault();
        contextLost = true;
        handlers.onError?.('三维绘图上下文已中断，实际运算数值仍可读取。请恢复画布。');
    }

    function restored() {
        contextLost = false;
        handlers.onError?.(null);
        resize();
    }

    function motionChanged(event) {
        reducedMotion = event.matches;
        controls.enableDamping = !reducedMotion;
        if (calculation) update(calculation, state, { page, cursor });
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    renderer.domElement.addEventListener('pointerdown', pointerDown);
    renderer.domElement.addEventListener('pointerup', pointerUp);
    renderer.domElement.addEventListener('webglcontextlost', lost);
    renderer.domElement.addEventListener('webglcontextrestored', restored);
    motionQuery.addEventListener('change', motionChanged);
    controls.enableDamping = !reducedMotion;
    controls.addEventListener('change', () => { projectLabels(); render(); });
    resetCamera();
    resize();
    function tick() {
        if (disposed) return;
        controls.update();
        request = requestAnimationFrame(tick);
    }
    request = requestAnimationFrame(tick);

    return {
        update,
        resetCamera,
        focusSelected,
        dispose() {
            disposed = true;
            cancelAnimationFrame(request);
            resizeObserver.disconnect();
            controls.dispose();
            renderer.domElement.removeEventListener('pointerdown', pointerDown);
            renderer.domElement.removeEventListener('pointerup', pointerUp);
            renderer.domElement.removeEventListener('webglcontextlost', lost);
            renderer.domElement.removeEventListener('webglcontextrestored', restored);
            motionQuery.removeEventListener('change', motionChanged);
            geometries.forEach(geometry => geometry.dispose());
            materials.forEach(item => item.dispose());
            renderer.dispose();
            renderer.domElement.remove();
        },
    };
}
