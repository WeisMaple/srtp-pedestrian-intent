import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { fitPerspectiveCamera } from './perspectiveFit.js';
import { MEMORY_BOUNDS } from './computationBounds.js';

const CYAN = 0x65d5ff, GOLD = 0xffb84c, CORAL = 0xf08783;
const clean = group => group.traverse(object => {
    object.geometry?.dispose();
    for (const material of (Array.isArray(object.material) ? object.material : object.material ? [object.material] : [])) {
        material.map?.dispose(); material.dispose();
    }
});

/** A perspective schematic: depth separates parallel gates, not physical body depth. */
export function createLSTMScene(host, { onInteract, onError } = {}) {
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x061225);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', '三维 LSTM 单元：输入与上一隐藏状态进入四个并行门，上层记忆通路更新 c，再生成 h。可拖动旋转。');
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 150);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.enablePan = false; controls.enableZoom = true;
    controls.minDistance = 9; controls.maxDistance = 55; controls.maxPolarAngle = Math.PI * .47;
    controls.addEventListener('start', () => onInteract?.());
    scene.add(new THREE.HemisphereLight(0xe1f6ff, 0x142d46, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(-4, 10, 9); scene.add(light);
    const rim = new THREE.DirectionalLight(0x419aff, 2); rim.position.set(4, 4, -6); scene.add(rim);
    const group = new THREE.Group(); scene.add(group);
    const objects = {}, valueLabels = {}, paths = [], particles = [];
    let frame, stopped = false, visible = true, auto = false, phase = 0, phaseStart = performance.now();
    let orbitStart = performance.now(), orbitBase = new THREE.Vector3();
    const reduced = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    function label(text, position, color = '#b5d5ec', height = .35) {
        const canvas = document.createElement('canvas'), ctx = canvas.getContext('2d');
        ctx.font = '600 48px system-ui'; canvas.width = Math.ceil(ctx.measureText(text).width + 24); canvas.height = 72;
        ctx.font = '600 48px system-ui'; ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width / 2, 36);
        const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
        sprite.scale.set(height * canvas.width / 72, height, 1); sprite.position.set(...position); group.add(sprite); return sprite;
    }
    function box(key, title, position, color, size = [1.2, 1.25, 1.1]) {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial({
            color, roughness: .3, metalness: .3, transparent: true, opacity: .82, emissive: color, emissiveIntensity: .08 }));
        mesh.position.set(...position); group.add(mesh); objects[key] = mesh;
        const edge = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8 }));
        edge.position.copy(mesh.position); group.add(edge);
        label(title, [position[0], position[1] + size[1] / 2 + .45, position[2]], key === 'c' || key === 'h' ? '#ffb84c' : '#c6e5ff');
        valueLabels[key] = label('—', [position[0], position[1], position[2] + size[2] / 2 + .08], '#ffffff', .4);
    }
    function path(points, activePhase) {
        const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)), false, 'catmullrom', .2);
        const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, .035, 6, false), new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: .22 }));
        group.add(tube); paths.push({ tube, curve, phase: activePhase });
        const particle = new THREE.Mesh(new THREE.SphereGeometry(.12, 12, 8), new THREE.MeshBasicMaterial({ color: GOLD }));
        group.add(particle); particles.push(particle);
    }
    box('x', '输入 xₜ', [-6.4, .5, 2], CYAN, [1.1, 1.8, .8]);
    box('prevh', 'hₜ₋₁', [-6.4, .5, -.5], CYAN, [1.1, 1.5, .8]);
    box('prevc', '旧记忆 cₜ₋₁', [-4.2, 3.3, -1.5], CYAN, [1.2, .8, .9]);
    for (const [key, name, x] of [['f','遗忘门 f',-3.4],['i','输入门 i',-1.2],['g','候选记忆 g',1],['o','输出门 o',3.2]]) {
        box(key, name, [x, .7, 1.1], key === 'g' ? CORAL : CYAN);
        path([[-5.8,.5,2],[-5.1,-.5,2.5],[x,-.5,2.5],[x,.1,1.1]], 0);
        path([[-5.8,.5,-.5],[-5.1,.2,-1],[x,.2,-1],[x,.1,1.1]], 0);
    }
    box('c', '新记忆 cₜ', [0,3.3,-1.5], GOLD, [1.5,.9,1.1]);
    box('tanh', 'tanh(cₜ)', [3.2,3.3,-1.5], CORAL, [1.25,.8,.9]);
    box('h', '输出 hₜ', [5.7,1.5,-.3], GOLD, [1.2,1.5,1.2]);
    path([[-3.6,3.3,-1.5],[-1.8,3.3,-1.5],[-.8,3.3,-1.5]], 1);
    path([[-3.4,1.4,1.1],[-3.4,2.4,-.2],[-1.8,3.3,-1.5]], 1);
    for (const x of [-1.2,1]) path([[x,1.4,1.1],[x,2.3,.2],[0,2.8,-1.5]], 2);
    path([[.8,3.3,-1.5],[2,3.3,-1.5],[2.6,3.3,-1.5]], 3);
    path([[3.9,3.3,-1.5],[5.7,3.3,-1.5],[5.7,2.3,-.3]], 3);
    path([[3.2,1.4,1.1],[4.5,1.5,1.1],[5.1,1.5,-.3]], 3);
    label('f × cₜ₋₁ + i × g', [-.2,4.5,-1.5], '#ffb84c', .42);
    label('o × tanh(cₜ)', [5.3,3.9,-1.3], '#ffb84c');
    label('四个门并行计算 · 上层为记忆通路', [-.3,-1.6,1.1], '#8fb2cf');
    const floor = new THREE.GridHelper(20, 40, 0x315573, 0x102e48); floor.position.y = -1.15;
    floor.material.transparent = true; floor.material.opacity = .45; group.add(floor);

    function reset() {
        controls.target.set(-.4,1.2,0);
        fitPerspectiveCamera(camera,MEMORY_BOUNDS,[-.4,1.2,0],[7,8,16],.18);
        controls.update();
        orbitBase.copy(camera.position).sub(controls.target); orbitStart = performance.now();
    }
    function resize() {
        const w = Math.max(1,host.clientWidth), h = Math.max(1,host.clientHeight);
        renderer.setSize(w,h,false); camera.aspect = w / h; camera.updateProjectionMatrix(); reset();
    }
    function update(model, nextPhase, playing) {
        if (playing && !auto) { orbitBase.copy(camera.position).sub(controls.target); orbitStart = performance.now(); }
        auto = playing;
        if (phase !== nextPhase) { phase = nextPhase; phaseStart = performance.now(); }
        const active = [['x','prevh'],['f','prevc'],['i','g','c'],['o','tanh','h']][phase];
        for (const [key, mesh] of Object.entries(objects)) {
            const value = key === 'prevc' ? model.gates?.c_previous : key === 'tanh'
                ? Number.isFinite(model.gates?.c) ? Math.tanh(model.gates.c) : null
                : key === 'prevh' ? model.gates ? model.hidden[0] : null : model.gates?.[key];
            const recorded = Number.isFinite(value) || (key === 'x' && model.input.length) || (key === 'prevh' && model.hidden.length);
            mesh.material.wireframe = !recorded;
            mesh.material.opacity = recorded ? .88 : .26;
            mesh.material.emissiveIntensity = active.includes(key) ? .55 : .07;
            const text = key === 'x' ? model.input.length ? `${model.input.length} 维` : '—'
                : Number.isFinite(value) ? value.toFixed(3) : '—';
            const sprite = valueLabels[key];
            if (sprite.userData.text !== text) {
                sprite.userData.text = text;
                const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 80;
                const ctx = canvas.getContext('2d'); ctx.font = '600 44px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillStyle = '#ffffff'; ctx.shadowColor = '#061225'; ctx.shadowBlur = 6; ctx.fillText(text,128,40);
                const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
                sprite.material.map.dispose(); sprite.material.map = texture; sprite.material.needsUpdate = true; sprite.scale.set(1.22,.38,1);
            }
        }
        paths.forEach(({tube,phase:p},i) => { tube.material.opacity = p === phase ? .95 : .15; tube.material.color.setHex(p === phase ? GOLD : CYAN); particles[i].visible = p === phase && !reduced; });
    }
    const observer = new ResizeObserver(resize); observer.observe(host);
    const visibility = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting !== false; }); visibility.observe(host);
    const lost = event => { event.preventDefault(); onError?.('三维渲染中断，门值和公式仍可查看。'); };
    renderer.domElement.addEventListener('webglcontextlost', lost);
    function render(now) {
        if (stopped) return;
        frame = requestAnimationFrame(render);
        if (!visible || document.hidden) return;
        if (auto && !reduced) camera.position.copy(orbitBase).applyAxisAngle(new THREE.Vector3(0,1,0), .18 * Math.sin((now-orbitStart)/6000)).add(controls.target);
        paths.forEach(({curve},i) => { particles[i].position.copy(curve.getPointAt(auto ? ((now - phaseStart) % 900) / 900 : .55)); });
        controls.update(); renderer.render(scene,camera);
    }
    resize(); frame = requestAnimationFrame(render);
    return { update, reset, dispose() {
        stopped = true; cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect(); controls.dispose();
        renderer.domElement.removeEventListener('webglcontextlost',lost); clean(scene); renderer.dispose(); renderer.domElement.remove();
    } };
}
