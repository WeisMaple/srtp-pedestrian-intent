import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { fitPerspectiveCamera } from './perspectiveFit.js';
import { INTENT_BOUNDS } from './computationBounds.js';

const CYAN = 0x65d5ff, GOLD = 0xffb84c, CORAL = 0xf08783;
const colors = [0x9cafe5, CYAN, GOLD];
const shown = value => Number.isFinite(value) ? value.toFixed(3) : '—';

export function createIntentScene(host, { onInteract, onClass, onError } = {}) {
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x061225);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5)); renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', '三维意图计算：时间与关节特征平面汇入池化向量，连接三个分类节点，再输出三类分数柱。');
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 200);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.enablePan = false; controls.minDistance = 10; controls.maxDistance = 90;
    controls.maxPolarAngle = Math.PI * .47; controls.addEventListener('start', () => onInteract?.());
    scene.add(new THREE.HemisphereLight(0xe1f6ff, 0x142d46, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(-6, 12, 10); scene.add(light);
    const group = new THREE.Group(); scene.add(group);
    const matrix = new THREE.Matrix4(), scale = new THREE.Vector3(), position = new THREE.Vector3(), quaternion = new THREE.Quaternion();
    const paths = [], nodes = [], columns = [], labels = [], pickable = [];
    let frame, stopped = false, visible = true, playing = false, detailed = true, phase = 0, start = performance.now();
    const orbit = new THREE.Vector3(), reduced = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    function setText(sprite, text, color = '#d6edff') {
        if (sprite.userData.text === text && sprite.userData.color === color) return;
        sprite.userData.text = text; sprite.userData.color = color;
        const canvas = document.createElement('canvas'), ctx = canvas.getContext('2d');
        ctx.font = '600 44px system-ui'; canvas.width = Math.ceil(ctx.measureText(text).width + 24); canvas.height = 68;
        ctx.font = '600 44px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = color;
        ctx.shadowColor = '#061225'; ctx.shadowBlur = 6; ctx.fillText(text,canvas.width/2,34);
        sprite.material.map?.dispose(); sprite.material.map = new THREE.CanvasTexture(canvas);
        sprite.material.map.colorSpace = THREE.SRGBColorSpace; sprite.material.needsUpdate = true;
        sprite.scale.set(sprite.userData.height * canvas.width / 68, sprite.userData.height, 1);
    }
    function label(text, point, height = .38, color) {
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false })); sprite.userData.height = height;
        sprite.position.set(...point); group.add(sprite); setText(sprite,text,color); return sprite;
    }
    function mesh(point, size, color) {
        const item = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial({ color,
            metalness: .25, roughness: .35, emissive: color, emissiveIntensity: .12 }));
        item.position.set(...point); group.add(item); return item;
    }
    function cells(count) {
        const object = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshStandardMaterial({ metalness: .2, roughness: .45 }), count);
        object.instanceMatrix.setUsage(THREE.DynamicDrawUsage); group.add(object); return object;
    }
    function route(points, stage, classIndex = null) {
        const curve = new THREE.CatmullRomCurve3(points.map(point => new THREE.Vector3(...point)));
        const tube = new THREE.Mesh(new THREE.TubeGeometry(curve,32,.035,6,false), new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: .2 }));
        const particle = new THREE.Mesh(new THREE.SphereGeometry(.1,10,8), new THREE.MeshBasicMaterial({ color: GOLD }));
        group.add(tube,particle); paths.push({ tube,particle,curve,stage,classIndex });
    }
    const poolCells = cells(221), features = cells(256);
    label('① 通道 0 · T × V',[-7.1,3.2,-.3],.44);
    const poolLabel = label('未采集',[-7.1,-.5,2.1]);
    label('② 每个通道独立求均值',[-1.6,3,-.3],.44);
    const featureLabel = label('256 维池化向量',[-1.6,-.5,2.2]);
    label('③ 加权求和 + 偏置',[3.2,5.8,-.3],.44);
    label('Softmax → 模型分数',[8,4.6,-.3],.44).userData.result = true;
    route([[-5.5,1,.1],[-4.2,1.8,.1],[-3,1,.1]],0);
    ['其他','不过街','过街'].forEach((name,i) => {
        const y = .7+i*1.6, x=6+i*2, z=.7;
        const node = mesh([3.3,y,0],[1.15,1.05,1.15],colors[i]); nodes.push(node); node.userData.classIndex = i;
        const column = mesh([x,.5,z],[1.05,1,1.05],colors[i]); columns.push(column); column.userData.classIndex = i; column.userData.result = true;
        pickable.push(node,column);
        label(name,[3.3,y+.8,0]);
        const logit = label('—',[3.3,y,.64],.36);
        const score = label('—',[x,1.6,z],.42); score.userData.result = true;
        label(name,[x,-.5,z],.4).userData.result = true;
        labels.push({ logit,score });
        route([[.1,.7,.1],[1.5,y,0],[2.7,y,0]],1,i);
        route([[3.9,y,0],[4.8,y,-1.4],[x,.35,z]],2,i);
    });
    const floor = new THREE.GridHelper(24,48,0x315573,0x102e48); floor.position.y = -.15;
    floor.material.transparent = true; floor.material.opacity = .45; scene.add(floor);
    function instances(object, values, columns, origin, spacing, maxHeight, selected) {
        const peak = Math.max(1e-6,...values.filter(Number.isFinite));
        object.count = values.length;
        values.forEach((value,i) => {
            const height = Number.isFinite(value) ? .025 + Math.max(0,value)/peak*maxHeight : .025;
            position.set(origin[0] + (i%columns)*spacing, height/2, origin[1] + Math.floor(i/columns)*spacing);
            scale.set(spacing*.78,height,spacing*.78); matrix.compose(position,quaternion,scale); object.setMatrixAt(i,matrix);
            object.setColorAt(i,new THREE.Color(i === selected ? GOLD : Number.isFinite(value) ? CYAN : 0x18344c));
        });
        object.instanceMatrix.needsUpdate = true; if(object.instanceColor) object.instanceColor.needsUpdate = true;
        object.computeBoundingSphere();
    }
    function update(model, nextPhase, channel, selectedClass, auto, detail = true) {
        if (detailed !== detail) { detailed=detail; reset(); }
        group.children.forEach(object => { object.visible = detailed || Boolean(object.userData.result); });
        if (auto && !playing) { orbit.copy(camera.position).sub(controls.target); start = performance.now(); }
        playing = auto; phase = nextPhase;
        const pooling = model.pooling;
        instances(poolCells,pooling?.values.flat() || [],pooling?.input_shape[1] || 17,[-8.6,-1.5],.19,2.2,-1);
        instances(features,model.pool.slice(0,256),16,[-3.1,-1.5],.19,1.7,channel);
        setText(poolLabel,pooling ? `${pooling.input_shape.join(' × ')} → ${shown(pooling.actual)}` : '完整池化平面未采集');
        setText(featureLabel,model.pool.length ? `c=${channel} · ${shown(model.pool[channel])}` : '池化值未采集');
        nodes.forEach((node,i) => {
            node.material.emissiveIntensity = i === selectedClass && phase === 1 ? .7 : .1;
            node.material.wireframe = !Number.isFinite(model.logits[i]);
            setText(labels[i].logit,shown(model.logits[i]));
            const probability = model.scores?.[i];
            const height = Number.isFinite(probability) ? Math.max(.015,probability*4) : .015;
            columns[i].visible = Number.isFinite(probability); columns[i].scale.y = height; columns[i].position.y = height/2;
            columns[i].material.emissiveIntensity = model.winners.includes(i) && phase === 2 ? .65 : .1;
            labels[i].score.position.y = height+.5;
            setText(labels[i].score,Number.isFinite(probability) ? `${(probability*100).toFixed(1)}%` : '—');
        });
        paths.forEach(path => {
            const active = path.stage === phase && (phase !== 1 || path.classIndex === selectedClass)
                && (phase === 0 ? Boolean(pooling) : phase === 1 ? model.classifiers.length > 0 : Boolean(model.softmax));
            path.tube.material.opacity = active ? .9 : .12; path.tube.material.color.setHex(active ? GOLD : CYAN);
            path.particle.visible = detailed && active && !reduced;
        });
    }
    function reset() {
        const target=detailed ? [0,1.3,0] : [8,1.7,0];
        controls.target.set(...target); fitPerspectiveCamera(camera,detailed ? INTENT_BOUNDS : [INTENT_BOUNDS[3]],target,[8,10,22],.1);
        controls.update(); orbit.copy(camera.position).sub(controls.target); start = performance.now();
    }
    function resize() { const w = Math.max(1,host.clientWidth), h = Math.max(1,host.clientHeight);
        renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); reset(); }
    const observer = new ResizeObserver(resize); observer.observe(host);
    const visibility = new IntersectionObserver(entries => { visible=entries[0]?.isIntersecting !== false; }); visibility.observe(host);
    const lost = event => { event.preventDefault(); onError?.('三维画布中断，下方公式与分数仍可查看。'); };
    const raycaster = new THREE.Raycaster(); let down;
    const pointerDown = event => { down=[event.clientX,event.clientY]; };
    const pointerUp = event => {
        if (!down || Math.hypot(event.clientX-down[0],event.clientY-down[1])>6) return;
        const rect=renderer.domElement.getBoundingClientRect();
        raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,1-(event.clientY-rect.top)/rect.height*2),camera);
        const hit=raycaster.intersectObjects(pickable.filter(object => object.visible))[0];
        if(hit) onClass?.(hit.object.userData.classIndex);
    };
    renderer.domElement.addEventListener('webglcontextlost',lost);
    renderer.domElement.addEventListener('pointerdown',pointerDown); renderer.domElement.addEventListener('pointerup',pointerUp);
    function render(now) {
        if(stopped) return; frame=requestAnimationFrame(render);
        if(!visible || document.hidden) return;
        if(playing && !reduced) camera.position.copy(orbit).applyAxisAngle(new THREE.Vector3(0,1,0),.1*Math.sin((now-start)/6000)).add(controls.target);
        paths.forEach(path => path.particle.position.copy(path.curve.getPointAt(playing ? ((now-start)%1300)/1300 : .55)));
        controls.update(); renderer.render(scene,camera);
    }
    resize(); frame=requestAnimationFrame(render);
    return { update,reset,dispose() {
        stopped=true; cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect(); controls.dispose();
        renderer.domElement.removeEventListener('webglcontextlost',lost);
        renderer.domElement.removeEventListener('pointerdown',pointerDown); renderer.domElement.removeEventListener('pointerup',pointerUp);
        scene.traverse(object => { object.geometry?.dispose(); const materials=Array.isArray(object.material)?object.material:[object.material];
            materials.filter(Boolean).forEach(material => { material.map?.dispose(); material.dispose(); }); });
        renderer.dispose(); renderer.domElement.remove();
    } };
}
