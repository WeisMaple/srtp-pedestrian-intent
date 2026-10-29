import * as THREE from 'three';
import { createNetworkTrace } from './NetworkTrace.js';
import { traceSceneNodes } from './networkTraceData.js';
import {validGraphSlice} from '../spatialGraph.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { finite, flattenCells, tensorCells, latticeSlots, evenlySpacedIndices, normalizePose, poseCoordinateFrame, poseSampleIndex, asPose, GRAPH_EDGES, STAGE_NAMES, eventStage } from './sceneData.js';

const COLORS = { background:0x071016, floor:0x0b1820, line:0x25424a, cyan:0x68ded2, mint:0xc1f5b1, violet:0xb7a2f4, dim:0x29414a, text:'#dfebe7', muted:'#7e9c9f' };
const V = (x=0,y=0,z=0)=>new THREE.Vector3(x,y,z);
const geometryCache = () => ({ cube:new THREE.BoxGeometry(1,1,1), sphere:new THREE.SphereGeometry(1,12,8) });
const signedColor = (value,peak,weight=false) => {
    if (!finite(value)) return new THREE.Color(COLORS.dim);
    const strength = Math.pow(Math.min(1,Math.abs(value)/(peak || 1)),.45);
    return new THREE.Color(weight ? 0x483b64 : 0x1c4249).lerp(new THREE.Color(weight ? COLORS.violet : value >= 0 ? COLORS.mint : COLORS.cyan),.2+strength*.8);
};

/** A retained Three.js scene: live tensor updates change instance buffers only. */
export function createNeuralScene(host, handlers={}) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);
    scene.fog = new THREE.Fog(COLORS.background,45,130);
    const renderer = new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    renderer.domElement.className = 'neural-three-canvas';
    renderer.domElement.setAttribute('aria-label','可旋转的三维神经网络。拖动旋转，滚轮缩放，点击张量单元查看真实值。');
    renderer.domElement.setAttribute('role','img');
    renderer.domElement.tabIndex = 0;
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(38,1,.1,200);
    const controls = new OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true; controls.dampingFactor=.09; controls.minDistance=3; controls.maxDistance=85;
    controls.maxPolarAngle=Math.PI*.86; controls.rotateSpeed=.65; controls.zoomSpeed=.75;
    scene.add(new THREE.HemisphereLight(0xd9ffef,0x102d3e,2));
    const sun = new THREE.DirectionalLight(0xc8fff3,3); sun.position.set(-8,18,12); scene.add(sun);
    const rim = new THREE.DirectionalLight(0x8b7ddd,2); rim.position.set(12,8,-14); scene.add(rim);
    const root = new THREE.Group(); scene.add(root);
    const attached=createNetworkTrace(scene,host,index=>handlers.onTermSelect?.(index));
    let calculation=null,calculationState=null,lastAspect=null,baseNodes={},bufferRevision=0,autoFollow=true,focusKey=null;
    const floor = new THREE.GridHelper(100,100,0x20434c,0x11252e); floor.position.y=-4.9; floor.material.transparent=true; floor.material.opacity=.22; scene.add(floor);
    const geometry = geometryCache();
    const raycaster = new THREE.Raycaster();
    let nodes={}, stage=0, group, pickables=[], registrations=[], skeletons=[], paths=[], labels=[], stageCenters=new Map();
    let selected=null, disposed=false, frame, cameraMotion=null, dragOrigin=null, lastEvent=null, eventStarted=0,autoBeforePointer=false;
    let running=false, frozen=false, progress=1, reduceMotion=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let holdTime=0, lastTime=performance.now(), contextLost=false;
    const nodePulses=new Map(),stagePulses=new Map();
    const selection = new THREE.Box3Helper(new THREE.Box3(),COLORS.mint);
    selection.visible=false; scene.add(selection);
    const materials = new Set(), textures = new Set(), geometries = new Set();
    function mat(options={}) { const m=new THREE.MeshStandardMaterial({color:COLORS.cyan,roughness:.42,metalness:.12,...options}); materials.add(m);return m; }
    function mesh(geom,material,parent=group) {const m=new THREE.Mesh(geom,material);parent.add(m);return m;}
    function line(points,color=COLORS.line,parent=group,opacity=.8) {
        const g=new THREE.BufferGeometry().setFromPoints(points.map(p=>p.isVector3?p:V(...p)));geometries.add(g);
        const m=new THREE.LineBasicMaterial({color,transparent:true,opacity});materials.add(m);
        const l=new THREE.Line(g,m);parent.add(l);return l;
    }
    function label(text,position,{size=.35,color=COLORS.text,parent=group,width,inspect}={}) {
        const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');
        const font=48;ctx.font=`500 ${font}px "SFMono-Regular", "Noto Sans", system-ui, sans-serif`;
        const w=Math.ceil(ctx.measureText(text).width)+24;canvas.width=Math.min(2048,w);canvas.height=72;
        ctx.font=`500 ${font}px "SFMono-Regular", "Noto Sans", system-ui, sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=color;ctx.fillText(text,canvas.width/2,36);
        const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;textures.add(texture);
        const material=new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false,depthWrite:false,sizeAttenuation:false});materials.add(material);
        const sprite=new THREE.Sprite(material);sprite.userData.labelPixels=Math.max(16,size*45);sprite.userData.labelAspect=canvas.width/72;const unit=sprite.userData.labelPixels*2*Math.tan(camera.fov*Math.PI/360)/Math.max(1,host.clientHeight);sprite.scale.set(unit*sprite.userData.labelAspect,unit,1);sprite.position.set(...position);sprite.renderOrder=5;parent.add(sprite);labels.push(sprite);
        if(inspect){sprite.userData.inspect=inspect;pickables.push(sprite);}return sprite;
    }
    function wireBox(position,size,color=COLORS.line,parent=group) {
        const box=new THREE.BoxGeometry(...size),edges=new THREE.EdgesGeometry(box);box.dispose();geometries.add(edges);
        const material=new THREE.LineBasicMaterial({color,transparent:true,opacity:.7});materials.add(material);
        const m=new THREE.LineSegments(edges,material);m.position.set(...position);parent.add(m);return m;
    }
    function route(points,{stage:routeStage=stage,color=COLORS.cyan,parent=group,node}={}) {
        const curve=new THREE.CatmullRomCurve3(points.map(p=>p.isVector3?p:V(...p)));
        line(curve.getPoints(36),color,parent,.46);
        const dot=mesh(geometry.sphere,mat({color,emissive:color,emissiveIntensity:.8,roughness:.2}),parent);dot.scale.setScalar(.075);dot.visible=false;
        paths.push({curve,dot,stage:routeStage,node});return curve;
    }
    function lattice(key,position,{cols=8,rows=4,depth=1,spacing=.21,cell=.14,weight=false,title,valueLabel,shape,stage:cellStage=stage,parent=group,getCells,capacity}={}) {
        const total=capacity || cols*rows*depth;
        const material=mat({color:0xffffff,emissive:weight?0x241535:0x10312b,emissiveIntensity:.2,transparent:false});
        const instanced=new THREE.InstancedMesh(geometry.cube,material,total);instanced.position.set(...position);instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);parent.add(instanced);
        const registration={key,mesh:instanced,cols,rows,depth,spacing,cell,weight,valueLabel:valueLabel || title,stage:cellStage,getCells,cells:[],shape,capacity:total};
        instanced.userData.registration=registration;registrations.push(registration);pickables.push(instanced);
        wireBox([position[0]+(cols-1)*spacing/2,position[1]-(rows-1)*spacing/2,position[2]+(depth-1)*spacing/2],[cols*spacing+.14,rows*spacing+.14,depth*spacing+.14],weight?0x4b4166:0x35525b,parent);
        if(title)label(title,[position[0]+(cols-1)*spacing/2,position[1]+.42,position[2]],{size:.26,parent,inspect:{node:key,label:title,stage:cellStage}});
        return registration;
    }
    function arrayCells(key,count=256){return n=>flattenCells(n[key]?.values || [],count).map(c=>({...c,index:c.index.map((i,axis)=>n[key]?.indices?.[axis]?.[i] ?? i)}));}
    function featureCells(key,limits=[4,6,6]){return n=>tensorCells(n[key],limits);}
    function skeleton(key,timestep,position,{scale=3,stage:poseStage=stage,parent=group,color=COLORS.cyan}={}) {
        const subgroup=new THREE.Group();subgroup.position.set(...position);parent.add(subgroup);
        const joints=new THREE.InstancedMesh(geometry.sphere,mat({color:0xffffff,roughness:.35,emissive:0x143b35,emissiveIntensity:.2}),17);subgroup.add(joints);
        const lineGeometry=new THREE.BufferGeometry();lineGeometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(GRAPH_EDGES.length*6),3));geometries.add(lineGeometry);
        const material=new THREE.LineBasicMaterial({color,transparent:true,opacity:.85});materials.add(material);
        const bones=new THREE.LineSegments(lineGeometry,material);subgroup.add(bones);
        const entry={key,timestep,group:subgroup,joints,bones,scale,stage:poseStage,points:[],color};joints.userData.skeleton=entry;skeletons.push(entry);pickables.push(joints);return entry;
    }
    function stagePlaque(title,subtitle,position,number,parent=group) {
        label(title,[position[0],position[1]+3.2,position[2]],{size:.52,parent,inspect:{node:`stage.${number}`,label:title,stage:number}});
        label(subtitle,[position[0],position[1]+2.55,position[2]],{size:.26,color:COLORS.muted,parent});
        stageCenters.set(number,V(...position));
    }
    function buildOverview(){
        const centers=[[-10,0,-4],[-3.3,0,-4],[3.3,0,-4],[10,0,-4],[10,0,4],[2.8,0,4],[-5,0,4]];
        const descriptors=[['像素 → 关节','23 层 · 三尺度特征',1],['轨迹记忆','50 次有效观测',2],['递归预测','10 层 LSTM · 5 轮',3],['未来骨架','50 × 17 × 3',4],['空间聚合','17 关节 · 3 个邻接分区',5],['时间聚合','9 帧卷积核 · 10 块',6],['行为输出','256 特征 → 3 分数',7]];
        descriptors.forEach(([title,sub,s],i)=>{
            const p=centers[i];stagePlaque(title,sub,p,s);
            if(s===1){for(let j=0;j<4;j++)lattice(`yolo.${[0,4,8,22][j]}`,[p[0]-.9+j*.27,1.5,p[2]+j*.52],{cols:4,rows:4,depth:2,spacing:.4,cell:.3,stage:s,getCells:featureCells(`yolo.${[0,4,8,22][j]}`,[2,4,4])});}
            else if(s===2 || s===4){for(let j=0;j<4;j++)skeleton(s===2?'history':'future',j*12,[p[0]-.6+j*.35,.2,p[2]+j*.38],{scale:3,stage:s,color:s===4?COLORS.mint:COLORS.cyan});}
            else if(s===3){for(let j=0;j<5;j++)lattice(`lstm.layer.${[0,2,4,6,9][j]}`,[p[0]-.8,1.6-j*.48,p[2]+j*.25],{cols:4,rows:2,depth:2,spacing:.38,cell:.26,stage:s,getCells:arrayCells(`lstm.layer.${[0,2,4,6,9][j]}`,16)});}
            else if(s===5){skeleton('future',25,[p[0],.3,p[2]],{scale:3.8,stage:s});for(let j=0;j<3;j++)lattice(`stgcn.${j*4}.spatial`,[p[0]+1.15,1.1,p[2]-1+j*.7],{cols:4,rows:4,spacing:.26,cell:.2,stage:s,getCells:featureCells(`stgcn.${j*4}.spatial`,[4,4])});}
            else if(s===6){for(let j=0;j<5;j++)lattice(`stgcn.${j*2}.temporal`,[p[0]-1.3+j*.55,1.5,p[2]+j*.22],{cols:3,rows:6,spacing:.26,cell:.18,stage:s,getCells:featureCells(`stgcn.${j*2}.temporal`,[6,3])});}
            else{lattice('stgcn.pool',[p[0]-1.8,1.5,p[2]],{cols:8,rows:8,depth:4,spacing:.22,cell:.15,stage:s,getCells:arrayCells('stgcn.pool')}); scoreBars([p[0]+.9,-1.2,p[2]],{small:true,stage:s});}
            if(i<centers.length-1){const next=centers[i+1];route([[p[0],-2.2,p[2]],[p[0]+(next[0]-p[0])*.5,-2.45,(p[2]+next[2])/2],[next[0],-2.2,next[2]]],{stage:s});}
        });
        label('点击任何层或单元，进入计算现场',[0,-3.3,0],{size:.32,color:COLORS.muted});
    }
    function buildYolo(){
        label('FEATURE PYRAMID  /  每个体素来自实际特征图',[-.5,6,0],{size:.42});
        for(let i=0;i<23;i++){
            const row=Math.floor(i/8),x=(i%8-3.5)*2.6,y=3.9-row*3.1,z=row*1.2;
            const key=`yolo.${i}`;
            lattice(key,[x,y,z],{cols:6,rows:6,depth:4,spacing:.22,cell:.16,title:`${String(i).padStart(2,'0')} · ${i<10?'骨干':'颈部'}`,getCells:featureCells(key),capacity:144});
            if(i%8<7 && i<22)route([[x+1.3,y-.7,z+.3],[x+1.8,y-.7,z-.15],[x+2.55,y-.7,z+.3]],{node:key});
        }
        label('RGB 输入块',[ -7.3,-4.3,5.7],{size:.3});label('卷积权重',[ -2.8,-4.3,5.7],{size:.3,color:'#c4b2ef'});label('BN → SiLU',[3.4,-4.3,5.7],{size:.3});
        lattice('yolo.patch',[-8,-2.6,5],{cols:3,rows:3,depth:3,spacing:.45,cell:.32,getCells:featureCells('yolo.patch',[3,3,3])});
        lattice('yolo.weights',[-3.5,-2.6,5],{cols:3,rows:3,depth:3,spacing:.45,cell:.32,weight:true,getCells:featureCells('yolo.weights',[3,3,3])});
        lattice('yolo.conv_output',[3.4,-3,5.4],{cols:1,rows:1,cell:.62,spacing:.7,getCells:arrayCells('yolo.conv_output',1)});
        route([[-6.5,-3,5.7],[-4.5,-2.1,6.5],[-3.7,-3,5.7]],{node:'yolo.0'});route([[-2,-3,5.7],[.5,-2.1,6.6],[3.4,-3,5.7]],{node:'yolo.0'});
        label('采样显示：4 通道 × 6 × 6 网格；点击可查原始坐标',[0,-5.5,4],{size:.28,color:COLORS.muted});
    }
    function buildHistory(future=false){
        const key=future?'future':'history';
        label(future?'RECURSIVE FORECAST / 五轮 × 10 帧':'TRACK MEMORY / 同一行人的有效观测',[0,4,0],{size:.42});
        for(let i=0;i<10;i++){
            const x=(i-4.5)*1.8,z=(i-4.5)*.55;
            skeleton(key,i*5,[x,.4,z],{scale:4,color:future?COLORS.mint:COLORS.cyan});
            label(future?`≈ +${i*5+1}`:`≈ t${i*5}`,[x,-2.5,z],{size:.29,color:COLORS.muted});
            if(i<9)route([[x,-2.1,z],[x+.9,-2.6,z+.25],[x+1.8,-2.1,z+.55]],{node:key});
        }
        line([[-10,-3.1,-2.8],[10,-3.1,2.8]],COLORS.cyan);
        label('时间 →',[10.3,-3.1,3],{size:.34});
        label('坐标为图像 x / y，第三量为置信度；纵深用于时间排列',[0,-4.1,3],{size:.29,color:COLORS.muted});
        label('稀疏时间点就近展示；点选骨架可读原始时间索引',[0,-4.8,3],{size:.26,color:COLORS.muted});
        if(future){for(let i=0;i<5;i++)label(`ROUND ${i+1}`,[-7.8+i*3.6,3.1,-2.4+i*1.1],{size:.26,color:'#b4a5df'});}
    }
    function buildLstm(){
        label('LSTM / 隐状态与记忆状态在十层中传递',[0,5.4,0],{size:.42});
        for(let i=0;i<10;i++){
            const x=(i%5-2)*3.5,y=i<5?3.4:.1,z=i<5?-1:1;
            const key=`lstm.layer.${i}`;
            lattice(key,[x-.8,y,z],{cols:4,rows:4,spacing:.38,cell:.27,title:`L${i+1}  hₜ`,getCells:arrayCells(key,16)});
            lattice(key,[x-.5,y,z+1.2],{cols:4,rows:4,spacing:.38,cell:.23,valueLabel:`L${i+1} · cₜ`,getCells:n=>flattenCells(n[key]?.cell || n[key]?.cell_values || [],16).map(c=>({...c,index:c.index.map((v,axis)=>n[key]?.cell_indices?.[axis]?.[v] ?? n[key]?.indices?.[axis]?.[v] ?? v)}))});
            label('cₜ',[x+.9,y-.65,z+1.5],{size:.27,color:'#a4d2c7'});
            if(i%5<4)route([[x+.8,y-.6,z+.3],[x+1.6,y-.3,z+.5],[x+2.6,y-.6,z+.3]],{node:key});
        }
        const gateNames=['i · 输入','f · 遗忘','g · 候选','o · 输出','c · 记忆','h · 隐态'];
        gateNames.forEach((name,i)=>{
            lattice('lstm.gates',[-7.4+i*2.9,-3.1,4.7],{cols:1,rows:1,cell:.58,spacing:.65,title:name,getCells:n=>[{value:n['lstm.gates']?.values?.[i] ?? null,index:[i]}]});
            if(i<5)route([[-7+i*2.9,-3.1,4.7],[-5.9+i*2.9,-2.5,5],[-4.9+i*2.9,-3.1,4.7]],{node:'lstm.gates'});
        });
        lattice('lstm.sequence',[-8.8,-4.4,-2],{cols:50,rows:8,spacing:.35,cell:.22,getCells:n=>tensorCells(n['lstm.sequence'],[50,8]).map(c=>({...c,layoutIndex:[c.layoutIndex[1],c.index[0]]})),capacity:400});
        label('50 步时间轴（采样）× 8 个采样隐藏单元',[0,-5,1.6],{size:.28,color:COLORS.muted});
        label('实时采样 h / c；门值仅记录第 1 轮、第 1 层、单元 0 提供',[0,-6.3,6.2],{size:.28,color:COLORS.muted});
    }
    function buildSpatial(){
        label('SPATIAL GRAPH / 由邻居特征形成一个关节的输出',[0,5,0],{size:.42});
        for(let i=0;i<3;i++){
            const x=-7+i*4;
            skeleton('future',25,[x,1,-1.5+i*.8],{scale:5,color:i===1?COLORS.mint:COLORS.cyan});
            label(['自身','自身 + 向心','离心邻居'][i],[x,-2.4,-1.5+i*.8],{size:.32});
            route([[x,1,-1+i*.8],[3.4,3,1],[5.2,1,1]],{node:'stgcn.0.spatial'});
        }
        for(let i=0;i<10;i++){
            const key=`stgcn.${i}.spatial`;
            lattice(key,[4.6+(i%2)*3.5,3.2-Math.floor(i/2)*1.6,.5+Math.floor(i/2)*.65],{cols:17,rows:8,spacing:.13,cell:.09,title:`BLOCK ${i+1} · C × 17`,getCells:n=>tensorCells(n[key],[8,17]),capacity:136});
        }
        label('图像 x / y 骨架的三个邻接分区；右侧为各块的真实聚合特征',[0,-4.8,4],{size:.28,color:COLORS.muted});
    }
    function buildTemporal(){
        label('TEMPORAL CONVOLUTION / 九个时间位置共同形成输出',[0,5,0],{size:.42});
        for(let i=0;i<9;i++){
            const x=-9+i*1.75,z=-2+i*.45;
            skeleton('future',21+i,[x,1.5,z],{scale:2.6});
            lattice('stgcn.window',[x-.2,-.5,z],{cols:1,rows:8,spacing:.22,cell:.15,getCells:n=>{
                const node=n['stgcn.window'];return evenlySpacedIndices(node?.values?.length || 0,8).map(c=>({value:node.values[c]?.[i],index:[c,node.indices?.[1]?.[i]??i]}));
            }});
            lattice('stgcn.kernel',[x+.25,-.5,z+.45],{cols:1,rows:8,spacing:.22,cell:.13,weight:true,getCells:n=>evenlySpacedIndices(n['stgcn.kernel']?.values?.length || 0,8).map(c=>({value:n['stgcn.kernel'].values[c]?.[i],index:[c,i]}))});
            label(`≈t${i<4?i-4:i===4?'':`+${i-4}`}`,[x,-2.8,z],{size:.26,color:COLORS.muted});
            route([[x,-2,z],[2,-3.5,4],[7,-1,3]],{node:'stgcn.0.temporal'});
        }
        wireBox([-2,1,-.2],[16.5,5.4,4.9],0x477c79);
        lattice('stgcn.0.temporal',[6.5,1.8,3],{cols:17,rows:8,spacing:.2,cell:.14,title:'Σ  C × 9 → 输出',getCells:n=>tensorCells(n['stgcn.0.temporal'],[8,17]),capacity:136});
        label('骨架仅作时间参照 · 紫色为当前块的实际卷积权重',[0,-4.6,4],{size:.29,color:COLORS.muted});
    }
    function scoreBars(position,{small=false,stage:barStage=stage}={}) {
        const factor=small?.6:1;
        for(let i=0;i<3;i++){
            const bar=mesh(geometry.cube,mat({color:i===2?COLORS.mint:COLORS.cyan,emissive:0x1d3528,emissiveIntensity:.35}));
            bar.position.set(position[0]+i*factor*1.5,position[1],position[2]);bar.scale.set(.7*factor,.04,.7*factor);
            bar.userData.score={i,base:position[1],factor,stage:barStage};pickables.push(bar);
            registrations.push({bar,key:'stgcn.probabilities'});
            label(['其他','不过街','过街'][i],[position[0]+i*factor*1.5,position[1]-.65*factor,position[2]],{size:.3*factor});
        }
    }
    function buildClassifier(){
        label('INTENT HEAD / 从 256 维特征到三个类别',[0,5,0],{size:.42});
        lattice('stgcn.pool',[-8,3,-1],{cols:8,rows:8,depth:4,spacing:.5,cell:.36,title:'GLOBAL POOL · 256',getCells:arrayCells('stgcn.pool')});
        for(let i=0;i<3;i++){
            lattice('stgcn.weights',[-1.8,2.7,-2+i*1.9],{cols:16,rows:4,depth:1,spacing:.2,cell:.13,weight:true,title:`W[${i}] · 256 → 1`,getCells:n=>evenlySpacedIndices(n['stgcn.weights']?.values?.[i]?.length || 0,64).map(j=>({value:n['stgcn.weights'].values[i][j],index:[i,j]}))});
            route([[-4,1,-.2],[-2.4,2,-2+i*1.9],[-1.8,1.9,-2+i*1.9]],{node:'stgcn.pool'});
            route([[1.4,2,-2+i*1.9],[3.4,1.3,i*.7],[5+i*1.5,-1,1]],{node:'stgcn.logits'});
        }
        scoreBars([5,-2.4,1]);
        label('softmax',[6.5,3.8,1],{size:.4});
        lattice('stgcn.logits',[4.7,-3.8,1],{cols:3,rows:1,spacing:1.5,cell:.45,title:'logits · 点击读取数值',getCells:arrayCells('stgcn.logits',3)});
        label('单元色彩分别按张量绝对值归一化；数值以检查器为准',[0,-5,3],{size:.28,color:COLORS.muted});
    }
    function disposeGroup(){
        if(group){root.remove(group);group.traverse(o=>{if(o.isInstancedMesh)o.dispose();});}
        materials.forEach(m=>m.dispose());materials.clear();textures.forEach(t=>t.dispose());textures.clear();geometries.forEach(g=>g.dispose());geometries.clear();
        pickables=[];registrations=[];skeletons=[];paths=[];labels=[];stageCenters=new Map();selected=null;selection.visible=false;
    }
    function build(nextStage){
        attached.clearHighlights();stage=nextStage;disposeGroup();group=new THREE.Group();root.add(group);
        [buildOverview,buildYolo,()=>buildHistory(false),buildLstm,()=>buildHistory(true),buildSpatial,buildTemporal,buildClassifier][stage]?.();
        updateBuffers();resetCamera(false);updateTrace(true);
    }
    const dummy=new THREE.Object3D();
    function updateBuffers(){
        attached.clearHighlights();bufferRevision++;
        registrations.forEach(r=>{
            if(r.bar){
                const d=r.bar.userData.score;const value=nodes['stgcn.probabilities']?.values?.[d.i];
                const height=finite(value)?Math.max(.025,value*5*d.factor):.035;
                r.bar.scale.y=height;r.bar.position.y=d.base+height/2;r.bar.material.color.set(finite(value)?d.i===2?COLORS.mint:COLORS.cyan:COLORS.dim);return;
            }
            const cells=r.getCells?r.getCells(nodes):flattenCells(nodes[r.key]?.values || [],r.capacity);
            r.cells=latticeSlots(cells,r);r.mesh.count=r.cells.length;
            const peak=Math.max(...cells.map(c=>finite(c.value)?Math.abs(c.value):0),1e-9);
            for(let i=0;i<r.cells.length;i++){
                const c=r.cells[i];let x,y,z;
                if(c?.layoutIndex?.length===3){[z,y,x]=c.layoutIndex;}
                else if(c?.layoutIndex?.length===2){[y,x]=c.layoutIndex;z=0;}
                else{x=i%r.cols;y=Math.floor(i/r.cols)%r.rows;z=Math.floor(i/(r.cols*r.rows));}
                dummy.position.set(x*r.spacing,-y*r.spacing,z*r.spacing);dummy.scale.setScalar(r.cell);dummy.updateMatrix();
                r.mesh.setMatrixAt(i,dummy.matrix);r.mesh.setColorAt(i,signedColor(c?.value,peak,r.weight));
            }
            r.mesh.instanceMatrix.needsUpdate=true;if(r.mesh.instanceColor)r.mesh.instanceColor.needsUpdate=true;r.mesh.computeBoundingSphere();
        });
        const poseFrames={history:poseCoordinateFrame(nodes.history?.values),future:poseCoordinateFrame(nodes.future?.values)};
        skeletons.forEach((s,slot)=>{
            const windowTime=s.stage===6&&calculation?.stage==='temporal'&&calculation.operation!=='residual_add'?nodes['stgcn.window']?.indices?.[1]?.[slot]:undefined;
            const time=Number.isFinite(windowTime)?windowTime:s.timestep;
            const values=nodes[s.key]?.values || [];const sampleIndex=poseSampleIndex(nodes[s.key],time,s.key==='future');const pose=asPose(values[sampleIndex]);
            const points=normalizePose(pose,s.scale,poseFrames[s.key]);s.points=points;s.pose=pose;s.actualTimestep=sampleIndex;
            points.forEach((p,i)=>{
                dummy.position.set(p[0],p[1],0);dummy.scale.setScalar(.064*s.scale/3);dummy.updateMatrix();s.joints.setMatrixAt(i,dummy.matrix);
                s.joints.setColorAt(i,new THREE.Color(pose?s.color:COLORS.dim));
            });
            s.joints.instanceMatrix.needsUpdate=true;if(s.joints.instanceColor)s.joints.instanceColor.needsUpdate=true;s.joints.computeBoundingSphere();
            const array=s.bones.geometry.attributes.position.array;GRAPH_EDGES.forEach(([a,b],i)=>{array.set([points[a][0],points[a][1],0,points[b][0],points[b][1],0],i*6);});
            s.bones.geometry.attributes.position.needsUpdate=true;s.bones.geometry.computeBoundingSphere();s.bones.material.opacity=pose?.85:.32;
        });
    }
    function cameraPose(){
        const views={0:{target:[0,.3,.2],position:[23,23,36]},1:{target:[0,-.5,1],position:[17,17,31]},2:{target:[0,.5,0],position:[14,12,26]},3:{target:[0,.2,.5],position:[16,15,29]},4:{target:[0,.5,0],position:[14,12,26]},5:{target:[.2,.3,0],position:[18,14,29]},6:{target:[-.5,0,0],position:[15,13,28]},7:{target:[0,0,0],position:[18,13,27]}};
        const result=views[stage]||views[0];const ratio=host.clientWidth/Math.max(1,host.clientHeight);
        const scale=ratio<1.35?1.35/Math.max(.65,ratio):1;
        const target=V(...result.target);
        return {target,position:V(...result.position).multiplyScalar(scale*(stage===0?.68:.82))};
    }
    function moveCamera(target,position,smooth=true){
        if(smooth&&!reduceMotion)cameraMotion={target,position};else{camera.position.copy(position);controls.target.copy(target);controls.update();cameraMotion=null;}
    }
    function resetCamera(smooth=true){const p=cameraPose();moveCamera(p.target,p.position,smooth);}
    function focusSelected(){
        if(!selected)return;const p=selected.point || V();const dir=camera.position.clone().sub(controls.target).normalize();moveCamera(p,p.clone().add(dir.multiplyScalar(7)));
    }
    function lastLayer(){const layers=Object.keys(baseNodes).filter(key=>/^lstm\.layer\.\d+$/.test(key)).map(key=>Number(key.split('.').at(-1)));return layers.length?Math.max(...layers):9;}
    function focusCurrent(outputOnly=false){
        const bounds=attached.focusBounds(outputOnly);if(!bounds)return;
        const target=bounds.getCenter(V()),size=bounds.getSize(V());
        // Reserve room for labels attached to the same objects, not a second panel.
        const horizontal=Math.max(4.5,size.x+3.5)/Math.max(.55,camera.aspect),vertical=Math.max(4,size.y+3);
        const distance=Math.max(8,Math.max(horizontal,vertical)/(2*Math.tan(camera.fov*Math.PI/360))*1.35+size.z*.5);
        const direction=V(.58,.42,1).normalize();moveCamera(target,target.clone().add(direction.multiplyScalar(distance)));
    }
    function updateTrace(force=false){
        attached.update(calculation,calculationState,{registrations,skeletons,nodes,stageCenters,revision:bufferRevision},reduceMotion);
        if(!calculation||!calculationState){focusKey=null;return;}
        const outputOnly=['bias','activation','result'].includes(calculationState.phaseName);
        const key=`${stage}:${calculation.id}:${outputOnly?'output':'inputs'}`;
        if(autoFollow&&(force||key!==focusKey))focusCurrent(outputOnly);
        focusKey=key;
    }
    function setAutoFollow(value){autoFollow=value;handlers.onAutoFollow?.(value);if(value)updateTrace(true);}
    function resize(){if(disposed)return;const width=Math.max(host.clientWidth,1),height=Math.max(host.clientHeight,1);const aspect=width/height;if(lastAspect!==null&&Math.abs(lastAspect-aspect)>.01){const scale=Math.max(1,1.35/aspect)/Math.max(1,1.35/lastAspect);camera.position.sub(controls.target).multiplyScalar(scale).add(controls.target);cameraMotion=null;}lastAspect=aspect;camera.aspect=aspect;camera.updateProjectionMatrix();renderer.setSize(width,height,false);labels.forEach(label=>{const unit=label.userData.labelPixels*2*Math.tan(camera.fov*Math.PI/360)/height;label.scale.set(unit*label.userData.labelAspect,unit,1);});if(autoFollow&&calculation)updateTrace(true);}
    const resizeObserver=new ResizeObserver(resize);resizeObserver.observe(host);resize();
    function select(event){
        if(dragOrigin&&Math.hypot(event.clientX-dragOrigin.x,event.clientY-dragOrigin.y)>5)return;
        const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),camera);
        const hit=raycaster.intersectObjects(pickables,false)[0];if(!hit)return;
        let info;const r=hit.object.userData.registration,s=hit.object.userData.skeleton,score=hit.object.userData.score;
        if(r){const c=r.cells[hit.instanceId];const n=nodes[r.key];info={node:r.key,label:r.valueLabel || n?.label || r.key,index:c?.index || [],value:finite(c?.value)?c.value:null,shape:n?.shape || r.shape,source:n?.source || '结构示意 · 尚无此层数值',stage:r.stage};
            const instance=new THREE.Matrix4();r.mesh.getMatrixAt(hit.instanceId,instance);const world=r.mesh.matrixWorld.clone().multiply(instance);const box=new THREE.Box3().setFromBufferAttribute(geometry.cube.attributes.position).applyMatrix4(world);selection.box.copy(box);selection.visible=true;
        }else if(s){const p=s.pose?.[hit.instanceId];info={node:s.key,label:`${s.key==='future'?'预测':'观测'} · 关节 ${hit.instanceId}`,index:s.pose?[nodes[s.key]?.indices?.[0]?.[s.actualTimestep] ?? s.actualTimestep,hit.instanceId]:[],value:p || null,shape:nodes[s.key]?.shape,source:s.pose?(nodes[s.key]?.source || '实际骨架观测'):'骨架拓扑示意 · 该时间尚无观测值',stage:s.stage};}
        else if(score){const n=nodes['stgcn.probabilities'];info={node:'stgcn.probabilities',label:['其他','不过街','过街'][score.i],index:[score.i],value:n?.values?.[score.i]??null,shape:[3],source:n?.source || '尚无分类值',stage:score.stage};}
        else info=hit.object.userData.inspect;
        if(s?.stage===5&&validGraphSlice(calculation)){handlers.onGraphTarget?.(hit.instanceId);if(autoBeforePointer)setAutoFollow(true);return;}
        if(info){selected={...info,point:hit.point.clone()};handlers.onInspect?.(info);const term=attached.termForHit(hit.object,hit.instanceId);if(term>=0)handlers.onTermSelect?.(term);else if(stage===0&&info.stage)handlers.onStageSelect?.(info.stage);}
    }
    const pointerDown=e=>{autoBeforePointer=autoFollow;dragOrigin={x:e.clientX,y:e.clientY};cameraMotion=null;};
    renderer.domElement.addEventListener('pointerdown',pointerDown,true);renderer.domElement.addEventListener('pointerup',select);
    const lost=e=>{e.preventDefault();contextLost=true;handlers.onError?.('三维渲染上下文已中断，请恢复三维画布。');};
    renderer.domElement.addEventListener('webglcontextlost',lost);
    controls.addEventListener('start',()=>{cameraMotion=null;setAutoFollow(false);});
    function animate(now){
        if(disposed)return;frame=requestAnimationFrame(animate);if(contextLost||document.hidden)return;
        const delta=Math.min(.05,(now-lastTime)/1000);lastTime=now;if(!frozen)holdTime+=delta;
        if(cameraMotion){camera.position.lerp(cameraMotion.position,.085);controls.target.lerp(cameraMotion.target,.085);if(camera.position.distanceTo(cameraMotion.position)<.015)cameraMotion=null;}
        paths.forEach((p,i)=>{
            const pulseAt=p.node ? nodePulses.get(p.node) : stagePulses.get(p.stage);
            const livePhase=pulseAt===undefined?2:(holdTime-pulseAt)/1.25;
            const active=lastEvent ? livePhase>=0 && livePhase<=1 : running;
            p.dot.visible=active&&!reduceMotion&&!calculation;
            if(active){const phase=lastEvent?livePhase:((progress*5+i*.053)%1);p.dot.position.copy(p.curve.getPoint(Math.max(0,Math.min(1,phase))));}
        });
        registrations.forEach(r=>{if(r.mesh){const pulseAt=nodePulses.get(r.key);const phase=pulseAt===undefined?2:(holdTime-pulseAt)/1.25;r.mesh.material.emissiveIntensity=r.traceActive?.7:lastEvent&&phase<1?.2+Math.max(0,1-phase)*.8:.2;}});
        controls.update();renderer.render(scene,camera);attached.render(camera,host.clientWidth,host.clientHeight);
        labels.forEach(label=>{
            if(calculation?.stage!=='spatial'){label.visible=true;label.material.opacity=1;return;}
            const point=label.getWorldPosition(V()).project(camera),y=(1-point.y)*host.clientHeight/2;
            label.visible=y>170&&y<host.clientHeight-150;label.material.opacity=.55;
        });
        renderer.domElement.dataset.autoFollow=String(autoFollow);renderer.domElement.dataset.focusKey=focusKey||'';renderer.domElement.dataset.cameraDistance=camera.position.distanceTo(controls.target).toFixed(3);renderer.domElement.dataset.cameraTarget=controls.target.toArray().map(v=>v.toFixed(3)).join(',');renderer.domElement.dataset.networkInstances=registrations.reduce((n,r)=>n+(r.mesh?.count||0),skeletons.length*17);
    }
    build(0);frame=requestAnimationFrame(animate);
    return {
        setStage(next){if(next!==stage)build(next);},
        update(nextNodes,{liveEvent,playing=false,isFrozen=false,phase=1}={}){
            const incoming=nextNodes||{};
            if(liveEvent){
                if(liveEvent.job_id!==lastEvent?.job_id || liveEvent.frame!==lastEvent?.frame){nodePulses.clear();stagePulses.clear();}
                Object.entries(incoming).forEach(([key,node])=>{
                    const previous=nodes[key];
                    if(!previous || node.seq!==previous.seq || node.seq===undefined&&node.values!==previous.values){
                        nodePulses.set(key,holdTime);const changedStage=key==='history'?2:key==='future'?4:eventStage({node:key});
                        if(changedStage!==null)stagePulses.set(changedStage,holdTime);
                    }
                });
                lastEvent=liveEvent;eventStarted=holdTime;
            }else{lastEvent=null;nodePulses.clear();stagePulses.clear();}
            baseNodes=incoming;nodes=traceSceneNodes(baseNodes,calculation,lastLayer());running=playing;frozen=isFrozen;progress=phase;updateBuffers();updateTrace();
        },
        setPlayback({playing,isFrozen,phase}){running=playing;frozen=isFrozen;progress=phase;},
        setCalculation(next,state){const changed=calculation!==next;calculation=next;calculationState=state;if(changed){nodes=traceSceneNodes(baseNodes,next,lastLayer());updateBuffers();}updateTrace();},
        setAutoFollow,
        setGraphEmphasis:attached.setGraphEmphasis,
        focusCalculation(){setAutoFollow(true);},
        resetCamera(){setAutoFollow(false);resetCamera();},focusSelected,clearSelection(){selected=null;selection.visible=false;},
        dispose(){disposed=true;cancelAnimationFrame(frame);resizeObserver.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointerdown',pointerDown,true);renderer.domElement.removeEventListener('pointerup',select);renderer.domElement.removeEventListener('webglcontextlost',lost);attached.dispose();disposeGroup();Object.values(geometry).forEach(g=>g.dispose());selection.geometry.dispose();selection.material.dispose();floor.geometry.dispose();floor.material.dispose();renderer.dispose();renderer.domElement.remove();},
    };
}
