import * as THREE from 'three';
import {calculationBinding,exactCellIndex,exactPoseAnchor} from './networkTraceData.js';
import {createGraphMessages} from './GraphMessages.js';
import {JOINT_NAMES,spatialNeighborhood} from '../spatialGraph.js';

const V=(x=0,y=0,z=0)=>new THREE.Vector3(x,y,z);
const number=value=>Number.isFinite(value)?value===0?'0':Math.abs(value)<.001||Math.abs(value)>99999?value.toExponential(3):Number(value.toPrecision(6)).toString():'未采集';
const COLOR={current:0xf2c779,used:0x72dcc1,pending:0x79a7cc,absent:0x34434c};

/** Only annotations: no replacement neurons, arithmetic blocks or second diagram. */
export function createNetworkTrace(scene,host,onTermSelect) {
    const group=new THREE.Group();scene.add(group);group.visible=false;
    const overlay=document.createElement('div');overlay.className='network-attached-trace';overlay.hidden=true;host.appendChild(overlay);
    const makeCard=kind=>{
        const element=document.createElement('div');element.className=`attached-value attached-${kind}`;element.dataset.role=kind;
        const title=document.createElement('span'),value=document.createElement('strong'),detail=document.createElement('small');element.append(title,value,detail);overlay.appendChild(element);
        return {element,title,value,detail,point:null};
    };
    const sourceCard=makeCard('source'),weightCard=makeCard('weight'),targetCard=makeCard('target'),packetCard=makeCard('packet');
    const graph=createGraphMessages(scene,overlay,onTermSelect);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('attached-leaders');overlay.prepend(svg);
    const cards=[sourceCard,weightCard,targetCard];
    cards.forEach(card=>{card.leader=document.createElementNS(svg.namespaceURI,'line');svg.appendChild(card.leader);});
    const sphere=new THREE.SphereGeometry(.065,12,8),packetMaterial=new THREE.MeshBasicMaterial({color:COLOR.current});
    const packet=new THREE.Mesh(sphere,packetMaterial);group.add(packet);
    const rings=[COLOR.current,COLOR.used].map(color=>{const box=new THREE.Box3Helper(new THREE.Box3(),color);box.visible=false;group.add(box);return box;});
    const edgeGeometry=new THREE.BufferGeometry();edgeGeometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(33*3),3));
    const edgeMaterial=new THREE.LineBasicMaterial({color:COLOR.current,transparent:true,opacity:.9});const edge=new THREE.Line(edgeGeometry,edgeMaterial);group.add(edge);
    const neighborGeometry=new THREE.BufferGeometry();neighborGeometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(51*2*3),3));
    const neighborMaterial=new THREE.LineBasicMaterial({color:COLOR.pending,transparent:true,opacity:.45});const neighbors=new THREE.LineSegments(neighborGeometry,neighborMaterial);group.add(neighbors);
    let context=null,current=null,state=null,bindings=null,cacheKey=null,curve=null,source=null,target=null,weight=null;
    const touched=[],active=new Set();
    function clearHighlights(){
        graph.restoreContext();
        touched.forEach(({mesh,index,color})=>{if(mesh.instanceColor){mesh.setColorAt(index,color);mesh.instanceColor.needsUpdate=true;}});touched.length=0;
        active.forEach(reg=>{reg.traceActive=false;});active.clear();
    }
    function mark(anchor,color){
        if(anchor?.registration){anchor.registration.traceActive=true;active.add(anchor.registration);}
        if(!anchor?.mesh||anchor.instance===undefined)return;
        const prior=touched.find(item=>item.mesh===anchor.mesh&&item.index===anchor.instance);
        if(prior){const rank=value=>value===COLOR.current?4:value===COLOR.used?3:value===COLOR.pending?2:1;if(rank(color)<=rank(prior.applied))return;prior.applied=color;anchor.mesh.setColorAt(anchor.instance,new THREE.Color(color));anchor.mesh.instanceColor.needsUpdate=true;return;}
        const original=new THREE.Color();anchor.mesh.getColorAt(anchor.instance,original);touched.push({mesh:anchor.mesh,index:anchor.instance,color:original,applied:color});
        anchor.mesh.setColorAt(anchor.instance,new THREE.Color(color));anchor.mesh.instanceColor.needsUpdate=true;
    }
    function resolve(binding,side,chapter) {
        if(!binding||!context)return null;
        const {registrations,skeletons,nodes,stageCenters}=context;
        const poses=skeletons.filter(s=>s.key===binding.key&&s.stage===chapter);
        if(binding.kind==='joint'&&poses.length){
            const s=binding.partition!=null?poses[Math.min(binding.partition,poses.length-1)]:poses.reduce((best,item)=>Math.abs(item.timestep-binding.time)<Math.abs(best.timestep-binding.time)?item:best,poses[0]);
            const joint=binding.joint??0,p=s.points[joint];if(!p)return null;
            const point=s.group.localToWorld(V(p[0],p[1],0));
            const port=binding.port||!exactPoseAnchor(binding,s,nodes[binding.key]);
            if(port)point.x-=.8;
            return {point,box:new THREE.Box3().setFromCenterAndSize(point,V(.32,.32,.32)),mesh:port?null:s.joints,instance:port?undefined:joint,scope:port?'port':'joint',key:binding.key,binding};
        }
        let candidates=registrations.filter(r=>r.key===binding.key),fallback=false;
        if(!candidates.length&&binding.fallback){candidates=registrations.filter(r=>r.key===binding.fallback);fallback=true;}
        let exact=-1,reg=null;
        if(!fallback){for(const r of candidates){const index=exactCellIndex(r.cells||[],binding,nodes[r.key]);if(index>=0){reg=r;exact=index;break;}}}
        if(!reg&&binding.slot!=null)reg=candidates[binding.slot];
        if(!reg&&binding.key==='stgcn.weights')reg=candidates.find(r=>r.cells?.some(c=>c.index?.[0]===binding.index?.[0]));
        if(!reg&&binding.classIndex!=null)reg=candidates.find(r=>r.bar?.userData.score.i===binding.classIndex);
        reg=reg||candidates[0];
        if(reg?.bar){
            const point=reg.bar.getWorldPosition(V());point.y+=reg.bar.scale.y/2;
            if(binding.port||fallback&&binding.fallbackPort)point.y=reg.bar.userData.score.base;
            return {point,box:new THREE.Box3().setFromObject(reg.bar),registration:reg,scope:binding.port||fallback?'port':'cell',key:reg.key,binding};
        }
        if(reg?.mesh){
            reg.mesh.updateWorldMatrix(true,false);
            if(exact>=0){const matrix=new THREE.Matrix4();reg.mesh.getMatrixAt(exact,matrix);const point=V().applyMatrix4(matrix).applyMatrix4(reg.mesh.matrixWorld);return {point,box:new THREE.Box3().setFromCenterAndSize(point,V(reg.cell*1.5,reg.cell*1.5,reg.cell*1.5)),mesh:reg.mesh,instance:exact,registration:reg,scope:'cell',key:reg.key,binding};}
            const box=new THREE.Box3().setFromObject(reg.mesh),point=box.getCenter(V()),size=box.getSize(V());
            point.x+=(side==='source'?-1:1)*Math.max(.5,size.x*.7);
            return {point,box,registration:reg,scope:'port',key:reg.key,binding};
        }
        // An unsampled operand is an annotation at its owning model stage,
        // never a fabricated cell in another layer.
        const owned=skeletons.find(s=>s.key===binding.key&&s.stage===chapter);
        const layer=registrations.find(r=>r.stage===chapter&&r.mesh);
        const point=owned?owned.group.getWorldPosition(V()):layer?new THREE.Box3().setFromObject(layer.mesh).getCenter(V()):(stageCenters.get(chapter)||V()).clone();
        point.x+=side==='source'?-1.1:1.1;
        return {point,box:new THREE.Box3().setFromCenterAndSize(point,V(1,1,1)),scope:'port',key:binding.key,binding};
    }
    function rebuild(c,ctx){
        context=ctx;scene.updateMatrixWorld(true);
        const lastLayer=Math.max(0,...Object.keys(ctx.nodes).filter(key=>/^lstm\.layer\.\d+$/.test(key)).map(key=>Number(key.split('.').at(-1))));
        bindings=c.terms.map((_,i)=>{const spec=calculationBinding(c,i,lastLayer||9);return spec?{source:resolve(spec.source,'source',c.chapter),target:resolve(spec.target,'target',c.chapter),weight:spec.weight?resolve(spec.weight,'weight',c.chapter):null}:null;});
    }
    function update(c,s,ctx,reduceMotion=false){
        clearHighlights();current=c;state=s;
        group.visible=Boolean(c&&s);overlay.hidden=!group.visible;
        if(!group.visible){graph.update(null,null,null,null);return;}
        const key=`${c.id}:${c.frame}:${c.track_id}:${ctx.revision}`;
        if(key!==cacheKey){rebuild(c,ctx);cacheKey=key;}
        ({source,target,weight}=bindings[s.termIndex]||{});if(!source||!target){group.visible=false;overlay.hidden=true;return;}
        graph.update(c,s,bindings,ctx);
        const term=c.terms[s.termIndex];
        bindings.forEach((binding,index)=>{if(!binding)return;const operand=c.terms[index];mark(binding.source,!operand.recorded||!operand.connected?COLOR.absent:index===s.termIndex&&!s.done?COLOR.current:index<s.completedCount?COLOR.used:COLOR.pending);});
        mark(target,COLOR.used);if(weight)mark(weight,0xb7a2f4);
        [source,target].forEach((anchor,i)=>{rings[i].visible=true;rings[i].box.copy(anchor.box);});
        const start=source.point.clone(),end=target.point.clone();
        if(start.distanceTo(end)<.5){start.x-=.75;end.x+=.75;}
        const middle=start.clone().lerp(end,.5);middle.y+=Math.min(1.8,Math.max(.7,start.distanceTo(end)*.2));
        curve=graph.curveForTerm(s.termIndex)||(weight&&weight.point.distanceTo(start)>.35&&weight.point.distanceTo(end)>.35
            ?new THREE.CatmullRomCurve3([start,weight.point.clone(),end],false,'centripetal')
            :new THREE.QuadraticBezierCurve3(start,middle,end));
        curve.getPoints(32).forEach((p,i)=>edge.geometry.attributes.position.setXYZ(i,p.x,p.y,p.z));edge.geometry.attributes.position.needsUpdate=true;edge.geometry.computeBoundingSphere();
        edge.visible=term.recorded&&term.connected;packet.visible=edge.visible&&!s.done&&s.cursor<c.terms.length&&!reduceMotion;
        if(c.stage==='spatial')edge.visible=false;
        // Input reaches the existing weight location, product holds there, then
        // the contribution arrives at the existing output before accumulation.
        const phase=reduceMotion?1:s.phase<.35?s.phase/.35*.5:s.phase<.65?.5:Math.min(1,.5+(s.phase-.65)/.2*.5);
        packet.position.copy(curve.getPoint(phase));
        let neighborCount=0;
        neighborGeometry.setDrawRange(0,neighborCount);neighborGeometry.attributes.position.needsUpdate=true;neighborGeometry.computeBoundingSphere();neighbors.visible=neighborCount>0;
        sourceCard.point=source.point;sourceCard.title.textContent=c.stage==='spatial'?'原关节 · 投影特征':source.scope==='cell'?'原张量单元 · 输入':'原网络输入位置';sourceCard.value.textContent=number(term.input);sourceCard.detail.textContent=`${term.label}${source.scope==='port'?' · 层级标注':''}`;
        weightCard.point=weight?.point||curve.getPoint(.5);weightCard.title.textContent=c.stage==='spatial'?'此连接的有效系数':'本项权重 / 系数';weightCard.value.textContent=number(term.weight);weightCard.detail.textContent=!term.recorded?'未采集':!term.connected?'无连接 · 不传递贡献':term.padding?'零填充':term.product===0?'连接存在 · 贡献为零':`本项乘积 ${number(term.product)}`;
        targetCard.point=target.point;targetCard.title.textContent=`原输出位置 · Σ ${s.completedCount}/${c.terms.length}`;targetCard.value.textContent=number(s.partialSum);targetCard.detail.textContent=`b ${number(c.bias)} · ${c.activation}\n${c.actual!==null?'实测':'重建'} y = ${number(c.actual??c.reconstructed)}`;
        if(c.stage==='spatial'){
            const n=spatialNeighborhood(c,s);targetCard.title.textContent=`${JOINT_NAMES[n.target]} J${n.target} · 聚合输出`;
            targetCard.detail.textContent=n.partitions.map(p=>`K${p.index} ${p.name}  Σ ${number(p.partial)}`).join('\n')+
                `\n${n.completed}/${n.messages.length} 条消息已累计\n${c.actual!==null?'独立实测':'重建'} y = ${number(c.actual??c.reconstructed)}`+
                (Number.isFinite(c.absoluteError)?`\n重建误差 ${number(c.absoluteError)}`:'');
        }
        overlay.dataset.graph=String(c.stage==='spatial');
        packetCard.point=packet.position;packetCard.title.textContent=s.phase<.35?'读取输入':s.phase<.65?'乘以系数':'计入输出';packetCard.value.textContent=number(s.phase<.35?term.input:term.product);packetCard.detail.textContent='';packetCard.element.hidden=!packet.visible;
        overlay.dataset.binding=c.id;overlay.dataset.sourceKey=source.key;overlay.dataset.targetKey=target.key;overlay.dataset.membership=term.membership;overlay.dataset.scope=source.scope;overlay.dataset.phase=s.phaseName;
    }
    function render(camera,width,height){
        if(!current||!group.visible)return;
        svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
        const project=point=>{const p=point.clone().project(camera);return {x:(p.x+1)*width/2,y:(1-p.y)*height/2,visible:p.z>-1&&p.z<1};};
        const occupied=[];
        cards.forEach((card,i)=>{
            const isGraph=current.stage==='spatial';
            if(isGraph&&i<2){card.element.hidden=true;card.leader.style.display='none';return;}
            const p=project(card.point),w=Math.min(i===2?isGraph?width<600?194:232:216:185,width-24),h=i===2?isGraph?191:105:91;
            card.element.style.width=`${w}px`;
            let x=p.x+(i===0?-w-24:i===2?24:-w/2),y=p.y+(i===1?-h-45:-h/2);
            x=Math.max(10,Math.min(width-w-10,x));y=Math.max(isGraph?155:105,Math.min(height-h-(isGraph?145:48),y));
            for(const rect of occupied){if(x<rect.x+rect.w+8&&x+w>rect.x-8&&y<rect.y+rect.h+8&&y+h>rect.y-8)y=rect.y+rect.h+12;}
            y=Math.max(isGraph?155:105,Math.min(height-h-(isGraph?145:42),y));occupied.push({x,y,w,h});
            if(isGraph&&width<600){x=width-w-12;y=315;}
            card.element.style.transform=`translate(${x}px,${y}px)`;card.element.hidden=!p.visible;card.leader.style.display=p.visible?'':'none';
            card.leader.setAttribute('x1',p.x);card.leader.setAttribute('y1',p.y);card.leader.setAttribute('x2',Math.max(x,Math.min(x+w,p.x)));card.leader.setAttribute('y2',Math.max(y,Math.min(y+h,p.y)));
        });
        graph.render(camera,width,height);
        packetCard.element.hidden=!packet.visible||current.stage==='spatial'&&width<600;
        if(!packetCard.element.hidden){const p=project(packet.position);packetCard.element.style.transform=`translate(${Math.max(5,Math.min(width-110,p.x-50))}px,${Math.max(85,Math.min(height-85,p.y+14))}px)`;}
    }
    return {
        update,render,clearHighlights,setGraphEmphasis:graph.setEmphasis,
        focusBounds(outputOnly=false){if(!source||!target)return null;const bounds=target.box.clone();if(!outputOnly){bounds.union(graph.bounds()||source.box);if(weight)bounds.union(weight.box);}return bounds;},
        termForHit(mesh,index){return bindings?.findIndex(binding=>binding?.source?.mesh===mesh&&binding.source.instance===index)??-1;},
        dispose(){clearHighlights();graph.dispose();scene.remove(group);overlay.remove();sphere.dispose();packetMaterial.dispose();edgeGeometry.dispose();edgeMaterial.dispose();neighborGeometry.dispose();neighborMaterial.dispose();rings.forEach(r=>{r.geometry.dispose();r.material.dispose();});},
    };
}
