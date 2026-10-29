import * as THREE from 'three';
import {JOINT_NAMES,spatialNeighborhood,spatialTarget} from '../spatialGraph.js';

const V=(x=0,y=0,z=0)=>new THREE.Vector3(x,y,z);
const fmt=n=>Number.isFinite(n)?n===0?'0':Number(n.toPrecision(5)).toString():'未采集';
const positive=0x7ee0c1,negative=0xf2a797,current=0xf2c779;

/** Message annotations share original skeleton/feature-map anchors. No extra neurons. */
export function createGraphMessages(scene,overlay,onTermSelect){
    const root=new THREE.Group();scene.add(root);root.visible=false;
    const layer=document.createElement('div');layer.className='graph-message-labels';overlay.appendChild(layer);
    const leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');leaders.classList.add('graph-message-leaders');layer.appendChild(leaders);
    let key=null,entries=[],rings=[],model=null,state=null,active=false,emphasis=true,context=null,bindings=null;
    const restored=[];
    function restoreContext(){restored.forEach(([m,opacity,transparent,depthWrite,color])=>{m.opacity=opacity;m.transparent=transparent;m.depthWrite=depthWrite;if(color)m.color.copy(color);});restored.length=0;}
    function dim(m,opacity){restored.push([m,m.opacity,m.transparent,m.depthWrite,m.color?.clone()]);m.transparent=true;m.opacity=opacity;m.depthWrite=false;if(m.color)m.color.multiplyScalar(.25);}
    function contextEmphasis(){
        restoreContext();if(!active||!context||!emphasis)return;
        const targetKey=bindings?.find(Boolean)?.target?.key;
        for(const r of context.registrations)if(r.mesh&&r.key!==targetKey)dim(r.mesh.material,.12);
        for(const s of context.skeletons){if(s.stage!==5){dim(s.joints.material,.1);dim(s.bones.material,.1);}else dim(s.bones.material,.22);}
    }
    function clear(){
        restoreContext();entries.forEach(e=>{e.button.remove();e.leader.remove();});entries=[];rings=[];
        root.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});root.clear();
    }
    function rebuild(c,all,ctx){
        clear();bindings=all;context=ctx;
        const n=spatialNeighborhood(c,state);model=n;
        for(const term of n.messages){
            const b=all[term.index];if(!b?.source||!b.target)continue;
            const start=b.source.point.clone(),end=b.target.point.clone(),middle=start.clone().lerp(end,.5);
            middle.y+=Math.max(.65,start.distanceTo(end)*.13)+Math.floor(term.index/17)*.12;
            middle.z+=Math.floor(term.index/17)*.15;
            const curve=new THREE.QuadraticBezierCurve3(start,middle,end);
            const color=term.product<0?negative:positive;
            const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(32)),new THREE.LineBasicMaterial({color,transparent:true,opacity:.45}));root.add(line);
            const arrow=new THREE.Mesh(new THREE.ConeGeometry(.055,.2,6),new THREE.MeshBasicMaterial({color}));
            arrow.position.copy(curve.getPoint(.84));arrow.quaternion.setFromUnitVectors(V(0,1,0),curve.getTangent(.84).normalize());root.add(arrow);
            const button=document.createElement('button');button.className='graph-message';button.type='button';button.dataset.term=term.index;
            button.setAttribute('aria-label',`追踪分区 ${Math.floor(term.index/17)}，${JOINT_NAMES[term.index%17]} J${term.index%17}，输入 ${term.input}，系数 ${term.weight}，贡献 ${term.product}`);
            const title=document.createElement('span'),formula=document.createElement('small'),value=document.createElement('strong'),bar=document.createElement('i');
            title.textContent=`${JOINT_NAMES[term.index%17]} J${term.index%17} · K${Math.floor(term.index/17)}`;
            formula.textContent=`${fmt(term.input)} × ${fmt(term.weight)}`;value.textContent=`m = ${fmt(term.product)}`;
            bar.style.setProperty('--magnitude',`${term.recorded?Math.abs(term.product)/n.peak*100:0}%`);
            button.dataset.sign=term.product<0?'negative':'positive';button.dataset.recorded=String(term.recorded);
            button.append(title,formula,value,bar);layer.appendChild(button);button.onclick=()=>onTermSelect?.(term.index);
            const leader=document.createElementNS(leaders.namespaceURI,'line');leaders.appendChild(leader);
            entries.push({term,anchor:start,curve,line,arrow,button,color,leader,title});
        }
        // Outline receiving joints in the original partition skeletons. Their
        // projected-input values remain distinct from the output feature cell.
        for(const s of ctx.skeletons.filter(s=>s.stage===5&&s.key==='future')){
            const p=s.points[n.target];if(!p)continue;
            const geometry=new THREE.SphereGeometry(.14*s.scale/3,16,10),material=new THREE.MeshBasicMaterial({color:0xc4b0f0,wireframe:true,transparent:true,opacity:.8});
            const ring=new THREE.Mesh(geometry,material);ring.position.copy(s.group.localToWorld(V(p[0],p[1],0)));root.add(ring);rings.push(ring);
        }
    }
    function update(c,s,all,ctx){
        active=c?.stage==='spatial'&&!!s;root.visible=active;layer.hidden=!active;
        if(!active){restoreContext();key=null;return;}
        state=s;
        const next=`${c.id}:${c.frame}:${c.track_id}:${ctx.revision}`;
        if(key!==next){rebuild(c,all,ctx);key=next;}
        model=spatialNeighborhood(c,s);contextEmphasis();
        const output=['bias','activation','result'].includes(s.phaseName);
        for(const e of entries){
            const selected=e.term.index===s.termIndex&&!output,used=e.term.index<s.completedCount;
            e.line.material.color.set(selected?current:e.color);e.line.material.opacity=selected?1:used?.75:.28;
            e.arrow.material.color.set(selected?current:e.color);e.arrow.visible=e.term.recorded;
            e.line.visible=e.term.recorded;e.button.dataset.active=String(selected);e.button.dataset.used=String(used);
            e.title.textContent=`${used?'✓ ':selected?'→ ':''}${JOINT_NAMES[e.term.index%17]} J${e.term.index%17} · K${Math.floor(e.term.index/17)}`;
            e.button.setAttribute('aria-pressed',String(selected));
        }
        layer.dataset.receiver=String(spatialTarget(c));layer.dataset.messages=entries.length;layer.dataset.completed=model.completed;
        layer.dataset.phase=output?'writeback':s.phaseName==='input'?'gather':'message';
    }
    function render(camera,width,height){
        if(!active)return;
        leaders.setAttribute('viewBox',`0 0 ${width} ${height}`);
        const output=['bias','activation','result'].includes(state.phaseName),occupied=[];
        const visible=entries.filter(e=>!output&&(width>=900||e.term.index===state.termIndex));
        const project=p=>{const q=p.clone().project(camera);return {x:(q.x+1)*width/2,y:(1-q.y)*height/2,z:q.z};};
        for(const e of entries){e.button.hidden=!visible.includes(e);e.leader.style.display=e.button.hidden?'none':'';}
        // Labels sit beside their source joints, never in a separate diagram.
        visible.sort((a,b)=>project(a.anchor).y-project(b.anchor).y).forEach(e=>{
            const p=project(e.anchor),w=142,h=68;
            if(p.z<-1||p.z>1||p.x<-100||p.x>width+100){e.button.hidden=true;e.leader.style.display='none';return;}
            let x=Math.max(10,Math.min(width-w-18,p.x-w-20)),y=Math.max(155,Math.min(height-h-145,p.y-h/2));
            if(width<600){x=12;y=210;}
            for(let pass=0;pass<entries.length;pass++){
                const hit=occupied.find(r=>Math.abs(x-r.x)<w+8&&Math.abs(y-r.y)<h+7);
                if(!hit)break;
                y=hit.y+h+9;
                if(y>height-h-137){x=Math.min(width-w-18,x+w+12);y=155;}
            }
            occupied.push({x,y});e.button.style.transform=`translate(${x}px,${y}px)`;
            e.leader.setAttribute('x1',p.x);e.leader.setAttribute('y1',p.y);
            e.leader.setAttribute('x2',Math.max(x,Math.min(x+w,p.x)));e.leader.setAttribute('y2',Math.max(y,Math.min(y+h,p.y)));
        });
    }
    return {update,render,restoreContext,
        curveForTerm(index){return active?entries.find(e=>e.term.index===index)?.curve:null;},
        bounds(){if(!active)return null;const box=new THREE.Box3();entries.forEach(e=>{box.expandByPoint(e.anchor);box.expandByPoint(e.curve.v2);});rings.forEach(r=>box.expandByPoint(r.position));return box.isEmpty()?null:box;},
        setEmphasis(value){emphasis=value;contextEmphasis();},
        dispose(){clear();scene.remove(root);layer.remove();},
    };
}
