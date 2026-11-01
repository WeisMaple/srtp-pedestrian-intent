import {useEffect,useMemo,useState} from 'react';
import {GRAPH_EDGES,TEMPLATE_POSE} from '../three/sceneData.js';
import {JOINT_NAMES,PARTITION_NAMES,spatialTarget,spatialNeighborhood,validGraphSlice} from '../spatialGraph.js';
import {sequenceFrames,frameStrip,projectedSequence,temporalKernel,temporalTerms} from '../skeletonExplorer.js';
import '../skeleton-explorer.css';

const fmt=x=>Number.isFinite(x)?Number(x.toPrecision(5)).toString():'—';
const colors=['#62d9c0','#8eafff','#f6b975'];
function Pose({points,cx,cy,joint,onJoint,scale=1,dim=false}) {
    if(!points)return null;
    const pos=i=>[cx+points[i][0]*scale,cy+points[i][1]*scale];
    return <g className={dim?'sx-dim':''}>
        {GRAPH_EDGES.map(([a,b])=><line key={`${a}-${b}`} x1={pos(a)[0]} y1={pos(a)[1]} x2={pos(b)[0]} y2={pos(b)[1]} stroke={a===joint||b===joint?'#e4c06a':'#527b88'} strokeWidth={a===joint||b===joint?3:2}/>)}
        {points.map((p,i)=><g key={i} role={onJoint?'button':undefined} tabIndex={onJoint?0:undefined} aria-label={onJoint?`查看${JOINT_NAMES[i]}时序`:undefined} onClick={()=>onJoint?.(i)} onKeyDown={e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();onJoint?.(i);}}}>
            <circle cx={pos(i)[0]} cy={pos(i)[1]} r={i===joint?8:4} fill={i===joint?'#ffce75':'#64d8c1'} stroke={i===joint?'#fff0cd':'#0d2734'} strokeWidth="2"/>
            {onJoint&&<circle cx={pos(i)[0]} cy={pos(i)[1]} r="11" fill="transparent"/>}
        </g>)}
    </g>;
}

function Sequence({nodes,stage}) {
    const future=stage===4;
    const frames=useMemo(()=>projectedSequence(sequenceFrames(nodes[future?'future':'history'])),[nodes,future]);
    const [index,setIndex]=useState(0),[joint,setJoint]=useState(9),[playing,setPlaying]=useState(false);
    useEffect(()=>{setIndex(0);setPlaying(false);},[stage,nodes[future?'future':'history']]);
    useEffect(()=>{if(!playing||frames.length<2)return;const id=setInterval(()=>setIndex(i=>(i+1)%frames.length),180);return()=>clearInterval(id);},[playing,frames.length]);
    const current=frames[Math.min(index,frames.length-1)];
    const strip=frameStrip(frames,index,5);
    const label=f=>future?`+${f.time+(nodes.future?.indices?.[0]?.[0]===1?0:1)}`:`t=${f.time}`;
    return <>
        <div className="sx-heading"><div><span>{future?'02 / FUTURE MOTION':'01 / OBSERVED MOTION'}</span><h2>{future?'把预测展开为连续姿态':'沿着同一个关节，观察动作变化'}</h2><p>{future?'五轮递归预测，每轮生成10步骨架。':'每个行人独立积累历史，50次有效观测组成一个输入。'}</p></div><b>{frames.length} 个已采样时刻</b></div>
        {!current?<div className="sx-empty">选择已有骨架的行人记录，或等待有效历史形成。</div>:<>
        <div className="sx-sequence-grid">
            <section className="sx-pose-focus"><div className="sx-section-label">当前姿态 <b>{label(current)}</b></div>
                <svg viewBox="0 0 280 310" aria-label="可选择关节的当前骨架"><Pose points={current.points} cx={140} cy={150} joint={joint} onJoint={setJoint}/><text x="140" y="292" textAnchor="middle">点击关节，沿时间追踪</text></svg>
                <div className="sx-coordinates"><strong>{JOINT_NAMES[joint]} · J{joint}</strong><span>x {fmt(current.pose[joint][0])}　y {fmt(current.pose[joint][1])}</span><small>置信度 {fmt(current.pose[joint][2])}</small></div>
            </section>
            <section className="sx-sequence-strip"><div className="sx-section-label">时空骨架序列 <span>同一关节沿虚线连接</span></div>
                <svg viewBox="0 0 850 340" aria-label="时空骨架序列与关节时间连接">
                    {strip.slice(1).map((f,i)=>{const a=strip[i],x1=70+i*710/Math.max(1,strip.length-1),x2=70+(i+1)*710/Math.max(1,strip.length-1);return <line key={`t${f.slot}`} x1={x1+a.points[joint][0]*.55} y1={166+a.points[joint][1]*.55} x2={x2+f.points[joint][0]*.55} y2={166+f.points[joint][1]*.55} stroke="#e4c06a" strokeWidth="2" strokeDasharray="5 5"/>;})}
                    {strip.map((f,i)=>{const x=70+i*710/Math.max(1,strip.length-1);return <g key={f.slot} role="button" tabIndex="0" aria-label={`选择时刻${f.time}`} onClick={()=>{setIndex(f.slot);setPlaying(false);}} onKeyDown={e=>{if(e.key==='Enter'){setIndex(f.slot);setPlaying(false);}}}>
                        <rect x={x-60} y="46" width="120" height="238" rx="14" fill={f.slot===index?'#173e48':'#102b37'} stroke={f.slot===index?'#62d9c0':'#284752'}/>
                        <Pose points={f.points} cx={x} cy={166} joint={joint} scale={.55}/><text x={x} y="313" textAnchor="middle">{label(f)}</text>
                    </g>;})}
                </svg>
                <div className="sx-axis-note"><span>空间：一帧内的关节连接</span><span>时间：同一关节在不同观测中的变化 →</span></div>
                <p className="sx-fine">所有姿态共用坐标尺度。时间轴展示实际采样位置，缺失时刻不补帧；x、y是图像坐标。</p>
            </section>
        </div>
        <div className="sx-transport"><button onClick={()=>setPlaying(!playing)} aria-label={playing?'暂停骨架序列':'播放骨架序列'}>{playing?'Ⅱ 暂停':'▶ 播放序列'}</button><input aria-label="骨架序列时间轴" type="range" min="0" max={Math.max(0,frames.length-1)} value={index} onChange={e=>{setIndex(Number(e.target.value));setPlaying(false);}}/><b>{label(current)}</b><select aria-label="选择时间追踪关节" value={joint} onChange={e=>setJoint(Number(e.target.value))}>{JOINT_NAMES.map((n,i)=><option key={i} value={i}>{n}</option>)}</select></div>
        </>}
    </>;
}

function Spatial({calculation:c,state,onGraphTarget,onTermSelect,onCursorChange,playing,onToggle}) {
    const n=spatialNeighborhood(c,state),target=spatialTarget(c),selectable=!!validGraphSlice(c);
    const points=TEMPLATE_POSE.map(([x,y])=>[182+x*112,200-y*67]);
    // Spread the five head joints in this topology diagram so incoming arrows stay readable.
    [[182,48],[155,24],[209,24],[127,44],[237,44]].forEach((p,i)=>{points[i]=p;});
    const current=state?.currentTerm;
    const targetPoint=points[target],labelSide=targetPoint?.[0]<182?-1:1;
    const labelY=target<5?-8:targetPoint?.[1];
    const labelX=labelSide<0?9:356;
    const messagePath=(source)=>{
        const p=points[source],q=targetPoint;
        const control=[(p[0]+q[0])/2+18,(p[1]+q[1])/2-12];
        const trim=(point,radius)=>{
            const dx=control[0]-point[0],dy=control[1]-point[1],length=Math.hypot(dx,dy)||1;
            return [point[0]+dx/length*radius,point[1]+dy/length*radius];
        };
        const start=trim(p,9),end=trim(q,17);
        return `M${start[0]} ${start[1]} Q${control[0]} ${control[1]} ${end[0]} ${end[1]}`;
    };
    return <>
        <div className="sx-heading"><div><span>03 / SPATIAL AGGREGATION</span><h2>邻居的特征，如何汇成一个关节的输出</h2><p>选定接收关节，查看三个分区的连接、加权消息与求和结果。</p></div><b>{c?`块 ${Number(c.metadata?.block??0)+(c.source==='live'?1:0)} · 通道 ${c.target?.channel??c.metadata?.target_indices?.[0]??'—'}`:'等待实际计算'}</b></div>
        {!n?<div className="sx-empty">这一阶段尚未采集到完整算式。选择“记录讲解”中的有效行人可查看聚合过程。</div>:<>
        <div className="sx-graph-grid">
            <section className="sx-topology"><div className="sx-section-label">接收关节 <select aria-label="骨架图接收关节" value={target??0} disabled={!selectable} onChange={e=>onGraphTarget?.(Number(e.target.value))}>{JOINT_NAMES.map((name,i)=><option key={i} value={i}>{name} · J{i}</option>)}</select></div>
                <svg viewBox="0 -22 365 392" aria-label="目标关节的有向邻域图"><defs>{colors.map((color,k)=><marker key={k} id={`sx-arrow-${k}`} markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" viewBox="0 0 7 7" refX="6.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill={color}/></marker>)}</defs>
                    {GRAPH_EDGES.map(([a,b])=><line key={`${a}-${b}`} x1={points[a][0]} y1={points[a][1]} x2={points[b][0]} y2={points[b][1]} stroke="#284652" strokeWidth="2"/>)}
                    {n.messages.filter(t=>t.index%17!==target).map(t=><path className="sx-message-edge" key={t.index} d={messagePath(t.index%17)} fill="none" stroke={colors[Math.floor(t.index/17)]} strokeWidth={current?.index===t.index?3:2} opacity={current?.index===t.index?1:.7} markerEnd={`url(#sx-arrow-${Math.floor(t.index/17)})`} pointerEvents="none"/>)}
                    {points.map(([x,y],i)=><g key={i} role={selectable?'button':undefined} tabIndex={selectable?0:undefined} aria-label={`聚合到${JOINT_NAMES[i]}`} onClick={()=>selectable&&onGraphTarget?.(i)} onKeyDown={e=>{if(selectable&&['Enter',' '].includes(e.key)){e.preventDefault();onGraphTarget?.(i);}}}><circle cx={x} cy={y} r={i===target?12:7} fill={i===target?'#ffcd78':n.neighbors.includes(i)?'#67d6c0':'#36525e'} stroke={i===target?'#fff1ce':'#0d2734'} strokeWidth="2"/><circle cx={x} cy={y} r="14" fill="transparent"/>{i!==target&&<text x={i===0?x:x+(x<182?-12:12)} y={i===1||i===2?y-11:i===0?y+22:y+5} textAnchor={i===0?'middle':x<182?'end':'start'}>{i}</text>}</g>)}
                    {targetPoint&&<g pointerEvents="none"><path d={target<5?`M${targetPoint[0]} ${targetPoint[1]-16} L${targetPoint[0]} ${labelY} L${labelSide<0?43:322} ${labelY}`:`M${targetPoint[0]+labelSide*17} ${labelY} L${labelSide<0?43:322} ${labelY}`} fill="none" stroke="#b79d65" strokeWidth="1"/><text className="sx-target-label" x={labelX} y={labelY+4} textAnchor={labelSide<0?'start':'end'}>{JOINT_NAMES[target]}</text></g>}
                </svg><p className="sx-fine">关节拓扑示意 · 箭头指向接收关节<br/>当前时刻 {c.target?.timestep??c.metadata?.target_indices?.[1]}，颜色区分邻接分区。</p>
            </section>
            <section className="sx-messages"><div className="sx-section-label">沿连接读取实际数值 <span>{n.messages.length} 项消息</span></div>
                <div className="sx-message-head"><span>来源 / 分区</span><span>投影特征 h̃</span><span>边权 A⊙E</span><span>贡献 h̃×权重</span></div>
                <div className="sx-message-list">{n.messages.map(t=><button key={t.index} className={t.index===current?.index?'active':''} onClick={()=>onTermSelect?.(t.index)} style={{'--partition':colors[Math.floor(t.index/17)]}}><span><i/>{JOINT_NAMES[t.index%17]}<small>K{Math.floor(t.index/17)}</small></span><span>{fmt(t.input)}</span><span>{fmt(t.weight)}</span><strong>{fmt(t.product)}</strong></button>)}</div>
                <div className="sx-partitions">{n.partitions.map(p=><div key={p.index} style={{'--partition':colors[p.index]}}><span>K{p.index} · {PARTITION_NAMES[p.index]}</span><strong>{fmt(p.partial)}</strong><small>完整分区和 {fmt(p.sum)}</small></div>)}</div>
                <div className="sx-sum"><span>各分区相加</span><strong>{fmt(state?.partialSum)}</strong><i>→</i><span>完整聚合输出</span><strong>{fmt(c.actual)}</strong></div>
                <p className="sx-fine">h̃ 是通道投影后的特征。聚合结果随后进入批归一化与时间卷积。</p>
            </section>
        </div>
        <div className="sx-transport"><button onClick={onToggle}>{playing?'Ⅱ 暂停消息':'▶ 顺序传递'}</button><button onClick={()=>onCursorChange?.(n.first+.1)}>读取邻域</button><button onClick={()=>onCursorChange?.(c.terms.length+2)}>查看求和结果</button><span>{n.completed} / {n.messages.length} 项已计入</span></div>
        </>}
    </>;
}

function Temporal({calculation:c,state,onTermSelect,onCursorChange}) {
    const [channel,setChannel]=useState(null);
    const allTerms=temporalTerms(c),channels=[...new Set(allTerms.map(t=>t.channel))];
    const contribution=new Map(channels.map(v=>[v,0]));
    allTerms.forEach(t=>contribution.set(t.channel,contribution.get(t.channel)+Math.abs(t.product||0)));
    const representative=channels.reduce((best,v)=>(contribution.get(v)>contribution.get(best)?v:best),channels[0]);
    const selected=channels.includes(channel)?channel:(representative??0),terms=temporalKernel(c,selected);
    const peak=Math.max(1e-9,...terms.map(t=>Math.abs(t.product||0)));
    return <>
        <div className="sx-heading"><div><span>04 / TEMPORAL RECEPTIVE FIELD</span><h2>同一关节，在九个时间位置上汇聚信息</h2><p>固定目标关节与输出通道，先观察一个输入通道的时间卷积，再查看全部通道的输出。</p></div><b>{c?`${JOINT_NAMES[c.target?.joint??c.metadata?.target_indices?.[2]]||'目标关节'} · t=${c.target?.timestep??c.metadata?.target_indices?.[1]??'—'}`:'等待实际计算'}</b></div>
        {!terms.length?<div className="sx-empty">选择时间卷积算式后显示真实输入窗口；残差相加可在三维视图和下方数值区查看。</div>:<>
        <div className="sx-kernel-control">输入通道<select aria-label="时间卷积输入通道" value={selected} onChange={e=>setChannel(Number(e.target.value))}>{channels.map(v=><option key={v} value={v}>C{v}</option>)}</select><span>{channel===null?'默认展开绝对贡献之和最大的通道。':'实际特征窗口。'}颜色表示贡献的正负。</span></div>
        <div className="sx-kernel-strip">{terms.map(t=><button key={t.index} className={`${state?.termIndex===t.index?'active':''} ${t.padding?'padding':''}`} onClick={()=>onTermSelect?.(t.index)}><span>{t.padding?'PAD':`t=${t.time??'—'}`}</span><small>输入</small><b>{fmt(t.input)}</b><i>×</i><small>核位置 {t.slot}</small><b>{fmt(t.weight)}</b><div className="sx-product-bar"><i style={{height:`${Math.abs(t.product||0)/peak*58}px`,background:t.product<0?'#e5ac9e':'#62d9c0'}}/></div><strong>{fmt(t.product)}</strong></button>)}</div>
        <div className="sx-temporal-footer"><span>当前通道贡献和 <b>{terms.every(t=>t.recorded)?fmt(terms.reduce((a,t)=>a+t.product,0)):'—'}</b></span><span>全部 {channels.length} 个输入通道 + 偏置 → <b>{fmt(c.actual)}</b></span><button onClick={()=>onCursorChange?.(c.terms.length+2)}>查看完整输出</button></div>
        <p className="sx-fine">这里展示空间聚合后的特征值；越界时间位置遵循模型的零填充规则。</p>
        </>}
    </>;
}

export default function SpatioTemporalExplorer({stage,nodes,calculation,state,onGraphTarget,onTermSelect,onCursorChange,playing,onToggle,onShow3D}) {
    return <section className="skeleton-explorer" data-testid="skeleton-explorer" data-stage={stage}>
        <div className="sx-view-bar"><span>骨架与计算关系</span><button onClick={onShow3D}>查看三维网络 ↗</button></div>
        {[2,4].includes(stage)?<Sequence nodes={nodes} stage={stage}/>:stage===5?<Spatial calculation={calculation} state={state} onGraphTarget={onGraphTarget} onTermSelect={onTermSelect} onCursorChange={onCursorChange} playing={playing} onToggle={onToggle}/>:<Temporal calculation={calculation} state={state} onTermSelect={onTermSelect} onCursorChange={onCursorChange}/>}
    </section>;
}
