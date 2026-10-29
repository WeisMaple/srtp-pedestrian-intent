import { useEffect, useMemo, useState } from 'react';
import { TensorMap } from './NetworkPrimitives';
import Icon from './Icon';

export function LiveControls({live,job}) {
    const [now,setNow]=useState(Date.now());
    useEffect(()=>{if(live.frozen||live.connection==='complete')return;const timer=setInterval(()=>setNow(Date.now()),500);return()=>clearInterval(timer);},[live.frozen,live.connection]);
    const snapshot=live.snapshot;
    const final=['completed','failed','cancelled'].includes(job?.status);
    const age=live.receivedAt?Math.max(0,now-live.receivedAt):null;
    const output=snapshot?.nodes?.['pipeline.output'];
    const count=Object.keys(snapshot?.nodes||{}).length;
    const streamStatus=live.frozen
        ? final?'画面已冻结 · 后台推理已结束':'画面已冻结，后台推理继续'
        : final?'推理已结束 · 保留最后采集结果'
        : live.connection==='reconnecting'?'连接恢复中'
        : age!=null?`最近收到 ${Math.round(age)} ms 前`:'等待模型开始运行';
    return <div className={`live-control-bar ${live.frozen?'is-frozen':''}`}>
        <button className="live-freeze" aria-pressed={live.frozen} disabled={!count} onClick={live.toggleFreeze}><Icon name={live.frozen?'play':'pause'} size={18}/>{live.frozen?'继续实时跟随':'冻结当前计算'}</button>
        <div className="live-metric"><span>{live.frozen?'冻结帧':'采集帧'}</span><b>{snapshot?.frame>=0?snapshot.frame+1:'—'}</b></div>
        <div className="live-metric"><span>已收到层输出</span><b>{count}<small> / 当前帧</small></b></div>
        <div className="live-metric"><span>当前帧耗时</span><b>{Number.isFinite(output?.duration_ms)?output.duration_ms.toFixed(0):'—'}<small> ms</small></b></div>
        <div className="live-stream-state"><i className={!final&&!live.frozen&&live.connection==='connected'?'receiving':''}/><span>{streamStatus}</span><small>序列 {snapshot?.seq||0}{live.frozen&&live.latestSeq>snapshot?.seq?` · 后台已到 ${live.latestSeq}`:''}</small></div>
    </div>;
}

export function LiveTensorView({nodes,stage}) {
    const [selected,setSelected]=useState(null);
    const entries=useMemo(()=>Object.entries(nodes||{}).filter(([key])=>stage===0||stage===1?stage===0||key.startsWith('yolo.'):stage===2?key==='history':stage===3?key.startsWith('lstm.'):stage===4?key==='future':stage===5?key.startsWith('stgcn.')&&key.endsWith('.spatial'):stage===6?key.startsWith('stgcn.')&&(key.endsWith('.temporal')||key.endsWith('.output')):key==='stgcn.pool'||key==='stgcn.logits'||key==='stgcn.probabilities'),[nodes,stage]);
    const key=selected&&entries.some(([name])=>name===selected)?selected:entries.at(-1)?.[0];
    const node=nodes?.[key];
    const rows=useMemo(()=>{if(!node?.values)return null;const values=node.values.flat(Infinity);const width=node.values.at(-1)?.length||Math.min(16,values.length);return Array.from({length:Math.ceil(values.length/width)},(_,i)=>values.slice(i*width,i*width+width));},[node]);
    return <div className="live-tensor-view"><div className="network-section-title"><span>LIVE / ACTUAL FORWARD OUTPUT</span><h2>正在运行的网络，实际输出了什么</h2><p>从本次前向计算直接采样。选择一层查看数值；详细逐项推导可在分析后切换到“记录讲解”。</p></div>
        <div className="live-node-list">{entries.map(([name,value])=><button key={name} aria-pressed={name===key} onClick={()=>setSelected(name)}><span>{value.label||name}</span><small>{value.shape?.join(' × ')}</small></button>)}</div>
        {node?<div className="live-tensor-detail"><h3>{node.label||key}</h3><code>原张量 {node.shape?.join(' × ')} · 帧 {(node.frame??0)+1}</code><TensorMap values={rows} label="真实采样 · 悬停读取具体值"/><p>{node.source}。以原采样顺序排列，省略的元素不填补为零。</p></div>:<div className="live-awaiting"><Icon name="person" size={30}/><strong>等待此阶段的实际输出</strong><p>YOLO 先产生图像特征；同一行人积累 50 次有效观测后，才运行 LSTM 与 ST-GCN。</p></div>}
    </div>;
}

export function LiveActivity({live}) {
    const events=(live.events||[]).filter(e=>e.node&&e.node!=='pipeline.frame').slice(-5).reverse();
    return <div className="live-activity"><div><span>FORWARD EVENTS</span><b>当前计算轨迹</b></div><ol>{events.map(e=><li key={e.seq}><i/><strong>{e.node}</strong><span>{Number.isFinite(e.duration_ms)?`${e.duration_ms.toFixed(1)} ms`:'已采集'}</span></li>)}</ol>{!events.length&&<p>开始分析后显示实际层输出事件</p>}<small>光迹延长用于看清层间顺序；数值与时间戳来自实际推理。</small></div>;
}
