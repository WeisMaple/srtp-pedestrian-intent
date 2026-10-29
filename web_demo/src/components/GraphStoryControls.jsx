import {JOINT_NAMES,spatialTarget,spatialNeighborhood,validGraphSlice} from '../spatialGraph.js';

export default function GraphStoryControls({calculation:c,state,onTarget,onSeek,emphasis,onEmphasis,playing,onToggle}){
    const n=spatialNeighborhood(c,state);if(!n)return null;
    const canSelect=!!validGraphSlice(c),target=spatialTarget(c),output=['bias','activation','result'].includes(state.phaseName);
    const step=output?2:state.phaseName==='input'?0:1;
    const t=c.metadata?.target_indices||[c.target?.channel,c.target?.timestep,target];
    const steps=[['读取邻域',n.first+.1],['沿边传递',(!output&&state.currentTerm?.connected?state.termIndex:n.first)+.5],['求和写回',c.terms.length+2]];
    return <>
        <div className="graph-story-caption"><b>{JOINT_NAMES[target]||'目标关节'} J{target} 正在接收什么？</b><span>{n.neighbors.length} 个来源关节 · {n.messages.length} 项连接 / {c.terms.filter(t=>!t.connected).length} 项无连接 · 通道 {t[0]} / 时刻 {t[1]}<br/>{!output&&state.currentTerm?.connected===false?`当前检查 J${state.termIndex%17} / K${Math.floor(state.termIndex/17)}：对本目标无连接，不传递消息。`:'h̃ 是投影后的特征；每条消息 m = h̃ × (A ⊙ E)，随后按分区求和。'}</span></div>
        <div className="graph-story-controls">
            <div className="graph-receiver-control"><label>接收关节<select aria-label="选择接收关节" value={target??0} disabled={!canSelect} onChange={e=>onTarget(Number(e.target.value))}>{JOINT_NAMES.map((name,j)=><option key={j} value={j}>{name} · J{j}</option>)}</select></label><button type="button" aria-pressed={emphasis} onClick={()=>onEmphasis(!emphasis)}>突出当前邻域</button><button type="button" className="graph-play" onClick={onToggle}>{playing?"Ⅱ 暂停消息":"▶ 顺序传递"}</button><small>{canSelect?'点选原骨架也可切换接收关节':'此旧记录仅采集了当前接收关节'}</small></div>
            <div className="graph-phase-controls" role="group" aria-label="图消息传递步骤">{steps.map(([label,cursor],i)=><button type="button" key={label} aria-pressed={step===i} onClick={()=>onSeek(cursor)}><i>0{i+1}</i>{label}</button>)}<span>{n.completed} / {n.messages.length} 条消息已计入 · 求和后进入 BN / 时间卷积</span></div>
        </div>
    </>;
}
