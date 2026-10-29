import { useEffect, useMemo, useRef, useState } from 'react';
import { normalizeLiveNodes } from '../liveComputation.js';
import { createNeuralScene } from '../three/NeuralScene.js';
import { sceneNodes, STAGE_NAMES, finite } from '../three/sceneData.js';
import GraphStoryControls from './GraphStoryControls.jsx';
import '../three-scene.css';

/** Live mode supplies only liveNodes; saved tensors are used for inspection mode. */
export default function NeuralScene3D({ stage=0, data, frontend, track, liveNodes, liveEvent=null, running=false, frozen=false, progress=1, onStageSelect, onInspect, onError, calculation=null, calculationState=null, onTermSelect, onCursorChange, onGraphTarget, calculationPlaying, onToggleCalculation, liveSnapshot, selectedId }) {
    const host=useRef(null),engine=useRef(null),callbacks=useRef({onStageSelect,onInspect,onError,onTermSelect,onGraphTarget});
    const [autoFollow,setAutoFollow]=useState(true);
    const [graphEmphasis,setGraphEmphasis]=useState(true);
    const [error,setError]=useState(null),[revision,setRevision]=useState(0),[selected,setSelected]=useState(null),[round,setRound]=useState(0);
    callbacks.current={onStageSelect,onInspect,onError,onTermSelect,onGraphTarget};
    const effectiveRound=calculation?Math.max(0,(calculation.metadata?.round??1)-1):round;
    const nodes=useMemo(()=>liveSnapshot&&calculation?.metadata?.round?normalizeLiveNodes(liveSnapshot,selectedId,{round:effectiveRound+1}):sceneNodes({data,frontend,track,liveNodes,round:effectiveRound}),[data,frontend,track,liveNodes,effectiveRound,liveSnapshot,selectedId,calculation?.metadata?.round]);
    const latest=useRef({nodes,liveEvent,running,frozen,progress,stage,calculation,calculationState});latest.current={nodes,liveEvent,running,frozen,progress,stage,calculation,calculationState};
    useEffect(()=>{
        setError(null);
        try {
            engine.current=createNeuralScene(host.current,{
                onAutoFollow:setAutoFollow,
                onStageSelect:value=>callbacks.current.onStageSelect?.(value),
                onTermSelect:index=>callbacks.current.onTermSelect?.(index),
                onGraphTarget:index=>{setSelected(null);callbacks.current.onGraphTarget?.(index);},
                onInspect:value=>{setSelected(value);callbacks.current.onInspect?.(value);},
                onError:message=>{setError(message);callbacks.current.onError?.(message);},
            });
            callbacks.current.onError?.(null);
            const props=latest.current;engine.current.update(props.nodes,{liveEvent:props.liveEvent,playing:props.running,isFrozen:props.frozen,phase:props.progress});engine.current.setStage(props.stage);engine.current.setCalculation(props.calculation,props.calculationState);
        } catch (cause) {
            const message=`当前浏览器未能创建三维画布：${cause.message || 'WebGL 不可用'}`;
            setError(message);callbacks.current.onError?.(message);
        }
        return ()=>{engine.current?.dispose();engine.current=null;};
    },[revision]);
    useEffect(()=>{engine.current?.setStage(stage);setSelected(null);},[stage]);
    useEffect(()=>{setSelected(null);engine.current?.clearSelection();},[liveEvent?.seq,liveEvent?.frame,liveEvent?.job_id,track?.track_id,data,frontend,round]);
    useEffect(()=>{engine.current?.update(nodes,{liveEvent,playing:running,isFrozen:frozen,phase:progress});},[nodes,liveEvent]);
    useEffect(()=>{engine.current?.setPlayback({playing:running,isFrozen:frozen,phase:progress});},[running,frozen,progress]);
    useEffect(()=>{engine.current?.setCalculation(calculation,calculationState);},[calculation,calculationState]);
    useEffect(()=>{engine.current?.setGraphEmphasis(graphEmphasis);},[graphEmphasis,revision]);
    const count=Object.keys(nodes).length;
    const graph=calculation?.stage==='spatial';
    const format=value=>finite(value)?Number(value.toPrecision(7)).toString():Array.isArray(value)?value.map(v=>finite(v)?Number(v.toPrecision(6)):String(v)).join(' / '):'尚无此单元数值';
    return <div className={`neural-scene ${calculation?'has-spatial-trace':''} ${calculation?.stage==='spatial'?'has-graph-story':''}`} data-calculation={calculation?.id||''} data-cursor={calculationState?.cursor??0} data-stage={stage} data-mode={liveEvent?'live':count?'record':'structure'}>
        <div className="neural-scene-toolbar">
            <div className="neural-scene-title"><span className="neural-dimension-mark">3D</span><div><strong>{stage===0&&calculation?`全流程 · 当前${STAGE_NAMES[calculation.chapter]}`:STAGE_NAMES[stage]}</strong><span>{calculation?`${calculation.title} · 帧 ${Number(calculation.frame)+1} · ${calculation.track_id==null?'整帧图像':`行人 ${calculation.track_id}`} · 代表输出的逐项演算`:liveEvent ? `实时事件 #${liveEvent.seq} · ${liveEvent.node}` : count ? '实际张量 / 空间检查' : '网络结构 / 等待计算'}</span></div></div>
            <div className="neural-camera-tools">
                {stage===3&&!calculation&&data?.lstm?.rounds?.length>1&&<label className="neural-round-select"><span>递归轮</span><select aria-label="三维 LSTM 递归轮" value={round} onChange={e=>setRound(Number(e.target.value))}>{data.lstm.rounds.map((_,i)=><option value={i} key={i}>{i+1} / {data.lstm.rounds.length}</option>)}</select></label>}
                {calculation&&<button className="neural-auto-follow" aria-pressed={autoFollow} onClick={()=>engine.current?.setAutoFollow(!autoFollow)}>{autoFollow?'自动跟随当前步骤':'恢复自动跟随'}</button>}
                <button onClick={()=>engine.current?.focusSelected()} disabled={!selected} title="将选中的单元移至镜头中央">聚焦单元</button>
                <button onClick={()=>engine.current?.resetCamera()} title="恢复当前章节的三维相机">重置视角</button>
            </div>
        </div>
        <div className="neural-scene-viewport" ref={host}/>
        {!error&&calculation?.stage==='spatial'&&<GraphStoryControls calculation={calculation} state={calculationState} onTarget={onGraphTarget} onSeek={onCursorChange} emphasis={graphEmphasis} onEmphasis={setGraphEmphasis} playing={calculationPlaying} onToggle={onToggleCalculation}/>}
        {error&&<div className="neural-scene-fallback" role="alert"><span>三维画布暂不可用</span><p>{error}</p><button onClick={()=>setRevision(v=>v+1)}>恢复三维画布</button></div>}
        {!error&&<div className="neural-scene-hint"><span>拖动旋转</span><i/><span>滚轮缩放</span><i/><span>右键平移</span><i/><span>点选真实单元</span></div>}
        {selected&&!error&&<div className="neural-cell-readout" aria-live="polite"><div><span>{selected.label}</span><button aria-label="关闭三维单元读数" onClick={()=>{setSelected(null);engine.current?.clearSelection();}}>×</button></div><strong>{format(selected.value)}</strong><p>{selected.index?.length ? `[${selected.index.join(', ')}]` : '层结构'}{selected.shape?.length ? ` · shape ${selected.shape.join(' × ')}` : ''}</p><small>{typeof selected.source==='string'?selected.source:'实际推理张量'}</small></div>}
        <div className="neural-scene-scale">{graph?<><span><i className="neural-used-key"/>＋贡献</span><span><i className="graph-negative-key"/>－贡献</span><span><i className="neural-current-key"/>当前消息</span><span><i className="neural-absent-key"/>无连接</span></>:<><span><i className={calculation?"neural-current-key":"neural-activation-key"}/>{calculation?"当前项":"激活"}</span><span><i className={calculation?"neural-used-key":"neural-weight-key"}/>{calculation?"已累计":"权重"}</span><span><i className={calculation?"neural-absent-key":"neural-empty-key"}/>{calculation?"无连接 / 未采集":"未采样"}</span></>}<em>{calculation?autoFollow?'镜头跟随当前步骤':'手动视角 · 数值继续更新':frozen?'已冻结':liveEvent?'事件驱动':running?'讲解时间':'静态检查'}</em></div>
    </div>;
}
