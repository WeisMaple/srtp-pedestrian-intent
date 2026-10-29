import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { collectCalculations, createCalculation, flowState } from '../calculationFlow';
import {retargetSpatial} from '../spatialGraph.js';
import { useCalculationPlayback } from '../hooks/useCalculationPlayback';
import '../calculation-workbench.css';

const CalculationScene3D=lazy(()=>import('./CalculationScene3D'));
const number=value=>Number.isFinite(value)?value===0?'0':Math.abs(value)<.0001?value.toExponential(3):Number(value.toPrecision(6)).toString():'未采集';
const phaseNames={input:'读取输入与权重',product:'得到乘积',accumulate:'送入累加器',accumulated:'本项已累计',bias:'加上偏置',activation:'执行输出变换',output:'完成本次计算',result:'完成本次计算'};
const keyOf=c=>c?`${c.source}:${c.frame}:${c.track_id}:${c.id}`:'';
const termState=(term,index,state)=>!term.recorded?'unknown':!term.connected?'disconnected':term.padding?'padding':index<state.completedCount?'accumulated':index===state.termIndex&&!state.done?'current':'pending';
const statusNames={unknown:'未采集',disconnected:'无连接',padding:'零填充',accumulated:'已累计',current:'当前项',pending:'待累计'};

export default function CalculationWorkbench({job,stage,snapshot,data,frontend,trackId,frame,isLive,renderScene,sceneContext}) {
    const calculations=useMemo(()=>collectCalculations({snapshot:isLive?snapshot:undefined,data:isLive?null:data,frontend:isLive?null:frontend,trackId,frame}),[snapshot,data,frontend,trackId,frame,isLive]);
    const available=useMemo(()=>calculations.filter(c=>stage===0||(c.chapter??c.stage)===stage),[calculations,stage]);
    const context=`${job?.id}:${isLive?'live':'record'}:${trackId}:${stage}:${isLive?'':frame}`;
    const [selection,setSelection]=useState(null);
    const [held,setHeld]=useState(null);
    const [mode,setMode]=useState('explain');
    const [auto,setAuto]=useState(true);
    const [filter,setFilter]=useState('all');
    const [page,setPage]=useState(null);
    const [sceneError,setSceneError]=useState('');
    const [detailsOpen,setDetailsOpen]=useState(false);
    const [graphTarget,setGraphTarget]=useState(null);
    const sceneContextRef=useRef(sceneContext);sceneContextRef.current=sceneContext;
    const latest=available.find(c=>c.id===selection)||available[0];
    const pinned=held?.context===context?held.calculation:null;
    const baseCalculation=mode==='explain'?pinned:latest;
    const calculation=useMemo(()=>{
        if(!baseCalculation||graphTarget?.key!==keyOf(baseCalculation))return baseCalculation;
        const raw=retargetSpatial(baseCalculation,graphTarget.joint);
        return raw?createCalculation(raw,baseCalculation.provenance):baseCalculation;
    },[baseCalculation,graphTarget]);
    const playback=useCalculationPlayback(calculation);
    const cursor=mode==='follow'?(calculation?.terms?.length||0)+2:playback.cursor;
    const state=useMemo(()=>calculation?flowState(calculation,cursor):null,[calculation,cursor]);
    const availableRef=useRef(available);availableRef.current=available;
    const previousContext=useRef(context);
    useEffect(()=>{
        if(previousContext.current===context)return;
        previousContext.current=context;setHeld(null);setSelection(null);setGraphTarget(null);setMode('explain');setAuto(true);setFilter('all');setPage(null);setSceneError('');
    },[context]);
    useEffect(()=>{
        if(mode!=='explain'||pinned||!latest)return;
        setHeld({context,calculation:latest,sceneContext:sceneContextRef.current});setSelection(latest.id);playback.start();
    },[context,mode,pinned,latest]);
    useEffect(()=>{
        if(!auto||mode!=='explain'||!state?.done||!calculation)return;
        const timer=setTimeout(()=>{
            const list=availableRef.current;
            const index=list.findIndex(c=>c.id===(calculation.base_id||calculation.id));
            const next=stage===0?list[index+1]:null;
            if(next){setSelection(next.id);setHeld({context,calculation:next,sceneContext:sceneContextRef.current});setPage(null);playback.start();}
            else if(isLive&&job?.status==='running'){
                const update=list.find(c=>c.id===(calculation.base_id||calculation.id));
                if(update&&keyOf(update)!==keyOf(calculation)){setHeld({context,calculation:update,sceneContext:sceneContextRef.current});setPage(null);playback.start();}
            }
        },1300);
        return()=>clearTimeout(timer);
    },[auto,mode,state?.done,calculation,context,stage,job?.status,isLive]);
    const pin=(value=latest,play=true)=>{if(!value)return;setHeld({context,calculation:value,sceneContext:sceneContextRef.current});setMode('explain');setSelection(value.id);setPage(null);setFilter('all');if(play)playback.start();else playback.seek(0);};
    const seek=value=>{if(mode==='follow'&&latest)setHeld({context,calculation:latest,sceneContext:sceneContextRef.current});setMode('explain');setAuto(false);setPage(null);playback.seek(value);};
    const choose=id=>{const next=available.find(c=>c.id===id);if(next){setAuto(false);pin(next,true);}};
    const chooseGraphTarget=joint=>{
        if(!baseCalculation||!retargetSpatial(baseCalculation,joint))return;
        if(mode==='follow')setHeld({context,calculation:baseCalculation,sceneContext:sceneContextRef.current});
        setMode('explain');setAuto(false);setPage(null);
        setGraphTarget({key:keyOf(baseCalculation),joint});playback.seek(0);
    };
    const terms=calculation?.terms||[];
    const filtered=useMemo(()=>terms.filter(term=>filter==='all'||filter==='disconnected'&&term.connected===false||filter==='zero'&&term.recorded&&term.connected&&term.product===0||filter==='unknown'&&!term.recorded),[terms,filter]);
    const cursorIndex=state?.termIndex??0;
    const currentPage=page??(filter==='all'?Math.floor(Math.max(0,cursorIndex)/8):0);
    const maxPage=Math.max(0,Math.ceil(filtered.length/8)-1);
    const visiblePage=Math.min(currentPage,maxPage);
    const visibleTerms=filtered.slice(visiblePage*8,visiblePage*8+8);
    const used=terms.filter(t=>t.connected&&t.recorded).length;
    const disconnected=terms.filter(t=>t.connected===false).length;
    const zero=terms.filter(t=>t.recorded&&t.connected&&t.product===0).length;
    const unknown=terms.filter(t=>!t.recorded).length;
    const snapshotFrame=snapshot?.frame;
    const actual=calculation?.actual;
    return <section className={`calculation-workbench ${renderScene?'spatial-workbench':''}`} data-calculation={calculation?.id||''} data-phase={state?.phaseName||'waiting'} data-cursor={cursor} data-frame={calculation?.frame??-1} data-source={calculation?.source||''}>
        {renderScene?.({calculation,state,cursor,context:mode==='explain'&&pinned?held.sceneContext:sceneContext,onTermSelect:index=>seek(index+.5),onCursorChange:seek,onGraphTarget:chooseGraphTarget,playing:playback.playing&&mode==='explain',onTogglePlay:()=>{if(mode==='follow')pin(latest,true);else playback.toggle();setAuto(false);}})}
        <header className="flow-header"><div><span>TRACE ONE OUTPUT / 沿着一个结果，追踪每一项</span><h2>{calculation?.title||'等待可追踪的计算'}</h2></div><div className="flow-mode"><button aria-pressed={mode==='explain'} onClick={()=>pin(latest,true)}>慢放本次计算</button><button aria-pressed={mode==='follow'} onClick={()=>{setMode('follow');setAuto(false);playback.pause();}}>跟随最新结果</button></div></header>
        <div className="flow-provenance"><b>{isLive?'实际前向采集':'所选记录重算'}</b><span>{calculation?`讲解帧 ${Number(calculation.frame??frame)+1} · ${calculation.track_id==null?'整帧图像':`行人 ${calculation.track_id}`}`:'模型运行后显示完整算式'}</span>{isLive&&snapshotFrame>=0&&<span className="flow-latest">最新采集帧 {snapshotFrame+1}{mode==='explain'&&snapshotFrame!==calculation?.frame?' · 后台继续，不替换正在讲解的数字':''}</span>}</div>
        <div className="flow-operation"><label>正在追踪的输出<select aria-label="选择数据流算式" value={selection&&available.some(c=>c.id===selection)?selection:latest?.id||''} disabled={!available.length} onChange={e=>choose(e.target.value)}>{!available.length&&<option value="">此阶段暂无完整算式</option>}{available.map(c=><option key={c.id} value={c.id}>{c.title} · {typeof c.target==='string'?c.target:JSON.stringify(c.target)}</option>)}</select></label><span className="flow-target">{calculation?`${used} / ${terms.length} 项有连接 · ${disconnected} 项无连接 · ${zero} 项贡献为零`:'选择网络结构仍可查看已采样层输出'}</span></div>
        {!calculation?<div className="flow-empty"><div>输入 <span>→</span> 权重 <span>→</span> 乘积 <span>→</span> 累加 <span>→</span> 输出</div><h3>{job?'等待这一阶段的完整输入与计算结果':'选择视频后，沿着数值观看网络计算'}</h3><p>{isLive?'新分析会记录代表性输出的完整参与项。LSTM 与 ST-GCN 在同一行人积累 50 次有效观测后运行；旧任务可切换“记录讲解”。':'先完成视频分析，再选择记录帧和行人。缺失的输入不会补成零。'}</p></div>:<>
            {!renderScene&&<Suspense fallback={<div className="flow-loading">正在准备三维数据流…</div>}><CalculationScene3D calculation={calculation} state={state} cursor={cursor} onTermSelect={index=>seek(index+.15)} onCursorChange={seek} onError={setSceneError}/></Suspense>}
            {sceneError&&<p className="flow-render-error" role="alert">{sceneError} · 下方数值与逐项控制仍可使用。</p>}
            <div className="flow-transport"><div className="flow-play-buttons"><button className="flow-play" aria-label={playback.playing&&mode==='explain'?'暂停数据流演算':'播放数据流演算'} onClick={()=>{if(mode==='follow')pin(latest,true);else playback.toggle();setAuto(false);}}>{playback.playing&&mode==='explain'?'Ⅱ 暂停':'▶ 播放'}</button><button aria-label="上一项数据流" onClick={()=>seek(Math.max(0,Math.floor(cursor)-1)+.15)}>← 上一项</button><button aria-label="下一项数据流" onClick={()=>seek(Math.min(terms.length+2,Math.floor(cursor)+1)+.15)}>下一项 →</button><button onClick={()=>seek(terms.length+2)}>查看输出</button></div><label className="flow-speed">速度<select aria-label="数据流演算速度" value={playback.speed} onChange={e=>playback.setSpeed(Number(e.target.value))}><option value=".5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select></label><label className="flow-auto"><input type="checkbox" checked={auto} onChange={e=>setAuto(e.target.checked)}/>连续讲解</label></div>
            <div className="flow-seek"><input type="range" aria-label="数据流逐项进度" min="0" max={terms.length+2} step=".01" value={cursor} onChange={e=>seek(Number(e.target.value))}/><div><strong>{phaseNames[state.phaseName]||state.phaseName}</strong><span>已累计 {state.completedCount} / {terms.length} 项 · 演算速度与模型耗时分开</span></div></div>
            <div className="flow-equation" aria-live="off"><div><span>当前项</span><strong>{state.currentTerm?`${number(state.currentTerm.input)} × ${number(state.currentTerm.weight)}`:'—'}</strong><small>来源 {state.currentTerm?.label||state.currentTerm?.inputIndex?.join(', ')||'完整输入集合'}</small></div><i>→</i><div><span>乘积</span><strong>{number(state.product)}</strong><small>{state.currentTerm?.connected===false?'有效邻接为 0：对这个目标无连接':state.currentTerm?.padding?'越界位置按模型规则零填充':state.product===0?'连接存在，本项贡献为 0':'计入当前目标的求和'}</small></div><i>→</i><div><span>当前累加值</span><strong data-testid="flow-partial-sum">{number(state.partialSum)}</strong><small>偏置 b = {number(calculation.bias)}</small></div><i>→</i><div className="flow-result"><span>{Number.isFinite(actual)?'独立采集的输出':'记录重建的输出'}</span><strong data-testid="flow-actual-result">{number(Number.isFinite(actual)?actual:calculation.reconstructed)}</strong><small>{Number.isFinite(calculation.absoluteError)?`完整重建误差 ${number(calculation.absoluteError)}`:'此项没有独立输出采样'}</small></div></div>
            {renderScene&&<button className="flow-expand-detail" aria-expanded={detailsOpen} onClick={()=>setDetailsOpen(value=>!value)}>{detailsOpen?'收起参与项表':'展开参与项表'}<span>数值已标在原三维网络中</span></button>}
            {(!renderScene||detailsOpen)&&<div className="flow-ledger"><div className="flow-ledger-heading"><div><span>PARTICIPATION / 针对当前目标</span><h3>哪些输入参与了这个结果</h3></div><p>{used} 项有连接 · {disconnected} 项无连接 · {zero} 项贡献为零{unknown?` · ${unknown} 项未采集`:''}</p></div><div className="flow-legend"><span className="current">当前参与</span><span className="accumulated">已经累计</span><span className="pending">等待累计</span><span className="disconnected">对本目标无连接</span><span className="unknown">未采集 ≠ 未使用</span></div><div className="flow-ledger-tools"><div>{[['all','全部项'],['disconnected',`无连接 ${disconnected}`],['zero',`零贡献 ${zero}`],['unknown',`未采集 ${unknown}`]].map(([value,label])=><button key={value} aria-pressed={filter===value} onClick={()=>{setFilter(value);setPage(0);}}>{label}</button>)}</div><div><button aria-label="上一页参与项" disabled={!visiblePage} onClick={()=>setPage(visiblePage-1)}>←</button><span>{visiblePage+1} / {maxPage+1}</span><button aria-label="下一页参与项" disabled={visiblePage===maxPage} onClick={()=>setPage(visiblePage+1)}>→</button></div></div><div className="flow-term-table" role="table" aria-label="完整算式参与项"><div role="row" className="flow-term-head"><span>来源 / 索引</span><span>输入值</span><span>权重 / 系数</span><span>乘积</span><span>作用</span></div>{visibleTerms.map(term=>{const status=termState(term,term.index,state);return <button role="row" key={term.index} className={`flow-term-row ${status}`} onClick={()=>seek(term.index+.5)} aria-label={`查看第 ${term.index+1} 项，${statusNames[status]}，输入 ${term.input}，权重 ${term.weight}，乘积 ${term.product}`}><span><b>#{term.index+1}</b> {term.label||term.inputIndex?.join(', ')||`x[${term.index}]`}</span><span title={String(term.input)}>{number(term.input)}</span><span title={String(term.weight)}>{number(term.weight)}</span><span title={String(term.product)}>{number(term.product)}</span><span>{statusNames[status]}{term.connected&&term.product===0?' · 贡献 0':''}</span></button>;})}{!visibleTerms.length&&<p className="flow-no-terms">这份完整算式中没有符合此条件的项。</p>}</div><p className="flow-scope">{calculation.formula}。每页仅展开 8 项，完整累加包含全部 {terms.length} 项。“无连接”仅针对这个目标；未展示的网络单元不能据此判定为没有使用。</p></div>}
        </>}
    </section>;
}
