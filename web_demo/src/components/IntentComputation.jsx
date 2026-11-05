import { Button, Details, Range, Summary } from './ui/Controls';
import { useEffect, useMemo, useRef, useState } from 'react';
import { CLASS_LABELS } from '../api';
import { intentDisplayModel } from '../domain/intent';
import { formatValue as fmt } from '../domain/ledger';
import { flowState } from '../domain/calculation';
import { createIntentScene } from '../scene/intentScene';
import '../styles/intent.css';

const PHASES = ['全局平均池化', '三类分类乘加', 'Softmax 与结果'];

export default function IntentComputation({ live, record, track, state, actions }) {
    const model = useMemo(() => intentDisplayModel(live, record?.result, track?.track_id), [live, record?.result, track?.track_id]);
    const calculation = model.classifiers.find(row => row.metadata.selected_class === state.intentClass);
    const detailed = state.intentDetail;
    const channel = Math.min(Math.max(0,Math.floor(state.termIndex)),Math.max(0,model.pool.length-1));
    const term = calculation?.terms[channel];
    const flow = calculation ? flowState(calculation,state.termIndex) : null;
    const playing = detailed && state.autoGraph && !state.frozen;
    const host = useRef(null), engine = useRef(null), current = useRef({}), [error,setError] = useState('');
    current.current = { actions, state, calculation };
    useEffect(() => {
        try { engine.current = createIntentScene(host.current, {
            onInteract: () => current.current.actions.cameraInteract(),
            onClass: index => { const { state,actions }=current.current;
                if(!state.intentDetail) actions.setIntentDetail(true); actions.selectIntentClass(index); },
            onError: message => { setError(message); engine.current?.dispose(); engine.current=null; },
        }); } catch(cause) { setError(cause.message || 'WebGL 不可用'); }
        return () => { engine.current?.dispose(); engine.current=null; };
    }, []);
    useEffect(() => { engine.current?.update(model,state.beat,channel,state.intentClass,playing,detailed); }, [model,state.beat,channel,state.intentClass,playing,detailed]);
    useEffect(() => {
        if (!playing || !model.softmax) return undefined;
        let hold=0;
        const timer=window.setInterval(() => {
            if(document.hidden) return;
            const { state,actions,calculation }=current.current;
            if(state.beat===0) { if(++hold>=20) { hold=0; actions.tickIntent(1,0); } }
            else if(state.beat===1) {
                if(calculation && state.termIndex < calculation.terms.length-1) actions.tickIntent(1,Math.floor(state.termIndex)+1);
                else { hold=0; actions.tickIntent(2,calculation?.terms.length || 0); }
            } else if(++hold>=35) actions.setGraphAuto(false);
        },100/state.speed);
        return () => window.clearInterval(timer);
    }, [playing,state.speed,model.source?.frame,Boolean(model.softmax)]);
    const pooling=model.pooling, softmax=model.softmax;
    const chooseClass=index => { if(!detailed) actions.setIntentDetail(true); actions.selectIntentClass(index); };
    const winner=model.winner == null ? model.winners.length>1 ? '最高分并列' : '尚无分类结果' : `最高分：${CLASS_LABELS[model.winner]}`;
    const sourceLabel=model.source ? '历史重算分数' : model.held ? '沿用最近一次分类分数' : '原分析分数';
    return <section className="intent-computation" aria-label="三维意图计算过程">
        <header className="temporal-heading"><div><span className="mono">INTENT / 3D CLASSIFICATION</span>
            <h3>{winner}</h3><p>源帧 #{model.frame ?? '—'} · ID {model.trackId ?? '—'} · {sourceLabel}
                {model.margin != null ? ` · 前两名差 ${(model.margin*100).toFixed(2)} 个百分点` : ''}</p></div>
            <Button type="button" className="process-button" onClick={() => { actions.cameraInteract(); engine.current?.reset(); }}>重置意图视角</Button>
        </header>
        {detailed && <div className="intent-phase-tabs" role="group" aria-label="意图计算阶段">{PHASES.map((name,i) =>
            <Button type="button" key={name} aria-pressed={state.beat===i} onClick={() => actions.seekBeat(i)}><span>0{i+1}</span>{name}</Button>)}</div>}
        <div className="intent-3d-canvas" ref={host} />
        {error && <p role="alert">{error} 下方数值仍可查看。</p>}
        <div className="intent-scores" role="group" aria-label="三类意图分数">
            {CLASS_LABELS.map((label,i) => <Button key={label} type="button" aria-label={`查看${label}分类计算`}
                aria-pressed={detailed && state.intentClass===i} onClick={() => chooseClass(i)}>
                <span>{label}<small>{model.winners.includes(i) ? '最高分' : `类别 ${i}`}</small></span>
                <strong>{model.scores ? `${(model.scores[i]*100).toFixed(2)}%` : '—'}</strong>
                <i><em style={{width:`${(model.scores?.[i] || 0)*100}%`}} /></i>
                {model.source && <small>原分析：{model.originalScores ? `${(model.originalScores[i]*100).toFixed(2)}%` : '未记录'}</small>}
            </Button>)}
        </div>
        <p className="intent-note">{model.source ? '当前公式与 3D 柱使用重算分数；左侧保留原分析结果。保存的骨架有舍入，因此两者可能略有差异。'
            : model.held ? '此帧没有重新分类，显示该 ID 上一次预测的分数。' : '分数来自模型输出，不代表已验证的识别正确率。'}</p>
        {record?.active && record.status!=='ready' && <div className="intent-status" role="status">
            {record.status==='error' ? record.error : record.status==='empty' ? '没有保存完整的 50 次骨架历史，可切到“逐帧结果”回看原始分数。' : '正在恢复池化、分类权重与 Softmax…'}
            {record.status==='error' && <Button className="process-button" onClick={record.retry}>重试意图计算</Button>}
        </div>}
        {detailed && <>
            <div className="intent-controls">
                <Button className="process-button" onClick={actions.restartIntent} disabled={!softmax}>从池化重播</Button>
                <Button className="process-button" disabled={state.beat===0} onClick={() => actions.seekBeat(state.beat-1)}>← 上个阶段</Button>
                <Button className="process-button" disabled={state.beat===2} onClick={() => actions.seekBeat(state.beat+1)}>下个阶段 →</Button>
                <span>{playing ? state.beat===1 ? '自动逐项扫描全部通道' : '自动演示计算阶段' : '已暂停，可切换阶段和通道回看'}</span>
            </div>
            <div className="intent-equation" aria-live="off">
                {state.beat===0 ? <><small>每个通道沿时间、关节求平均；完整展开通道 {pooling?.channel ?? 0}</small>
                    <strong>pool[{pooling?.channel ?? 0}] = {fmt(pooling?.sum)} ÷ {pooling?.count ?? '—'} = {fmt(pooling?.mean)}</strong>
                    <span>PyTorch 实际值 {fmt(pooling?.actual)} · 误差 {fmt(pooling?.absolute_error)}</span></>
                    : state.beat===1 ? <><small>{CLASS_LABELS[state.intentClass]} · 通道 c={channel} · 第 {model.pool.length ? channel+1 : '—'} / {model.pool.length || '—'} 项</small>
                        <strong>{fmt(term?.input)} × {fmt(term?.weight)} = {fmt(term?.product)}</strong>
                        <span>当前已累计 {flow?.completedCount ?? '—'} 项 {fmt(flow?.partialSum)} · 全部乘积和 {fmt(calculation?.dot)} + 偏置 {fmt(calculation?.bias)} = {fmt(calculation?.reconstructed)}</span></>
                    : <><small>数值稳定的 Softmax：pᵢ = exp(zᵢ − max(z)) / Σ exp(zⱼ − max(z))</small>
                        <strong>最大 logit = {fmt(softmax?.max)} · 指数总和 = {fmt(softmax?.denominator)}</strong>
                        <span>与模型分数的最大绝对误差 {fmt(model.softmaxError)}</span></>}
            </div>
            {state.beat===1 && <div className="intent-channel-control"><Button className="process-button" disabled={!model.pool.length || channel===0}
                onClick={() => actions.seekTerm(channel-1)}>← 上一通道</Button>
                <label>分类输入通道 <Range type="range" aria-label="意图分类输入通道" min="0" max={Math.max(0,model.pool.length-1)} value={channel}
                    disabled={!model.pool.length} onChange={event => actions.seekTerm(Number(event.target.value))} /></label>
                <Button className="process-button" disabled={!model.pool.length || channel>=model.pool.length-1} onClick={() => actions.seekTerm(channel+1)}>下一通道 →</Button></div>}
            {state.beat===1 && !calculation && <p className="intent-note">所选类别的权重未采集。实时数据只保留代表分类算式，分析完成后可重算全部三类。</p>}
        </>}
        <p className="intent-note">{detailed ? '左侧为通道 0 的完整池化示例；中间展示全部已采集的池化特征。小柱高度在各自矩阵内缩放，右侧分数柱统一使用 0–100% 刻度。' : '三根立体柱使用统一的 0–100% 刻度。切到计算拆解可查看形成分数的算式。'}可拖动旋转、滚轮缩放。</p>
        {detailed && <>
            <div className="intent-table-scroll"><table className="intent-softmax-table"><caption>三类 logits → 指数 → 分数</caption>
                <thead><tr><th>类别</th><th>logit z</th><th>z − max(z)</th><th>exp(z − max)</th><th>归一化分数</th><th>模型实际分数</th></tr></thead>
                <tbody>{CLASS_LABELS.map((label,i) => <tr key={label}><th>{label}</th><td>{fmt(model.logits[i])}</td><td>{fmt(softmax?.shifted[i])}</td>
                    <td>{fmt(softmax?.exponents[i])}</td><td>{fmt(softmax?.probabilities[i])}</td><td>{fmt(model.scores?.[i])}</td></tr>)}</tbody></table></div>
            {!pooling && <p className="intent-note">此帧未保存完整池化平面；未采集的位置保持为空。分析完成后可从保存的完整历史重算。</p>}
            {pooling && <Details className="intent-pool-details"><Summary>展开通道 {pooling.channel} 的全部 {pooling.count} 项池化输入</Summary>
                <div className="intent-table-scroll"><table><caption>行：时间索引 t；列：关节 v；均值除数 = {pooling.count}</caption>
                    <thead><tr><th>t / v</th>{pooling.values[0].map((_,v) => <th key={v}>{v}</th>)}</tr></thead>
                    <tbody>{pooling.values.map((row,t) => <tr key={t}><th>{t}</th>{row.map((value,v) => <td key={v}>{fmt(value)}</td>)}</tr>)}</tbody>
                </table></div></Details>}
        </>}
    </section>;
}
