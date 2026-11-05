import { Button, Details, Range, Summary } from './ui/Controls';
import { useEffect, useMemo, useRef, useState } from 'react';
import { lstmDisplayModel } from '../domain/memory';
import { formatValue as fmt } from '../domain/ledger';
import { createLSTMScene } from '../scene/lstmScene';

const PHASES = ['并行计算四个门', '保留旧记忆', '写入新记忆', '生成隐藏输出'];

export default function LSTMComputation({ live, record, track, state, actions }) {
    const [index, setIndex] = useState(0), [phase, setPhase] = useState(0), [error, setError] = useState('');
    const host = useRef(null), engine = useRef(null), current = useRef({});
    const model = useMemo(() => lstmDisplayModel(live, record?.result, track?.track_id, index), [live, record?.result, track?.track_id, index]);
    const playing = state.autoGraph && !state.frozen;
    current.current = { model, phase, index, actions };
    useEffect(() => {
        try {
            engine.current = createLSTMScene(host.current, {
                onInteract: () => current.current.actions.cameraInteract(),
                onError: message => { setError(message); engine.current?.dispose(); engine.current = null; },
            });
        } catch (cause) { setError(cause.message || 'WebGL 不可用'); }
        return () => { engine.current?.dispose(); engine.current = null; };
    }, []);
    useEffect(() => { engine.current?.update(model, phase, playing); }, [model, phase, playing]);
    useEffect(() => {
        if (!playing) return undefined;
        const timer = window.setInterval(() => {
            if (document.hidden) return;
            const { phase, index, model, actions } = current.current;
            if (phase < 3) setPhase(phase + 1);
            else {
                setPhase(0);
                if (model.steps && index >= model.steps - 1) actions.selectStep(5);
                else if (model.steps) setIndex(index + 1);
            }
        }, 900 / state.speed);
        return () => window.clearInterval(timer);
    }, [playing, state.speed]);
    useEffect(() => { setIndex(0); setPhase(0); }, [record?.selection?.key, track?.track_id]);
    const inspect = (nextIndex, nextPhase = phase) => { actions.cameraInteract(); setIndex(nextIndex); setPhase(nextPhase); };
    const gates = model.gates;
    return <section className="lstm-computation" aria-label="三维 LSTM 计算过程">
        <header className="temporal-heading"><div><span className="mono">LSTM / 3D RECURRENT CELL</span>
            <h3>四个门，逐步更新一份记忆</h3>
            <p>{gates ? `第 1 轮 · 第 1 层 · 单元 0 · 时间步 ${gates.t} / ${model.steps}` : `ID ${track?.track_id ?? '—'} · 有效历史 ${track?.history_length || 0} / 50`}</p></div>
            <Button type="button" className="process-button" onClick={() => { actions.cameraInteract(); engine.current?.reset(); }}>重置立体视角</Button>
        </header>
        <p className="lstm-source-note">{gates ? '使用全部隐藏单元重建第 1 层，仅展开单元 0；已与 PyTorch 的最终 h、c 核对。'
            : record?.active ? record.status === 'error' ? record.error : record.status === 'empty' ? '没有保存完整的 50 次有效历史。可换用更长或行人更清晰的视频。' : '正在读取历史并恢复 LSTM 门值。'
                : '实时显示已采集的输入、隐藏状态和预测轮次；四个门暂以结构线框展示，分析完成后自动恢复逐时间步门值。'}</p>
        {record?.status === 'error' && <Button type="button" className="process-button" onClick={record.retry}>重试记录重算</Button>}
        <div className="lstm-phase-tabs" role="group" aria-label="LSTM 演算阶段">
            {PHASES.map((name, i) => <Button type="button" key={name} aria-pressed={phase === i} onClick={() => inspect(index, i)}><span>0{i + 1}</span>{name}</Button>)}
        </div>
        <div className="lstm-time-control"><label htmlFor="lstm-time">时间步 <b>{model.timestep ?? '—'} / {model.steps || '—'}</b></label>
            <Button type="button" className="process-button" aria-label="LSTM 上一个时间步" disabled={!model.steps || index === 0}
                onClick={() => inspect(index - 1, 0)}>← 上一步</Button>
            <Range id="lstm-time" type="range" aria-label="LSTM 时间步" min="0" max={Math.max(0,model.steps - 1)} value={index}
                disabled={!model.steps} onChange={event => inspect(Number(event.target.value), 0)} />
            <Button type="button" className="process-button" aria-label="LSTM 下一个时间步" disabled={!model.steps || index >= model.steps - 1}
                onClick={() => inspect(index + 1, 0)}>下一步 →</Button>
            <span>{playing ? '自动逐门演示，50 步结束后进入 ST-GCN' : '已暂停，可拖动选择任意时间步'}</span>
        </div>
        <div className="lstm-3d-canvas" ref={host} />
        {error && <p role="alert">{error} 下方数值仍可查看。</p>}
        <p className="lstm-scene-note">实体方块与管线均为 3D 对象，可拖动旋转、滚轮缩放。金色光路表示当前演算阶段；几何位置用于解释运算关系。</p>
        <div className="lstm-gate-values">
            {model.gateTerms.map(gate => <div key={gate.name}><small>{({i:'输入门 i', f:'遗忘门 f', g:'候选记忆 g', o:'输出门 o'})[gate.name]}</small>
                <strong className="mono">{fmt(gate.value)}</strong><span>{gate.activation}({fmt(gate.z)})</span>
                <Details><Summary>展开加权计算</Summary><p className="mono">Wₓ·x = {fmt(gate.input)}<br/>Wₕ·h = {fmt(gate.recurrent)}<br/>b = {fmt(gate.bias)}</p></Details></div>)}
        </div>
        <div className="lstm-memory-equations">
            <div><small>保留 + 写入 → 新记忆 cₜ</small><b className="mono">{fmt(gates?.f)} × {fmt(gates?.c_previous)} + {fmt(gates?.i)} × {fmt(gates?.g)} = {fmt(gates?.c)}</b></div>
            <div><small>输出门 × 记忆激活 → 隐状态 hₜ</small><b className="mono">{fmt(gates?.o)} × tanh({fmt(gates?.c)}) = {fmt(gates?.h)}</b></div>
        </div>
        <div className="lstm-rounds" aria-label="五轮预测状态">{Array.from({length:5},(_,i) => {
            const round = model.forecasts.find(row => row.round === i + 1);
            return <div key={i} className={round?.poses?.length ? 'is-ready' : ''}><small>递归第 {i + 1} 轮</small><strong>未来 {i * 10 + 1}–{i * 10 + 10}</strong><span>{round?.poses?.length ? `${round.poses.length} 帧已采集` : '等待采集'}</span></div>;
        })}</div>
        <p className="lstm-source-note">每轮重新初始化 h₀、c₀，以 50 帧为输入预测 10 帧，滚动五轮得到未来 50 帧。门内演示限定第 1 轮；不是把一个门值复制到后四轮。</p>
    </section>;
}
