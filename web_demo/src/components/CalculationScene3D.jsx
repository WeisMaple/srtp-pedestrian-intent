import { useEffect, useMemo, useRef, useState } from 'react';
import { createCalculationScene } from '../three/CalculationScene.js';
import { flowState } from '../calculationFlow.js';
import '../calculation-scene.css';

const PHASES = { input: '读取输入', product: '乘以权重', accumulate: '送入累加器', accumulated: '已计入累加', bias: '加入偏置', activation: '执行激活', result: '核对实际输出', complete: '计算完成' };
const MEMBERSHIP = { connected: '已连接', zeroContribution: '已连接 · 本项贡献为 0', disconnected: '无结构连接', padding: '边界补零', unrecorded: '未记录' };
const ACTIVATION = { identity: '恒等映射', relu: 'ReLU', tanh: 'tanh', sigmoid: 'sigmoid', silu: 'SiLU', batch_norm_silu: 'BN → SiLU', softmax: 'softmax' };
const number = (value, precision = 6) => typeof value === 'number' && Number.isFinite(value) ? (value!==0&&(Math.abs(value)<.001||Math.abs(value)>=1e6)?value.toExponential(Math.min(4,precision-1)):Number(value.toPrecision(precision)).toString()).replaceAll('-', '−') : '未记录';
const description = value => typeof value === 'string' ? value : Array.isArray(value) ? `[${value.join(', ')}]` : value && typeof value === 'object' ? Object.entries(value).map(([key, item]) => `${key}=${Array.isArray(item) ? `[${item.join(',')}]` : item}`).join(' · ') : '';
const indexLabel = term => Array.isArray(term?.inputIndex) ? `[${term.inputIndex.join(',')}]` : term?.inputIndex !== undefined ? String(term.inputIndex) : `i=${term?.index ?? '—'}`;
const shortNumber = value => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    return number(value, 2).replace(/^0\./, '.').replace(/^−0\./, '−.');
};

function Dock({ name, title, symbol, value, detail, active, children, className = '' }) {
    return <div className={`calc-label calc-operator calc-operator-${name} ${active ? 'is-current' : ''} ${className}`} data-calc-anchor={name} data-dock={name}>
        <div className="calc-operator-heading"><span>{title}</span><b aria-hidden="true">{symbol}</b></div>
        {value !== undefined && <strong className="calc-value" title={typeof value === 'number' ? String(value) : undefined}>{number(value)}</strong>}
        {detail && <div className="calc-operator-detail" title={detail}>{detail}</div>}
        {children}
    </div>;
}

/** The playback cursor belongs to the owner; this component never starts a timer. */
export default function CalculationScene3D({ calculation, state: suppliedState, cursor = 0, onTermSelect, onCursorChange, onError }) {
    const host = useRef(null);
    const overlay = useRef(null);
    const engine = useRef(null);
    const callbacks = useRef({ onTermSelect, onCursorChange, onError });
    const [page, setPage] = useState(0);
    const [revision, setRevision] = useState(0);
    const [error, setError] = useState(null);
    const state = useMemo(() => suppliedState || (calculation ? flowState(calculation, cursor) : null), [suppliedState, calculation, cursor]);
    const terms = calculation?.terms || [];
    const current = state?.currentTerm || terms[state?.termIndex ?? 0];
    const activeIndex = state?.termIndex ?? 0;
    const phase = state?.phaseName || 'input';
    const pageCount = Math.max(1, Math.ceil(terms.length / 32));
    const visiblePage = Math.min(page, pageCount - 1);
    const start = visiblePage * 32;
    const visibleTerms = terms.slice(start, start + 32);
    callbacks.current = { onTermSelect, onCursorChange, onError };
    const latest = useRef(null);
    latest.current = { calculation, state, page: visiblePage, cursor };

    function selectTerm(index) {
        if (callbacks.current.onTermSelect) callbacks.current.onTermSelect(index);
        else callbacks.current.onCursorChange?.(index);
    }

    useEffect(() => {
        setError(null);
        try {
            engine.current = createCalculationScene(host.current, overlay.current, {
                onTermSelect: index => selectTerm(index),
                onError: message => { setError(message); callbacks.current.onError?.(message); },
            });
            const props = latest.current;
            engine.current.update(props.calculation, props.state, props);
            callbacks.current.onError?.(null);
        } catch (cause) {
            const message = `三维画布暂不可用：${cause.message || 'WebGL 不可用'}。以下保留实际运算数值。`;
            setError(message);
            callbacks.current.onError?.(message);
        }
        return () => { engine.current?.dispose(); engine.current = null; };
    }, [revision]);

    useEffect(() => { setPage(0); }, [calculation?.id]);
    // Follow the data when playback reaches a new page. Manual paging remains possible while paused.
    useEffect(() => {
        if (activeIndex >= 0 && activeIndex < terms.length) setPage(Math.floor(activeIndex / 32));
    }, [activeIndex, terms.length]);
    useEffect(() => {
        engine.current?.update(calculation, state, { page: visiblePage, cursor });
    }, [calculation, state, visiblePage, cursor]);

    const target = description(calculation?.target) || '当前目标单元';
    const source = calculation?.source === 'live' ? '实时记录' : '保存记录';
    const scope = [calculation?.frame != null && `帧 ${calculation.frame+1}`, calculation?.track_id != null ? `行人 ${calculation.track_id}` : '整帧图像'].filter(Boolean).join(' · ');
    const member = current?.membership || 'unrecorded';
    const excluded = ['disconnected', 'padding', 'unrecorded'].includes(member);
    const inputName = current?.label || indexLabel(current);
    const activationName = ACTIVATION[calculation?.activation] || calculation?.activation || '恒等映射';
    const formula = current && !excluded ? `${number(current.input)} × ${number(current.weight)} = ${number(current.product)}` : member === 'padding' ? '边界补零：本项贡献为 0' : member === 'disconnected' ? '该来源与当前目标没有结构连接' : '没有记录该项数值，无法重算';
    const packetValue = phase === 'bias' ? calculation?.bias : phase === 'activation' ? state?.activationPhase >= .5 ? calculation?.reconstructed : calculation?.preactivationValue : phase === 'result' ? calculation?.reconstructed : phase === 'accumulate' || phase === 'accumulated' ? current?.product : phase === 'product' ? current?.weight : current?.input;
    const packetName = phase === 'bias' ? '+ b' : phase === 'activation' ? state?.activationPhase >= .5 ? activationName : 'Σ + b' : phase === 'result' ? activationName : phase === 'accumulate' || phase === 'accumulated' ? 'pᵢ' : phase === 'product' ? 'wᵢ' : 'xᵢ';
    const sumDetail = `已计入 ${state?.completedCount || 0} / ${terms.length} 项`;
    const actualRecorded = typeof calculation?.actual === 'number' && Number.isFinite(calculation.actual);

    return <section className={`calculation-scene ${error ? 'is-fallback' : ''}`} data-calculation-id={calculation?.id} data-stage={calculation?.stage ?? calculation?.chapter} data-phase={phase} aria-label="实际运算的三维数据流">
        <header className="calculation-scene-toolbar">
            <div className="calculation-scene-identity"><span className="calc-dimension">3D</span><div><strong>{calculation?.title || '等待可核对的实际运算'}</strong><span>{source}{scope && ` · ${scope}`}</span></div></div>
            <div className="calc-camera-tools"><button type="button" onClick={() => engine.current?.focusSelected()} disabled={!!error || !current} title="轻转视角，查看输入与权重的纵深">查看纵深</button><button type="button" onClick={() => engine.current?.resetCamera()} disabled={!!error}>重置视角</button></div>
        </header>
        {error && <div className="calc-error" role="alert"><span>{error}</span><button type="button" onClick={() => setRevision(value => value + 1)}>恢复画布</button></div>}
        <div className="calculation-scene-scroll" tabIndex={0} aria-label="三维计算画布；窄屏可横向滚动">
            <div className="calculation-scene-space">
                <div className="calculation-scene-viewport" ref={host} />
                <div className="calc-overlay" ref={overlay}>
                    <div className="calc-label calc-grid-heading" data-calc-anchor="gridTitle">
                        <div><span>输入来源</span><strong>{terms.length ? `${start + 1}–${Math.min(start + 32, terms.length)}` : '—'} <small>/ {terms.length} 项</small></strong></div>
                        <nav aria-label="三维输入分页"><button type="button" aria-label="上一页输入" onClick={() => setPage(value => Math.max(0, value - 1))} disabled={visiblePage === 0}>‹</button><span>{visiblePage + 1}/{pageCount}</span><button type="button" aria-label="下一页输入" onClick={() => setPage(value => Math.min(pageCount - 1, value + 1))} disabled={visiblePage === pageCount - 1}>›</button></nav>
                    </div>
                    {visibleTerms.map((term, slot) => {
                        const selected = term.index === activeIndex;
                        const status = state?.statusByTerm?.[term.index] || 'pending';
                        const membership = term.membership || 'connected';
                        const marker = membership === 'disconnected' ? '×' : membership === 'unrecorded' ? '?' : membership === 'padding' ? '∅' : selected ? '›' : status === 'accumulated' ? '✓' : '·';
                        const accessible = `项 ${term.index}，来源 ${term.label || indexLabel(term)}，输入 ${number(term.input, 12)}，权重 ${number(term.weight, 12)}，乘积 ${number(term.product, 12)}，${MEMBERSHIP[membership]}，${selected ? '当前项' : status === 'accumulated' ? '已累加' : '待处理'}`;
                        return <button type="button" key={`${start}-${slot}`} className={`calc-label calc-grid-cell is-${membership} status-${status} ${selected ? 'is-selected' : ''}`} data-calc-anchor={`slot-${slot}`} data-term-index={term.index} data-membership={membership} data-status={status} aria-label={accessible} aria-pressed={selected} title={accessible} onClick={() => selectTerm(term.index)}><span className="calc-cell-index"><span>{term.index}</span><b aria-hidden="true">{marker}</b></span><strong>{shortNumber(term.input)}</strong></button>;
                    })}
                    <Dock name="input" title={`输入 x[${current?.index ?? '—'}]`} symbol="xᵢ" value={current?.input} detail={indexLabel(current)} active={phase === 'input'}>
                        <span className="calc-source-name" title={inputName}>{inputName}</span>
                    </Dock>
                    <Dock name="weight" title="当前连接的权重" symbol="wᵢ" value={current?.weight} detail={MEMBERSHIP[member]} active={phase === 'product'} className={`is-${member}`} />
                    <Dock name="product" title="这一项的贡献" symbol="×" value={current?.product} detail={`x[${current?.index ?? '—'}] × w[${current?.index ?? '—'}]`} active={phase === 'product' || phase === 'accumulate'}>
                        <span className="calc-source-name">{excluded ? MEMBERSHIP[member] : current?.product === 0 ? '参与计算 · 贡献为 0' : '送入当前目标的累加器'}</span>
                    </Dock>
                    <Dock name="sum" title="运行中的部分和" symbol="Σ" value={state?.partialSum} detail={sumDetail} active={phase === 'accumulate' || phase === 'accumulated' || phase === 'bias'}>
                        <span className="calc-source-name">{phase === 'bias' || phase === 'activation' || state?.done ? 'Σ pᵢ + b' : 'Σ 已完成项的 pᵢ'}</span>
                    </Dock>
                    <Dock name="bias" title="偏置 b" symbol="+" value={calculation?.bias} active={phase === 'bias'} />
                    <Dock name="activation" title="完成求和后" symbol="ƒ" detail={activationName} active={phase === 'activation'} />
                    <Dock name="output" title="实际记录的目标输出" symbol="y" value={calculation?.actual} detail={actualRecorded ? '独立读取模型输出' : '当前没有输出记录'} active={phase === 'result' || !!state?.done}>
                        <span className="calc-output-reconstruction">重算 <b title={String(calculation?.reconstructed ?? '')}>{number(calculation?.reconstructed)}</b></span>
                    </Dock>
                    <div className="calc-label calc-destination" data-calc-anchor="destination"><span>本次只计算这个目标</span><strong title={target}>{target}</strong></div>
                    <div className="calc-label calc-moving-value" data-calc-anchor="packet" aria-hidden="true"><span>{packetName}</span><strong>{number(packetValue, 5)}</strong></div>
                </div>
                {!calculation && <div className="calc-empty">选择已记录的运算，即可逐项查看输入、权重与输出。</div>}
            </div>
        </div>
        <div className="calc-equation-strip" data-testid="calculation-current-formula"><span className="calc-phase-name">{PHASES[phase] || phase}</span><strong>{phase === 'bias' ? `${number(calculation?.dot)} + ${number(calculation?.bias)} = ${number(calculation?.preactivationValue)}` : phase === 'activation' || phase === 'result' || state?.done ? calculation?.formula || `${activationName}(${number(calculation?.preactivationValue)}) = ${number(calculation?.reconstructed)}` : formula}</strong></div>
        <footer className="calculation-scene-legend"><div><span><i className="is-current" />当前项</span><span><i className="is-accumulated" />已累加</span><span><i className="is-pending" />待处理</span><span><i className="is-disconnected">×</i>无连接</span><span><i className="is-unrecorded" />未记录</span></div><p>拖动旋转 · 滚轮缩放 · 点选输入<span> · 窄屏横向滑动</span></p></footer>
    </section>;
}
