import { Button, Details, Select, Summary } from './ui/Controls';
import { useEffect, useRef, useState } from 'react';
import { temporalWindow, temporalPartialSum } from '../domain/temporal';
import { formatValue } from '../domain/ledger';
import TemporalStage3D from './TemporalStage3D';

export default function TemporalConvolution({ calculation, flow, frozen, onSeek, onFreeze, auto = false, speed = 1,
    onTick, onNext, onInteract, record, job, track }) {
    const [playing, setPlaying] = useState(false);
    const timerState = useRef({});
    const lastChannel = useRef(0);
    const kernel = calculation?.metadata?.kernel_size || 1;
    if (flow && !flow.biasApplied) lastChannel.current = Math.floor(flow.termIndex / kernel);
    const channel = lastChannel.current;
    const window = temporalWindow(calculation, channel);
    timerState.current = { flow, window, onSeek, onTick, onNext, calculation };
    useEffect(() => {
        if (!auto || frozen || !calculation) return undefined;
        let tick = 0;
        const reduced = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        const timer = globalThis.setInterval(() => {
            if (document.hidden) return;
            const current = timerState.current;
            if (!current.window) return;
            const { kernel, offset } = current.window;
            const count = current.calculation.terms.length;
            const position = tick / 6;
            if (position < kernel) current.onTick(offset + position);
            else if (position < kernel + 1) current.onTick(count + (position - kernel));
            else if (position < kernel + 2) current.onTick(count + 1);
            else if (position < kernel + 4) current.onTick(count + 2);
            else { tick = 0; current.onNext(); return; }
            tick += reduced ? 6 : 1;
        }, (reduced ? 600 : 100) / speed);
        return () => globalThis.clearInterval(timer);
    }, [auto, frozen, calculation?.id, speed]);
    useEffect(() => {
        if (!playing || !frozen) { setPlaying(false); return undefined; }
        const timer = globalThis.setInterval(() => {
            const { flow, window, onSeek } = timerState.current;
            if (!window || flow.cursor >= window.offset + window.kernel - 0.1) { setPlaying(false); return; }
            onSeek(Math.min(window.offset + window.kernel - 0.1, flow.cursor + 1));
        }, 450);
        return () => globalThis.clearInterval(timer);
    }, [playing, frozen, calculation?.id]);
    if (!window) return <section className="temporal-stage temporal-empty" aria-label="时序卷积窗口" role="status">
        <h3>{record?.active ? record.status === 'error' ? '记录重算未完成' : record.status === 'empty'
            ? '这段视频没有保存完整的 50 次骨架历史' : record.status === 'ready' ? '所选窗口没有有效记录' : '正在准备完整时间卷积'
            : !job ? '选择视频并开始分析' : (track?.history_length || 0) < 50 ? '正在积累骨架历史' : '正在接收时间卷积数据'}</h3>
        <p>{record?.active ? record.error || (record.status === 'empty' ? '需要同一个行人累积 50 次有效观测，才能展示 LSTM → ST-GCN。可换用更长或行人更清晰的视频。'
            : record.status === 'ready' ? '可切换已就绪的卷积块，或重试记录重算。'
            : `从保存的骨架历史恢复全部卷积块${record.selection ? ` · 帧 #${record.selection.frame} · ID ${record.selection.track.track_id}` : ''}。`)
            : `当前有效观测 ${track?.history_length || 0} / 50。自动模式会选中已采集的卷积块；分析完成后自动接入历史重算。`}</p>
        {record?.active && ['error', 'ready'].includes(record.status) && <Button type="button" className="process-button" onClick={record.retry}>重试记录重算</Button>}
    </section>;
    const activeTerm = flow.biasApplied ? window.terms.at(-1) : flow.currentTerm;
    const channelCount = Math.max(0, Math.min(kernel, flow.completedCount - window.offset));
    const channelSum = temporalPartialSum(window.terms, channelCount);
    return <section className="temporal-stage" aria-label="时序卷积窗口">
        <header className="temporal-heading"><div><span className="mono">TEMPORAL CONVOLUTION / 代表输出</span>
            <h3>{window.channels} 个通道 × {window.kernel} 个时间位置 → 1 个输出</h3>
            <p>第 {(calculation.metadata.block ?? 0) + 1} 块 · ID {calculation.track_id} · 输出通道 {window.target[0]} · 输出时刻 {window.target[1]} · 关节 {window.target[2]}</p></div>
            <label>输入通道 <Select aria-label="时序卷积输入通道" value={channel} onChange={event => { setPlaying(false); onSeek(Number(event.target.value) * kernel); }}>
                {Array.from({ length: window.channels }, (_, i) => <option key={i} value={i}>c = {i}</option>)}</Select></label>
        </header>
        <div className="temporal-parameters mono">步长 {window.stride} · 填充 {window.padding} · 膨胀 {window.dilation} · t输入 = {window.target[1]} × {window.stride} − {window.padding} + q × {window.dilation}</div>
        <div className="temporal-parameters mono">y = Σc Σq (x[c, t输入, v] × w[c输出, c, q]) + b[c输出]</div>
        <p className="temporal-explanation">固定一个输出通道、时刻和关节，沿时间轴逐项乘加，再跨输入通道求和。t 是本层特征序列索引，x 是空间聚合并经 BN / ReLU 后的特征值。</p>
        <div className="temporal-live-equation" aria-label="当前三维乘加">
            <span>{auto ? '自动演示' : '定点查看'} · {flow.biasApplied ? '全部通道汇总' : `c=${channel} / q=${flow.termIndex % kernel}`}</span>
            <strong className="mono">{formatValue(activeTerm.input)} × {formatValue(activeTerm.weight)} = {formatValue(activeTerm.product)}</strong>
            <span>本通道累计 <b className="mono">{formatValue(channelSum)}</b></span>
        </div>
        <TemporalStage3D calculation={calculation} channel={channel} flow={flow} onSeek={onSeek} onInteract={onInteract} auto={auto && !frozen} />
        <Details className="temporal-grid-details"><Summary>展开本通道 {kernel} 项输入、权重和乘积</Summary>
        <span className="temporal-scroll-hint">左右滑动查看全部 {kernel} 个时间位置 →</span>
        <div className="temporal-scroll"><div className="temporal-window" style={{ gridTemplateColumns: `repeat(${window.kernel}, minmax(82px, 1fr))` }}>
            {window.terms.map((term, q) => <Button type="button" key={q}
                className={`${term.padding ? 'zero-pad' : ''} ${flow.termIndex === window.offset + q ? 'active-term' : ''}`}
                aria-label={`通道 ${window.channel} 时间 ${term.inputIndex[1]}${term.padding ? ' 零填充' : ''} 输入 ${formatValue(term.input)} 权重 ${formatValue(term.weight)}`}
                onClick={() => { setPlaying(false); onSeek(window.offset + q + 0.5); }}>
                <b>t = {term.inputIndex[1]}</b><small>{term.padding ? '边界零填充' : `核位置 q = ${q}`}</small>
                <span><small>输入 x</small>{formatValue(term.input)}</span>
                <i>×</i><span><small>权重 w</small>{formatValue(term.weight)}</span>
                <i>↓</i><strong><small>乘积 x·w</small>{formatValue(term.product)}</strong>
            </Button>)}
        </div></div></Details>
        <div className="temporal-sums">
            <div><small>① 本通道 {window.kernel} 项乘积和</small><b className="mono">{formatValue(window.sum)}</b></div>
            <span>→</span><div><small>② 全部 {calculation.terms.length} 项求和 + 偏置</small><b className="mono">{formatValue(calculation.dot)} + {formatValue(calculation.bias)}</b></div>
            <span>→</span><div><small>③ 卷积输出 · 后续 BN / 残差 / ReLU 之前</small><b className="mono">{formatValue(calculation.actual)}</b></div>
        </div>
        <footer><Button type="button" className="process-button" aria-pressed={playing} onClick={() => {
            if (playing) setPlaying(false);
            else { onFreeze(); onSeek(window.offset); setPlaying(true); }
        }}>{playing ? '暂停通道演算' : `演示本通道 ${kernel} 项`}</Button>
            <p>自动演示当前通道后汇总全部 {window.channels} 个通道，再切换已就绪窗口。动画速度用于讲解；实时数值按采样更新。手动通道演示会冻结数值。</p></footer>
    </section>;
}
