import { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Details, Select, Summary } from './ui/Controls';
import NumericLedger from './NumericLedger';
import { JOINT_LABELS, spatiotemporalModel } from '../domain/spatiotemporal';
import { temporalWindow } from '../domain/temporal';
import { flowState } from '../domain/calculation';
import { filterTerms, formatValue } from '../domain/ledger';
import SpatiotemporalGraph from './SpatiotemporalGraph';
import '../styles/spatiotemporal.css';

function ArithmeticCard({ kind, calculation, index, onSeek, children }) {
    const [page, setPage] = useState(0), [filter, setFilter] = useState('all');
    const spatial = kind === 'spatial';
    const term = calculation?.terms[index];
    const target = calculation?.metadata?.target_indices;
    return <section className={`st-arithmetic ${kind}`} aria-label={spatial ? '空间聚合计算' : '时间卷积计算'}>
        <header><span>{spatial ? 'S / 空间维度' : 'T / 时间维度'}</span><h3>{spatial ? '同帧邻域聚合' : '跨帧时间卷积'}</h3>
            {calculation && <p className="st-target">输出 c={target?.[0]} · t={target?.[1]} · {JOINT_LABELS[target?.[2]]} v{target?.[2]}</p>}</header>
        {calculation && term ? <>
            {children}
            <div className="st-equation"><small>{spatial ? `分区 k=${term.inputIndex[0]} · 源关节 ${JOINT_LABELS[term.inputIndex.at(-1)]} v${term.inputIndex.at(-1)}`
                : `输入 c=${term.inputIndex[0]} · t=${term.inputIndex[1]}${term.padding ? ' · 零填充' : ''}`}</small>
                <strong>{formatValue(term.input)} <i>×</i> {formatValue(term.weight)} <i>=</i> {formatValue(term.product)}</strong>
                <small>{spatial ? '投影特征 × 实际邻接权重' : '空间聚合经 BN / ReLU 后的特征 × 时间核权重'}</small></div>
            <div className="st-totals"><div><small>全部 {calculation.terms.length} 项 + 偏置</small><b>{formatValue(calculation.reconstructed)}</b></div>
                <div><small>实测输出</small><b>{formatValue(calculation.actual)}</b></div>
                <div><small>重建误差</small><b>{calculation.absoluteError?.toExponential(2) ?? '—'}</b></div></div>
            <Details className="st-ledger"><Summary>查看完整 {calculation.terms.length} 项乘加</Summary>
                <NumericLedger calculation={calculation} state={flowState(calculation, index + .5)} terms={filterTerms(calculation.terms, filter)}
                    page={page} filter={filter} onPage={setPage} onFilter={value => { setFilter(value); setPage(0); }} onSeek={value => onSeek(Math.floor(value))} />
            </Details>
        </> : <p className="st-wait">等待当前帧、当前行人和当前块的{spatial ? '空间' : '时间'}计算。</p>}
    </section>;
}

export default function SpatiotemporalComputation({ job, live, record, track, structure, state, actions }) {
    const model = useMemo(() => spatiotemporalModel({ snapshot: live, result: record?.result, track,
        jobId: job?.id, block: state.block, boundary: state.boundary, structure }),
    [job?.id, live, record?.result, track, state.block, state.boundary, structure]);
    const [spatialIndex, setSpatialIndex] = useState(0), [temporalIndex, setTemporalIndex] = useState(0);
    const [graphZoom, setGraphZoom] = useState(1);
    const [showLowScores, setShowLowScores] = useState(false);
    const connected = useMemo(() => model.spatial?.terms.map((term, index) => ({ term, index })).filter(({ term }) => term.connected) || [], [model.spatial]);
    const spatialCursor = Math.min(spatialIndex, Math.max(0, (model.spatial?.terms.length || 1) - 1));
    const temporalCursor = Math.min(temporalIndex, Math.max(0, (model.temporal?.terms.length || 1) - 1));
    const spatialTerm = model.spatial?.terms[spatialCursor], temporalTerm = model.temporal?.terms[temporalCursor];
    const kernel = model.temporal?.metadata?.kernel_size || 1;
    const channel = Math.floor(temporalCursor / kernel), window = temporalWindow(model.temporal, channel);
    const stepRef = useRef({}); stepRef.current = { connected, kernel, channel, spatialCursor, temporalCursor };
    useEffect(() => {
        setSpatialIndex(connected[0]?.index || 0); setTemporalIndex(0);
    }, [state.block, state.boundary, model.source.frame, model.source.trackId, model.source.jobId, Boolean(model.spatial), Boolean(model.temporal)]);
    useEffect(() => {
        if (!state.autoGraph || state.frozen || !model.temporal) return undefined;
        const timer = setInterval(() => {
            if (document.hidden) return;
            const item = stepRef.current;
            setTemporalIndex(item.channel * item.kernel + (item.temporalCursor % item.kernel + 1) % item.kernel);
            const position = item.connected.findIndex(row => row.index === item.spatialCursor);
            setSpatialIndex(item.connected[(position + 1) % item.connected.length]?.index || 0);
        }, 850 / state.speed);
        return () => clearInterval(timer);
    }, [state.autoGraph, state.frozen, state.speed, Boolean(model.temporal)]);
    const inspect = () => actions.cameraInteract();
    const seekSpatial = index => { inspect(); setSpatialIndex(index); };
    const seekTemporal = index => { inspect(); setTemporalIndex(index); };
    const selectedSlot = model.slots.find(slot => slot.time === temporalTerm?.inputIndex?.[1]);
    const pick = point => {
        inspect();
        if (point.spatial) {
            const found = connected.find(({ term }) => term.inputIndex.at(-1) === point.joint);
            if (found) setSpatialIndex(found.index);
        }
        if (point.temporal && point.joint === model.target) {
            const q = window?.terms.findIndex(term => term.inputIndex[1] === point.time);
            if (q >= 0) setTemporalIndex(channel * kernel + q);
        }
    };
    return <section className="spatiotemporal" aria-label="骨架时空卷积同页展示">
        <header className="st-source"><div><span className="st-kicker">SKELETON → SPACE × TIME</span><h2>同一副骨架，沿空间与时间计算</h2></div>
            <span className="st-source-id">{record?.active ? '记录重算' : '实时采集'} · ID {track?.track_id ?? '—'} · 源帧 #{model.source.frame ?? '—'} · 第 {state.block} 块</span></header>
        <div className="st-provenance"><span>YOLO 观测骨架 <b>{track?.history_length ?? model.history.length} / 50</b></span><i>→</i>
            <span>LSTM 预测骨架 <b>{model.forecasts.length} / 50</b></span><i>→</i><span>归一化 / 特征变换</span><i>→</i><span>ST-GCN <b>空间 → 时间</b></span></div>
        <div className="st-layout"><div className="st-visual">
            {model.forecasts.length && model.slots.some(slot => slot.pose) ? <SpatiotemporalGraph key={`${model.source.jobId}:${model.source.trackId}:${state.block}:${state.boundary}`}
                model={model} spatialTerm={spatialTerm} temporalTerm={temporalTerm} zoom={graphZoom} onZoom={setGraphZoom}
                showLowScores={showLowScores} onShowLowScores={setShowLowScores}
                onSelect={pick} onInteract={inspect} /> : <div className="st-empty" role="status">
                <h3>{record?.status === 'error' ? '记录重算未完成' : (track?.history_length || 0) < 50 ? '等待同一行人的 50 次骨架观测' : '等待同帧骨架与卷积数据'}</h3>
                <p>{record?.error || (record?.active ? '从已保存的骨架历史重算；就绪后空间和时间同时展示。' : '当前未采集完整来源，骨架与计算值保持为空。')}</p>
                {record?.active && record.status === 'error' && <Button onClick={record.retry}>重试记录重算</Button>}
            </div>}
            <div className="st-time-slots" aria-label="时间卷积骨架窗口">{window?.terms.map((term, q) => <Button key={q}
                aria-pressed={q === temporalCursor % kernel} className={term.padding ? 'is-padding' : ''}
                aria-label={`选择时间位置 ${term.inputIndex[1]}${term.padding ? '，零填充' : ''}`}
                onClick={() => seekTemporal(channel * kernel + q)}><b>t={term.inputIndex[1]}</b><span>{term.padding ? '补零' : `特征 ${formatValue(term.input)}`}</span></Button>)}</div>
            <div className="st-receptive"><strong>{selectedSlot?.padding ? '该时间位置超出序列：实际输入为 0，不存在对应骨架。'
                : selectedSlot?.range ? `所选特征的时间感受野：预测 +${selectedSlot.range[0] + 1} … +${selectedSlot.range[1] + 1}` : '等待时间索引'}</strong>
                <span>{state.block === 1 ? '首块在归一化后的骨架特征上计算。' : `本块输入时间步对应原序列间隔 ${model.mapping?.jump ?? '—'}；骨架显示定位中心，深层输入为聚合特征。`}</span>
                <div className="st-sequence" aria-label="50 帧预测骨架中的时间范围">{Array.from({ length: 50 }, (_, i) => <i key={i}
                    className={selectedSlot?.range && i >= selectedSlot.range[0] && i <= selectedSlot.range[1] ? 'in-range' : ''}
                    data-anchor={i === selectedSlot?.anchor || undefined} />)}</div>
            </div>
        </div><div className="st-calculations">
            <ArithmeticCard kind="spatial" calculation={model.spatial} index={spatialCursor} onSeek={seekSpatial}>
                <div className="st-neighbors">{connected.map(({ term, index }) => <Button key={index} aria-pressed={index === spatialCursor}
                    onClick={() => seekSpatial(index)}>k{term.inputIndex[0]} · {JOINT_LABELS[term.inputIndex.at(-1)]} v{term.inputIndex.at(-1)}</Button>)}</div>
            </ArithmeticCard>
            <ArithmeticCard kind="temporal" calculation={model.temporal} index={temporalCursor} onSeek={seekTemporal}>
                <label className="st-channel">输入通道 <Select aria-label="时空图时间输入通道" value={channel} onChange={event => seekTemporal(Number(event.target.value) * kernel)}>
                    {Array.from({ length: window?.channels || 0 }, (_, i) => <option key={i} value={i}>c={i}</option>)}</Select>
                    <span>{window?.kernel} 个时间位置 · 步长 {window?.stride}</span></label>
            </ArithmeticCard>
        </div></div>
        <footer className="st-footer"><Details className="st-explanation"><Summary>数据来源与坐标说明</Summary><p>骨架来自同一行人的 50 次观测经 LSTM 预测的结果；各时间列使用同一个比例缩放，空间边来自模型邻接矩阵。左图仅放大空间邻域；名称引线的端点就是原坐标。默认隐藏第三通道预测值 ≤0.2 的点及相关连线，避免不可靠头部点连成正常骨架；开启“显示原始低分点”可用空心点和虚线核对原值。显示过滤不调整模型输入或乘加结果；该分数不是经过校准的检测置信度。空间代表 t={model.spatial?.metadata?.target_indices?.[1] ?? '—'}，时间输出 t={model.temporal?.metadata?.target_indices?.[1] ?? '—'}；中间经过 BN / ReLU。{record?.active && '记录重算使用保存的骨架及原权重，舍入可能造成与原分析的数值差异。'}</p></Details>
            <Button aria-pressed={state.frozen} onClick={state.frozen ? () => actions.setGraphAuto(true) : actions.freeze}>{state.frozen ? '跟随最新' : '冻结当前数据'}</Button>
            <label>演示速度 <Select aria-label="时空图演示速度" value={state.speed} onChange={event => actions.setSpeed(Number(event.target.value))}>
                <option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></Select></label>
            <Button onClick={() => actions.selectStep(6)}>查看意图输出 →</Button></footer>
    </section>;
}
