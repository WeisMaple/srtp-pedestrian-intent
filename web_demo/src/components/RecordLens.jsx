import { useEffect, useMemo, useState } from 'react';
import { request, yoloConvUrl } from '../api';
import { flowState, normalizeCalculation } from '../domain/calculation';
import { filterTerms, formatValue } from '../domain/ledger';
import NumericLedger from './NumericLedger';

/** On-demand record lens for a completed job's YOLO representative convolution. */
export default function RecordLens({ job, structure, onClose }) {
    const modules = structure?.yoloModules || [];
    const frameCount = Number(job?.metadata?.frame_count) || 0;
    const enabled = job?.status === 'completed' && frameCount > 0 && modules.length > 0;
    const [frame, setFrame] = useState(0);
    const [module, setModule] = useState(modules[0]?.index ?? 0);
    const [revision, setRevision] = useState(0);
    const [row, setRow] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [filter, setFilter] = useState('all');
    const [page, setPage] = useState(0);
    const [cursor, setCursor] = useState(0);
    useEffect(() => {
        if (modules.length && !modules.some(item => item.index === module)) setModule(modules[0].index);
    }, [modules, module]);
    useEffect(() => {
        if (!enabled) { setRow(null); return undefined; }
        let disposed = false;
        let timer = null;
        const controller = new AbortController();
        const load = async (attempt = 0) => {
            if (disposed) return;
            let retrying = false;
            setLoading(true); setError(''); setRow(null);
            try {
                const result = await request(yoloConvUrl(job.id, frame, module), { signal: controller.signal });
                if (!disposed) { setRow(result); setCursor(0); setPage(0); }
            } catch (cause) {
                if (disposed || cause.name === 'AbortError') return;
                // Aborting a fetch cannot cancel the server's CPU replay. A
                // quick selector change can therefore meet the previous replay
                // as a 409; wait for admission and retry instead of surfacing it.
                if (cause.status === 409 && attempt < 8) {
                    retrying = true;
                    timer = window.setTimeout(() => load(attempt + 1), Math.min(2000, 400 * (attempt + 1)));
                    return;
                }
                setError(cause.message || '记录重算失败。');
            } finally {
                if (!disposed && !retrying) setLoading(false);
            }
        };
        load();
        return () => { disposed = true; if (timer !== null) window.clearTimeout(timer); controller.abort(); };
    }, [enabled, job?.id, frame, module, revision]);
    const calculation = useMemo(() => normalizeCalculation(row), [row]);
    const terms = useMemo(() => calculation ? filterTerms(calculation.terms, filter) : [], [calculation, filter]);
    const flow = calculation ? flowState(calculation, cursor) : null;
    if (!enabled) return <aside className="record-lens blueprint-panel" aria-label="记录重算">
        <header><div><span className="mono">RECORD / 记录重算</span><strong>任务完成后可用</strong></div>
            <button onClick={onClose} aria-label="关闭记录重算">×</button></header>
        <p className="record-note">记录重算只在分析完成后读取源帧；运行中仍使用实时前向数据。</p>
    </aside>;
    return <aside className="record-lens blueprint-panel" aria-label="记录重算">
        <header><div><span className="mono">RECORD / 记录重算</span>
            <strong>YOLO 模块 {module} · 源帧 {frame}</strong></div>
            <button onClick={onClose} aria-label="关闭记录重算">×</button></header>
        <div className="record-controls">
            <label><span className="mono">源帧</span>
                <input type="number" min="0" max={Math.max(0, frameCount - 1)} value={frame}
                    aria-label="记录源帧"
                    onChange={event => setFrame(Math.max(0, Math.min(frameCount - 1, Number(event.target.value) || 0)))} />
            </label>
            <label><span className="mono">YOLO 模块</span>
                <select aria-label="记录 YOLO 模块" value={module}
                    onChange={event => setModule(Number(event.target.value))}>
                    {modules.map(item => <option key={item.index} value={item.index}>
                        #{item.index} · {item.type || '模块'}
                    </option>)}
                </select>
            </label>
            <button className="blueprint-button" onClick={() => setRevision(value => value + 1)}
                disabled={loading}>{loading ? '重算中…' : '重新计算'}</button>
        </div>
        <p className="record-note mono">仅已完成任务 · 保留独立 Conv / BatchNorm / SiLU · 与融合推理可能存在浮点舍入差异。</p>
        {error && <p className="record-error" role="alert">{error}</p>}
        {calculation && <div className="lens-summary mono">
            <span>实际输出 <b>{formatValue(calculation.actual)}</b></span>
            <span>重建 <b>{formatValue(calculation.reconstructed)}</b></span>
            <span>部分和 <b>{formatValue(flow?.partialSum)}</b></span>
            {calculation.absoluteError != null && <span>误差 <b>{formatValue(calculation.absoluteError)}</b></span>}
        </div>}
        {calculation ? <NumericLedger calculation={calculation} state={flow} terms={terms} filter={filter}
            page={page} onFilter={value => { setFilter(value); setPage(0); }} onPage={setPage}
            onSeek={index => setCursor(index)} /> : <div className="ledger-empty">
            {loading ? '正在重算源帧…' : '选择源帧与模块以读取记录重算。'}
        </div>}
    </aside>;
}
