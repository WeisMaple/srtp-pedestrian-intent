import { useRef, useState } from 'react';
import { formatTime, jobUrl } from '../api';
import { formatStatus, uploadHint } from '../domain/workbench';

export default function Workbench({ job, health, error, pending, onUpload, onStart, onCancel, onOpenCorridor, onOpenResults }) {
    const input = useRef(null);
    const [dragging, setDragging] = useState(false);
    const busy = ['loading', 'running', 'encoding'].includes(job?.status);
    const ready = Boolean(health?.models_ready && health?.encoder_ready);
    const choose = file => { if (file && !busy) onUpload(file); };
    return <section className="workbench">
        <div className="workbench-copy">
            <span className="mono workbench-kicker">DEEP LEARNING COURSE PROJECT / 2026</span>
            <h1>看见一次预测的发生</h1>
            <p>把一个真实行人的骨架送进模型，沿着六步长廊看清检测、绑定、预测与意图。</p>
            <div className="workbench-actions">
                <input ref={input} type="file" accept=".mp4,.mov,.avi" hidden
                    onChange={event => choose(event.target.files?.[0])} />
                <button className="blueprint-button primary" onClick={() => input.current?.click()}
                    disabled={busy || !ready}>选择视频</button>
                {job?.status === 'ready' && <button className="blueprint-button" onClick={onStart} disabled={!ready}>开始分析</button>}
                {busy && <button className="blueprint-button" onClick={onCancel}>停止分析</button>}
                {job && <button className="blueprint-button" onClick={onOpenCorridor}>进入长廊</button>}
                {job?.status === 'completed' && <button className="blueprint-button" onClick={onOpenResults}>查看结果</button>}
            </div>
            <div className={`drop-zone ${dragging ? 'dragging' : ''}`}
                onDragOver={event => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files?.[0]); }}>
                <strong>{uploadHint(job ? { name: job.filename } : null, ready)}</strong>
                <span className="mono">{job ? `${job.metadata?.frame_count || '—'} 帧 · ${formatTime(job.metadata?.duration)}` : 'MP4 / MOV / AVI'}</span>
            </div>
            {error && <p className="workbench-error" role="alert">{error}</p>}
            {job && <div className="workbench-job">
                <span className="mono">{job.id.slice(0, 8)}</span>
                <strong>{formatStatus(job)}</strong>
                <div className="job-progress"><i style={{ transform: `scaleX(${job.progress || 0})` }} /></div>
            </div>}
        </div>
        <div className="workbench-light-table blueprint-panel">
            {job ? <video src={jobUrl(job.id, '/files/source')} poster={jobUrl(job.id, '/files/poster')}
                muted playsInline preload="metadata" /> : <div className="light-table-empty">
                <span className="mono">LIGHT TABLE</span><b>源视频将在这里展开</b>
                <small>真实骨架、ID 与分数不会在这里伪造</small>
            </div>}
            <div className="light-table-caption mono">{job
                ? `${job.metadata?.width || '—'}×${job.metadata?.height || '—'} · ${job.metadata?.fps || '—'} fps`
                : '等待视频'}</div>
        </div>
        <div className="workbench-steps">
            {['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图'].map((label, index) =>
                <div key={label}><span className="mono">{String(index + 1).padStart(2, '0')}</span><b>{label}</b><i /></div>)}
        </div>
    </section>;
}
