import { Button, FilePicker } from './ui/Controls';
import VideoPlayer from './ui/VideoPlayer';
import { useState } from 'react';
import { formatTime, jobUrl } from '../api';
import { formatStatus, uploadHint } from '../domain/workbench';
import { visibleTracks } from '../domain/process';

export default function Workbench({ job, live, health, error, pending, onUpload, onStart, onCancel, onOpenCorridor, onOpenResults }) {
    const [dragging, setDragging] = useState(false);
    const busy = ['loading', 'running', 'encoding'].includes(job?.status);
    const ready = Boolean(health?.models_ready && health?.encoder_ready);
    const tracks = visibleTracks(live, job);
    const choose = file => { if (file && !busy) onUpload(file); };
    return <section className="workbench">
        <div className="workbench-copy">
            <span className="mono workbench-kicker">DEEP LEARNING COURSE PROJECT / 2026</span>
            <h1>看见一次预测的发生</h1>
            <p>把一个真实行人的骨架送进模型，沿着六步长廊看清检测、绑定、预测与意图。</p>
            <div className="workbench-actions">
                <FilePicker accept=".mp4,.mov,.avi" label="选择本地视频文件" onFile={choose} className="blueprint-button primary"
                    disabled={busy || !ready || Boolean(pending)}>{pending === 'upload' ? '正在上传…' : '选择视频'}</FilePicker>
                {job?.status === 'ready' && <Button className="blueprint-button" onClick={onStart} disabled={!ready || Boolean(pending)}>开始分析</Button>}
                {busy && <Button className="blueprint-button" onClick={onCancel} disabled={Boolean(pending)}>停止分析</Button>}
                {job && <Button className="blueprint-button" onClick={onOpenCorridor}>查看计算过程 →</Button>}
                {job?.status === 'completed' && <Button className="blueprint-button" onClick={onOpenResults}>查看结果</Button>}
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
                <span className="mono">已处理 {job.processed_frames || 0} / {job.metadata?.frame_count || '—'} 帧 · 当前 {tracks.length} 位行人</span>
                <div className="job-progress"><i style={{ transform: `scaleX(${job.progress || 0})` }} /></div>
                {job.logs?.length > 0 && <p className="workbench-log">{job.logs.at(-1).message}</p>}
            </div>}
        </div>
        <div className="workbench-light-table blueprint-panel">
            {job?.preview_version > 0 ? <img src={jobUrl(job.id, `/preview?v=${job.preview_version}`)} alt="最近处理帧的骨架、行人 ID 与预测标注" />
                : job ? <VideoPlayer src={jobUrl(job.id, '/files/source')} poster={jobUrl(job.id, '/files/poster')}
                    muted label="源视频" /> : <div className="light-table-empty">
                <span className="mono">LIGHT TABLE</span><b>源视频将在这里展开</b>
                <small>真实骨架、ID 与分数不会在这里伪造</small>
            </div>}
            <div className="light-table-caption mono">{job
                ? `${job.preview_version > 0 ? '最近处理帧 · ' : '源视频 · '}${job.metadata?.width || '—'}×${job.metadata?.height || '—'} · ${job.metadata?.fps || '—'} fps`
                : '等待视频'}</div>
        </div>
        <div className="workbench-steps">
            {['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图'].map((label, index) =>
                <Button type="button" key={label} onClick={() => onOpenCorridor(index + 1)}><span className="mono">{String(index + 1).padStart(2, '0')}</span><b>{label}</b><i /></Button>)}
        </div>
    </section>;
}
