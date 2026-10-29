import { useRef, useState } from 'react';
import { ACTIVE_STATUSES, formatBytes, formatTime, STATUS_LABELS } from '../api';
import Icon from './Icon';

export default function InputPanel({ job, health, pending, upload, start, cancel }) {
    const input = useRef(null);
    const [dragging, setDragging] = useState(false);
    const [saveOpen, setSaveOpen] = useState(false);
    const busy = ACTIVE_STATUSES.has(job?.status);
    const disabled = busy || Boolean(pending);
    const metadata = job?.metadata;
    const canStart = job?.status === 'ready' && health?.models_ready && health?.encoder_ready && !pending;

    function onDrop(event) {
        event.preventDefault();
        setDragging(false);
        if (!disabled) upload(event.dataTransfer.files?.[0]);
    }

    return <aside className="panel input-panel">
        <h2>输入视频</h2>
        <input ref={input} className="visually-hidden" type="file" accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo" aria-label="选择本地视频" disabled={disabled} onChange={(event) => { upload(event.target.files?.[0]); event.target.value = ''; }} />
        <button className={`upload-zone ${dragging ? 'dragging' : ''}`} disabled={disabled} onClick={() => input.current?.click()} onDragOver={(event) => { event.preventDefault(); if (!disabled) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <Icon name="upload" size={38} />
            <strong>{pending === 'upload' ? '正在上传视频…' : '选择视频'}</strong>
            <span>拖放文件，或点击选择</span>
            <small>支持 MP4、MOV、AVI</small>
        </button>
        {job ? <div className="input-file">
            <div className="file-icon"><Icon name="video" size={20} /></div>
            <div className="file-details"><strong title={job.filename}>{job.filename}</strong><small>{metadata?.width} × {metadata?.height}<span> {formatTime(metadata?.duration)} </span>{formatBytes(metadata?.bytes)}</small></div>
        </div> : <p className="input-hint">选择一段包含行人的视频，开始本地分析。</p>}

        <div className="input-actions">
            {busy ? <button className="button primary" onClick={cancel} disabled={Boolean(pending)}><Icon name="stop" size={19} />{pending === 'cancel' ? '正在请求停止…' : '停止分析'}</button> : <button className="button primary" onClick={start} disabled={!canStart}><Icon name="play" size={21} />{pending === 'start' ? '正在启动…' : '开始分析'}</button>}
            <div className="save-control" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setSaveOpen(false); }}>
                <button className="button secondary" disabled={job?.status !== 'completed'} onClick={() => setSaveOpen(!saveOpen)} aria-expanded={saveOpen} aria-haspopup="menu"><Icon name="download" size={20} />保存结果<Icon name="chevron" size={16} /></button>
                {saveOpen && <div className="save-menu" role="menu">
                    {[["video", "标注视频", "MP4"], ["data", "分析数据", "JSONL"], ["bundle", "完整结果", "ZIP"]].map(([key, label, extension]) => job.outputs?.[key] ? <a key={key} role="menuitem" href={job.outputs[key]} download onClick={() => setSaveOpen(false)}><span>{label}</span><small>{extension}</small></a> : null)}
                </div>}
            </div>
        </div>
        {job && <div className={`job-status ${job.status}`} role="status"><i className={busy ? 'pulsing' : ''} />{STATUS_LABELS[job.status] || job.status}{busy && <span>{Math.round((job.progress || 0) * 100)}%</span>}</div>}
        {job?.error && <p className="inline-error">{job.error}</p>}
        {health && !health.unavailable && (!health.models_ready || !health.encoder_ready) && <p className="environment-note">{health.detail || '请检查本地模型文件和 ffmpeg 编码器。'}</p>}
        <div className="output-note"><span>输出内容包含</span><p>骨架 / 行人 ID / 预测类别</p><small>模型在本地运行，视频保存在本机。</small></div>
    </aside>;
}
