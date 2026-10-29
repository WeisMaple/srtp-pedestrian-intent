import { useEffect, useRef, useState } from 'react';
import { ACTIVE_STATUSES, formatTime, jobUrl } from '../api';
import Icon from './Icon';

export default function VideoPanel({ job, seekTime, onInspectTime }) {
    const [mode, setMode] = useState('source');
    const [playing, setPlaying] = useState(false);
    const [time, setTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [mediaError, setMediaError] = useState(false);
    const video = useRef(null);
    const frame = useRef(null);
    const busy = ACTIVE_STATUSES.has(job?.status);
    const completed = job?.status === 'completed';
    const resultVideo = mode === 'result' && completed && job.outputs?.video;
    const still = mode === 'result' && !resultVideo && job?.preview_version > 0;
    const playable = Boolean(job && (mode === 'source' || resultVideo));
    const source = job ? resultVideo || jobUrl(job.id, '/files/source') : '';
    const poster = job ? resultVideo && job.preview_version > 0 ? jobUrl(job.id, `/preview?v=${job.preview_version}`) : jobUrl(job.id, '/files/poster') : undefined;
    const elapsed = still ? (job.frame || 0) / (job.metadata?.fps || 1) : time;
    const total = duration || job?.metadata?.duration || 0;
    const currentFrame = still ? (job.frame || 0) + 1 : Math.min(job?.metadata?.frame_count || Infinity, Math.floor(time * (job?.metadata?.fps || 0)) + (job ? 1 : 0));

    useEffect(() => { setMode('source'); setTime(0); setDuration(0); setMediaError(false); }, [job?.id]);
    useEffect(() => { if (job?.status === 'running' || job?.status === 'completed') setMode('result'); }, [job?.status, job?.id]);
    useEffect(() => { setPlaying(false); setTime(0); setMediaError(false); }, [source, mode]);
    useEffect(() => {
        if (!Number.isFinite(seekTime) || !video.current) return;
        video.current.pause();
        video.current.currentTime = seekTime;
        setTime(seekTime);
    }, [seekTime, source, mode]);

    const togglePlayback = () => {
        if (!video.current || !playable) return;
        if (video.current.paused) video.current.play().catch(() => setMediaError(true));
        else { video.current.pause(); onInspectTime?.(video.current.currentTime); }
    };

    return <section className="panel video-panel" aria-label="视频预览">
        <div className="video-tabs" role="tablist" aria-label="视频模式">
            <button id="tab-source" role="tab" aria-selected={mode === 'source'} className={mode === 'source' ? 'active' : ''} onClick={() => setMode('source')}>原始视频</button>
            <button id="tab-result" role="tab" aria-selected={mode === 'result'} className={mode === 'result' ? 'active' : ''} onClick={() => setMode('result')}>分析结果<span>（骨架与预测）</span></button>
            {busy && <span className="live-label"><i />实时预览</span>}
        </div>
        <div className="video-stage" ref={frame}>
            {playable ? <video ref={video} src={source} poster={poster} playsInline preload="metadata" onLoadedMetadata={(event) => { setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0); if (Number.isFinite(seekTime)) { event.currentTarget.currentTime = seekTime; setTime(seekTime); } }} onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setMediaError(true)} onClick={togglePlayback} /> : still ? <img src={jobUrl(job.id, `/preview?v=${job.preview_version}`)} alt="当前处理帧，包含真实骨架、行人 ID 与已有预测类别" /> : <div className="video-empty">
                <div className="empty-frame"><Icon name={mode === 'result' ? 'person' : 'video'} size={40} /><i /><i /><i /><i /></div>
                <strong>{mode === 'result' && job ? '等待第一帧分析结果' : '让视频中的每一步，都有迹可循'}</strong>
                <span>{mode === 'result' && job ? '开始分析后，骨架、行人 ID 与预测将在这里呈现。' : '选择本地视频，查看骨架与行人行为倾向。'}</span>
            </div>}
            {still && <div className="preview-badge"><i className={busy ? 'pulsing' : ''} />{busy ? '分析中' : '已处理帧'}<span>第 {currentFrame} 帧</span></div>}
            {mediaError && <div className="media-error"><Icon name="info" size={18} /><span>浏览器暂时无法播放此视频。{mode === 'source' ? '仍可开始分析，并查看处理后的兼容视频。' : '请下载标注视频查看。'}</span></div>}
        </div>
        <div className="video-controls">
            <button className="icon-button playback-toggle" aria-label={playing ? '暂停视频' : '播放视频'} disabled={!playable || mediaError} onClick={togglePlayback}><Icon name={playing ? 'pause' : 'play'} size={19} /></button>
            <span className="time-code">{formatTime(elapsed)} <span>/ {job ? formatTime(total) : '--:--'}</span></span>
            <input aria-label="视频播放进度" type="range" min="0" max={total || 1} step="0.01" value={Math.min(elapsed, total || 1)} disabled={!playable || mediaError} onChange={(event) => { const value = Number(event.target.value); if (video.current) video.current.currentTime = value; setTime(value); onInspectTime?.(value); }} style={{ '--fill': `${total ? Math.min(100, elapsed / total * 100) : 0}%` }} />
            <span className="frame-code">当前帧 <strong>{job ? currentFrame : '—'}</strong><span> / {job?.metadata?.frame_count || '—'}</span></span>
            <button className="icon-button fullscreen-button" aria-label="全屏预览" disabled={!job} onClick={() => frame.current?.requestFullscreen?.().catch(() => {})}><Icon name="expand" size={19} /></button>
        </div>
        {job && <div className="analysis-progress" aria-label="分析进度"><div style={{ transform: `scaleX(${job.progress || 0})` }} /></div>}
    </section>;
}
