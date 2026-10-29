import { CLASS_LABELS, trackLabel } from '../api';
import Icon from './Icon';
import Skeleton from './Skeleton';

export function ScoreBars({ probabilities, dark = false }) {
    return <div className={`score-bars ${dark ? 'dark' : ''}`}>{CLASS_LABELS.map((label, index) => {
        const score = Array.isArray(probabilities) && Number.isFinite(probabilities[index]) ? probabilities[index] : null;
        return <div className={`score-row class-${index}`} key={label}><span>{label}</span><div className="score-track"><i style={{ transform: `scaleX(${score === null ? 0 : Math.max(0, Math.min(1, score))})` }} /></div><small>{score === null ? '—' : score.toFixed(2)}</small></div>;
    })}</div>;
}

export default function TrackPanel({ job, selectedId, onSelect }) {
    const tracks = job?.tracks || [];
    const logs = (job?.logs || []).slice(-3).reverse();
    return <aside className="panel track-panel">
        <div className="section-heading"><h2>{job?.status === 'completed' ? '末帧行人' : '当前行人'}</h2>{tracks.length > 0 && <span>{tracks.length} 位</span>}</div>
        <div className="track-list">
            {tracks.length ? tracks.map((track) => <button key={track.track_id} className={`track-card ${String(track.track_id) === String(selectedId) ? 'selected' : ''}`} onClick={() => onSelect(track.track_id)} aria-pressed={String(track.track_id) === String(selectedId)}>
                <div className="track-skeleton"><Skeleton pose={track.keypoints} color={track.class_id === 1 ? '#d6a343' : '#289a7f'} /></div>
                <div className="track-info"><div className="track-title"><strong>ID {String(track.track_id).padStart(2, '0')}</strong><span className={`class-tag class-${track.class_id ?? 'pending'}`}>{trackLabel(track)}</span></div><small className="score-label">{track.probabilities ? '模型分数' : `有效观测 ${track.history_length || 0} / 50`}</small><ScoreBars probabilities={track.probabilities} /></div>
            </button>) : <div className="tracks-empty"><Icon name="person" size={30} /><strong>{job?.status === 'running' ? '正在寻找行人' : job && job.status !== 'ready' ? '当前帧没有可用行人' : '等待行人进入画面'}</strong><p>{job?.status === 'running' ? '检测到行人后，将显示骨架与跟踪 ID。' : '分析开始后，在这里查看跟踪状态与模型分数。'}</p></div>}
        </div>
        <div className="progress-logs"><h2>处理进度日志</h2>{logs.length ? <ol aria-live="polite">{logs.map((log, index) => <li key={`${log.time}-${index}`}><span className="log-check"><Icon name="check" size={11} /></span><span title={log.message}>{log.message}</span><time>{log.time}</time></li>)}</ol> : <p>实际处理步骤将在这里同步更新。</p>}</div>
    </aside>;
}
