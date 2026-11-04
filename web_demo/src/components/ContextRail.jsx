import { CLASS_LABELS, formatTime, jobUrl } from '../api';
import { COCO_EDGES, asPose } from '../domain/pose';
import { historyPoses, trackIsWarm } from '../domain/tracks';

function MiniSkeleton({ pose }) {
    const points = asPose(pose);
    if (!points) return <div className="rail-skeleton empty">暂无真实骨架</div>;
    return <svg className="rail-skeleton" viewBox="0 0 100 120" aria-label="所选行人骨架">
        {COCO_EDGES.map(([a, b], index) => <line key={index} x1={points[a][0] / 12} y1={points[a][1] / 12}
            x2={points[b][0] / 12} y2={points[b][1] / 12} stroke="currentColor" strokeWidth="1.6"
            vectorEffect="non-scaling-stroke" />)}
        {points.map((point, index) => <circle key={index} cx={point[0] / 12} cy={point[1] / 12} r="1.5" fill="currentColor" />)}
    </svg>;
}

export default function ContextRail({ job, tracks, track, onSelect }) {
    const history = historyPoses(track);
    const warm = trackIsWarm(track);
    return <aside className="context-rail">
        <div className="rail-heading"><span>01 / VIDEO CONTEXT</span><b>视频与行人</b></div>
        <div className="rail-video">
            {job ? <video src={jobUrl(job.id, '/files/source')} poster={jobUrl(job.id, '/files/poster')}
                muted playsInline preload="metadata" /> : <div className="rail-empty">等待视频</div>}
            <div className="rail-meta mono">{job ? `${job.metadata?.frame_count || '—'} 帧 · ${formatTime(job.metadata?.duration)}` : '—'}</div>
        </div>
        <div className="rail-person">
            <MiniSkeleton pose={track?.keypoints} />
            <div>
                <span className="rail-id mono">{track ? `ID ${String(track.track_id).padStart(2, '0')}` : '无行人'}</span>
                <strong>{track ? (track.class_id != null ? CLASS_LABELS[track.class_id] : warm ? '等待预测' : '积累历史中') : '等待检测'}</strong>
                <small className="mono">历史 {track?.history_length || history.length} / 50 · 17 关节</small>
            </div>
        </div>
        <div className="rail-scores">
            {(track?.probabilities || [null, null, null]).map((value, index) => <div key={index}>
                <span>{CLASS_LABELS[index]}</span><i><em style={{ width: `${Math.max(0, value || 0) * 100}%` }} /></i>
                <b className="mono">{typeof value === 'number' ? value.toFixed(3) : '—'}</b>
            </div>)}
        </div>
        <div className="rail-people">{tracks.map(person => <button key={person.track_id}
            aria-pressed={person.track_id === track?.track_id} onClick={() => onSelect(person.track_id)}>
            ID {String(person.track_id).padStart(2, '0')}
        </button>)}</div>
        <p className="rail-note">ID 为分析时 ByteTrack 的真实归属；匹配代价未记录，因此不显示虚构分数。</p>
    </aside>;
}
