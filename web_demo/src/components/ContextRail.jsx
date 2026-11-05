import { Button } from './ui/Controls';
import VideoPlayer from './ui/VideoPlayer';
import { useEffect, useRef } from 'react';
import { CLASS_LABELS, formatTime, jobUrl } from '../api';
import { COCO_EDGES, asPose, poseBounds, validPoint } from '../domain/pose';
import { historyPoses, trackIsWarm } from '../domain/tracks';

function MiniSkeleton({ pose }) {
    const points = asPose(pose);
    if (!points) return <div className="rail-skeleton empty">暂无真实骨架</div>;
    const bounds = poseBounds([points]);
    if (!bounds) return <div className="rail-skeleton empty">姿态置信度不足</div>;
    return <svg className="rail-skeleton" viewBox={`${bounds.minX - 16} ${bounds.minY - 16} ${bounds.maxX - bounds.minX + 32} ${bounds.maxY - bounds.minY + 32}`} aria-label="所选行人骨架">
        {COCO_EDGES.filter(([a,b]) => validPoint(points[a]) && validPoint(points[b])).map(([a, b], index) => <line key={index} x1={points[a][0]} y1={points[a][1]}
            x2={points[b][0]} y2={points[b][1]} stroke="currentColor" strokeWidth="1.6"
            vectorEffect="non-scaling-stroke" />)}
        {points.map((point, index) => validPoint(point) && <circle key={index} cx={point[0]} cy={point[1]} r="3" fill="currentColor" />)}
    </svg>;
}

function HistoricalFrame({ job, frame, onFrameReady }) {
    const video = useRef(null), target = useRef(frame); target.current = frame;
    const fps = job.metadata?.fps || 30;
    const seek = () => {
        const element = video.current;
        if (!element || element.readyState < 1) return;
        if (Math.abs(element.currentTime - target.current / fps) < 0.25 / fps && element.readyState >= 2)
            onFrameReady?.(target.current);
        else if (!element.seeking) element.currentTime = target.current / fps;
    };
    useEffect(seek, [frame, fps, onFrameReady]);
    return <video ref={video} src={jobUrl(job.id, '/files/source')} muted playsInline preload="auto"
        onLoadedMetadata={seek} onLoadedData={seek} onSeeked={seek} aria-label={`历史源帧 ${frame}`} />;
}

export default function ContextRail({ job, live, tracks, track, onSelect, recorded = false, onFrameReady }) {
    const history = historyPoses(track);
    const warm = trackIsWarm(track);
    return <aside className="context-rail">
        <div className="rail-heading"><span>01 / VIDEO CONTEXT</span><b>视频与行人</b></div>
        <div className="rail-video">
            {recorded ? job && live?.frame >= 0 ? <HistoricalFrame key={job.id} job={job} frame={live.frame} onFrameReady={onFrameReady} /> : <div className="rail-empty">正在读取历史帧</div>
                : job?.preview_version > 0 ? <img src={jobUrl(job.id, `/preview?v=${job.preview_version}`)} alt="最近处理帧：骨架与行人 ID" />
                : job ? <VideoPlayer src={jobUrl(job.id, '/files/source')} poster={jobUrl(job.id, '/files/poster')}
                    muted label="源视频" /> : <div className="rail-empty">等待视频</div>}
            <div className="rail-meta mono">{job ? `${job.metadata?.frame_count || '—'} 帧 · ${formatTime(job.metadata?.duration)}` : '—'}</div>
            {live?.frame >= 0 && <div className="rail-meta mono">{recorded ? '历史记录帧' : '数值采样帧'} #{live.frame} · {formatTime(live.time_seconds)}</div>}
            {recorded ? <small className="rail-note">当前展示所选历史帧；下方分数是原分析记录。</small>
                : job?.preview_version > 0 && <small className="rail-note">图像为最近处理帧，数值按限流采样帧展示。</small>}
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
        <div className="rail-people">{tracks.map(person => <Button key={person.track_id}
            aria-pressed={person.track_id === track?.track_id} onClick={() => onSelect(person.track_id)}>
            ID {String(person.track_id).padStart(2, '0')}
        </Button>)}</div>
        <p className="rail-note">ID 为分析时 ByteTrack 的真实归属；匹配代价未记录，因此不显示虚构分数。</p>
    </aside>;
}
