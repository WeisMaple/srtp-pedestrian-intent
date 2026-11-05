import { Button, Progress } from './ui/Controls';
import { useEffect, useRef, useState } from 'react';
import { jobUrl } from '../api';
import { asPose, COCO_EDGES, validPoint } from '../domain/pose';

const PHASES = ['源视频帧', 'YOLO 检测框', 'COCO17 骨架', 'ByteTrack ID 绑定'];

export default function PoseBindingStage({ job, live, tracks, track, step, beat, onSelect, onInteract, onFrameReady }) {
    const video = useRef(null);
    const [override, setOverride] = useState(null);
    const [readyFrame, setReadyFrame] = useState(null);
    const [failed, setFailed] = useState(false);
    const frame = live?.frame >= 0 ? live.frame : null;
    const currentFrame = useRef(frame); currentFrame.current = frame;
    const width = job?.metadata?.width || 1280, height = job?.metadata?.height || 720;
    const fps = job?.metadata?.fps || 30;
    const phase = override ?? (step === 1 ? 0 : step === 3 ? 3 : Math.min(3, beat + 1));
    useEffect(() => { setOverride(null); }, [step]);
    const seek = () => {
        const element = video.current;
        if (!element || element.readyState < 1 || currentFrame.current == null) return;
        const time = currentFrame.current / fps;
        if (Math.abs(element.currentTime - time) < 0.25 / fps && element.readyState >= 2) {
            setReadyFrame(currentFrame.current);
            onFrameReady?.(currentFrame.current);
        } else if (!element.seeking) element.currentTime = time;
    };
    useEffect(() => { seek(); }, [frame, fps, job?.id, onFrameReady]);
    const aligned = frame != null && readyFrame === frame && !failed;
    return <section className="pose-binding" aria-label="骨架与身份绑定过程">
        <div className="binding-phases" role="group" aria-label="绑定讲解步骤">
            {PHASES.map((title, index) => <Button type="button" key={title} aria-pressed={phase === index}
                onClick={() => { onInteract(); setOverride(index); }}><span>0{index + 1}</span> {title}</Button>)}
        </div>
        <div className="pose-layout">
            <div className="pose-picture">
                {job ? <video key={job.id} ref={video} src={jobUrl(job.id, '/files/source')} muted playsInline preload="auto"
                    onLoadedMetadata={seek} onLoadedData={seek} onSeeked={seek} onError={() => setFailed(true)} />
                    : <p>选择视频并开始分析后，在这里查看同一帧的检测与绑定。</p>}
                {aligned && phase > 0 && <svg viewBox={`0 0 ${width} ${height}`} aria-label={`采样帧 ${frame} 的检测与骨架`}>
                    <defs><clipPath id="binding-frame-clip"><rect width={width} height={height} /></clipPath></defs>
                    <g clipPath="url(#binding-frame-clip)">{tracks.map((person, index) => {
                        const points = asPose(person.keypoints);
                        const selected = String(person.track_id) === String(track?.track_id);
                        const box = person.bbox;
                        if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return null;
                        return <g key={person.track_id} className={selected ? 'selected-person' : 'other-person'}
                            role="button" tabIndex="0" aria-label={`选择行人 ID ${person.track_id}`}
                            onClick={() => { onInteract(); onSelect(person.track_id); }}
                            onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onInteract(); onSelect(person.track_id); } }}>
                            <rect x={box[0]} y={box[1]} width={Math.max(0, box[2] - box[0])} height={Math.max(0, box[3] - box[1])}
                                fill="transparent" stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                            {phase >= 2 && points && <g className="binding-skeleton">
                                {COCO_EDGES.filter(([a,b]) => validPoint(points[a]) && validPoint(points[b])).map(([a,b]) => <line key={`${a}-${b}`}
                                    x1={points[a][0]} y1={points[a][1]} x2={points[b][0]} y2={points[b][1]} stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />)}
                                {points.map((p, i) => validPoint(p) && <circle key={i} cx={p[0]} cy={p[1]} r="4" fill="currentColor" />)}
                            </g>}
                            <text x={Math.max(0, Math.min(width - 150, box[0]))} y={Math.max(25, box[1] - 12)} fill="currentColor" fontSize="24"
                                paintOrder="stroke" stroke="black" strokeWidth="3">{phase === 3 ? `ID ${person.track_id}` : `候选 ${index + 1}`}</text>
                        </g>;
                    })}</g>
                </svg>}
                {job && !aligned && <div className="binding-loading" role="status">{failed ? '源视频无法解码，暂不叠加骨架' : frame == null ? '等待真实采样帧' : `正在对齐帧 #${frame}…`}</div>}
                <span className="binding-frame-label mono">{frame == null ? '尚无采样帧' : `源帧 #${frame} · ${(frame / fps).toFixed(2)} s`}</span>
            </div>
            <aside className="binding-associations">
                <h3>{PHASES[phase]}</h3>
                <p>{phase === 0 ? '原始画面与数值采样帧对齐。' : phase === 1 ? '每个框对应本帧采集到的行人候选。' : phase === 2
                    ? '在检测框内定位 17 个关节，连线组成该行人的骨架。低置信度关节不绘制。' : '按记录中的身份归属，将检测骨架写入对应 ID 的历史序列。'}</p>
                {tracks.map((person, index) => <Button type="button" key={person.track_id}
                    aria-pressed={String(person.track_id) === String(track?.track_id)}
                    onClick={() => { onInteract(); onSelect(person.track_id); }}>
                    <span>候选 {index + 1}{phase >= 2 ? ' · 17 关节' : ''}</span>
                    <strong>{phase === 3 ? `→ ByteTrack ID ${person.track_id}` : typeof person.detection_confidence === 'number' ? `检测置信度 ${(person.detection_confidence * 100).toFixed(1)}%` : '置信度未采集'}</strong>
                    {phase === 3 && <><small>{person.pose_accepted === false ? '本次姿态不足，未加入历史' : `有效历史 ${person.history_length ?? 0} / 50`}</small>
                        <Progress aria-label={`ID ${person.track_id} 有效历史`} value={person.history_length || 0} max="50" /></>}
                </Button>)}
                {!tracks.length && <p>当前没有已采集的行人绑定。</p>}
            </aside>
        </div>
        <p className="binding-footnote">同一采样帧的记录讲解 · 候选编号仅用于本帧展示；身份来自 ByteTrack 实际结果，匹配代价与候选匹配搜索未采集。</p>
    </section>;
}
