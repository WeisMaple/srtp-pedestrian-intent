import { useEffect, useMemo, useState } from 'react';
import { Button, Select } from './ui/Controls';
import { JOINT_LABELS, nearestJoint, skeletonLayout } from '../domain/spatiotemporal';
import { validPoint } from '../domain/pose';

const SCALE = 40, COLUMN = 2.15 * SCALE;
const fixed = value => Number.isFinite(value) ? value.toFixed(2) : '—';
const pointerPosition = event => {
    const svg = event.currentTarget.ownerSVGElement;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    return [point.x, point.y];
};
const keyboard = callback => event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); callback(); }
};

function SpatialFocus({ model, spatialTerm, showLowScores, onSelect }) {
    const slot = model.slots.find(item => item.spatial && item.pose);
    const target = model.spatial?.metadata?.target_indices?.[2];
    const neighbors = [...new Set(model.edges.filter(([, , joint]) => joint === target).map(([, joint]) => joint))];
    if (!slot || !neighbors.length) return null;
    const visible = neighbors.map(joint => showLowScores || validPoint(slot.pose[joint]));
    const points = neighbors.filter((_, i) => visible[i]).map(joint => slot.pose[joint]);
    if (!points.length) return <section className="st-focus"><h3>S · 空间邻域</h3><p className="st-wait">本帧邻域点的预测分数不足；原始数值仍在下方参与计算。</p></section>;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min(88 / Math.max(1, maxX - minX), 168 / Math.max(1, maxY - minY));
    const point = joint => [60 + (slot.pose[joint][0] - (minX + maxX) / 2) * scale,
        120 + (slot.pose[joint][1] - (minY + maxY) / 2) * scale];
    const selected = spatialTerm?.inputIndex?.at(-1);
    const [tx, ty] = point(target);
    return <section className="st-focus" aria-label="空间邻域局部放大">
        <h3>S · 空间邻域放大</h3><p>t={slot.time} · 聚合到 {JOINT_LABELS[target]} v{target}</p>
        <svg viewBox="0 0 250 240" aria-label="同帧空间邻域原坐标等比放大，名称引线指向真实关节点">
            {neighbors.filter((joint, i) => joint !== target && visible[i] && (showLowScores || validPoint(slot.pose[target]))).map(joint => {
                const [x, y] = point(joint);
                return <line key={joint} x1={x} y1={y} x2={tx} y2={ty} className={`st-focus-edge ${!validPoint(slot.pose[joint]) || !validPoint(slot.pose[target]) ? 'is-low-score' : ''}`} />;
            })}
            {neighbors.map((joint, index) => {
                const [x, y] = point(joint), labelY = 24 + index * 43;
                const select = () => onSelect({ time: slot.time, joint, spatial: true, temporal: slot.temporal });
                return <g key={joint} role="button" tabIndex="0" aria-label={`选择空间关节 ${JOINT_LABELS[joint]} v${joint}`}
                    aria-pressed={selected === joint} className={`st-focus-joint ${selected === joint ? 'is-selected' : ''}`}
                    onClick={event => {
                        const picked = event.target.tagName === 'circle' && event.detail
                            ? neighbors[nearestJoint(neighbors.map(point), pointerPosition(event), visible)] : joint;
                        onSelect({ time: slot.time, joint: picked, spatial: true, temporal: slot.temporal });
                    }} onKeyDown={keyboard(select)} data-joint={joint}>
                    {visible[index] && <path d={`M ${x} ${y} L 123 ${labelY} H 137`} className="st-callout-line" />}
                    <rect x="138" y={labelY - 16} width="106" height="34" rx="4" />
                    <text x="148" y={labelY + 5}>v{joint} {JOINT_LABELS[joint]}{!validPoint(slot.pose[joint]) ? ' · 低' : joint === target ? ' · 自身' : ''}</text>
                    {visible[index] && <circle cx={x} cy={y} r={selected === joint ? 6 : 4} className={slot.pose[joint][2] <= .2 ? 'is-uncertain' : ''} />}
                </g>;
            })}
        </svg>
    </section>;
}

export default function SpatiotemporalGraph({ model, spatialTerm, temporalTerm, zoom, onZoom, showLowScores, onShowLowScores, onSelect, onInteract }) {
    const [inspected, setInspected] = useState(null);
    const layout = useMemo(() => skeletonLayout(model.slots, { showLowScores }), [model.slots, showLowScores]);
    const topology = useMemo(() => [...new Map(model.edges.filter(([, a, b]) => a !== b)
        .map(([, a, b]) => [[Math.min(a, b), Math.max(a, b)].join(':'), [a, b]])).values()], [model.edges]);
    const spatialTarget = model.spatial?.metadata?.target_indices?.[2];
    const connected = new Set(model.edges.filter(([, , target]) => target === spatialTarget).map(([, joint]) => joint));
    const hasSpatial = connected.size > 0 && model.slots.some(slot => slot.spatial && slot.pose);
    const width = layout.length * COLUMN + 20, height = 238;
    const project = (slot, joint) => [width / 2 + (slot.x + slot.points[joint][0]) * SCALE, 188 - slot.points[joint][1] * SCALE];
    const selectedTime = temporalTerm?.inputIndex?.[1];
    useEffect(() => { setInspected(null); }, [selectedTime, model.source.frame]);
    const selection = inspected || { time: selectedTime, joint: model.target };
    const selectedSlot = model.slots.find(slot => slot.time === selection.time);
    const selectedPoint = selectedSlot?.pose?.[selection.joint];
    const inspect = point => { setInspected(point); onSelect(point); };
    const timeline = layout.filter(slot => slot.temporal && slot.points[model.target] && slot.visible[model.target]);
    const hiddenCount = layout.reduce((count, slot) => count + slot.visible.filter(visible => !visible).length, 0);
    return <div className="st-graph" data-render-ready="true">
        <div className={`st-graph-panels ${hasSpatial ? '' : 'is-temporal-only'}`}>
            <SpatialFocus model={model} spatialTerm={spatialTerm} showLowScores={showLowScores} onSelect={inspect} />
            <section className="st-timeline" aria-label="跨帧骨架时空图">
                <header><div><h3>T · 同一关节沿时间连接</h3><span>每列一帧 · 原坐标等比绘制</span></div>
                    <div className="st-zoom" aria-label="骨架图缩放"><Button className="st-low-score-toggle" aria-pressed={showLowScores}
                        onClick={() => { onInteract(); onShowLowScores(!showLowScores); }}>显示原始低分点</Button>{[1, 1.5, 2].map(value => <Button key={value}
                        aria-label={`骨架图放大 ${value} 倍`} aria-pressed={zoom === value} onClick={() => { onInteract(); onZoom(value); }}>{value}×</Button>)}</div></header>
                <div className="st-timeline-scroll" tabIndex="0" aria-label="骨架时间轴，可横向滚动查看">
                    <svg className="st-timeline-svg" viewBox={`0 0 ${width} ${height}`} style={{ width: `${zoom * 100}%`, minWidth: width * zoom, height: height * zoom }}
                        aria-label="时间窗口中的预测骨架；青色为同帧邻域，金色为跨帧同一关节">
                        {layout.map(slot => {
                            const x = width / 2 + slot.x * SCALE;
                            return <g key={slot.time} data-time={slot.time} className={`st-slice ${slot.spatial ? 'is-spatial' : ''} ${slot.time === selectedTime ? 'is-active' : ''}`}>
                                <rect x={x - COLUMN / 2 + 3} y="3" width={COLUMN - 6} height="221" rx="5" />
                                <text x={x} y="22" className="st-time-label">t={slot.time}{slot.spatial ? ' · S' : ''}</text>
                                {!slot.points.length ? <g className="st-padding"><rect x={x - 29} y="47" width="58" height="141" rx="5" />
                                    <text x={x} y="128">{slot.padding ? '补零' : slot.pose ? '分数不足' : '未采集'}</text></g> : topology.filter(([a, b]) => slot.visible[a] && slot.visible[b]).map(([a, b]) => {
                                    const [ax, ay] = project(slot, a), [bx, by] = project(slot, b);
                                    const relevant = slot.spatial && ((a === spatialTarget && connected.has(b)) || (b === spatialTarget && connected.has(a)));
                                    const low = !validPoint(slot.pose[a]) || !validPoint(slot.pose[b]);
                                    return <line key={`${a}:${b}`} data-source-joint={a} data-target-joint={b} x1={ax} y1={ay} x2={bx} y2={by}
                                        className={`${relevant ? 'st-spatial-edge' : 'st-bone'} ${low ? 'is-low-score' : ''}`} />;
                                })}
                                <text x={x} y="213" className="st-anchor-label">{slot.padding ? '无对应骨架' : `预测 +${slot.anchor + 1}`}</text>
                            </g>;
                        })}
                        {timeline.slice(1).map((slot, index) => {
                            const previous = timeline[index];
                            if (slot.time !== previous.time + 1) return null;
                            const [ax, ay] = project(previous, model.target), [bx, by] = project(slot, model.target);
                            return <line key={slot.time} x1={ax} y1={ay} x2={bx} y2={by} className="st-temporal-edge" />;
                        })}
                        {layout.flatMap(slot => slot.points.map((_, joint) => {
                            if (!slot.visible[joint]) return null;
                            const [x, y] = project(slot, joint);
                            const spatial = slot.spatial && connected.has(joint), temporal = slot.temporal && joint === model.target;
                            const active = (spatial && joint === spatialTerm?.inputIndex?.at(-1)) || (temporal && slot.time === selectedTime);
                            const select = () => inspect({ time: slot.time, joint, spatial: slot.spatial, temporal: slot.temporal });
                            return <g key={`${slot.time}:${joint}`} className={`st-joint ${spatial ? 'is-spatial' : temporal ? 'is-temporal' : ''}`}
                                role="button" tabIndex="0" aria-label={`检查 t=${slot.time} ${JOINT_LABELS[joint]} v${joint}`}
                                onClick={event => {
                                    const picked = event.detail ? nearestJoint(slot.points.map((_, i) => project(slot, i)), pointerPosition(event), slot.visible) : joint;
                                    inspect({ time: slot.time, joint: picked, spatial: slot.spatial, temporal: slot.temporal });
                                }} onKeyDown={keyboard(select)} data-time={slot.time} data-joint={joint}
                                data-source-x={slot.pose[joint][0]} data-source-y={slot.pose[joint][1]}>
                                <title>{`t=${slot.time} · ${JOINT_LABELS[joint]} v${joint}\nx=${fixed(slot.pose[joint][0])}, y=${fixed(slot.pose[joint][1])}`}</title>
                                <circle cx={x} cy={y} r="8" className="st-hit" />
                                {active && <circle cx={x} cy={y} r="8" className="st-selection-ring" />}
                                {inspected?.time === slot.time && inspected.joint === joint && <circle cx={x} cy={y} r="10" className="st-inspected-ring" />}
                                <circle cx={x} cy={y} r={spatial || temporal ? 3.6 : 2.7} className={`st-point ${slot.pose[joint][2] <= .2 ? 'is-uncertain' : ''}`} />
                            </g>;
                        }))}
                        <path d={`M 12 230 H ${width - 12} l -6 -4 m 6 4 l -6 4`} className="st-time-arrow" />
                    </svg>
                </div>
            </section>
        </div>
        <div className="st-coordinate-readout">
            <label>核对关节 <Select aria-label="核对骨架关节" value={selection.joint ?? 0} onChange={event => {
                onInteract(); setInspected({ time: selection.time, joint: Number(event.target.value) });
            }}>{JOINT_LABELS.map((name, joint) => <option key={joint} value={joint}>v{joint} {name}</option>)}</Select></label>
            <span>t={selection.time ?? '—'} · {selectedSlot?.padding ? '零填充，无骨架坐标' : `原图坐标 x=${fixed(selectedPoint?.[0])}, y=${fixed(selectedPoint?.[1])}`}</span>
            <span>{selectedPoint ? `预测分数 ${fixed(selectedPoint[2])}${!showLowScores && !validPoint(selectedPoint) ? ' · 低分点未绘制' : ''}` : ''}</span>
            <span className="st-coordinate-hint">白环核对 · {showLowScores ? '空心点 / 虚线为低分预测' : `已隐藏 ${hiddenCount} 个低分点`}</span>
            {inspected && <Button onClick={() => { onInteract(); setInspected(null); }}>跟随计算</Button>}
        </div>
    </div>;
}
