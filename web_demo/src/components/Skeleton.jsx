import { memo } from 'react';

export const COCO_EDGES = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];

export function validPoint(point) {
    return Array.isArray(point) && Number.isFinite(point[0]) && Number.isFinite(point[1]) && (point.length < 3 || point[2] > 0.05);
}

export function poseBounds(poses) {
    const points = poses.flat().filter(validPoint);
    if (!points.length) return null;
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const padding = Math.max(maxX - minX, maxY - minY, 1) * 0.14;
    return { x: minX - padding, y: minY - padding, width: Math.max(1, maxX - minX) + padding * 2, height: Math.max(1, maxY - minY) + padding * 2 };
}

function Skeleton({ pose, color = '#66dfb4', className = '', dashed = false, bounds, label }) {
    const box = bounds || (pose ? poseBounds([pose]) : null);
    if (!box) return <div className={`skeleton-placeholder ${className}`} aria-label="暂无真实骨架"><span /></div>;
    return <svg className={`skeleton ${className}`} viewBox={`${box.x} ${box.y} ${box.width} ${box.height}`} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true} fill="none" style={{ color }}>
        {COCO_EDGES.map(([a,b]) => validPoint(pose[a]) && validPoint(pose[b]) ? <line key={`${a}-${b}`} x1={pose[a][0]} y1={pose[a][1]} x2={pose[b][0]} y2={pose[b][1]} stroke="currentColor" strokeWidth="1.6" strokeDasharray={dashed ? '4 3' : undefined} vectorEffect="non-scaling-stroke" /> : null)}
        {pose.map((point, index) => validPoint(point) ? <circle key={index} cx={point[0]} cy={point[1]} r={Math.max(box.height * 0.012, 0.3)} fill="currentColor" className="joint" /> : null)}
    </svg>;
}

export default memo(Skeleton);

export function CocoGraph({ progress = 0, reducedMotion = false }) {
    // The fixed node layout is an explicitly labelled COCO17 topology diagram.
    // It is never represented as an observed or predicted pose.
    const structure = [[50,12],[46,9],[54,9],[42,12],[58,12],[35,32],[65,32],[27,53],[73,53],[22,74],[78,74],[40,76],[60,76],[37,108],[63,108],[32,139],[68,139]];
    const amount = Math.min(1, Math.max(0, progress));
    return <svg className="coco-graph" viewBox="10 0 80 153" aria-label="COCO17 图连接结构示意">
        {COCO_EDGES.map(([a,b], index) => {
            const local = Math.max(0, Math.min(1, amount * 3 - index / COCO_EDGES.length * 2));
            return <g key={`${a}-${b}`}>
                <line x1={structure[a][0]} y1={structure[a][1]} x2={structure[b][0]} y2={structure[b][1]} className="graph-edge" />
                <line x1={structure[a][0]} y1={structure[a][1]} x2={structure[b][0]} y2={structure[b][1]} className="graph-signal" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - local} />
                {!reducedMotion && local > 0 && local < 1 && <circle cx={structure[a][0] + (structure[b][0] - structure[a][0]) * local} cy={structure[a][1] + (structure[b][1] - structure[a][1]) * local} r="2.1" fill="#d0ffec" />}
            </g>;
        })}
        {structure.map(([x,y], index) => <circle key={index} cx={x} cy={y} r="2.8" className="graph-joint" style={{ fill: amount > index / 17 ? '#55dcb3' : '#113b3c' }} />)}
    </svg>;
}
