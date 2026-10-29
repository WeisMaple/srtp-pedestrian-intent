import { memo, useEffect, useRef, useState } from 'react';

export const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
export const shape = (value) => value?.join(' × ') || '—';

export const TensorMap = memo(function TensorMap({ values, label, rows = 8, cols = 17 }) {
    const canvas = useRef(null);
    const [point, setPoint] = useState(null);
    const height = values?.length || rows;
    const width = values?.[0]?.length || cols;
    useEffect(() => {
        const context = canvas.current.getContext('2d');
        context.clearRect(0, 0, width * 7, height * 7);
        const all = values?.flat().filter(Number.isFinite) || [];
        const peak = Math.max(...all.map(Math.abs), .00001);
        for (let r = 0; r < height; r += 1) for (let c = 0; c < width; c += 1) {
            const v = values?.[r]?.[c];
            const intensity = Number.isFinite(v) ? Math.min(1, Math.abs(v) / peak) : 0;
            context.fillStyle = Number.isFinite(v) ? v < 0 ? `rgba(99,157,246,${.13 + intensity * .87})` : `rgba(69,222,169,${.13 + intensity * .87})` : '#203640';
            context.fillRect(c * 7, r * 7, 6, 6);
        }
    }, [values, width, height]);
    return <figure className="tensor-map"><canvas ref={canvas} width={width * 7} height={height * 7} role="img" aria-label={`${label}，${height} 行 ${width} 列${values ? '，真实张量采样' : '，尚无计算数据'}`} onMouseMove={(event) => { const b = event.currentTarget.getBoundingClientRect(); const c = Math.min(width - 1, Math.floor((event.clientX - b.left) / b.width * width)); const r = Math.min(height - 1, Math.floor((event.clientY - b.top) / b.height * height)); setPoint({r,c,v:values?.[r]?.[c]}); }} onMouseLeave={() => setPoint(null)} /><figcaption>{point && values ? `[${point.r}, ${point.c}] = ${fmt(point.v, 4)}` : label}</figcaption></figure>;
});

export function Wire({ d, active = false, progress = 1, color = '#68dcb4', label, x, y }) {
    return <g className="network-wire"><path d={d} /><path d={d} pathLength="1" style={{ stroke: color }} strokeDasharray="1" strokeDashoffset={1 - (active ? progress : 0)} className="wire-signal" />{label && <text x={x} y={y}>{label}</text>}</g>;
}

export function Box({ x, y, width = 160, height = 78, title, subtitle, detail, active, accent, onClick }) {
    return <g className={`network-box ${active ? 'active' : ''} ${onClick ? 'clickable' : ''}`} style={accent ? { '--node-accent': accent } : undefined} onClick={onClick} role={onClick ? 'button' : undefined} tabIndex={onClick ? 0 : undefined} onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } } : undefined} aria-label={onClick ? title : undefined}>
        <rect x={x} y={y} width={width} height={height} rx="9" /><text x={x + 16} y={y + 28} className="node-title">{title}</text><text x={x + 16} y={y + 49} className="node-subtitle">{subtitle}</text>{detail && <text x={x + 16} y={y + 69} className="node-detail">{detail}</text>}
    </g>;
}

export function Operator({ x, y, text, active = false }) {
    return <g className={`network-operator ${active ? 'active' : ''}`}><circle cx={x} cy={y} r="15" /><text x={x} y={y + 5} textAnchor="middle">{text}</text></g>;
}

export function ValueStrip({ values, label }) {
    return <div className="value-strip"><span>{label}</span><div>{Array.from({length:16}, (_,i) => <i key={i} title={fmt(values?.[i], 5)} style={Number.isFinite(values?.[i]) ? { background: values[i] >= 0 ? '#63dbaf' : '#78a8f3', opacity: .2 + .8 * Math.min(1, Math.abs(values[i])) } : undefined} />)}</div><small>前 16 维</small></div>;
}
