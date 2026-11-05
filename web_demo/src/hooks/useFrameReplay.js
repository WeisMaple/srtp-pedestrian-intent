import { useCallback, useEffect, useMemo, useState } from 'react';
import { jobUrl, request } from '../api.js';

/** Replay every original inference row, including empty frames. Never reuse live poses. */
export function useFrameReplay(job, enabled, speed = 1) {
    const jobId = job?.id;
    const active = Boolean(enabled && job?.status === 'completed');
    const fps = Number(job?.metadata?.fps) || 30;
    const [data, setData] = useState(null);
    const [cursor, setCursor] = useState(null);
    const [decoded, setDecoded] = useState(null);
    const [revision, setRevision] = useState(0);
    useEffect(() => {
        if (!active) { setCursor(current => current ? { ...current, playing: false } : current); return undefined; }
        let disposed = false;
        const controller = new AbortController();
        setData({ jobId, loading: true });
        request(jobUrl(jobId, '/files/data'), { signal: controller.signal }, 'text').then(text => {
            const rows = text.split('\n').filter(line => line.trim()).map(line => JSON.parse(line))
                .filter(row => Number.isInteger(row.frame) && row.frame >= 0 && Array.isArray(row.tracks))
                .sort((a, b) => a.frame - b.frame);
            if (!disposed) setData({ jobId, rows });
        }).catch(error => {
            if (!disposed && error.name !== 'AbortError') setData({ jobId, error: error.message || '无法读取逐帧记录。' });
        });
        return () => { disposed = true; controller.abort(); };
    }, [active, jobId, revision]);
    const rows = useMemo(() => active && data?.jobId === jobId ? data.rows || [] : [], [active, data, jobId]);
    const byFrame = useMemo(() => new Map(rows.map(row => [row.frame, row])), [rows]);
    const minFrame = rows[0]?.frame ?? 0, maxFrame = rows.at(-1)?.frame ?? 0;
    const clamp = useCallback(value => Math.max(minFrame, Math.min(maxFrame, Math.round(Number(value) || 0))), [minFrame, maxFrame]);
    const frame = cursor && cursor.jobId === jobId ? clamp(cursor.frame) : minFrame;
    const playing = Boolean(active && rows.length && cursor?.jobId === jobId && cursor.playing);
    const seek = useCallback(value => setCursor({ jobId, frame: clamp(value), playing: false }), [jobId, clamp]);
    const pause = useCallback(() => setCursor(current => current?.playing ? { ...current, playing: false } : current), []);
    const onFrameReady = useCallback(frame => setDecoded(current => current?.jobId === jobId && current.frame === frame
        ? current : { jobId, frame }), [jobId]);
    const play = useCallback(() => setCursor(current => {
        const from = current && current.jobId === jobId ? clamp(current.frame) : minFrame;
        return { jobId, frame: from >= maxFrame ? minFrame : from, playing: true };
    }), [jobId, clamp, minFrame, maxFrame]);
    const step = useCallback(delta => setCursor(current => ({ jobId,
        frame: clamp((current && current.jobId === jobId ? current.frame : minFrame) + delta), playing: false })), [jobId, minFrame, clamp]);
    useEffect(() => {
        // Apply back-pressure: a slow decoder must show this frame before the
        // cursor advances, otherwise the skeleton would chase it indefinitely.
        if (!playing || decoded?.jobId !== jobId || decoded.frame !== frame) return undefined;
        const timer = window.setTimeout(() => setCursor(current => {
            if (current?.jobId !== jobId || !current.playing) return current;
            const next = clamp(current.frame + 1);
            return { ...current, frame: next, playing: next < maxFrame };
        }), 1000 / fps / speed);
        return () => window.clearTimeout(timer);
    }, [playing, frame, fps, speed, jobId, clamp, maxFrame, decoded]);
    const snapshot = useMemo(() => rows.length ? {
        frame, time_seconds: frame / fps, tracks: [], replayMissing: !byFrame.has(frame),
        ...byFrame.get(frame), nodes: {}, details: {},
    } : null, [rows, frame, fps, byFrame]);
    const error = active && data?.jobId === jobId ? data.error || '' : '';
    const status = !active ? 'inactive' : error ? 'error' : data?.jobId !== jobId || data.loading ? 'loading' : rows.length ? 'ready' : 'empty';
    const retry = useCallback(() => setRevision(value => value + 1), []);
    const firstPersonFrame = rows.find(row => row.tracks.length)?.frame;
    return { active, status, error, frame, minFrame, maxFrame, fps, playing, snapshot,
        firstPersonFrame, seek, pause, play, step, retry, onFrameReady };
}
