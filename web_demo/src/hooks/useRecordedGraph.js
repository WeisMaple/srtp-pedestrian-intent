import { useCallback, useEffect, useMemo, useState } from 'react';
import { jobUrl, request } from '../api.js';
import { recordedGraphDetails, recordedGraphSelections } from '../domain/temporal.js';
import { recordedMemoryNodes } from '../domain/memory.js';
import { recordedIntentNodes } from '../domain/intent.js';

/** Completed jobs replay saved histories; live jobs never trigger a second inference. */
export function useRecordedGraph(job, enabled, replay = null) {
    const active = Boolean(enabled && job?.status === 'completed');
    const jobId = job?.id;
    const [trace, setTrace] = useState(null);
    const [choice, setChoice] = useState(null);
    const [network, setNetwork] = useState(null);
    const [revision, setRevision] = useState(0);
    useEffect(() => {
        if (!active) return undefined;
        const controller = new AbortController(); let disposed = false;
        setTrace({ jobId, loading: true });
        request(jobUrl(jobId, '/trace'), { signal: controller.signal }).then(data => {
            if (!disposed) setTrace({ jobId, data });
        }).catch(error => {
            if (!disposed && error.name !== 'AbortError') setTrace({ jobId, error: error.message });
        });
        return () => { disposed = true; controller.abort(); };
    }, [active, jobId, revision]);
    const choices = useMemo(() => active && trace?.jobId === jobId ? recordedGraphSelections(trace.data) : [], [active, trace, jobId]);
    const selection = useMemo(() => {
        if (!replay) return choices.find(item => choice?.jobId === jobId && item.key === choice.key) || choices[0] || null;
        const sameTrack = choices.filter(item => String(item.track.track_id) === String(replay.trackId));
        return (sameTrack.length ? sameTrack : choices).reduce((nearest, item) =>
            !nearest || Math.abs(item.frame - replay.frame) < Math.abs(nearest.frame - replay.frame) ? item : nearest, null);
    }, [choices, choice, jobId, replay?.frame, replay?.trackId, Boolean(replay)]);
    const key = selection ? `${jobId}:${selection.key}` : null;
    useEffect(() => {
        if (!active || !selection) return undefined;
        const controller = new AbortController(); let disposed = false, timer;
        setNetwork({ key, loading: true });
        const load = async (attempt = 0) => {
            try {
                const result = await request(jobUrl(jobId, `/network?frame=${selection.frame}&track_id=${selection.track.track_id}`), { signal: controller.signal });
                if (disposed) return;
                if (result.source?.job_id !== jobId || result.source.frame !== selection.frame
                    || String(result.source.track_id) !== String(selection.track.track_id)) throw new Error('计算记录与所选帧或行人不一致，请重试。');
                setNetwork({ key, result });
            } catch (error) {
                if (disposed || error.name === 'AbortError') return;
                if (error.status === 409 && attempt < 8) { timer = setTimeout(() => load(attempt + 1), Math.min(2000, 400 * (attempt + 1))); return; }
                setNetwork({ key, error: error.message });
            }
        };
        load();
        return () => { disposed = true; clearTimeout(timer); controller.abort(); };
    }, [active, jobId, key, revision]);
    const result = active && network?.key === key ? network.result || null : null;
    const error = active ? (trace?.jobId === jobId && trace.error) || (network?.key === key && network.error) || '' : '';
    const status = !active ? 'inactive' : error ? 'error' : trace?.jobId !== jobId || trace.loading ? 'loading'
        : !selection ? 'empty' : !result ? 'loading' : 'ready';
    const details = useMemo(() => recordedGraphDetails(result), [result]);
    const snapshot = useMemo(() => selection ? { ...selection.sample, nodes: { ...recordedMemoryNodes(result), ...recordedIntentNodes(result) }, details,
        tracks: choices.filter(item => item.frame === selection.frame).map(item => item.track) } : null, [selection, choices, details, result]);
    const onSelect = replay?.onSelect;
    const select = useCallback(key => {
        const next = choices.find(item => item.key === key);
        if (!next) return;
        if (onSelect) onSelect(next.frame, next.track.track_id);
        else setChoice({ jobId, key });
    }, [jobId, choices, onSelect]);
    const selectTrack = useCallback(id => {
        const next = choices.find(item => item.frame === selection?.frame && String(item.track.track_id) === String(id));
        if (next) select(next.key);
    }, [choices, selection, select]);
    const retry = useCallback(() => setRevision(value => value + 1), []);
    return { active, status, error, choices, selection, snapshot, result, select, selectTrack, retry };
}
