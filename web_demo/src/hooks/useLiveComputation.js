import { useCallback, useEffect, useRef, useState } from 'react';
import { jobUrl } from '../api';
import { createLiveState, reduceLiveEvent, setLiveFrozen, visibleLiveState } from '../liveComputation';

const PUBLISH_INTERVAL_MS = 100;

/** One SSE connection per job; ingest in order, publish at most ten times per second. */
export function useLiveComputation(job, enabled = true) {
    const jobId = job?.id || null;
    const [published, setPublished] = useState(() => createLiveState(jobId));
    const [revision, setRevision] = useState(0);
    const current = useRef({ state: createLiveState(jobId), publish: null });

    useEffect(() => {
        const slot = current.current.state.jobId === jobId
            ? current.current : { state: createLiveState(jobId), publish: null };
        current.current = slot;
        let disposed = false, timer = null, stream = null;
        let lastPublished = -Infinity;
        const publish = () => {
            if (disposed || current.current !== slot) return;
            if (timer !== null) clearTimeout(timer);
            timer = null;
            lastPublished = performance.now();
            setPublished(slot.state);
        };
        const schedule = () => {
            if (disposed || timer !== null) return;
            timer = setTimeout(publish, Math.max(0, PUBLISH_INTERVAL_MS - (performance.now() - lastPublished)));
        };
        slot.publish = publish;
        slot.state = { ...slot.state, connection: !jobId ? 'idle' : enabled ? 'connecting' : 'disabled', error: '' };
        publish();
        if (jobId && enabled) {
            if (typeof EventSource === 'undefined') {
                slot.state = { ...slot.state, connection: 'error', error: '当前浏览器不支持实时事件连接。' };
                schedule();
            } else {
                // Leave reconnection to this EventSource so Last-Event-ID is preserved.
                stream = new EventSource(jobUrl(jobId, '/live'));
                stream.onopen = () => {
                    if (disposed) return;
                    slot.state = { ...slot.state, connection: 'connected', error: '' };
                    schedule();
                };
                stream.onerror = () => {
                    if (disposed) return;
                    slot.state = { ...slot.state, connection: 'reconnecting', error: '实时连接中断，正在自动重连。' };
                    schedule();
                };
                for (const type of ['snapshot', 'computation', 'gap', 'terminal']) {
                    stream.addEventListener(type, message => {
                        if (disposed) return;
                        let payload;
                        try { payload = JSON.parse(message.data); }
                        catch {
                            slot.state = { ...slot.state, error: '收到无法解析的实时事件，正在等待下一条有效记录。' };
                            schedule();
                            return;
                        }
                        const previous = slot.state;
                        slot.state = reduceLiveEvent(previous, type, payload, Date.now());
                        if (slot.state === previous) return;
                        if (type === 'terminal') stream.close();
                        schedule();
                    });
                }
            }
        }
        return () => {
            disposed = true;
            if (timer !== null) clearTimeout(timer);
            stream?.close();
            slot.publish = null;
        };
    }, [jobId, enabled, revision]);

    // Effects run after rendering: never expose the previous job for even one render.
    const state = visibleLiveState(published.jobId === jobId ? published : createLiveState(jobId));
    const freeze = useCallback((value = true) => {
        const slot = current.current;
        if (slot.state.jobId !== jobId) return;
        const displayed = visibleLiveState(published.jobId === jobId ? published : createLiveState(jobId));
        slot.state = setLiveFrozen(slot.state, Boolean(value), displayed);
        slot.publish?.();
    }, [jobId, published]);
    const toggleFreeze = useCallback(() => freeze(!current.current.state.frozen), [freeze]);
    const resume = useCallback(() => freeze(false), [freeze]);
    const retry = useCallback(() => setRevision(value => value + 1), []);

    return { snapshot: state.snapshot, latestEvent: state.latestEvent, events: state.events,
        connection: state.connection, error: state.error, frozen: state.frozen,
        freeze, toggleFreeze, resume, retry, receivedAt: state.receivedAt, latestSeq: state.latestSeq };
}
