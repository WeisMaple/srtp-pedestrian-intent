import { useCallback, useEffect, useRef, useState } from 'react';
import { ACTIVE_STATUSES, jobUrl, request } from '../api';

const STORAGE_KEY = 'srtp.web.last-job';

function savedJobId() {
    try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

export function useJob() {
    const [job, setJob] = useState(null);
    const [health, setHealth] = useState(null);
    const [error, setError] = useState('');
    const [pending, setPending] = useState('');
    const [connection, setConnection] = useState('connecting');
    const selectedId = useRef(null);
    const mounted = useRef(true);
    const terminal = ['completed', 'failed', 'cancelled'].includes(job?.status);

    const acceptSnapshot = useCallback((snapshot) => {
        if (!snapshot?.id || (selectedId.current && selectedId.current !== snapshot.id)) return;
        setJob((previous) => previous?.id === snapshot.id && previous.revision > snapshot.revision ? previous : snapshot);
    }, []);

    useEffect(() => {
        mounted.current = true;
        let disposed = false;
        const loadHealth = async () => {
            try {
                const result = await request('/api/health');
                if (!disposed) setHealth(result);
                return result;
            } catch {
                if (!disposed) setHealth({ models_ready: false, encoder_ready: false, unavailable: true, detail: '无法连接本地分析服务' });
                return null;
            }
        };
        (async () => {
            const result = await loadHealth();
            const active = typeof result?.active_job === 'string' ? result.active_job : result?.active_job?.id;
            const id = active || savedJobId();
            if (!id || disposed) return;
            selectedId.current = id;
            try { const snapshot = await request(jobUrl(id)); if (!disposed) acceptSnapshot(snapshot); }
            catch { if (!disposed) { selectedId.current = null; try { localStorage.removeItem(STORAGE_KEY); } catch { /* Storage may be disabled. */ } } }
        })();
        const interval = window.setInterval(loadHealth, 15000);
        return () => { disposed = true; mounted.current = false; window.clearInterval(interval); };
    }, [acceptSnapshot]);

    useEffect(() => {
        if (!job?.id) return;
        try { localStorage.setItem(STORAGE_KEY, job.id); } catch { /* The live session works without storage. */ }
        if (terminal) { setConnection('connected'); return; }
        const id = job.id;
        const stream = new EventSource(jobUrl(id, '/events'));
        let disposed = false;
        stream.onopen = () => {
            setConnection('connected');
            request(jobUrl(id)).then((snapshot) => { if (!disposed) acceptSnapshot(snapshot); }).catch(() => {});
        };
        stream.onmessage = (event) => {
            try { acceptSnapshot(JSON.parse(event.data)); } catch { /* Ignore an incomplete event; the next is a full snapshot. */ }
        };
        stream.onerror = () => setConnection('reconnecting');
        return () => { disposed = true; stream.close(); };
    }, [job?.id, terminal, acceptSnapshot]);

    const upload = useCallback(async (file) => {
        if (!file || pending || (job && ACTIVE_STATUSES.has(job.status))) return;
        setError('');
        setPending('upload');
        const body = new FormData();
        body.append('file', file);
        try {
            const snapshot = await request('/api/jobs', { method: 'POST', body });
            if (!mounted.current) return;
            selectedId.current = snapshot.id;
            setJob(snapshot);
        } catch (cause) { if (mounted.current) setError(cause.message || '视频上传失败，请重试。'); }
        finally { if (mounted.current) setPending(''); }
    }, [pending, job]);

    const action = useCallback(async (name) => {
        if (!job?.id || pending) return;
        setPending(name);
        setError('');
        try { acceptSnapshot(await request(jobUrl(job.id, `/${name}`), { method: 'POST' })); }
        catch (cause) { setError(cause.message || '操作失败，请重试。'); }
        finally { setPending(''); }
    }, [job?.id, pending, acceptSnapshot]);

    return { job, health, error, dismissError: () => setError(''), pending, connection, upload, start: () => action('start'), cancel: () => action('cancel') };
}
