import { useEffect, useState } from 'react';
import { ACTIVE_STATUSES, jobUrl } from '../api';

export function useNetwork(job, sample, track) {
    const [data, setData] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [revision, setRevision] = useState(0);
    const ready = track?.history?.length === 50 && !ACTIVE_STATUSES.has(job?.status);
    const key = `${job?.id}:${sample?.frame}:${track?.track_id}:${ready}`;
    useEffect(() => {
        setData(null); setError(''); setLoading(false);
        if (!ready || !job?.id || sample?.frame == null || track?.track_id == null) return;
        const controller = new AbortController();
        let timer;
        setLoading(true);
        const inspect = async (attempt = 0) => {
            try {
                const response = await fetch(jobUrl(job.id, `/network?frame=${sample.frame}&track_id=${track.track_id}`), { signal: controller.signal });
                const result = await response.json();
                if (controller.signal.aborted) return;
                if (response.status === 409 && attempt < 5) {
                    // Aborting HTTP cannot interrupt a model call already executing.
                    timer = setTimeout(() => inspect(attempt + 1), Math.min(1800, 400 * (attempt + 1)));
                    return;
                }
                if (!response.ok) throw new Error(result.detail || `读取失败（${response.status}）`);
                setData({ key, result }); setLoading(false);
            } catch (cause) {
                if (!controller.signal.aborted) { setError(cause.message); setLoading(false); }
            }
        };
        timer = setTimeout(() => inspect(), 180);
        return () => { controller.abort(); clearTimeout(timer); };
    }, [job?.id, sample?.frame, track?.track_id, ready, revision]);
    return { data: data?.key === key ? data.result : null, error, loading, retry: () => setRevision(v => v + 1) };
}
