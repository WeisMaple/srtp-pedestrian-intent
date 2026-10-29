import { useEffect, useState } from 'react';
import { jobUrl, request } from '../api';

export function useTrace(job, enabled) {
    const [record, setRecord] = useState(null);
    const [error, setError] = useState('');
    const [revision, setRevision] = useState(0);
    useEffect(() => { setRecord(null); setError(''); }, [job?.id]);
    useEffect(() => {
        if (!enabled || !job?.id) return;
        let disposed = false;
        const controller = new AbortController();
        request(jobUrl(job.id, '/trace'), { signal: controller.signal }).then((result) => {
            if (!disposed) { setRecord({ jobId: job.id, samples: Array.isArray(result.samples) ? result.samples : [] }); setError(''); }
        }).catch((cause) => { if (!disposed && cause.name !== 'AbortError') setError('暂时无法读取流程记录，请稍后重试。'); });
        return () => { disposed = true; controller.abort(); };
    }, [job?.id, job?.trace_count, job?.status, enabled, revision]);
    return { samples: record && record.jobId === job?.id ? record.samples : [], error, retry: () => setRevision(value => value + 1) };
}
