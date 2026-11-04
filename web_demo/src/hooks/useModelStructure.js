import { useCallback, useEffect, useState } from 'react';
import { modelUrl, request } from '../api';
import { normalizeModelStructure } from '../domain/model';

export function useModelStructure() {
    const [raw, setRaw] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const [revision, setRevision] = useState(0);
    useEffect(() => {
        let disposed = false;
        let timer = null;
        const run = async (attempt = 0) => {
            if (disposed) return;
            let retrying = false;
            setLoading(true); setError('');
            try {
                const result = await request(modelUrl());
                if (!disposed) setRaw(result);
            } catch (cause) {
                if (disposed) return;
                // /api/model answers 409 while another inference owns the
                // model; retry automatically before surfacing an error.
                if (cause.status === 409 && attempt < 8) {
                    retrying = true;
                    timer = window.setTimeout(() => run(attempt + 1), Math.min(2000, 400 * (attempt + 1)));
                    return;
                }
                setError(cause.message || '模型结构未就绪。');
            } finally {
                if (!disposed && !retrying) setLoading(false);
            }
        };
        run();
        return () => { disposed = true; if (timer !== null) window.clearTimeout(timer); };
    }, [revision]);
    const retry = useCallback(() => setRevision(value => value + 1), []);
    return { structure: raw ? normalizeModelStructure(raw) : null, loading, error, retry };
}
