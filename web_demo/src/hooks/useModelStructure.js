import { useCallback, useEffect, useState } from 'react';
import { modelUrl, request } from '../api';
import { normalizeModelStructure } from '../domain/model';

export function useModelStructure() {
    const [raw, setRaw] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const load = useCallback(async () => {
        setLoading(true); setError('');
        try { setRaw(await request(modelUrl())); }
        catch (cause) { setError(cause.message || '模型结构未就绪。'); }
        finally { setLoading(false); }
    }, []);
    useEffect(() => { load(); }, [load]);
    return { structure: raw ? normalizeModelStructure(raw) : null, loading, error, retry: load };
}
