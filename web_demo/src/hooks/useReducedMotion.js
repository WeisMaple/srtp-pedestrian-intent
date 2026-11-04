import { useEffect, useState } from 'react';
export function useReducedMotion() {
    const query = '(prefers-reduced-motion: reduce)';
    const [reduced, setReduced] = useState(() => window.matchMedia?.(query).matches || false);
    useEffect(() => {
        const media = window.matchMedia?.(query);
        if (!media) return undefined;
        const onChange = event => setReduced(event.matches);
        media.addEventListener?.('change', onChange);
        return () => media.removeEventListener?.('change', onChange);
    }, []);
    return reduced;
}
