import { useEffect, useRef, useState } from 'react';
import { NETWORK_DURATION, networkStart, networkPosition } from '../networkTimeline';

export function useNetworkPlayback() {
    const [time, setTime] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    const [reducedMotion, setReducedMotion] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
    const clock = useRef(0);
    useEffect(() => {
        const query = matchMedia('(prefers-reduced-motion: reduce)');
        const motion = () => setReducedMotion(query.matches);
        const visibility = () => { if (document.hidden) setPlaying(false); };
        query.addEventListener('change', motion); document.addEventListener('visibilitychange', visibility);
        return () => { query.removeEventListener('change', motion); document.removeEventListener('visibilitychange', visibility); };
    }, []);
    useEffect(() => {
        if (!playing) return;
        let raf, previous = performance.now(), painted = previous;
        const tick = now => {
            clock.current = Math.min(NETWORK_DURATION, clock.current + Math.min(100, now - previous) * speed / 1000); previous = now;
            if (now - painted >= 33 || clock.current >= NETWORK_DURATION) { setTime(clock.current); painted = now; }
            if (clock.current >= NETWORK_DURATION) setPlaying(false); else raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
    }, [playing, speed]);
    const seek = value => { clock.current = Math.max(0, Math.min(NETWORK_DURATION, value)); setTime(clock.current); setPlaying(false); };
    return { ...networkPosition(time), time, playing, speed, reducedMotion, setSpeed, seek,
        chapterSeek: index => seek(networkStart(index)), pause: () => setPlaying(false),
        toggle: () => { if (time >= NETWORK_DURATION) seek(0); setPlaying(v => !v); },
        restart: () => { seek(0); setPlaying(true); } };
}
