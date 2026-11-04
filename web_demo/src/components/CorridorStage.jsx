import { useEffect, useRef, useState } from 'react';
import { createCorridorScene } from '../scene/corridorScene';
import '../styles/corridor.css';

export default function CorridorStage({ step, track, onPick, onError, children }) {
    const host = useRef(null);
    const engine = useRef(null);
    const [error, setError] = useState('');
    useEffect(() => {
        try {
            engine.current = createCorridorScene(host.current, {
                onPick, onError: message => {
                    setError(message);
                    engine.current?.dispose(); engine.current = null;
                    onError?.(message);
                },
            });
        } catch (cause) { setError(cause.message || 'WebGL 不可用'); }
        return () => { engine.current?.dispose(); engine.current = null; };
    }, []);
    useEffect(() => { engine.current?.setStep(step); }, [step]);
    useEffect(() => { engine.current?.setTrack(track); }, [track]);
    if (error) return <div className="corridor-fallback" role="alert">
        <strong>三维画布暂不可用</strong><p>{error}</p>{children}
    </div>;
    return <div className="corridor-stage" ref={host} />;
}
