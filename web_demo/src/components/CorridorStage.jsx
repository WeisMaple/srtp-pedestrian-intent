import { useEffect, useRef, useState } from 'react';
import { createCorridorScene } from '../scene/corridorScene';
import { liveLayerModel } from '../domain/liveLayers';
import '../styles/corridor.css';

export default function CorridorStage({ step, track, live, trackId, structure, onPick, onCameraInteract, onError, children }) {
    const host = useRef(null);
    const engine = useRef(null);
    const [error, setError] = useState('');
    // The scene is created once; route callbacks through refs so it never
    // holds a stale pick/camera handler from the first render.
    const pickRef = useRef(onPick); pickRef.current = onPick;
    const interactRef = useRef(onCameraInteract); interactRef.current = onCameraInteract;
    const errorRef = useRef(onError); errorRef.current = onError;
    useEffect(() => {
        try {
            engine.current = createCorridorScene(host.current, {
                onPick: packet => pickRef.current?.(packet),
                onCameraInteract: () => interactRef.current?.(),
                onError: message => {
                    setError(message);
                    engine.current?.dispose(); engine.current = null;
                    errorRef.current?.(message);
                },
            });
        } catch (cause) { setError(cause.message || 'WebGL 不可用'); }
        return () => { engine.current?.dispose(); engine.current = null; };
    }, []);
    useEffect(() => { engine.current?.setStep(step); }, [step]);
    useEffect(() => {
        // With no live snapshot, still show the selected job track.
        if (!live) engine.current?.setTrack(track);
    }, [track, live]);
    useEffect(() => {
        if (!live || !engine.current) return;
        engine.current.setLive(liveLayerModel(live, trackId, structure, track));
    }, [live, trackId, structure, track]);
    if (error) return <div className="corridor-fallback" role="alert">
        <strong>三维画布暂不可用</strong><p>{error}</p>{children}
    </div>;
    return <div className="corridor-stage-shell">
        <div className="corridor-stage" ref={host} />
        <button className="camera-reset" type="button" aria-label="重置视角"
            onClick={() => engine.current?.reset()}>重置视角</button>
    </div>;
}
