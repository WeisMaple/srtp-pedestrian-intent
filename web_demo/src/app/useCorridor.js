import { useEffect, useMemo, useReducer, useRef } from 'react';
import { STEPS, beatsForStep, stepReady } from '../domain/timeline.js';

export const BEAT_INTERVAL_MS = 2200;
export const SPEEDS = [0.5, 1, 2];
export const usesComputationClock = state => state.step === 4 || (state.step === 5 && state.operation === 'temporal')
    || (state.step === 6 && state.intentDetail);

export function createCorridorState() {
    return { step: 1, beat: 0, paused: false, following: true, frozen: false, selectedTrackId: null,
        module: 0, block: 1, operation: 'temporal', boundary: false, autoGraph: true, termIndex: 0, lens: null, ready: true, speed: 1,
        intentDetail: true, intentClass: 2 };
}

export function corridorReducer(state, action) {
    switch (action.type) {
        case 'step': return { ...state, step: action.step, beat: 0, termIndex: 0, paused: true, following: false, lens: null,
            autoGraph: [4, 5, 6].includes(action.step) ? true : state.autoGraph };
        case 'restart': return { ...state, step: 1, beat: 0, paused: false, following: true,
            frozen: false, termIndex: 0, lens: null };
        // Camera interaction pauses the lecture but must not resume a snapshot
        // the user explicitly froze with the transport control.
        case 'camera': return { ...state, paused: true, following: false, autoGraph: false };
        case 'pause-narration': return { ...state, paused: true, following: false };
        case 'beat': return { ...state, beat: action.beat, paused: true, following: false, autoGraph: false };
        case 'beat-step': {
            const available = beatsForStep(state.step).length;
            return { ...state, beat: Math.max(0, Math.min(available - 1, state.beat + action.delta)),
                paused: true, following: false, autoGraph: false };
        }
        case 'pause': return { ...state, paused: true, following: false, frozen: true, autoGraph: false };
        case 'play': return { ...state, paused: false, following: true, frozen: false };
        case 'ready': return state.ready === Boolean(action.ready) ? state : { ...state, ready: Boolean(action.ready) };
        case 'speed': return { ...state, speed: SPEEDS.includes(action.speed) ? action.speed : 1 };
        case 'track': return { ...state, selectedTrackId: action.trackId, termIndex: 0 };
        case 'module': return { ...state, module: action.module, termIndex: 0 };
        case 'block': return { ...state, block: action.block, termIndex: 0, autoGraph: false };
        case 'operation': return { ...state, operation: action.operation, termIndex: 0, paused: true, following: false };
        case 'boundary': return { ...state, boundary: action.boundary, termIndex: 0, paused: true, following: false, autoGraph: false };
        case 'term': return { ...state, termIndex: action.index, paused: true, following: false, autoGraph: false };
        case 'graph-auto': return { ...state, autoGraph: action.value, frozen: action.value ? false : state.frozen, paused: true, following: false };
        case 'graph-tick': return state.autoGraph && !state.frozen ? { ...state, termIndex: action.index } : state;
        case 'graph-next': return state.autoGraph ? { ...state, block: action.block, boundary: action.boundary, termIndex: 0 } : state;
        case 'intent-mode': return { ...state, intentDetail: action.value, autoGraph: action.value, frozen: false, beat: 0, termIndex: 0, paused: true, following: false };
        case 'intent-class': return { ...state, intentClass: action.index, beat: 1, termIndex: 0, autoGraph: false, paused: true, following: false };
        case 'intent-tick': return state.autoGraph && !state.frozen ? { ...state, beat: action.beat, termIndex: action.index } : state;
        case 'intent-restart': return { ...state, beat: 0, termIndex: 0, autoGraph: true, frozen: false, paused: true, following: false };
        case 'next': return { ...state, step: Math.min(STEPS.length, state.step + 1), beat: 0,
            paused: true, following: false, termIndex: 0, lens: null };
        case 'beat-advance': return advanceNarration(state, action);
        default: return state;
    }
}

export function advanceNarration(state, { snapshot, job } = {}) {
    if (state.paused || !state.ready || (state.autoGraph && usesComputationClock(state))) return state;
    const available = beatsForStep(state.step).length;
    const beat = state.beat + 1;
    if (beat >= available) {
        // The last beat of the final step is the end of the lecture: stop
        // instead of wrapping back to the first step.
        if (state.step >= STEPS.length) {
            return { ...state, beat: Math.max(0, available - 1), paused: true, following: false };
        }
        const nextStep = state.step + 1;
        return { ...state, step: nextStep, beat: 0, ready: stepReady(nextStep, snapshot, job) };
    }
    return { ...state, beat };
}

export function useCorridor({ job, live, trackId, externalReadiness = false }) {
    const [state, dispatch] = useReducer(corridorReducer, undefined, createCorridorState);
    const stateRef = useRef(state); stateRef.current = state;
    // Snapshot and job identities change on every publish/poll. Keep them in
    // refs so the beat timer is not torn down before it can fire.
    const liveRef = useRef(live); liveRef.current = live;
    const jobRef = useRef(job); jobRef.current = job;
    const speedRef = useRef(state.speed); speedRef.current = state.speed;
    useEffect(() => {
        if (trackId != null && stateRef.current.selectedTrackId !== trackId) dispatch({ type: 'track', trackId });
    }, [trackId]);
    useEffect(() => {
        if (!externalReadiness) dispatch({ type: 'ready', ready: stepReady(state.step, live, job) });
    }, [state.step, live, job, externalReadiness]);
    useEffect(() => {
        if (state.paused || (state.autoGraph && usesComputationClock(state))) return undefined;
        let disposed = false;
        let timer = null;
        const arm = () => {
            if (disposed) return;
            timer = window.setTimeout(() => {
                timer = null;
                if (disposed) return;
                const current = stateRef.current;
                if (current.paused) return;
                if (!current.ready) { arm(); return; }
                dispatch({ type: 'beat-advance', snapshot: liveRef.current, job: jobRef.current });
                // If the reducer could not advance, keep the clock running for
                // the next data-gated attempt; otherwise the effect re-arms.
                arm();
            }, BEAT_INTERVAL_MS / speedRef.current);
        };
        arm();
        return () => { disposed = true; if (timer !== null) window.clearTimeout(timer); };
    }, [state.step, state.beat, state.paused, state.operation, state.autoGraph, state.intentDetail]);
    const actions = useMemo(() => ({
        selectStep: step => dispatch({ type: 'step', step }),
        selectTrack: id => dispatch({ type: 'track', trackId: id }),
        selectModule: module => dispatch({ type: 'module', module }),
        selectBlock: block => dispatch({ type: 'block', block }),
        selectOperation: operation => dispatch({ type: 'operation', operation }),
        selectBoundary: boundary => dispatch({ type: 'boundary', boundary }),
        setGraphAuto: value => dispatch({ type: 'graph-auto', value }),
        tickGraph: index => dispatch({ type: 'graph-tick', index }),
        nextGraph: (block, boundary) => dispatch({ type: 'graph-next', block, boundary }),
        setIntentDetail: value => dispatch({ type: 'intent-mode', value }),
        selectIntentClass: index => dispatch({ type: 'intent-class', index }),
        tickIntent: (beat, index) => dispatch({ type: 'intent-tick', beat, index }),
        restartIntent: () => dispatch({ type: 'intent-restart' }),
        freeze: () => dispatch({ type: 'pause' }),
        seekTerm: index => dispatch({ type: 'term', index }),
        seekBeat: beat => dispatch({ type: 'beat', beat }),
        togglePause: () => dispatch({ type: stateRef.current.paused ? 'play' : 'pause' }),
        followLatest: () => dispatch({ type: 'play' }),
        restart: () => dispatch({ type: 'restart' }),
        setSpeed: speed => dispatch({ type: 'speed', speed }),
        cameraInteract: () => dispatch({ type: 'camera' }),
        pauseNarration: () => dispatch({ type: 'pause-narration' }),
        setReady: ready => dispatch({ type: 'ready', ready }),
        next: () => dispatch({ type: 'next', snapshot: liveRef.current, job: jobRef.current }),
        previousBeat: () => dispatch({ type: 'beat-step', delta: -1 }),
        stepBeat: delta => dispatch({ type: 'beat-step', delta }),
    }), []);
    return { state, actions };
}
