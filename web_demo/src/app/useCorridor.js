import { useEffect, useMemo, useReducer, useRef } from 'react';
import { STEPS, beatsForStep, stepReady } from '../domain/timeline.js';

export function createCorridorState() {
    return { step: 1, beat: 0, paused: false, following: true, selectedTrackId: null,
        module: 0, block: 1, termIndex: 0, lens: null, ready: true };
}

export function corridorReducer(state, action) {
    switch (action.type) {
        case 'step': return { ...state, step: action.step, beat: 0, paused: true, following: false, lens: null };
        case 'beat': return { ...state, beat: action.beat };
        case 'pause': return { ...state, paused: true, following: false };
        case 'play': return { ...state, paused: false, following: true };
        case 'ready': return { ...state, ready: Boolean(action.ready) };
        case 'track': return { ...state, selectedTrackId: action.trackId, termIndex: 0 };
        case 'module': return { ...state, module: action.module, termIndex: 0 };
        case 'block': return { ...state, block: action.block, termIndex: 0 };
        case 'term': return { ...state, termIndex: action.index };
        case 'next': return state.paused ? state : advanceNarration(state, action);
        case 'beat-advance': return advanceNarration(state, action);
        default: return state;
    }
}

export function advanceNarration(state, { snapshot, job } = {}) {
    if (state.paused || !state.ready) return state;
    const available = beatsForStep(state.step).length;
    const beat = state.beat + 1;
    if (beat >= available) {
        const nextStep = Math.min(STEPS.length, state.step + 1);
        return { ...state, step: nextStep, beat: 0, ready: stepReady(nextStep, snapshot, job) };
    }
    return { ...state, beat };
}

export function useCorridor({ job, live, trackId }) {
    const [state, dispatch] = useReducer(corridorReducer, undefined, createCorridorState);
    const stateRef = useRef(state); stateRef.current = state;
    useEffect(() => {
        if (trackId != null && stateRef.current.selectedTrackId !== trackId) dispatch({ type: 'track', trackId });
    }, [trackId]);
    useEffect(() => { dispatch({ type: 'ready', ready: stepReady(state.step, live, job) }); }, [state.step, live, job]);
    useEffect(() => {
        if (state.paused) return undefined;
        const timer = window.setTimeout(() => dispatch({ type: 'beat-advance', snapshot: live, job }), 2200);
        return () => window.clearTimeout(timer);
    }, [state.step, state.beat, state.paused, live, job]);
    const actions = useMemo(() => ({
        selectStep: step => dispatch({ type: 'step', step }),
        selectTrack: id => dispatch({ type: 'track', trackId: id }),
        selectModule: module => dispatch({ type: 'module', module }),
        selectBlock: block => dispatch({ type: 'block', block }),
        seekTerm: index => dispatch({ type: 'term', index }),
        seekBeat: beat => dispatch({ type: 'beat', beat }),
        togglePause: () => dispatch({ type: stateRef.current.paused ? 'play' : 'pause' }),
        followLatest: () => dispatch({ type: 'play' }),
        next: () => dispatch({ type: 'next', snapshot: live, job }),
        previous: () => dispatch({ type: 'term', index: Math.max(0, stateRef.current.termIndex - 1) }),
    }), [live, job]);
    return { state, actions };
}
