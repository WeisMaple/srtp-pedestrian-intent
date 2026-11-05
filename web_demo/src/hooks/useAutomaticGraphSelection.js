import { useLayoutEffect } from 'react';
import { automaticGraphSelection } from '../domain/temporal.js';

export function useAutomaticGraphSelection(snapshot, cachedRows, state, trackId, { onWindow, onTrack }) {
    const pick = state.step === 5 && state.autoGraph
        ? automaticGraphSelection(snapshot, cachedRows, state, trackId) : null;
    // Persist the visible automatic choice before manual input can pause playback.
    // Otherwise disabling auto would restore the earlier unavailable person/block.
    useLayoutEffect(() => {
        if (!pick) return;
        if (pick.block !== state.block || pick.boundary !== state.boundary) onWindow(pick.block, pick.boundary);
        if (String(pick.trackId) !== String(trackId)) onTrack?.(pick.trackId);
    }, [pick?.block, pick?.boundary, pick?.trackId, state.block, state.boundary, trackId, onWindow, onTrack]);
    return pick;
}
