import test from 'node:test';
import assert from 'node:assert/strict';
import React, { useReducer, useState } from 'react';
import TestRenderer, { act } from 'react-test-renderer';
import { useAutomaticGraphSelection } from '../src/hooks/useAutomaticGraphSelection.js';
import { corridorReducer, createCorridorState } from '../src/app/useCorridor.js';

test('pausing or inspecting retains the automatically recovered person and convolution window', async () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const snapshot = { tracks: [{ track_id: 18 }, { track_id: 11 }], details: { stgcn: { 5: [
        { id: 'live', stage: 'temporal', track_id: 11, inputs: [2], metadata: { block: 5, boundary: true } },
    ] } } };
    let api, renderer;
    function Probe() {
        const [state, dispatch] = useReducer(corridorReducer, { ...createCorridorState(), step: 5 });
        const [trackId, setTrackId] = useState(18);
        const pick = useAutomaticGraphSelection(snapshot, [], state, trackId, {
            onWindow: (block, boundary) => dispatch({ type: 'graph-next', block, boundary }), onTrack: setTrackId,
        });
        api = { state, trackId, pick, dispatch };
        return null;
    }
    try {
        await act(async () => { renderer = TestRenderer.create(React.createElement(Probe)); });
        assert.equal(api.state.block, 6);
        assert.equal(api.state.boundary, true);
        assert.equal(api.trackId, 11);
        await act(async () => { api.dispatch({ type: 'camera' }); });
        assert.equal(api.pick, null);
        assert.equal(api.state.block, 6);
        assert.equal(api.state.boundary, true);
        assert.equal(api.trackId, 11);
        await act(async () => { api.dispatch({ type: 'graph-auto', value: true }); });
        await act(async () => { api.dispatch({ type: 'term', index: 2.5 }); });
        assert.equal(api.state.block, 6);
        assert.equal(api.state.termIndex, 2.5);
        assert.equal(api.trackId, 11);
    } finally {
        await act(async () => { renderer?.unmount(); });
        delete globalThis.IS_REACT_ACT_ENVIRONMENT;
    }
});
