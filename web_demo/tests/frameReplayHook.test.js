import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import TestRenderer, { act } from 'react-test-renderer';
import { useFrameReplay } from '../src/hooks/useFrameReplay.js';

test('replay seeks actual frame records, pauses on scrub, stops at the end, and isolates jobs', async () => {
    const originalFetch = globalThis.fetch, originalWindow = globalThis.window;
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const timers = new Map(); let timerId = 0, api, renderer, finishB;
    globalThis.window = { setTimeout: fn => { timers.set(++timerId, fn); return timerId; }, clearTimeout: id => timers.delete(id) };
    const rows = [0, 1, 2].map(frame => ({ frame, time_seconds: frame / 10,
        tracks: frame === 1 ? [{ track_id: 11, keypoints: [[100, 200, 0.9]] }] : [] }));
    const requests = [];
    globalThis.fetch = async url => {
        requests.push(url);
        if (url.includes('/b/')) return new Promise(resolve => { finishB = resolve; });
        return { ok: true, text: async () => rows.map(row => JSON.stringify(row)).join('\n') };
    };
    function Probe({ id = 'a', status = 'completed', enabled = true }) {
        api = useFrameReplay(id === null ? null : { id, status, metadata: { fps: 10, frame_count: 3 } }, enabled, 1); return null;
    }
    try {
        await act(async () => { renderer = TestRenderer.create(React.createElement(Probe, { id: null })); });
        assert.equal(api.active, false); assert.equal(api.snapshot, null);
        await act(async () => { renderer.update(React.createElement(Probe, { status: 'running' })); });
        assert.equal(api.active, false); assert.deepEqual(requests, []);
        await act(async () => { renderer.update(React.createElement(Probe)); });
        assert.equal(api.status, 'ready'); assert.equal(api.frame, 0);
        await act(async () => { api.seek(1); });
        assert.equal(api.snapshot.frame, 1); assert.equal(api.snapshot.tracks[0].track_id, 11);
        assert.deepEqual(api.snapshot.tracks[0].keypoints, [[100, 200, 0.9]]);
        await act(async () => { api.play(); });
        assert.equal(api.playing, true);
        await act(async () => { api.seek(0); });
        assert.equal(api.playing, false); assert.equal(timers.size, 0);
        await act(async () => { api.seek(1); api.play(); });
        assert.equal(timers.size, 0, 'playback waits for the video decoder');
        await act(async () => { api.onFrameReady(0); });
        assert.equal(timers.size, 0, 'a stale decoded frame cannot advance playback');
        await act(async () => { api.onFrameReady(1); });
        await act(async () => { [...timers.values()].at(-1)(); });
        assert.equal(api.frame, 2); assert.equal(api.playing, false);
        await act(async () => { api.play(); });
        assert.equal(api.frame, 0, 'play at the end restarts the video');
        await act(async () => { api.step(-10); });
        assert.equal(api.frame, 0); assert.equal(api.playing, false);
        await act(async () => { api.seek(999); });
        assert.equal(api.frame, 2);
        await act(async () => { renderer.update(React.createElement(Probe, { id: 'b' })); });
        assert.equal(api.snapshot, null); assert.equal(api.status, 'loading');
        await act(async () => { finishB({ ok: false, status: 500, json: async () => ({ detail: '读取失败' }) }); });
        assert.equal(api.status, 'error'); assert.equal(api.snapshot, null);
    } finally {
        await act(async () => { renderer?.unmount(); });
        globalThis.fetch = originalFetch; globalThis.window = originalWindow;
        delete globalThis.IS_REACT_ACT_ENVIRONMENT;
    }
});
