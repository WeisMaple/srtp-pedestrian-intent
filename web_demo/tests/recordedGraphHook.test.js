import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import TestRenderer, { act } from 'react-test-renderer';
import { useRecordedGraph } from '../src/hooks/useRecordedGraph.js';

const history = Array.from({ length: 50 }, () => Array.from({ length: 17 }, () => [1,2,0.9]));
const samples = { samples: [{ frame: 151, time_seconds: 15.1, tracks: [{ track_id: 11, history }] }] };
const response = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });

test('network replay follows the nearest saved source frame and sends manual choices back to the video cursor', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    let api, renderer, choice;
    globalThis.fetch = async url => {
        if (url.endsWith('/trace')) return response({ samples: [141, 151].map(frame => ({ frame,
            tracks: [{ track_id: 11, history }] })) });
        const parsed = new URL(url, 'http://localhost');
        return response({ source: { job_id: 'a', frame: Number(parsed.searchParams.get('frame')), track_id: 11 }, stgcn: { blocks: [] } });
    };
    function Probe({ frame }) { api = useRecordedGraph({ id: 'a', status: 'completed' }, true,
        { frame, trackId: 11, onSelect: (frame, trackId) => { choice = { frame, trackId }; } }); return null; }
    try {
        await act(async () => { renderer = TestRenderer.create(React.createElement(Probe, { frame: 140 })); });
        assert.equal(api.selection.frame, 141);
        assert.equal(api.result.source.frame, 141);
        await act(async () => { api.select(api.choices.find(row => row.frame === 151).key); });
        assert.deepEqual(choice, { frame: 151, trackId: 11 });
        await act(async () => { renderer.update(React.createElement(Probe, { frame: choice.frame })); });
        assert.equal(api.snapshot.frame, 151);
        assert.equal(api.result.source.frame, 151);
        await act(async () => { renderer.update(React.createElement(Probe, { frame: 0 })); });
        assert.equal(api.selection.frame, 141, 'returning to an earlier video frame must not restore the last graph selection');
    } finally {
        await act(async () => { renderer?.unmount(); });
        globalThis.fetch = originalFetch; delete globalThis.IS_REACT_ACT_ENVIRONMENT;
    }
});

test('completed empty-tail jobs load historical arithmetic and never expose another job during loading or error', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    let api, renderer, completeB;
    const requests = [];
    globalThis.fetch = async url => {
        requests.push(url);
        if (url === '/api/jobs/a/trace') return response(samples);
        if (url === '/api/jobs/b/trace') return new Promise(resolve => { completeB = resolve; });
        if (url === '/api/jobs/a/network?frame=151&track_id=11') return response({
            source: { job_id: 'a', frame: 151, track_id: 11 }, stgcn: { blocks: [] } });
        throw new Error(`Unexpected request: ${url}`);
    };
    function Probe({ job }) { api = useRecordedGraph(job, true); return null; }
    try {
        await act(async () => { renderer = TestRenderer.create(React.createElement(Probe, { job: { id: 'a', status: 'running' } })); });
        assert.equal(api.active, false); assert.deepEqual(requests, []);
        await act(async () => { renderer.update(React.createElement(Probe, { job: { id: 'a', status: 'completed', tracks: [] } })); });
        assert.equal(api.status, 'ready');
        assert.equal(api.selection.track.track_id, 11);
        assert.equal(api.selection.frame, 151);
        assert.equal(api.result.source.job_id, 'a');
        await act(async () => { renderer.update(React.createElement(Probe, { job: { id: 'b', status: 'completed', tracks: [] } })); });
        assert.equal(api.result, null);
        assert.equal(api.selection, null);
        assert.equal(api.status, 'loading');
        await act(async () => { completeB(response({ detail: '读取过程记录失败' }, 500)); });
        assert.equal(api.status, 'error');
        assert.equal(api.result, null);
        await act(async () => { api.retry(); });
        await act(async () => { completeB(response({ samples: [{ frame: 199, tracks: [] }] })); });
        assert.equal(api.status, 'empty');
        assert.equal(api.selection, null);
    } finally {
        await act(async () => { renderer?.unmount(); });
        globalThis.fetch = originalFetch;
        delete globalThis.IS_REACT_ACT_ENVIRONMENT;
    }
});
