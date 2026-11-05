import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import TestRenderer, { act } from 'react-test-renderer';
import { createServer } from 'vite';

let server, Stage;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
before(async () => {
    server = await createServer({ server: { middlewareMode: true, hmr: false, watch: null },
        optimizeDeps: { noDiscovery: true, include: [] } });
    Stage = (await server.ssrLoadModule('/src/components/SpatiotemporalComputation.jsx')).default;
});
after(async () => { await server?.close(); });

function props(frame = 80, spatial = true) {
    const pose = t => Array.from({ length: 17 }, (_, joint) => [100 + joint * 2 + t + frame, 50 + joint * 8, .9]);
    const nodes = Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`lstm.${i + 1}.forecast`, {
        frame, job_id: 'job', track_ids: [7], tensors: { future: { values: [Array.from({ length: 10 }, (_, j) => pose(i * 10 + j))] } },
    }]));
    if (spatial) nodes.spatial = { frame, job_id: 'job', calculations: [{ stage: 'spatial', track_id: 7,
        inputs: [2, 3], weights: [.5, .25], actual: 1.75, input_indices: [[0, 25, 5], [1, 25, 0]],
        metadata: { block: 0, target_indices: [0, 25, 0], graph_slice: { edges: [[0, 5, 0, .5], [1, 0, 0, .25]] } } }] };
    nodes.temporal = { frame, job_id: 'job', calculations: [{ stage: 'temporal', track_id: 7,
        inputs: [2, 3, 4], weights: [1, 2, 3], actual: 20, input_indices: [[0, 24, 0], [0, 25, 0], [0, 26, 0]],
        metadata: { block: 0, target_indices: [0, 25, 0], kernel_size: 3, stride: 1, boundary: false } }] };
    return { job: { id: 'job' }, live: { job_id: 'job', frame, nodes }, track: { track_id: 7, history_length: 50 },
        structure: { stgcnBlocks: [] }, state: { block: 1, boundary: false, autoGraph: false, frozen: false, speed: 1 },
        actions: { cameraInteract() {} } };
}

const button = (root, name) => root.findAllByType('button').find(item => item.props['aria-label'] === name);

test('new live frames retain zoom while inspected coordinates reset to the new source', async () => {
    let tree;
    try {
        await act(async () => { tree = TestRenderer.create(React.createElement(Stage, props())); });
        await act(async () => { button(tree.root, '骨架图放大 2 倍').props.onClick(); });
        const joint = tree.root.findAllByType('g').find(item => item.props['aria-label'] === '检查 t=25 左肩 v5');
        await act(async () => { joint.props.onKeyDown({ key: 'Enter', preventDefault() {}, stopPropagation() {} }); });
        assert.ok(tree.root.findAllByType('button').find(item => item.props['aria-label'] === '核对骨架关节').props.children[0].props.children.includes('左肩'));
        await act(async () => { tree.update(React.createElement(Stage, props(81))); });
        assert.equal(button(tree.root, '骨架图放大 2 倍').props['aria-pressed'], true);
        const nose = tree.root.findAllByType('g').find(item => item.props['data-time'] === 24 && item.props['data-joint'] === 0);
        assert.equal(nose.props['data-source-x'], 205);
        const labels = tree.root.findAllByType('span').map(item => item.children.filter(child => typeof child === 'string').join(''));
        assert.ok(labels.some(text => text.includes('x=205.00, y=50.00')));
        const waiting = props(82);
        delete waiting.live.nodes['lstm.3.forecast'];
        await act(async () => { tree.update(React.createElement(Stage, waiting)); });
        assert.equal(button(tree.root, '骨架图放大 2 倍'), undefined);
        await act(async () => { tree.update(React.createElement(Stage, props(82))); });
        assert.equal(button(tree.root, '骨架图放大 2 倍').props['aria-pressed'], true);
    } finally { if (tree) await act(async () => tree.unmount()); }
});

test('temporal-only data requests the full-width graph layout', async () => {
    let tree;
    try {
        await act(async () => { tree = TestRenderer.create(React.createElement(Stage, props(80, false))); });
        const panels = tree.root.findAllByType('div').find(item => item.props.className?.includes('st-graph-panels'));
        assert.ok(panels.props.className.includes('is-temporal-only'));
        assert.equal(tree.root.findAllByType('section').filter(item => item.props['aria-label'] === '跨帧骨架时空图').length, 1);
    } finally { if (tree) await act(async () => tree.unmount()); }
});
