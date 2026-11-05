import test from 'node:test';
import assert from 'node:assert/strict';
import { formatStatus, latestCalculation, uploadHint } from '../src/domain/workbench.js';

test('current-frame arithmetic wins over an older cached calculation', () => {
    const old = { stage: 'decoder', track_id: 4, id: 'old', actual: 1 };
    const current = { stage: 'decoder', track_id: 4, id: 'current', actual: 2 };
    const snapshot = { nodes: { 'lstm.1': { calculations: [current] } } };
    assert.equal(latestCalculation(snapshot, 4, {}, 0, 1, 4, [old]).actual, 2);
});

test('an explicitly absent pedestrian cannot expose another person\'s cached arithmetic', () => {
    const cached = ['decoder', 'spatial', 'classifier'].map(stage => ({
        id: stage, stage, track_id: 4, metadata: { block: 0 },
    }));
    for (const step of [4, 5, 6]) {
        assert.equal(latestCalculation({ nodes: {} }, step, {}, 0, 1, null, cached), null);
    }
});

test('most recently collected graph detail is used for the selected pedestrian', () => {
    const snapshot = { details: { stgcn: { 0: [
        { stage: 'spatial', track_id: 4, id: 'old', actual: 1 },
        { stage: 'spatial', track_id: 9, id: 'other', actual: 9 },
        { stage: 'spatial', track_id: 4, id: 'new', actual: 2 },
    ] } } };
    assert.equal(latestCalculation(snapshot, 5, {}, 0, 1, 4).actual, 2);
});

test('status labels distinguish ready, busy and terminal jobs', () => {
    assert.equal(formatStatus({ status: 'ready' }), '等待开始');
    assert.equal(formatStatus({ status: 'running', progress: 0.4 }), '正在分析 40%');
    assert.equal(formatStatus({ status: 'completed' }), '分析完成');
    assert.equal(formatStatus({ status: 'failed', error: '坏了' }), '分析失败 · 坏了');
});

test('upload hint reflects file and model readiness', () => {
    assert.equal(uploadHint(null, true), '拖入一段视频，或点击选择文件');
    assert.equal(uploadHint(null, false), '模型未就绪，请先检查权重与 ffmpeg');
    assert.equal(uploadHint({ name: 'a.mp4' }, true), 'a.mp4 · 可以开始分析');
});

test('calculation lookup stays on the selected module and block', () => {
    const structure = { yoloModules: [{ index: 0 }, { index: 23 }], stgcnBlocks: Array.from({ length: 10 }, (_, i) => ({ index: i + 1 })) };
    const yoloRows = [{ stage: 'yolo', module: 23, id: 'yolo.conv.23' }];
    const stgcnRows = [{ stage: 'spatial', metadata: { block: 9 }, id: 'stgcn.spatial.block9' }];
    const snapshot = {
        nodes: {
            'yolo.23': { calculations: yoloRows },
            'flow.stgcn.spatial.block9': { calculations: stgcnRows },
            'flow.stgcn.classifier': { calculations: [{ stage: 'classifier', id: 'classifier' }] },
        },
        details: {
            yolo: { 0: { id: 'yolo.conv.0' } },
            stgcn: { 0: [{ id: 'stgcn.block0' }], 9: [{ id: 'stgcn.block9' }] },
        },
    };
    assert.equal(latestCalculation(snapshot, 2, structure, 0, 1).id, 'yolo.conv.0');
    assert.equal(latestCalculation(snapshot, 2, structure, 23, 1).id, 'yolo.conv.23');
    assert.equal(latestCalculation(snapshot, 5, structure, 0, 10).id, 'stgcn.block9');
    assert.equal(latestCalculation(snapshot, 5, structure, 0, 1).id, 'stgcn.block0');
    assert.equal(latestCalculation(snapshot, 6, structure, 0, 1).id, 'classifier');
    assert.equal(latestCalculation(snapshot, 2, null, 0, 1), null);
});

test('LSTM step prefers the decoder row and stays on the selected pedestrian', () => {
    const structure = { yoloModules: [], stgcnBlocks: [] };
    const snapshot = {
        nodes: {
            'lstm.1': { calculations: [
                { stage: 'normalization', track_id: 4, id: 'normalize-4' },
                { stage: 'decoder', track_id: 4, id: 'decoder-4' },
                { stage: 'decoder', track_id: 9, id: 'decoder-9' },
            ] },
        },
        details: { yolo: {}, stgcn: {} },
    };
    assert.equal(latestCalculation(snapshot, 4, structure, 0, 1, 4).id, 'decoder-4');
    assert.equal(latestCalculation(snapshot, 4, structure, 0, 1, 9).id, 'decoder-9');
    assert.equal(latestCalculation(snapshot, 4, structure, 0, 1, 7), null);
});

test('cached event calculations survive frame resets for the selected pedestrian', () => {
    const structure = { yoloModules: [], stgcnBlocks: [] };
    const snapshot = { nodes: { 'pipeline.frame': { calculations: [] } }, tracks: [], details: { yolo: {}, stgcn: {} } };
    const cached = [
        { stage: 'decoder', track_id: 2, id: 'cached-decoder-2' },
        { stage: 'classifier', track_id: 2, id: 'cached-classifier-2' },
        { stage: 'classifier', track_id: 5, id: 'cached-classifier-5' },
    ];
    assert.equal(latestCalculation(snapshot, 4, structure, 0, 1, 2, cached).id, 'cached-decoder-2');
    assert.equal(latestCalculation(snapshot, 6, structure, 0, 1, 2, cached).id, 'cached-classifier-2');
    assert.equal(latestCalculation(snapshot, 6, structure, 0, 1, 5, cached).id, 'cached-classifier-5');
    assert.equal(latestCalculation(snapshot, 6, structure, 0, 1, 9, cached), null);
});

test('per-track calculations never borrow another pedestrian', () => {
    const structure = { yoloModules: [{ index: 0 }], stgcnBlocks: [{ index: 1 }] };
    const snapshot = {
        nodes: {},
        details: { yolo: {}, stgcn: { 0: [{ track_id: 3, id: 'track-3' }] } },
    };
    assert.equal(latestCalculation(snapshot, 5, structure, 0, 1, 7), null);
    assert.equal(latestCalculation(snapshot, 5, structure, 0, 1, 3).id, 'track-3');
    assert.equal(latestCalculation(snapshot, 5, structure, 0, 1).id, 'track-3');
});
