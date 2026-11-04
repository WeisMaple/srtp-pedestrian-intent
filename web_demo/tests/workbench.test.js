import test from 'node:test';
import assert from 'node:assert/strict';
import { formatStatus, latestCalculation, uploadHint } from '../src/domain/workbench.js';

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
