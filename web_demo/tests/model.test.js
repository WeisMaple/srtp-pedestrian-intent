import test from 'node:test';
import assert from 'node:assert/strict';
import { mergeDetail, normalizeModelStructure } from '../src/domain/model.js';

const raw = {
    yolo: { modules: [
        { index: 0, type: 'Conv', from: [-1], representative: { kernel: [3, 3], in_channels: 3, out_channels: 32 } },
        { index: 1, type: 'C3k2', from: [0], representative: null },
    ] },
    stgcn: { blocks: [{ index: 1, out_channels: 64 }, { index: 2, out_channels: 64 }], classes: ['a', 'b', 'c'], labels: ['其他', '不过街', '过街'] },
    versions: { stgcn_epoch: 121 },
};

test('normalizes module and block lookups', () => {
    const model = normalizeModelStructure(raw);
    assert.equal(model.yoloById.get(0).type, 'Conv');
    assert.equal(model.stgcnBlocks.length, 2);
    assert.deepEqual(model.labels, ['其他', '不过街', '过街']);
    assert.equal(model.versions.stgcn_epoch, 121);
});

test('merges only same-frame detail payloads', () => {
    const snapshot = { frame: 7, details: { yolo: {}, stgcn: {} } };
    const payload = { frame: 7, calculations: [
        { stage: 'yolo', module: 4, metadata: { detail: true } },
        { stage: 'stgcn', metadata: { detail: true, detail_block: 2 } },
        { stage: 'stgcn', metadata: { detail: false, detail_block: 3 } },
    ] };
    const details = mergeDetail(snapshot, payload);
    assert.equal(details.yolo[4].module, 4);
    assert.deepEqual(details.stgcn[2].map(row => row.stage), ['stgcn']);
    assert.equal(details.stgcn[3], undefined);
    assert.deepEqual(mergeDetail(snapshot, { frame: 8, calculations: payload.calculations }), snapshot.details);
});
