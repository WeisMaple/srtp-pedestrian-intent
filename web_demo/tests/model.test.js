import test from 'node:test';
import assert from 'node:assert/strict';
import { detailCoverage, detailForBlock, detailForModule, normalizeModelStructure } from '../src/domain/model.js';

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

test('reads latest detail rows from a live snapshot', () => {
    const snapshot = { details: { yolo: { 4: { id: 'yolo.conv.4', module: 4 } }, stgcn: { 2: [{ id: 's' }] } } };
    assert.equal(detailForModule(snapshot, 4).id, 'yolo.conv.4');
    assert.equal(detailForModule(snapshot, 5), null);
    assert.equal(detailForBlock(snapshot, 2).length, 1);
    assert.deepEqual(detailCoverage(snapshot), { yolo: [4], stgcn: [2] });
});
