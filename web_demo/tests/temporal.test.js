import test from 'node:test';
import assert from 'node:assert/strict';
import { latestCalculation } from '../src/domain/workbench.js';
import { temporalWindow, temporalPartialSum } from '../src/domain/temporal.js';
import { normalizeCalculation } from '../src/domain/calculation.js';

test('time windows retain channel ordering, true padding and all-channel accumulation', () => {
    const raw = { stage: 'temporal', inputs: [1,2,3,0,4,5], weights: [1,2,3,9,1,-1], bias: 0.5,
        actual: 13.5, input_indices: [[0,-1,2],[0,0,2],[0,1,2],[1,-1,2],[1,0,2],[1,1,2]],
        padding: [false,false,false,true,false,false], metadata: { kernel_size: 3, stride: 1,
            convolution_padding: 1, dilation: 1, target_indices: [0,0,2] } };
    const calculation = normalizeCalculation(raw);
    const first = temporalWindow(calculation, 0), second = temporalWindow(calculation, 1);
    assert.equal(first.sum, 14);
    assert.equal(second.sum, -1);
    assert.equal(second.offset, 3);
    assert.equal(second.terms[0].padding, true);
    assert.deepEqual(second.terms.map(term => term.inputIndex[1]), [-1,0,1]);
    assert.equal(calculation.reconstructed, 13.5);
    assert.equal(temporalWindow(normalizeCalculation({ ...raw, inputs: [1,2,3,0,null,5] }), 1).sum, null);
});

test('temporal inspection selects the requested block, person and boundary without borrowing spatial rows', () => {
    const row = (id, stage, boundary, track_id = 4) => ({ id, stage, track_id, metadata: { block: 0, boundary } });
    const snapshot = { details: { stgcn: { 0: [row('middle', 'temporal', false), row('spatial', 'spatial'),
        row('boundary', 'temporal', true), row('other', 'temporal', true, 9)] } } };
    const lookup = (operation, boundary, id = 4, block = 1) => latestCalculation(snapshot, 5, {}, 0, block, id, [], { operation, boundary });
    assert.equal(lookup('temporal', false)?.id, 'middle');
    assert.equal(lookup('temporal', true)?.id, 'boundary');
    assert.equal(lookup('spatial', false)?.id, 'spatial');
    assert.equal(lookup('temporal', true, 7), null);
    assert.equal(lookup('temporal', true, 4, 2), null);
});

test('partial accumulation preserves missing products instead of substituting zero', () => {
    const terms = [{ product: 2 }, { product: null }, { product: -1 }];
    assert.equal(temporalPartialSum(terms, 0), 0);
    assert.equal(temporalPartialSum(terms, 1), 2);
    assert.equal(temporalPartialSum(terms, 2), null);
    assert.equal(temporalPartialSum(terms, 3), null);
    assert.equal(temporalPartialSum([{ product: 0 }, { product: -1 }], 2), -1);
});
