import test from 'node:test';
import assert from 'node:assert/strict';
import { spatiotemporalModel, temporalMapping, skeletonLayout, currentGraphRows, nearestJoint } from '../src/domain/spatiotemporal.js';
import { recordedGraphDetails } from '../src/domain/temporal.js';

const pose = t => Array.from({ length: 17 }, (_, j) => [100 + j * 2 + t, 50 + j * 8, .9]);
const blocks = Array.from({ length: 10 }, (_, i) => ({ index: i + 1, stride: [i === 4 || i === 7 ? 2 : 1, 1], temporal_kernel: [9, 1] }));
function fixture() {
    const spatial = { stage: 'spatial', track_id: 7, inputs: [2, 3], weights: [.5, .25], bias: 0, actual: 1.75,
        input_indices: [[0, 0, 25, 5], [1, 0, 25, 0]], metadata: { block: 0, target_indices: [0, 25, 0],
            graph_slice: { edges: [[0, 5, 0, .5], [1, 0, 0, .25]] } } };
    const temporal = { stage: 'temporal', track_id: 7, inputs: [2, 3, 4], weights: [1, 2, 3], bias: .5, actual: 20.5,
        input_indices: [[0, 24, 0], [0, 25, 0], [0, 26, 0]], metadata: { block: 0, target_indices: [0, 25, 0],
            kernel_size: 3, stride: 1, convolution_padding: 1, dilation: 1, boundary: false } };
    const nodes = Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`lstm.${i + 1}.forecast`, {
        frame: 80, job_id: 'job', track_ids: [7], tensors: { future: { values: [Array.from({ length: 10 }, (_, j) => pose(i * 10 + j))] } },
    }]));
    nodes.spatial = { frame: 80, job_id: 'job', calculations: [spatial] };
    nodes.temporal = { frame: 80, job_id: 'job', calculations: [temporal] };
    return { snapshot: { job_id: 'job', frame: 80, nodes }, jobId: 'job', track: { track_id: 7, history: Array.from({ length: 50 }, (_, t) => pose(t - 50)) }, block: 1, structure: { stgcnBlocks: blocks } };
}

test('both dimensions use the same real forecast and actual graph weights', () => {
    const model = spatiotemporalModel(fixture());
    assert.equal(model.forecasts.length, 50);
    assert.equal(model.history.length, 50);
    assert.equal(model.spatial.actual, 1.75);
    assert.equal(model.temporal.actual, 20.5);
    assert.deepEqual(model.edges, [[0, 5, 0, .5], [1, 0, 0, .25]]);
    assert.deepEqual(model.slots.map(s => s.time), [24, 25, 26]);
    assert.deepEqual(model.slots[1].pose, pose(25));
});

test('stale frames, wrong people and missing forecast rounds never borrow data', () => {
    const input = fixture();
    input.snapshot.nodes.spatial.frame = 79;
    input.snapshot.details = { stgcn: { 0: [input.snapshot.nodes.spatial.calculations[0]] } };
    assert.equal(spatiotemporalModel(input).spatial, null);
    input.snapshot.nodes.temporal.calculations[0].track_id = 8;
    assert.equal(spatiotemporalModel(input).temporal, null);
    input.snapshot.nodes['lstm.3.forecast'].frame = 79;
    assert.equal(spatiotemporalModel(input).forecasts.length, 0);
});

test('deep feature time is mapped through actual preceding strides and receptive fields', () => {
    assert.deepEqual(temporalMapping(blocks, 1), { jump: 1, radius: 0 });
    assert.deepEqual(temporalMapping(blocks, 8), { jump: 2, radius: 36 });
    assert.deepEqual(temporalMapping(blocks, 9), { jump: 4, radius: 44 });
    assert.equal(temporalMapping([], 4), null);
});

test('padding has no skeleton; spatial and temporal representative positions remain distinct', () => {
    const input = fixture(); input.boundary = true;
    const row = input.snapshot.nodes.temporal.calculations[0];
    row.metadata.boundary = true; row.metadata.target_indices = [0, 0, 0];
    row.input_indices = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]; row.padding = [true, false, false];
    const model = spatiotemporalModel(input);
    assert.equal(model.slots[0].padding, true);
    assert.equal(model.slots[0].pose, null);
    assert.deepEqual(model.slots.find(s => s.time === 0).pose, pose(0));
    assert.equal(model.slots.at(-1).time, 25);
    assert.equal(model.slots.at(-1).spatial, true);
});

test('record adapters preserve graph targets and reject mismatched source records', () => {
    const input = fixture();
    const result = { source: { job_id: 'job', frame: 80, track_id: 7 }, future: { values: Array.from({ length: 50 }, (_, t) => pose(t)) },
        stgcn: { adjacency: [[[1, 0], [0, 1]]], blocks: [{ index: 1, input_shape: [1, 3, 50, 17], effective_adjacency: [[[1, 0], [0, 2]]],
            arithmetic: { spatial: { output_channel: 0, output_joint: 1, timestep: 25, projected_values: [[2, 3]], adjacency: [[0, 2]], actual: 6 } } }] } };
    const row = recordedGraphDetails(result).stgcn[0][0];
    assert.deepEqual(row.metadata.target_indices, [0, 25, 1]);
    assert.deepEqual(row.metadata.graph_slice.edges, [[0, 0, 0, 1], [0, 1, 1, 2]]);
    assert.equal(spatiotemporalModel({ ...input, result }).spatial.actual, 6);
    result.source.frame = 79;
    const invalid = spatiotemporalModel({ ...input, result });
    assert.equal(invalid.spatial, null);
    assert.equal(invalid.forecasts.length, 0);
});

test('skeleton layout keeps a common scale and leaves padding empty', () => {
    const slots = skeletonLayout([{ pose: pose(0) }, { pose: pose(10) }, { padding: true, pose: null }]);
    assert.equal(slots[2].points.length, 0);
    assert.equal(slots[0].points.length, 17);
    assert.ok(slots[1].points[0][0] > slots[0].points[0][0]);
    assert.equal(slots[1].points[0][1], slots[0].points[0][1]);
    assert.ok(slots[1].x > slots[0].x);
});

test('flat pose tensors are converted to joint triples before placing points', () => {
    const input = fixture();
    for (const event of Object.values(input.snapshot.nodes)) {
        if (event.tensors?.future) event.tensors.future.values[0] = event.tensors.future.values[0].map(frame => frame.flat());
    }
    const model = spatiotemporalModel(input);
    assert.deepEqual(model.forecasts[25], pose(25));
    assert.ok(skeletonLayout(model.slots).every(slot => slot.points.length === 17 && slot.points.flat().every(Number.isFinite)));
});

test('wide poses remain inside their time column instead of overlapping the next frame', () => {
    const wide = Array.from({ length: 17 }, (_, joint) => [joint * 30, joint * 2, .9]);
    const slots = skeletonLayout([{ pose: wide, time: 0 }, { pose: wide, time: 1 }]);
    const right = Math.max(...slots[0].points.map(point => point[0] + slots[0].x));
    const left = Math.min(...slots[1].points.map(point => point[0] + slots[1].x));
    assert.ok(left - right >= .3, `separation was ${left - right}`);
});

test('unreliable forecast head points do not distort or connect the displayed skeleton', () => {
    // Real head score pattern from walk sample, source frame 141 / ID 11 / forecast 25.
    const sample = pose(25);
    sample[0] = [500.5823, 281.5007, .390];
    sample[1] = [499.1864, 279.4518, .438];
    sample[2] = [503.5977, 281.3596, -.030];
    sample[3] = [508.0259, 281.8316, .963];
    sample[4] = [517.6093, 280.6259, .058];
    const slot = skeletonLayout([{ time: 25, pose: sample }])[0];
    assert.deepEqual(slot.visible.slice(0, 5), [true, true, false, true, false]);
    const raw = skeletonLayout([{ time: 25, pose: sample }], { showLowScores: true })[0];
    assert.deepEqual(raw.visible.slice(0, 5), [true, true, true, true, true]);
    assert.deepEqual(sample[2], [503.5977, 281.3596, -.030], 'display filtering must not modify model inputs');
    const outlier = pose(0); outlier[4] = [-100000, -100000, .01];
    const clean = skeletonLayout([{ pose: outlier }])[0];
    const visibleYs = clean.points.filter((_, i) => clean.visible[i]).map(p => p[1]);
    assert.ok(Math.max(...visibleYs) - Math.min(...visibleYs) > 3, 'invalid head outliers must not shrink the whole person');
});

test('dense facial joints select the nearest actual coordinate, independent of draw order', () => {
    const points = [[30, 31], [27, 28], [32, 28], [23, 30], [36, 30]];
    points.forEach(([x, y], joint) => assert.equal(nearestJoint(points, [x, y]), joint));
    assert.equal(nearestJoint(points, [27.5, 28]), 1);
    assert.equal(nearestJoint([[0, 0], [.1, 0]], [.1, 0], [true, false]), 0);
    assert.equal(nearestJoint([], [0, 0]), null);
});

test('automatic block choices discard retained details from older frames and other jobs', () => {
    const { snapshot } = fixture();
    const row = snapshot.nodes.temporal.calculations[0];
    snapshot.details = { stgcn: { 7: [{ ...row, source: { frame: 79, job_id: 'job' } },
        { ...row, source: { frame: 80, job_id: 'foreign' } }] } };
    snapshot.nodes.stale = { frame: 79, calculations: [row] };
    assert.equal(currentGraphRows(snapshot).length, 2);
    snapshot.nodes.foreign = { frame: 80, job_id: 'foreign', calculations: [row] };
    assert.equal(currentGraphRows(snapshot).length, 2);
    assert.deepEqual(currentGraphRows(null), []);
});

test('reconnect snapshots recover current detail-only arithmetic without using retained old frames', () => {
    const input = fixture(), snapshot = input.snapshot;
    snapshot.details = { stgcn: { 0: ['spatial', 'temporal'].map(key => ({
        ...snapshot.nodes[key].calculations[0], source: { job_id: 'job', frame: 80, kind: 'live_forward' },
    })) } };
    delete snapshot.nodes.spatial; delete snapshot.nodes.temporal;
    assert.equal(spatiotemporalModel(input).temporal.actual, 20.5);
    assert.equal(spatiotemporalModel(input).spatial.actual, 1.75);
    snapshot.details.stgcn[0][0].source.frame = 79;
    assert.equal(spatiotemporalModel(input).spatial, null);
});
