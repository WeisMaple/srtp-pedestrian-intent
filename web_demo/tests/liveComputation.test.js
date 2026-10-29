import test from 'node:test';
import assert from 'node:assert/strict';
import { createLiveState, reduceLiveEvent, setLiveFrozen, visibleLiveState, normalizeLiveNodes } from '../src/liveComputation.js';

const event = (seq, frame, node = 'yolo.0', extra = {}) => ({
    seq, job_id: 'job-a', frame, time_seconds: frame / 25, model: 'yolo', node,
    stage: 1, track_ids: [], captured_at_ms: 123, duration_ms: 2, tensors: {}, ...extra,
});
const tensor = (axes, shape, indices, values) => ({ axes, shape, sample_shape: indices.map(axis => axis.length), indices, values });

test('ordered events ignore duplicates, old sequence IDs, and other jobs', () => {
    let state = reduceLiveEvent(createLiveState('job-a'), 'computation', event(2, 3), 1000);
    for (const row of [event(2, 3), event(1, 2), { ...event(3, 3), job_id: 'job-b' }]) {
        assert.equal(reduceLiveEvent(state, 'computation', row, 1100), state);
    }
    assert.equal(state.latestSeq, 2);
    assert.equal(state.receivedAt, 1000);
});

test('a new captured frame removes all previous frame nodes and tracks', () => {
    let state = reduceLiveEvent(createLiveState('job-a'), 'snapshot', {
        seq: 4, job_id: 'job-a', frame: 8, nodes: { 'lstm.1': event(4, 8, 'lstm.1') }, tracks: [{ track_id: 7 }],
    }, 1000);
    state = reduceLiveEvent(state, 'computation', event(5, 9, 'pipeline.frame'), 1100);
    assert.deepEqual(Object.keys(state.snapshot.nodes), ['pipeline.frame']);
    assert.deepEqual(state.snapshot.tracks, []);
    assert.equal(state.snapshot.frame, 9);
});

test('a delayed older frame cannot replace a newer displayed frame even with a newer sequence', () => {
    const state = reduceLiveEvent(createLiveState('job-a'), 'computation', event(5, 9), 1000);
    assert.equal(reduceLiveEvent(state, 'computation', event(6, 8), 1100), state);
});

test('gap discards incomplete nodes and accepts the recovery snapshot at the same sequence', () => {
    let state = reduceLiveEvent(createLiveState('job-a'), 'computation', event(2, 1), 1000);
    state = reduceLiveEvent(state, 'gap', { seq: 20, job_id: 'job-a', reset: true, dropped_events: 17 }, 1200);
    assert.equal(state.snapshot, null);
    state = reduceLiveEvent(state, 'snapshot', { seq: 20, job_id: 'job-a', frame: 4, nodes: { 'yolo.1': event(19, 4, 'yolo.1') }, tracks: [] }, 1300);
    assert.equal(state.snapshot.nodes['yolo.1'].seq, 19);
    assert.equal(state.latestSeq, 20);
    assert.equal(state.awaitingSnapshot, false);
});

test('freeze holds nodes, latest event, activity and receive time while ingestion continues', () => {
    let state = reduceLiveEvent(createLiveState('job-a'), 'computation', event(1, 1), 1000);
    state = setLiveFrozen(state, true);
    state = reduceLiveEvent(state, 'computation', event(2, 2), 1100);
    const held = visibleLiveState(state);
    assert.equal(held.snapshot.frame, 1);
    assert.equal(held.latestEvent.seq, 1);
    assert.equal(held.events.length, 1);
    assert.equal(held.receivedAt, 1000);
    assert.equal(held.latestSeq, 2);
    state = setLiveFrozen(state, false);
    assert.equal(visibleLiveState(state).snapshot.frame, 2);
});

test('freezing during a pending React flush holds the frame the user is viewing', () => {
    const displayed = reduceLiveEvent(createLiveState('job-a'), 'computation', event(1, 1), 1000);
    const ingested = reduceLiveEvent(displayed, 'computation', event(2, 2), 1050);
    const frozen = visibleLiveState(setLiveFrozen(ingested, true, displayed));
    assert.equal(frozen.snapshot.frame, 1);
    assert.equal(frozen.latestSeq, 2);
});

test('terminal retains the last nodes even when terminal and snapshot share a sequence ID', () => {
    let state = reduceLiveEvent(createLiveState('job-a'), 'snapshot', {
        seq: 5, job_id: 'job-a', status: 'completed', frame: 4, nodes: { 'yolo.0': event(4, 4) }, tracks: [],
    }, 1000);
    state = reduceLiveEvent(state, 'terminal', { seq: 5, job_id: 'job-a', status: 'completed', frame: 4 }, 1100);
    assert.equal(state.connection, 'complete');
    assert.equal(state.snapshot.nodes['yolo.0'].seq, 4);
    assert.equal(reduceLiveEvent(state, 'computation', event(6, 5), 1200), state);
});

test('explicit gap recovery can reset a stale cursor after server buffer replacement', () => {
    let state = reduceLiveEvent(createLiveState('job-a'), 'computation', event(100, 99), 1000);
    state = reduceLiveEvent(state, 'gap', { job_id: 'job-a', seq: 5, requested_after: 100, reset: true }, 1200);
    state = reduceLiveEvent(state, 'snapshot', { job_id: 'job-a', seq: 5, frame: 4, nodes: { 'yolo.0': event(5, 4) } }, 1300);
    assert.equal(state.latestSeq, 5);
    assert.equal(state.snapshot.frame, 4);
});

test('activity history is bounded without losing the current reduced frame', () => {
    let state = createLiveState('job-a');
    for (let seq = 1; seq <= 90; seq++) state = reduceLiveEvent(state, 'computation', event(seq, seq), seq * 100);
    assert.equal(state.events.length, 60);
    assert.equal(state.events[0].seq, 31);
    assert.equal(state.snapshot.frame, 90);
});

test('normalization selects the explicit person axis and preserves actual sampled neuron indices', () => {
    const recurrent = event(3, 4, 'lstm.2', { model: 'lstm', track_ids: [7, 42], step: 2, tensors: {
        hidden: tensor(['layer', 'track', 'hidden'], [2, 2, 256], [[0, 1], [0, 1], [0, 15]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        cell: tensor(['layer', 'track', 'hidden'], [2, 2, 256], [[0, 1], [0, 1], [0, 15]], [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]),
        sequence: tensor(['track', 'time', 'hidden'], [2, 10, 256], [[0, 1], [0, 9], [0, 15]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    } });
    const nodes = normalizeLiveNodes({ frame: 4, nodes: { 'lstm.2': recurrent }, tracks: [] }, 42);
    assert.deepEqual(nodes['lstm.layer.1'].values, [7, 8]);
    assert.deepEqual(nodes['lstm.layer.1'].cell, [70, 80]);
    assert.deepEqual(nodes['lstm.layer.1'].shape, [256]);
    assert.deepEqual(nodes['lstm.layer.1'].indices, [[0, 15]]);
    assert.equal(nodes['lstm.layer.1'].round, 2);
    assert.equal(nodes['lstm.layer.1'].track_id, 42);
    assert.deepEqual(nodes['lstm.sequence'].values, [[5, 6], [7, 8]]);
    assert.match(nodes['lstm.layer.1'].source, /^实际推理/);
    assert.deepEqual(normalizeLiveNodes({ frame: 4, nodes: { 'lstm.2': recurrent } }, 99), {});
});

test('a representative first-round calculation selects that round instead of later LSTM states',()=>{
    const recurrent=round=>event(round,4,`lstm.${round}`,{track_ids:[42],step:round,tensors:{
        hidden:tensor(['layer','track','hidden'],[1,1,256],[[0],[0],[0]],[[[round]]]),
    }});
    const snapshot={frame:4,nodes:{'lstm.1':recurrent(1),'lstm.5':recurrent(5)}};
    assert.deepEqual(normalizeLiveNodes(snapshot,42)['lstm.layer.0'].values,[5]);
    assert.deepEqual(normalizeLiveNodes(snapshot,42,{round:1})['lstm.layer.0'].values,[1]);
    assert.equal(normalizeLiveNodes(snapshot,42,{round:2})['lstm.layer.0'],undefined);
    assert.equal(normalizeLiveNodes(snapshot,7,{round:1})['lstm.layer.0'],undefined);
});

test('spatial tensors use their recorded time index and YOLO remains available for any selected person', () => {
    const nodes = normalizeLiveNodes({ frame: 4, nodes: {
        'yolo.0': event(1, 4, 'yolo.0', { tensors: { output: tensor(['batch', 'channel', 'y', 'x'], [1, 64, 8, 8], [[0], [2], [1], [4, 7]], [[[[2, 3]]]]) } }),
        'stgcn.0.spatial': event(2, 4, 'stgcn.0.spatial', { model: 'stgcn', track_ids: [42], tensors: { output: tensor(['track', 'channel', 'time', 'joint'], [1, 64, 50, 17], [[0], [1, 5], [49], [0, 16]], [[[[11, 12]], [[13, 14]]]]) } }),
        'lstm.1': event(3, 3, 'lstm.1', { model: 'lstm', track_ids: [42] }),
    } }, 42);
    assert.deepEqual(nodes['yolo.0'].values, [[[2, 3]]]);
    assert.deepEqual(nodes['yolo.0'].indices, [[2], [1], [4, 7]]);
    assert.deepEqual(nodes['stgcn.0.spatial'].values, [[11, 12], [13, 14]]);
    assert.deepEqual(nodes['stgcn.0.spatial'].indices, [[1, 5], [0, 16]]);
    assert.equal(nodes['stgcn.0.spatial'].timestep, 49);
    assert.equal(nodes['lstm.layer.0'], undefined);
});

test('forecast blocks concatenate only contiguous rounds for the selected person in the same frame', () => {
    const forecast = (round, id, values) => event(round + 1, 4, `lstm.${round}.forecast`, { model: 'lstm', track_ids: [id], step: round, horizon_start: (round - 1) * 2 + 1,
        tensors: { future: tensor(['track', 'time', 'joint', 'coordinate'], [1, 2, 1, 3], [[0], [0, 1], [0], [0, 1, 2]], [values]) } });
    const snapshot = { frame: 4, nodes: {
        'lstm.1.forecast': forecast(1, 42, [[[1, 2, 1]], [[3, 4, 1]]]),
        'lstm.2.forecast': forecast(2, 42, [[[5, 6, 1]], [[7, 8, 1]]]),
        'lstm.3.forecast': forecast(3, 7, [[[9, 10, 1]], [[11, 12, 1]]]),
    } };
    const future = normalizeLiveNodes(snapshot, 42).future;
    assert.equal(future.values.length, 4);
    assert.deepEqual(future.values[3], [[7, 8, 1]]);
    assert.deepEqual(future.indices[0], [1, 2, 3, 4]);
    assert.equal(future.track_id, 42);
});

test('probabilities require an actual current-frame prediction for the selected track', () => {
    const snapshot = { frame: 4, nodes: { 'pipeline.output': event(2, 4, 'pipeline.output') }, tracks: [
        { track_id: 42, probabilities: [0.1, 0.2, 0.7], prediction_frame: 4, prediction_updated: true },
        { track_id: 7, probabilities: [0.8, 0.1, 0.1], prediction_frame: 2, prediction_updated: false },
    ] };
    assert.deepEqual(normalizeLiveNodes(snapshot, 42)['stgcn.probabilities'].values, [0.1, 0.2, 0.7]);
    assert.equal(normalizeLiveNodes(snapshot, 7)['stgcn.probabilities'], undefined);
});
