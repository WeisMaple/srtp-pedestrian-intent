const ACTIVITY_LIMIT = 60;
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

export function createLiveState(jobId = null) {
    return { jobId, snapshot: null, latestEvent: null, events: [], latestSeq: -1,
        receivedAt: null, connection: 'idle', error: '', frozen: false, held: null,
        awaitingSnapshot: false };
}

function appendActivity(events, event) {
    return [...events.slice(-(ACTIVITY_LIMIT - 1)), event];
}

function currentNodes(nodes, frame, jobId, receivedAt) {
    return Object.fromEntries(Object.entries(nodes || {}).filter(([, event]) =>
        event && event.frame === frame && (!event.job_id || event.job_id === jobId)
    ).map(([key, event]) => [key, { ...event, received_at_ms: receivedAt }]));
}

/** Reduce every event before the React publication throttle; no model call occurs here. */
export function reduceLiveEvent(state, type, payload, receivedAt = Date.now()) {
    if (!payload || payload.job_id !== state.jobId || !Number.isInteger(payload.seq) || payload.seq < 0) return state;
    if (type === 'computation' && (TERMINAL_STATUSES.has(state.snapshot?.status)
        || typeof payload.node !== 'string' || !Number.isInteger(payload.frame)
        || (state.snapshot && payload.frame < state.snapshot.frame))) return state;
    const seq = payload.seq;
    const resetCursor = type === 'gap' && payload.reset && payload.requested_after === state.latestSeq;
    if (seq < state.latestSeq && !resetCursor) return state;
    if (seq === state.latestSeq && type === 'computation') return state;
    if (type === 'snapshot' && seq === state.latestSeq && state.snapshot && !state.awaitingSnapshot) return state;
    if (type === 'terminal' && state.connection === 'complete') return state;
    if (!['snapshot', 'computation', 'gap', 'terminal'].includes(type)) return state;

    const row = { ...payload, type, received_at_ms: receivedAt };
    const next = { ...state, latestSeq: seq, receivedAt, error: '',
        connection: 'connected', events: appendActivity(state.events, row) };
    if (type === 'gap') {
        return { ...next, snapshot: null, latestEvent: null, awaitingSnapshot: true };
    }
    if (type === 'snapshot') {
        const nodes = currentNodes(payload.nodes, payload.frame, state.jobId, receivedAt);
        const latestEvent = Object.values(nodes).reduce((latest, node) =>
            !latest || node.seq > latest.seq ? node : latest, null);
        return { ...next, snapshot: { ...payload, nodes, tracks: payload.tracks || [] },
            latestEvent, awaitingSnapshot: false };
    }
    if (type === 'terminal') {
        return { ...next, connection: 'complete', snapshot: state.snapshot ? {
            ...state.snapshot, seq, status: payload.status, dropped: payload.dropped || state.snapshot.dropped,
        } : { ...payload, nodes: {}, tracks: [] } };
    }
    // A missed frame boundary is also safe: the first event for the next frame resets nodes.
    const sameFrame = state.snapshot?.frame === payload.frame;
    const previous = sameFrame ? state.snapshot : {};
    return { ...next, latestEvent: row, awaitingSnapshot: false, snapshot: {
        ...previous, seq, job_id: state.jobId, status: previous.status || 'running',
        frame: payload.frame, time_seconds: payload.time_seconds,
        captured_at_ms: payload.captured_at_ms,
        nodes: { ...(previous.nodes || {}), [payload.node]: row },
        tracks: payload.tracks || previous.tracks || [],
        dropped: { ...(state.snapshot?.dropped || {}), ...(payload.dropped || {}) },
        sampling: payload.sampling || state.snapshot?.sampling,
    } };
}

export function visibleLiveState(state) {
    return state.frozen && state.held ? { ...state, ...state.held } : state;
}

export function setLiveFrozen(state, frozen, displayed = state) {
    if (state.frozen === frozen) return state;
    return { ...state, frozen, held: frozen ? {
        snapshot: displayed.snapshot, latestEvent: displayed.latestEvent,
        events: displayed.events, receivedAt: displayed.receivedAt,
    } : null };
}

function sameTrack(left, right) {
    return left !== null && left !== undefined && right !== null && right !== undefined && String(left) === String(right);
}

function validTensor(tensor) {
    return tensor && Array.isArray(tensor.axes) && Array.isArray(tensor.shape)
        && Array.isArray(tensor.indices) && Array.isArray(tensor.values)
        && tensor.axes.length === tensor.shape.length && tensor.axes.length === tensor.indices.length;
}

function valueAtAxis(values, axis, index) {
    if (!Array.isArray(values)) return undefined;
    return axis === 0 ? values[index] : values.map(value => valueAtAxis(value, axis - 1, index));
}

function sliceAxis(tensor, name, sampleIndex) {
    const axis = tensor.axes.indexOf(name);
    if (axis < 0 || !tensor.indices[axis]?.length || sampleIndex < 0 || sampleIndex >= tensor.indices[axis].length) return null;
    const without = list => list.filter((_, index) => index !== axis);
    return { ...tensor, values: valueAtAxis(tensor.values, axis, sampleIndex),
        axes: without(tensor.axes), shape: without(tensor.shape),
        sample_shape: without(tensor.sample_shape || tensor.indices.map(values => values.length)),
        indices: without(tensor.indices),
        selection: { ...(tensor.selection || {}), [name]: tensor.indices[axis][sampleIndex] } };
}

function selectTrack(tensor, event, selectedId) {
    if (!validTensor(tensor) || !tensor.axes.includes('track')) return null;
    const position = event.track_ids?.findIndex(id => sameTrack(id, selectedId)) ?? -1;
    return position < 0 ? null : sliceAxis(tensor, 'track', position);
}

function descriptor(event, original, selected, label, selectedId) {
    if (!selected) return null;
    return { ...selected, source: `实际推理 · ${label}`, label,
        source_shape: original.shape, source_axes: original.axes, source_indices: original.indices,
        frame: event.frame, seq: event.seq, time_seconds: event.time_seconds,
        captured_at_ms: event.captured_at_ms, received_at_ms: event.received_at_ms,
        duration_ms: event.duration_ms, timing: event.timing, node: event.node,
        coordinate_system: event.coordinate_system, track_id: selectedId,
        model: event.model, stage: event.stage };
}

/** Adapt recorded axes only; absent people or tensors intentionally produce no node. */
export function normalizeLiveNodes(snapshot, selectedId, {round:requestedRound} = {}) {
    if (!snapshot?.nodes) return {};
    const result = {};
    const events = Object.values(snapshot.nodes).filter(event => event?.frame === snapshot.frame)
        .sort((a, b) => a.seq - b.seq);
    let latestRecurrent = null;
    const forecasts = [];
    for (const event of events) {
        const tensors = event.tensors || {};
        if (/^yolo\.\d+$/.test(event.node) && validTensor(tensors.output)) {
            const selected = sliceAxis(tensors.output, 'batch', 0);
            if (selected) result[event.node] = descriptor(event, tensors.output, selected, `${event.node} 输出特征`, null);
        } else if (/^lstm\.\d+$/.test(event.node)) {
            if (event.track_ids?.some(id => sameTrack(id, selectedId))
                && (requestedRound==null || (event.step??Number(event.node.split('.')[1]))===requestedRound)) latestRecurrent = event;
        } else if (/^stgcn\.\d+\.(spatial|temporal|output)$/.test(event.node)) {
            let selected = selectTrack(tensors.output, event, selectedId);
            if (selected) {
                const timeAxis = selected.axes.indexOf('time');
                if (timeAxis < 0) continue;
                const timeIndex = Math.floor(selected.indices[timeAxis].length / 2);
                selected = sliceAxis(selected, 'time', timeIndex);
                result[event.node] = { ...descriptor(event, tensors.output, selected, `${event.node} 特征`, selectedId),
                    timestep: selected.selection.time };
            }
        } else if (event.node === 'stgcn.pool' || event.node === 'stgcn.logits') {
            let selected = selectTrack(tensors.output, event, selectedId);
            for (const axis of ['y', 'x']) {
                const index = selected?.axes.indexOf(axis) ?? -1;
                if (index >= 0 && selected.shape[index] === 1) selected = sliceAxis(selected, axis, 0);
            }
            if (selected) result[event.node] = descriptor(event, tensors.output, selected, event.node === 'stgcn.pool' ? '全局池化输出' : '分类 logits', selectedId);
        }
        if (tensors.history) {
            const selected = selectTrack(tensors.history, event, selectedId);
            if (selected) result.history = descriptor(event, tensors.history, selected, '输入历史骨架', selectedId);
        }
        if (tensors.future) {
            const selected = selectTrack(tensors.future, event, selectedId);
            if (selected) forecasts.push({ event, tensor: tensors.future, selected });
        }
    }

    if (latestRecurrent) {
        const event = latestRecurrent, tensors = event.tensors;
        const round = event.step ?? Number(event.node.split('.')[1]);
        for (const [name, label] of [['input', 'LSTM 输入'], ['sequence', 'LSTM 序列输出']]) {
            const selected = selectTrack(tensors[name], event, selectedId);
            if (selected) result[`lstm.${name}`] = { ...descriptor(event, tensors[name], selected, label, selectedId), round };
        }
        const hidden = selectTrack(tensors.hidden, event, selectedId);
        const cell = selectTrack(tensors.cell, event, selectedId);
        const layers = hidden?.indices[hidden.axes.indexOf('layer')] || [];
        layers.forEach((layer, position) => {
            const selected = sliceAxis(hidden, 'layer', position);
            const cellPosition = cell?.indices[cell.axes.indexOf('layer')]?.indexOf(layer) ?? -1;
            const selectedCell = cellPosition < 0 ? null : sliceAxis(cell, 'layer', cellPosition);
            result[`lstm.layer.${layer}`] = { ...descriptor(event, tensors.hidden, selected, `LSTM 第 ${layer + 1} 层隐藏态`, selectedId),
                round, layer, cell: selectedCell?.values, cell_values: selectedCell?.values,
                cell_indices: selectedCell?.indices, cell_shape: selectedCell?.shape };
        });
    }

    // Forecast rounds contain actual denormalized poses. Keep only a contiguous prefix.
    forecasts.sort((a, b) => a.event.horizon_start - b.event.horizon_start);
    let future = null, expectedStart = 1;
    for (const { event, tensor, selected } of forecasts) {
        if (event.horizon_start !== expectedStart || selected.axes.join(',') !== 'time,joint,coordinate') break;
        const horizonIndices = selected.indices[0].map(index => event.horizon_start + index);
        const reference = { node: event.node, seq: event.seq, shape: tensor.shape, indices: tensor.indices };
        if (!future) {
            future = { ...descriptor(event, tensor, selected, 'LSTM 预测骨架', selectedId),
                indices: [horizonIndices, ...selected.indices.slice(1)], source_references: [reference], round: event.step };
        } else {
            future = { ...future, values: [...future.values, ...selected.values],
                indices: [[...future.indices[0], ...horizonIndices], ...future.indices.slice(1)],
                shape: [future.shape[0] + selected.shape[0], ...future.shape.slice(1)],
                sample_shape: [future.sample_shape[0] + selected.sample_shape[0], ...future.sample_shape.slice(1)],
                source_references: [...future.source_references, reference], seq: event.seq, round: event.step,
                captured_at_ms: event.captured_at_ms, received_at_ms: event.received_at_ms };
        }
        expectedStart += selected.shape[0];
    }
    if (future) result.future = future;

    const output = snapshot.nodes['pipeline.output'];
    const track = snapshot.tracks?.find(row => sameTrack(row.track_id, selectedId));
    if (output?.frame === snapshot.frame && track?.prediction_updated === true
        && track.prediction_frame === snapshot.frame && Array.isArray(track.probabilities)) {
        const probabilities = { values: track.probabilities, axes: ['class'], shape: [track.probabilities.length],
            sample_shape: [track.probabilities.length], indices: [track.probabilities.map((_, index) => index)] };
        result['stgcn.probabilities'] = descriptor(output, probabilities, probabilities, '实际分类概率', selectedId);
    }
    return result;
}
