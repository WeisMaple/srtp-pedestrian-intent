export function temporalPartialSum(terms, count) {
    const completed = terms.slice(0, count);
    return completed.every(term => Number.isFinite(term.product))
        ? completed.reduce((sum, term) => sum + term.product, 0) : null;
}

export function temporalWindow(calculation, channel = 0) {
    if (calculation?.stage !== 'temporal') return null;
    const kernel = calculation.metadata?.kernel_size;
    if (!Number.isInteger(kernel) || kernel < 1 || !calculation.terms?.length || calculation.terms.length % kernel) return null;
    const channels = calculation.terms.length / kernel;
    const selected = Math.min(channels - 1, Math.max(0, Math.floor(channel)));
    const offset = selected * kernel;
    const terms = calculation.terms.slice(offset, offset + kernel);
    const sum = temporalPartialSum(terms, terms.length);
    return { kernel, channels, channel: selected, offset, terms, sum,
        target: calculation.metadata.target_indices || [],
        stride: calculation.metadata.stride, padding: calculation.metadata.convolution_padding,
        dilation: calculation.metadata.dilation };
}

export function automaticGraphSelection(snapshot, cachedRows, state, trackId) {
    const people = new Set((snapshot?.tracks || []).map(track => String(track.track_id)));
    const rows = [...(cachedRows || []), ...Object.values(snapshot?.details?.stgcn || {}).flat()].filter(row =>
        row.stage === state.operation && people.has(String(row.track_id)) && row.inputs?.length);
    const onTrack = row => String(row.track_id) === String(trackId);
    const onWindow = row => row.stage !== 'temporal' || Boolean(row.metadata?.boundary) === Boolean(state.boundary);
    const newest = [...rows].reverse();
    const row = newest.find(row => onTrack(row) && onWindow(row) && row.metadata?.block === state.block - 1)
        || newest.find(row => onTrack(row) && onWindow(row))
        || newest.find(onWindow) || newest[0];
    return row ? { trackId: row.track_id, block: row.metadata.block + 1, boundary: Boolean(row.metadata.boundary) } : null;
}

/** A last frame without people does not erase valid saved histories. */
export function recordedGraphSelections(trace) {
    return (trace?.samples || []).flatMap(sample => (sample.tracks || []).filter(track =>
        Array.isArray(track.history) && track.history.length === 50 && track.history.every(pose =>
            Array.isArray(pose) && pose.length === 17 && pose.every(point =>
                Array.isArray(point) && point.length === 3 && point.every(Number.isFinite))))
        .map(track => ({ key: `${sample.frame}:${track.track_id}`, frame: sample.frame, sample, track })))
        .sort((a, b) => b.frame - a.frame || a.track.track_id - b.track.track_id);
}

/** Adapt the existing replay API to the same complete arithmetic ledger as live capture. */
export function recordedGraphDetails(result) {
    const source = result?.source;
    const stgcn = {};
    if (!source) return { stgcn };
    for (const block of result.stgcn?.blocks || []) {
        const rows = [];
        const common = { track_id: source.track_id, activation: 'identity', source,
            metadata: { block: block.index - 1, source_frame: source.frame, source_kind: source.kind } };
        const spatial = block.arithmetic?.spatial;
        const graphEdges = (block.effective_adjacency || []).flatMap((partition, k) => partition.flatMap((targets, v) =>
            targets.flatMap((weight, w) => (result.stgcn.adjacency?.[k]?.[v]?.[w] ?? weight) !== 0 ? [[k, v, w, weight]] : [])));
        if (spatial) rows.push({ ...common, id: `record.${source.job_id}.${source.frame}.${source.track_id}.${block.index}.spatial`,
            stage: 'spatial', operation: 'spatial_aggregation', title: `记录重算 · 第 ${block.index} 块 · 空间聚合`,
            inputs: spatial.projected_values.flat(), weights: spatial.adjacency.flat(), bias: 0, actual: spatial.actual,
            connected: spatial.adjacency.flat().map(value => value !== 0), formula: spatial.formula,
            input_indices: spatial.projected_values.flatMap((values, k) => values.map((_, v) => [k, spatial.timestep, v])),
            metadata: { ...common.metadata, target_indices: [spatial.output_channel, spatial.timestep, spatial.output_joint],
                graph_slice: { edges: graphEdges, timestep: spatial.timestep } } });
        for (const boundary of [false, true]) {
            const value = block.arithmetic?.[boundary ? 'temporal_boundary' : 'temporal'];
            if (!value) continue;
            const indices = value.input_values.flatMap((values, c) => values.map((_, q) => [c, value.input_timesteps[q], value.output_joint]));
            rows.push({ ...common, id: `record.${source.job_id}.${source.frame}.${source.track_id}.${block.index}.${boundary}`,
                stage: 'temporal', operation: 'temporal_convolution',
                title: `记录重算 · 第 ${block.index} 块 · 时间卷积${boundary ? '（边界）' : ''}`,
                inputs: value.input_values.flat(), weights: value.weights.flat(), bias: value.bias, actual: value.actual,
                input_indices: indices, padding: indices.map(([, t]) => t < 0 || t >= value.input_length), formula: value.formula,
                metadata: { ...common.metadata, boundary, kernel_size: value.kernel_size, stride: value.stride,
                    convolution_padding: value.padding, dilation: value.dilation, input_length: value.input_length,
                    target_indices: [value.output_channel, value.timestep, value.output_joint] } });
        }
        stgcn[block.index - 1] = rows;
    }
    return { stgcn };
}
