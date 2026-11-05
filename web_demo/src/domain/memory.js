const finite = values => values.every(Number.isFinite);
const multiply = (a, b) => finite([a, b]) ? a * b : null;

function forTrack(event, name, trackId) {
    const tensor = event?.tensors?.[name];
    const axis = tensor?.axes?.indexOf('track');
    const position = event?.track_ids?.findIndex(id => String(id) === String(trackId));
    if (!tensor || !Number.isInteger(axis) || axis < 0 || position == null || position < 0) return null;
    const take = (values, depth) => !Array.isArray(values) ? null : depth === 0 ? values[position] : values.map(value => take(value, depth - 1));
    return take(tensor.values, axis);
}

export function recordedMemoryNodes(result) {
    if (!result?.source || !result.lstm) return {};
    const source = result.source;
    return Object.fromEntries((result.lstm.rounds || []).map(round => {
        const node = `lstm.${round.index}.forecast`, example = round.decoder_example;
        return [node, { node, track_ids: [source.track_id], frame: source.frame,
            tensors: { future: { axes: ['track','time','joint','coordinate'], values: [round.forecast_poses || []] } },
            calculations: round.index === 1 && example ? [{
                id: `record.${source.job_id}.${source.frame}.${source.track_id}.decoder`, stage: 'decoder',
                title: '记录重算 · LSTM 第 1 轮隐藏状态解码', track_id: source.track_id,
                inputs: example.input_values, weights: example.weights, bias: example.bias, actual: example.actual,
                activation: 'identity', input_indices: example.input_values.map((_,i) => i),
                metadata: { source_frame: source.frame, source_kind: source.kind },
            }] : [] }];
    }));
}

export function lstmDisplayModel(snapshot, result, trackId, index = 0) {
    const record = result && String(result.source?.track_id) === String(trackId) ? result.lstm : null;
    const steps = record?.gates?.steps || [];
    const gates = steps[Math.min(steps.length - 1, Math.max(0, index))] || null;
    const nodes = snapshot?.nodes || {};
    const events = Object.entries(nodes).filter(([key, event]) => /^lstm\.\d+$/.test(key)
        && event.track_ids?.some(id => String(id) === String(trackId))).sort(([a], [b]) => Number(a.split('.')[1]) - Number(b.split('.')[1]));
    const hidden = gates?.previous_hidden || record?.rounds?.[0]?.layers?.[0]?.h || forTrack(events.at(-1)?.[1], 'hidden', trackId)?.[0] || [];
    const forecasts = record ? (record.rounds || []).map(round => ({ round: round.index, poses: round.forecast_poses || [] }))
        : Object.entries(nodes).filter(([key]) => /^lstm\.\d+\.forecast$/.test(key)).flatMap(([key, event]) => {
            const poses = forTrack(event, 'future', trackId);
            return poses ? [{ round: Number(key.split('.')[1]), poses }] : [];
        }).sort((a, b) => a.round - b.round);
    const retained = multiply(gates?.f, gates?.c_previous), written = multiply(gates?.i, gates?.g);
    const reconstructedCell = finite([retained, written]) ? retained + written : null;
    return { gates, steps: steps.length, timestep: gates?.t ?? null, hidden, forecasts,
        input: gates?.input_values || forTrack(events.at(-1)?.[1], 'input', trackId)?.[0] || [],
        retained, written, reconstructedCell, source: record ? result.source : null,
        validationError: record?.gates?.validation_max_error,
        gateTerms: ['i', 'f', 'g', 'o'].map((name, n) => {
            const input = gates?.input_dot?.[n] ?? null, recurrent = gates?.recurrent_dot?.[n] ?? null, bias = gates?.bias?.[n] ?? null;
            return { name, input, recurrent, bias, z: finite([input, recurrent, bias]) ? input + recurrent + bias : null,
                value: gates?.[name] ?? null, activation: name === 'g' ? 'tanh' : 'sigmoid' };
        }) };
}
