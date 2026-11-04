function valueAtAxis(values, axis, index) {
    if (!Array.isArray(values)) return undefined;
    return axis === 0 ? values[index] : values.map(value => valueAtAxis(value, axis - 1, index));
}

function withoutAxis(list, axis) {
    return axis < 0 ? list : list.filter((_, index) => index !== axis);
}

/**
 * The live backend publishes the actual 50-observation history under
 * `pipeline.history`, sampled to at most 10 time steps before it leaves the
 * server. Select the chosen track and keep the sampled time indices honest.
 */
function sampledHistory(event, trackId) {
    const tensor = event?.tensors?.history;
    if (!tensor || !Array.isArray(tensor.axes) || !Array.isArray(tensor.values)) return null;
    const trackAxis = tensor.axes.indexOf('track');
    const position = Array.isArray(event.track_ids)
        ? event.track_ids.findIndex(id => String(id) === String(trackId)) : -1;
    const trackCount = tensor.sample_shape?.[trackAxis] ?? tensor.indices?.[trackAxis]?.length ?? 0;
    if (trackAxis < 0 || position < 0 || position >= trackCount) return null;
    return {
        node: event.node, frame: event.frame, seq: event.seq, track_id: trackId,
        axes: withoutAxis(tensor.axes, trackAxis),
        shape: withoutAxis(tensor.shape, trackAxis),
        indices: withoutAxis(tensor.indices, trackAxis),
        poses: valueAtAxis(tensor.values, trackAxis, position),
        sampled: true,
        sample_note: '实时采样历史，最多 10 个时间步；不是完整 50 帧。',
    };
}

export function liveLayerModel(snapshot, trackId, structure, fallbackTrack = null) {
    const track = (snapshot?.tracks || []).find(item => String(item.track_id) === String(trackId))
        || (String(fallbackTrack?.track_id) === String(trackId) ? fallbackTrack : null);
    const nodes = snapshot?.nodes || {};
    const features = (structure?.yoloModules || []).map(module => ({
        module: module.index,
        output: nodes[`yolo.${module.index}`]?.tensors?.output || null,
        detail: snapshot?.details?.yolo?.[module.index] || null,
    }));
    const rounds = Object.keys(nodes).filter(node => /^lstm\.\d+$/.test(node))
        .sort((left, right) => Number(left.split('.')[1]) - Number(right.split('.')[1]));
    // Real forecast events are named `lstm.N.forecast`; they mark completed
    // prediction rounds. Future poses are not rendered by the scene yet, so the
    // model carries round numbers rather than an unused fabricated skeleton.
    const forecasts = Object.keys(nodes).filter(node => /^lstm\.\d+\.forecast$/.test(node))
        .map(node => Number(node.split('.')[1])).sort((a, b) => a - b);
    const memory = { rounds, forecasts };
    const graph = {
        blocks: (structure?.stgcnBlocks || []).map(block => ({
            block: block.index,
            output: nodes[`stgcn.${block.index - 1}.output`]?.tensors?.output || null,
            details: snapshot?.details?.stgcn?.[block.index - 1] || [],
        })),
    };
    return { frame: snapshot?.frame ?? -1, track, features, memory, graph,
        history: sampledHistory(nodes['pipeline.history'], trackId),
        intent: { probabilities: Array.isArray(track?.probabilities) ? track.probabilities : null,
            classId: Number.isInteger(track?.class_id) ? track.class_id : null } };
}
