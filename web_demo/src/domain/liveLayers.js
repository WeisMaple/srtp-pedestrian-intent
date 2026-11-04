export function liveLayerModel(snapshot, trackId, structure) {
    const track = (snapshot?.tracks || []).find(item => String(item.track_id) === String(trackId)) || null;
    const nodes = snapshot?.nodes || {};
    const features = (structure?.yoloModules || []).map(module => ({
        module: module.index,
        output: nodes[`yolo.${module.index}`]?.tensors?.output || null,
        detail: snapshot?.details?.yolo?.[module.index] || null,
    }));
    const memory = {
        rounds: Object.keys(nodes).filter(node => /^lstm\.\d+$/.test(node)).sort(),
        history: nodes.history?.tensors?.history || null,
        future: nodes.future?.tensors?.future || null,
    };
    const graph = {
        blocks: (structure?.stgcnBlocks || []).map(block => ({
            block: block.index,
            output: nodes[`stgcn.${block.index - 1}.output`]?.tensors?.output || null,
            details: snapshot?.details?.stgcn?.[block.index - 1] || [],
        })),
    };
    return { frame: snapshot?.frame ?? -1, track, features, memory, graph,
        intent: { probabilities: Array.isArray(track?.probabilities) ? track.probabilities : null,
            classId: Number.isInteger(track?.class_id) ? track.class_id : null } };
}
