const EMPTY = { yolo: {}, stgcn: {} };

export function normalizeModelStructure(raw) {
    const modules = raw?.yolo?.modules || [];
    const blocks = raw?.stgcn?.blocks || [];
    return {
        yoloModules: modules,
        yoloById: new Map(modules.map(module => [module.index, module])),
        stgcnBlocks: blocks,
        stgcnById: new Map(blocks.map(block => [block.index, block])),
        classes: raw?.stgcn?.classes || ['irrelevant', 'not crossing', 'crossing'],
        labels: raw?.stgcn?.labels || ['其他', '不过街', '过街'],
        versions: raw?.versions || {},
    };
}

export function mergeDetail(snapshot, payload) {
    const rows = payload?.calculations || [];
    if (!rows.length || snapshot?.frame !== payload?.frame) return snapshot?.details || EMPTY;
    const details = {
        yolo: { ...(snapshot?.details?.yolo || {}) },
        stgcn: { ...(snapshot?.details?.stgcn || {}) },
    };
    for (const row of rows) {
        if (!row?.metadata?.detail) continue;
        if (row.stage === 'yolo') {
            details.yolo[row.module] = row;
        } else {
            const block = row.metadata.detail_block;
            if (block == null) continue;
            details.stgcn[block] = [...(details.stgcn[block] || []), row].slice(-12);
        }
    }
    return details;
}
