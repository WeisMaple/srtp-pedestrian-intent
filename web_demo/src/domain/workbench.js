export function formatStatus(job) {
    if (!job) return '等待视频';
    if (job.status === 'running') return `正在分析 ${Math.round((job.progress || 0) * 100)}%`;
    if (job.status === 'loading') return '正在加载模型';
    if (job.status === 'encoding') return '正在编码结果';
    if (job.status === 'completed') return '分析完成';
    if (job.status === 'failed') return `分析失败 · ${job.error || '未知错误'}`;
    if (job.status === 'cancelled') return '已停止分析';
    return '等待开始';
}

export function uploadHint(file, modelsReady) {
    if (!modelsReady) return '模型未就绪，请先检查权重与 ffmpeg';
    if (!file) return '拖入一段视频，或点击选择文件';
    return `${file.name} · 可以开始分析`;
}

/**
 * Pick the recorded calculation for the inspected object only.
 * YOLO module indices are zero-based end to end. ST-GCN structure blocks are
 * labelled 1–10, while live detail rows use zero-based block ordinals.
 */
const sameTrack = (left, right) => left != null && right != null && String(left) === String(right);

export function latestCalculation(snapshot, step, structure, moduleIndex, blockIndex, trackId = null) {
    if (!snapshot || !structure) return null;
    const rows = Object.values(snapshot.nodes || {}).flatMap(node => node.calculations || []);
    const onTrack = row => trackId == null || sameTrack(row.track_id, trackId);
    if (step === 2) {
        return snapshot.details?.yolo?.[moduleIndex]
            || rows.find(row => row.stage === 'yolo' && row.module === moduleIndex) || null;
    }
    if (step === 5) {
        const ordinal = blockIndex - 1;
        const detailRows = snapshot.details?.stgcn?.[ordinal] || [];
        return detailRows.find(row => onTrack(row))
            || rows.find(row => row.stage === 'spatial' && row.metadata?.block === ordinal && onTrack(row)) || null;
    }
    if (step === 6) return rows.find(row => row.stage === 'classifier' && onTrack(row)) || null;
    return null;
}
