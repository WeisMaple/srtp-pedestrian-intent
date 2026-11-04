export const STEPS = [
    { step: 1, title: '视频帧', kicker: 'FRAME', beats: ['读取源帧', 'Letterbox / RGB', '送入模型'] },
    { step: 2, title: 'YOLO 骨架', kicker: 'POSE', beats: ['检测行人', '定位 17 关节', '写入置信度'] },
    { step: 3, title: 'ID 绑定', kicker: 'TRACK', beats: ['检测关联', '累积历史', '等待 50 帧'] },
    { step: 4, title: 'LSTM 预测', kicker: 'MEMORY', beats: ['编码历史', '五轮预测', '还原坐标'] },
    { step: 5, title: 'ST-GCN', kicker: 'GRAPH', beats: ['空间邻居', '时间卷积', '残差输出'] },
    { step: 6, title: '意图', kicker: 'CLASS', beats: ['全局池化', '分类头', 'Softmax'] },
];

export function beatsForStep(step) {
    return STEPS.find(item => item.step === step)?.beats || [];
}

/** A step is ready only when the live snapshot or job supplies its evidence. */
export function stepReady(step, snapshot = {}, job = {}) {
    const tracks = snapshot?.tracks || [];
    const nodes = snapshot?.nodes || {};
    if (step === 1) return Boolean(job?.metadata);
    if (step === 2) return tracks.length > 0;
    if (step === 3) return tracks.some(track => (track.history_length || 0) > 0
        || (Array.isArray(track.keypoints) && track.keypoints.length > 0));
    if (step === 4) return Object.keys(nodes).some(node => node.startsWith('lstm.'))
        || tracks.some(track => (track.history_length || 0) >= 50);
    if (step === 5) return Object.keys(nodes).some(node => node.startsWith('stgcn.'));
    if (step === 6) return Boolean(nodes['stgcn.logits'])
        || tracks.some(track => Array.isArray(track.probabilities) && track.probabilities.length > 0);
    return false;
}
