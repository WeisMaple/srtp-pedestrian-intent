import { CLASS_LABELS } from '../api.js';

export function visibleTracks(live, job) {
    return live && Array.isArray(live.tracks) ? live.tracks : (job?.tracks || []);
}

const EXPLANATIONS = [
    '将源视频逐帧解码，缩放与补边后转换为 RGB，再归一化送入姿态检测模型。',
    'YOLO 从画面中检测行人，并定位 COCO17 关节点。代表卷积展示一个输出位置的完整乘加。',
    'ByteTrack 为检测结果关联 ID，同一个 ID 的有效骨架按时间积累；达到 50 次观测后才能预测。',
    '将 50 次骨架观测编码为时序特征，每轮预测 10 帧，递归五轮得到未来 50 帧骨架。',
    '沿骨架邻接关系聚合空间特征，再用时间卷积提取动作变化，经过十个时空图卷积块。',
    '池化特征与分类权重逐项相乘、求和并加偏置，得到 logits；Softmax 转为三类模型分数。',
];

/** Actual context only; architecture descriptions do not imply executed work. */
export function processMetrics(step, { job, live, track, calculation, module = 0, block = 1 } = {}) {
    const tracks = visibleTracks(live, job);
    const history = Number.isFinite(track?.history_length) ? track.history_length : 0;
    const remaining = Math.max(0, 50 - history);
    const frame = Number.isInteger(live?.frame) && live.frame >= 0 ? live.frame : null;
    const scores = Array.isArray(track?.probabilities) && track.probabilities.length === 3
        && track.probabilities.every(value => typeof value === 'number' && Number.isFinite(value))
        ? track.probabilities : null;
    const rounds = Object.values(live?.nodes || {}).filter(node => /^lstm\.\d+\.forecast$/.test(node.node)
        && node.track_ids?.some(id => String(id) === String(track?.track_id))).length;
    const size = job?.metadata?.width && job?.metadata?.height
        ? `${job.metadata.width} × ${job.metadata.height} px` : '等待源视频';
    const metrics = [
        [size, '解码 → RGB → 归一化', frame == null ? '等待采样帧' : `采样帧 #${frame}`],
        [frame == null ? '等待视频帧' : `采样帧 #${frame}`, `YOLO 模块 #${module} · 代表卷积`, `${tracks.length} 位行人 · 每人 17 关节`],
        [track ? `行人 ID ${track.track_id}` : '等待行人 ID', 'ID 关联 → 有效骨架入列', `${history} / 50 次有效观测`],
        [`${history} / 50 次有效观测`, 'LSTM · 5 轮 × 10 帧', rounds ? `本采样帧已记录 ${rounds} / 5 轮` : '等待预测输出'],
        ['未来骨架序列', `ST-GCN 第 ${block} 块`, calculation ? `${calculation.terms.length} 项代表运算` : '等待本块输出'],
        ['池化特征', '加权求和 → Softmax', scores ? (CLASS_LABELS[track?.class_id] || '已记录三类分数') : '尚无分类分数'],
    ][step - 1] || [];
    const waiting = step === 1 ? frame == null : step === 2 ? !tracks.length
        : step === 3 ? !track : step === 4 ? !rounds && !calculation
            : step === 5 ? !calculation : !scores;
    let notice = '';
    if (!job) notice = '先到工作台选择视频并开始分析，六个步骤将展示同一任务的真实数据。';
    else if (job.status === 'ready') notice = '视频已就绪，请在工作台点击“开始分析”。';
    else if (job.status === 'failed' || job.status === 'cancelled') notice = job.error || '分析已停止，保留已采集的数据。';
    else if (step >= 4 && track && remaining) notice = `还需 ${remaining} 次有效骨架观测。视频帧数不等于有效观测数。`;
    else if (step >= 2 && !track) notice = '当前采样帧没有可用行人，等待检测与身份关联。';
    else if (!calculation && [2, 4, 5, 6].includes(step)) notice = job.status === 'completed'
        ? '本次任务未保留当前对象的逐项算式。可选择已采集的模块或块；YOLO 也可使用“记录重算”。'
        : '等待当前对象的详细采集；各模块轮转采样，尚未采集的值保持为空。';
    return { description: EXPLANATIONS[step - 1], input: metrics[0], operation: metrics[1],
        output: metrics[2], history, remaining, frame, scores, waiting, notice };
}
