export async function request(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
        let message = `请求失败（${response.status}）`;
        try {
            const body = await response.json();
            message = typeof body.detail === 'string' ? body.detail : body.message || message;
        } catch { /* Non-JSON errors still receive a readable status. */ }
        throw new Error(message);
    }
    return response.json();
}

export const jobUrl = (id, suffix = '') => `/api/jobs/${encodeURIComponent(id)}${suffix}`;
export const ACTIVE_STATUSES = new Set(['loading', 'running', 'encoding']);
export const STATUS_LABELS = {
    ready: '等待开始', loading: '正在加载模型', running: '正在分析', encoding: '正在编码结果',
    completed: '分析完成', failed: '分析失败', cancelled: '已停止分析',
};
export const CLASS_LABELS = ['其他', '不过街', '过街'];

export function formatTime(seconds) {
    if (!Number.isFinite(Number(seconds))) return '--:--';
    const value = Math.max(0, Math.floor(Number(seconds)));
    return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
}

export function formatBytes(bytes) {
    if (!Number.isFinite(Number(bytes))) return '—';
    return `${(Number(bytes) / 1024 / 1024).toFixed(1)} MB`;
}

export function trackLabel(track) {
    if (track.class_id !== null && track.class_id !== undefined) return CLASS_LABELS[track.class_id] || '未知类别';
    if (track.pose_accepted === false) return '姿态不足';
    if ((track.history_length || 0) < 50) return '积累历史中';
    return '等待预测';
}
