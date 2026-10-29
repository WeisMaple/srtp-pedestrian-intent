export const NETWORK_CHAPTERS = [
    { title: '全流程', subtitle: '网络全景', duration: 8 },
    { title: '视觉编码', subtitle: 'YOLO26 · 卷积', duration: 24 },
    { title: '跟踪与输入', subtitle: 'ID · 50 帧张量', duration: 14 },
    { title: '门控记忆', subtitle: 'LSTM · 逐项乘加', duration: 26 },
    { title: '未来骨架', subtitle: '五轮预测 · 归一化', duration: 16 },
    { title: '空间图卷积', subtitle: '邻接 · 关节聚合', duration: 22 },
    { title: '时间卷积', subtitle: '9 步窗口 · 残差', duration: 22 },
    { title: '意图输出', subtitle: '池化 · Softmax', duration: 18 },
];
export const NETWORK_DURATION = NETWORK_CHAPTERS.reduce((sum, x) => sum + x.duration, 0);
export const networkStart = (index) => NETWORK_CHAPTERS.slice(0, index).reduce((sum, x) => sum + x.duration, 0);
export function networkPosition(time) {
    for (let i = NETWORK_CHAPTERS.length - 1; i >= 0; i -= 1) {
        if (time >= networkStart(i)) return { chapter: i, progress: Math.max(0, Math.min(1, (time - networkStart(i)) / NETWORK_CHAPTERS[i].duration)) };
    }
    return { chapter: 0, progress: 0 };
}
