export const BAYS = Array.from({ length: 6 }, (_, index) => ({
    step: index + 1,
    position: [(index - 2.5) * 7, 0, (index - 2.5) * 1.2],
    label: ['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图'][index],
}));

export function bayForStep(step) {
    return BAYS.find(bay => bay.step === step) || BAYS[0];
}

export function cameraForStep(step, { focus = null, aspect = 1.6 } = {}) {
    const bay = bayForStep(step);
    const target = [bay.position[0], 1.4, bay.position[2]];
    if (focus) return { target: focus, position: [focus[0] + 5.5, focus[1] + 4.2, focus[2] + 12] };
    const distance = Math.max(5.8 / Math.max(0.2, aspect), 4.2) / (2 * Math.tan(17 * Math.PI / 180)) * 1.45 + 2;
    return { target, position: [target[0] + distance * 0.12, target[1] + distance * 0.18, target[2] + distance] };
}
