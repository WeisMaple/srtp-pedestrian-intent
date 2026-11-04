export const BAYS = Array.from({ length: 6 }, (_, index) => ({
    step: index + 1,
    position: [(index - 2.5) * 7, 0, (index - 2.5) * 1.2],
    label: ['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图'][index],
}));

export function bayForStep(step) {
    return BAYS.find(bay => bay.step === step) || BAYS[0];
}

export function cameraForStep(step, { focus = null } = {}) {
    const bay = bayForStep(step);
    const target = [bay.position[0], 2.1, bay.position[2]];
    if (focus) return { target: focus, position: [focus[0] + 5.5, focus[1] + 4.2, focus[2] + 12] };
    return { target, position: [target[0] + 1.5, 8.2, target[2] + 16] };
}
