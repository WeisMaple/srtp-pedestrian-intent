export function historyPoses(track) {
    const history = track?.history;
    if (Array.isArray(history)) return history.slice(-50);
    return track?.keypoints ? [track.keypoints] : [];
}

export function historyGaps(track) {
    return historyPoses(track).reduce((gaps, pose, index) => {
        if (!pose) gaps.push(index);
        return gaps;
    }, []);
}

export function trackIsWarm(track) {
    return (track?.history_length || 0) >= 50;
}
