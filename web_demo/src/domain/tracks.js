export function historyPoses(track) {
    const history = track?.history;
    if (Array.isArray(history)) return history.slice(-50);
    return track?.keypoints ? [track.keypoints] : [];
}

export function trackIsWarm(track) {
    return (track?.history_length || 0) >= 50;
}

function pointSignature(pose) {
    if (!Array.isArray(pose)) return null;
    return pose.map(point => Array.isArray(point)
        ? point.slice(0, 3).map(value => (typeof value === 'number' ? Math.round(value * 1000) / 1000 : value))
        : point);
}

/**
 * Content signature for the step-③ ribbon. A track keeps the same ByteTrack ID
 * while its pose and observation count change every snapshot, so the scene
 * must compare the data, not only the ID.
 */
export function ribbonSignature(track, { frame = null, poses = null } = {}) {
    if (!track) return 'none';
    return JSON.stringify([
        track.track_id ?? null,
        track.history_length ?? 0,
        frame,
        pointSignature(track.keypoints),
        Array.isArray(poses) ? poses.length : 0,
        Array.isArray(poses) ? poses.slice(-50).map(pointSignature) : null,
    ]);
}
