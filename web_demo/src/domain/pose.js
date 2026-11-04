export const COCO_EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6]];
export const JOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'];
const COCO_JOINTS = JOINT_NAMES.length;
const MIN_VISIBLE_CONFIDENCE = 0.2;
const finite = value => typeof value === 'number' && Number.isFinite(value);

export function validPoint(point) {
    return Array.isArray(point) && finite(point[0]) && finite(point[1])
        && (point.length < 3 || point[2] > MIN_VISIBLE_CONFIDENCE);
}

/** Accept a flat 51-value pose or an already nested joint list. */
function posePoints(value) {
    const source = value?.keypoints || value?.pose || value;
    if (!Array.isArray(source) || !source.length) return null;
    if (source.length === COCO_JOINTS * 3 && !Array.isArray(source[0])) {
        const points = Array.from({ length: COCO_JOINTS },
            (_, index) => source.slice(index * 3, index * 3 + 3));
        return points.every(point => finite(point[0]) && finite(point[1])) ? points : null;
    }
    return source.every(point => Array.isArray(point) && finite(point[0]) && finite(point[1]))
        ? source : null;
}

/** COCO poses for rendering must carry all 17 joints. */
export function asPose(value) {
    const points = posePoints(value);
    return points && points.length === COCO_JOINTS ? points : null;
}

export function poseBounds(poses) {
    const list = Array.isArray(poses) ? poses : [poses];
    const points = list.flatMap(pose => posePoints(pose) || []).filter(validPoint);
    if (!points.length) return null;
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

export function normalizePose(pose, size = 3.6, frame = null) {
    const points = posePoints(pose);
    if (!points) return null;
    const bounds = frame || poseBounds([points]);
    if (!bounds) return null;
    const midX = (bounds.minX + bounds.maxX) / 2;
    const midY = (bounds.minY + bounds.maxY) / 2;
    const scale = size / Math.max(bounds.maxY - bounds.minY, bounds.maxX - bounds.minX, 1e-6);
    return points.map(point => [(point[0] - midX) * scale, -(point[1] - midY) * scale,
        finite(point[2]) ? point[2] : null]);
}
