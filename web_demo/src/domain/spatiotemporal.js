import { asPose, validPoint } from './pose.js';
import { normalizeCalculation } from './calculation.js';
import { recordedGraphDetails, temporalWindow } from './temporal.js';

const same = (a, b) => a != null && b != null && String(a) === String(b);
const axis = value => Array.isArray(value) ? value[0] : value;
const poses = values => {
    const normalized = Array.isArray(values) ? values.map(asPose) : [];
    return normalized.every(Boolean) ? normalized : [];
};

export const JOINT_LABELS = ['鼻尖', '左眼', '右眼', '左耳', '右耳', '左肩', '右肩', '左肘', '右肘',
    '左腕', '右腕', '左髋', '右髋', '左膝', '右膝', '左踝', '右踝'];

/** Resolve overlapping hit areas by coordinate distance, never SVG paint order. */
export function nearestJoint(points, [x, y], visible = null) {
    let nearest = null, distance = Infinity;
    points.forEach((point, joint) => {
        if (visible && !visible[joint]) return;
        const next = Math.hypot(point[0] - x, point[1] - y);
        if (next < distance) { nearest = joint; distance = next; }
    });
    return nearest;
}

export function currentGraphRows(snapshot) {
    if (!Number.isInteger(snapshot?.frame)) return [];
    const live = Object.values(snapshot.nodes || {}).filter(event => event.frame === snapshot.frame
        && (!event.job_id || !snapshot.job_id || same(event.job_id, snapshot.job_id))).flatMap(event => event.calculations || []);
    const recorded = Object.values(snapshot.details?.stgcn || {}).flat().filter(row => row.source?.frame === snapshot.frame
        && (!snapshot.job_id || same(row.source?.job_id, snapshot.job_id)));
    return [...live, ...recorded].filter(row => ['spatial', 'temporal'].includes(row.stage));
}

/** Layer-input coordinates: symmetric kernels preserve the time origin. */
export function temporalMapping(blocks, block) {
    let jump = 1, radius = 0;
    for (let index = 1; index < block; index++) {
        const previous = blocks.find(item => item.index === index);
        const stride = axis(previous?.stride), kernel = axis(previous?.temporal_kernel);
        if (!Number.isInteger(stride) || !Number.isInteger(kernel)) return null;
        radius += (kernel - 1) / 2 * jump;
        jump *= stride;
    }
    return { jump, radius };
}

/** Strict event envelopes prevent retained arithmetic from borrowing a newer pose. */
export function spatiotemporalModel({ snapshot, result, track, block = 1, boundary = false, structure, jobId }) {
    const frame = snapshot?.frame, trackId = track?.track_id;
    const source = { jobId: jobId ?? snapshot?.job_id, frame, trackId, recorded: Boolean(result) };
    const empty = { source, history: [], forecasts: [], spatial: null, temporal: null, edges: [], slots: [], mapping: null, target: null };
    if (!Number.isInteger(frame) || trackId == null) return empty;
    if (result && (!same(result.source?.job_id, source.jobId) || result.source?.frame !== frame
        || !same(result.source?.track_id, trackId))) return empty;
    const events = Object.values(snapshot?.nodes || {}).filter(event => event.frame === frame
        && (!event.job_id || same(event.job_id, source.jobId)));
    const rows = result ? recordedGraphDetails(result).stgcn[block - 1] || []
        : currentGraphRows({ ...snapshot, job_id: source.jobId });
    const matching = rows.filter(row => same(row.track_id, trackId) && row.metadata?.block === block - 1);
    const spatial = normalizeCalculation(matching.find(row => row.stage === 'spatial'));
    const temporal = normalizeCalculation(matching.find(row => row.stage === 'temporal' && Boolean(row.metadata?.boundary) === boundary));
    let forecasts = [];
    if (result) forecasts = poses(result.future?.values);
    else {
        const rounds = Array.from({ length: 5 }, (_, i) => {
            const event = snapshot.nodes?.[`lstm.${i + 1}.forecast`];
            if (!events.includes(event)) return [];
            const index = event.track_ids?.findIndex(id => same(id, trackId));
            return index >= 0 ? poses(event.tensors?.future?.values?.[index]) : [];
        });
        if (rounds.every(round => round.length === 10)) forecasts = rounds.flat();
    }
    if (forecasts.length !== 50) forecasts = [];
    let history = poses(track.history);
    if (!history.length) {
        const event = snapshot?.nodes?.['pipeline.history'];
        const index = events.includes(event) ? event.track_ids?.findIndex(id => same(id, trackId)) : -1;
        if (index >= 0) history = poses(event.tensors?.history?.values?.[index]);
    }
    const edges = spatial?.metadata?.graph_slice?.edges || [];
    const mapping = temporalMapping(result?.stgcn?.blocks || structure?.stgcnBlocks || [], block);
    const window = temporalWindow(temporal);
    const spatialTime = spatial?.metadata?.target_indices?.[1];
    const times = window ? window.terms.map(term => ({ time: term.inputIndex[1], padding: term.padding })) : [];
    if (Number.isInteger(spatialTime) && !times.some(item => item.time === spatialTime)) times.push({ time: spatialTime, padding: false });
    times.sort((a, b) => a.time - b.time);
    const slots = times.map(({ time, padding }) => {
        const anchor = mapping ? time * mapping.jump : null;
        const pose = !padding && anchor != null ? forecasts[anchor] || null : null;
        return { time, padding, pose, anchor, spatial: time === spatialTime,
            temporal: Boolean(window?.terms.some(term => term.inputIndex[1] === time)),
            range: !padding && mapping && forecasts.length ? [Math.max(0, anchor - mapping.radius), Math.min(49, anchor + mapping.radius)] : null };
    });
    return { source, history, forecasts, spatial, temporal, edges, slots, mapping,
        target: temporal?.metadata?.target_indices?.[2] ?? spatial?.metadata?.target_indices?.[2] ?? null };
}

/** Shared spatial scale keeps real pose changes visible across time slices. */
export function skeletonLayout(slots, { showLowScores = false } = {}) {
    const visibleSlots = slots.map(slot => ({ ...slot, visible: (slot.pose || []).map(point => showLowScores || validPoint(point)) }));
    const points = visibleSlots.flatMap(slot => (slot.pose || []).filter((_, joint) => slot.visible[joint]));
    if (!points.length) return visibleSlots.map((slot, i) => ({ ...slot, x: (i - (slots.length - 1) / 2) * 2.15, points: [] }));
    const xs = points.map(point => point[0]), ys = points.map(point => point[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    // A wide pose must fit its own 2.15-unit time column as well as its height.
    const scale = Math.min(3.8 / Math.max(maxY - minY, 1), 1.8 / Math.max(maxX - minX, 1));
    return visibleSlots.map((slot, i) => ({ ...slot, x: (i - (slots.length - 1) / 2) * 2.15,
        points: (slot.pose || []).map(point => [(point[0] - (minX + maxX) / 2) * scale,
            (maxY - point[1]) * scale, 0]) }));
}
