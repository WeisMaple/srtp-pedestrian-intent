import { normalizeCalculation } from './calculation.js';
import { normalizeLiveNodes } from '../liveComputation.js';

const validScores = values => Array.isArray(values) && values.length === 3
    && values.every(value => Number.isFinite(value) && value >= 0 && value <= 1);

export function stableSoftmax(logits) {
    if (!Array.isArray(logits) || logits.length !== 3 || !logits.every(Number.isFinite)) return null;
    const max = Math.max(...logits), shifted = logits.map(value => value - max);
    const exponents = shifted.map(Math.exp), denominator = exponents.reduce((sum, value) => sum + value, 0);
    return { max, shifted, exponents, denominator, probabilities: exponents.map(value => value / denominator) };
}

export function recordedIntentNodes(result) {
    const source = result?.source, graph = result?.stgcn;
    if (!source || !graph) return {};
    const event = (node, values, axis, calculations = []) => ({ node, frame: source.frame, seq: 0,
        track_ids: [source.track_id], calculations,
        tensors: { output: { axes: ['track', axis], shape: [1, values.length], sample_shape: [1, values.length],
            indices: [[0], values.map((_,i) => i)], values: [values] } } });
    const rows = (graph.head_weights || []).map((weights, classIndex) => ({
        id: `record.${source.job_id}.${source.frame}.${source.track_id}.classifier.${classIndex}`,
        stage: 'classifier', title: `记录重算 · 类别 ${classIndex} 的分类乘加`, track_id: source.track_id,
        inputs: graph.pool || [], weights, bias: graph.head_bias?.[classIndex], actual: graph.logits?.[classIndex],
        input_indices: weights.map((_, i) => [i]), activation: 'identity', source,
        metadata: { selected_class: classIndex, target_indices: [classIndex], source_frame: source.frame,
            source_kind: source.kind, logits: graph.logits, probabilities: graph.probabilities, complete_sum: true },
    }));
    return { 'stgcn.pool': event('stgcn.pool', graph.pool || [], 'channel'),
        'stgcn.logits': event('stgcn.logits', graph.logits || [], 'class', rows) };
}

function vector(tensor) {
    if (!tensor || tensor.axes.length !== 1) return [];
    const values = Array(tensor.shape[0]).fill(null);
    tensor.indices[0].forEach((index, position) => { if (index < values.length && Number.isFinite(tensor.values[position])) values[index] = tensor.values[position]; });
    return values;
}

export function intentDisplayModel(snapshot, result, trackId) {
    const track = snapshot?.tracks?.find(item => String(item.track_id) === String(trackId));
    const record = track && result?.source?.frame === snapshot.frame && String(result.source.track_id) === String(trackId) ? result : null;
    const nodes = normalizeLiveNodes(snapshot, trackId);
    const rawRows = Object.values(snapshot?.nodes || {}).filter(event => event.frame === snapshot.frame)
        .flatMap(event => event.calculations || []).filter(row => row.stage === 'classifier' && String(row.track_id) === String(trackId));
    const classifiers = rawRows.map(normalizeCalculation).filter(Boolean);
    const pool = record?.stgcn?.pool || (classifiers[0]?.inputs?.length ? classifiers[0].inputs : vector(nodes['stgcn.pool']));
    const logits = record?.stgcn?.logits || classifiers[0]?.metadata?.logits || vector(nodes['stgcn.logits']);
    const originalScores = validScores(track?.probabilities) ? track.probabilities : null;
    const scores = record ? validScores(record.stgcn?.probabilities) ? record.stgcn.probabilities : null : originalScores;
    const sorted = scores ? scores.map((value,index) => ({ value,index })).sort((a,b) => b.value-a.value) : [];
    const winners = sorted.filter(item => Math.abs(item.value - sorted[0].value) < 1e-10).map(item => item.index);
    const arithmetic = record?.stgcn?.pool_arithmetic;
    const pooling = arithmetic?.values?.flat().every(Number.isFinite) && arithmetic.count > 0 ? arithmetic : null;
    const softmax = stableSoftmax(logits);
    return { source: record?.source || null, frame: snapshot?.frame, trackId: track?.track_id ?? null,
        pool, pooling, classifiers, logits, softmax, scores, originalScores,
        winner: winners.length === 1 ? winners[0] : null, winners,
        margin: sorted.length ? sorted[0].value - sorted[1].value : null,
        held: !record && Boolean(originalScores) && track?.prediction_updated === false,
        softmaxError: softmax && scores ? Math.max(...scores.map((value,i) => Math.abs(value - softmax.probabilities[i]))) : null };
}
