/** Complete representative sums shared by the 3D scene and numerical ledger.
 * Missing operands remain null. Actual outputs are never replaced by a replayed
 * sum, and presentation paging never changes the terms used for accumulation.
 */
const CHAPTERS = { yolo: 1, normalization: 2, decoder: 3, denormalization: 4,
    spatial: 5, temporal: 6, pool: 7, classifier: 7, softmax: 7 };
const ACTIVATIONS = new Set(['identity', 'relu', 'tanh', 'sigmoid', 'silu', 'batch_norm_silu', 'softmax']);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const nullable = value => value === null || finite(value);
const same = (left, right) => left != null && right != null && String(left) === String(right);
const at = (value, ...indices) => indices.reduce((item, index) => item?.[index], value);
const sigmoid = value => value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
const scalar = value => finite(value) ? value : null;

/** Null is an explicitly unrecorded operand; NaN/Infinity and shifted arrays are invalid. */
export function validateCalculation(raw) {
    const errors = [];
    if (!raw || typeof raw !== 'object') return { valid: false, errors: ['calculation is missing'] };
    const count = raw.inputs?.length;
    if (!Array.isArray(raw.inputs) || !count) errors.push('inputs must be a nonempty array');
    if (!Array.isArray(raw.weights) || raw.weights.length !== count) errors.push('weights must align with inputs');
    for (const key of ['inputs', 'weights']) {
        if (Array.isArray(raw[key]) && raw[key].some(value => !nullable(value))) errors.push(`${key} contains a nonfinite or absent value`);
    }
    for (const key of ['connected', 'padding', 'input_indices']) {
        if (raw[key] !== undefined && (!Array.isArray(raw[key]) || raw[key].length !== count)) errors.push(`${key} must align with inputs`);
    }
    for (const key of ['connected', 'padding']) {
        if (Array.isArray(raw[key]) && raw[key].some(value => typeof value !== 'boolean')) errors.push(`${key} must contain booleans`);
    }
    if (!nullable(raw.bias)) errors.push('bias must be finite or explicitly unrecorded');
    for (const key of ['actual', 'preactivation']) {
        if (raw[key] !== undefined && !nullable(raw[key])) errors.push(`${key} must be finite or explicitly unrecorded`);
    }
    if (!ACTIVATIONS.has(raw.activation || 'identity')) errors.push('unsupported activation');
    if (!errors.length) {
        let sum = 0;
        for (let index = 0; index < count; index += 1) {
            const input = raw.inputs[index], weight = raw.weights[index];
            if (raw.padding?.[index] && finite(input) && input !== 0) errors.push('padding input must be zero');
            if (!finite(input) || !finite(weight)) continue;
            const product = input * weight;
            if (!finite(product) || !finite(sum + product)) errors.push('products or sum overflow');
            if (raw.connected?.[index] === false && product !== 0) errors.push('disconnected term has nonzero contribution');
            sum += product;
        }
        if (finite(raw.bias) && !finite(sum + raw.bias)) errors.push('biased sum overflows');
    }
    return { valid: errors.length === 0, errors };
}

function transform(value, activation, metadata) {
    if (!finite(value)) return null;
    if (activation === 'identity') return value;
    if (activation === 'relu') return Math.max(0, value);
    if (activation === 'tanh') return Math.tanh(value);
    if (activation === 'sigmoid') return sigmoid(value);
    if (activation === 'silu') return value * sigmoid(value);
    if (activation === 'softmax') return finite(metadata.numerator) && value > 0 ? metadata.numerator / value : null;
    const bn = metadata.bn;
    if (activation === 'batch_norm_silu' && bn
        && [bn.gamma, bn.beta, bn.running_mean, bn.running_var, bn.epsilon].every(finite)
        && bn.running_var + bn.epsilon > 0) {
        const normalized = bn.gamma * (value - bn.running_mean) / Math.sqrt(bn.running_var + bn.epsilon) + bn.beta;
        return normalized * sigmoid(normalized);
    }
    return null;
}

/** Returns null for malformed records; retains explicitly incomplete records for honest display. */
export function createCalculation(raw, provenance = {}) {
    if (!validateCalculation(raw).valid) return null;
    const activation = raw.activation || 'identity';
    const metadata = { ...(raw.metadata || {}) };
    const inputs = [...raw.inputs], weights = [...raw.weights];
    const inputIndices = raw.input_indices || inputs.map((_, index) => index);
    const connected = raw.connected || inputs.map(() => true);
    const padding = raw.padding || inputs.map(() => false);
    const counts = { connected: 0, disconnected: 0, zeroContribution: 0, padding: 0, unrecorded: 0, total: inputs.length };
    const terms = inputs.map((input, index) => {
        const weight = weights[index], recorded = finite(input) && finite(weight);
        const product = recorded ? input * weight : null;
        const membership = !recorded ? 'unrecorded' : !connected[index] ? 'disconnected'
            : padding[index] ? 'padding' : product === 0 ? 'zeroContribution' : 'connected';
        if (recorded && connected[index]) counts.connected += 1;
        if (membership !== 'connected') counts[membership] += 1;
        const inputIndex = inputIndices[index];
        const axisNames={RGB_channel:'RGB',hidden_unit:'隐藏单元',partition:'分区',channel:'通道',time:'时刻',joint:'关节',coordinate:'坐标',y:'行',x:'列'};
        const namedIndex=Array.isArray(inputIndex)&&metadata.input_axes?.length===inputIndex.length ? inputIndex.map((value,axis)=>`${axisNames[metadata.input_axes[axis]]||metadata.input_axes[axis]} ${value}`).join(' · ') : null;
        const label = metadata.term_labels?.[index] ?? metadata.input_names?.[index] ?? namedIndex ?? (typeof inputIndex === 'string' ? inputIndex : JSON.stringify(inputIndex));
        return { index, input, weight, product, inputIndex, label, connected: connected[index],
            padding: padding[index], recorded, membership,
            ...(activation === 'softmax' ? { sourceValue: metadata.logits?.[index] } : {}) };
    });
    // Kahan accumulation uses every supplied product, including offscreen cells.
    const prefixes = [0];
    let sum = 0, compensation = 0;
    for (const term of terms) {
        if (sum === null || term.product === null) sum = null;
        else {
            const corrected = term.product - compensation;
            const next = sum + corrected;
            compensation = (next - sum) - corrected;
            sum = next;
        }
        prefixes.push(sum);
    }
    const dot = prefixes.at(-1);
    const preactivationValue = finite(dot) && finite(raw.bias) ? dot + raw.bias : null;
    const reconstructed = scalar(transform(preactivationValue, activation, metadata));
    const actual = scalar(raw.actual);
    const context = { ...provenance, track_id: raw.track_id ?? provenance.track_id ?? null };
    const source = context.source === 'live' || raw.source === 'live' ? 'live' : 'record';
    return { ...raw, ...context, source, provenance: { ...context, source },
        stage: raw.stage, chapter: CHAPTERS[raw.stage] ?? raw.chapter ?? null,
        title: raw.title || raw.id || '计算', operation: raw.operation || 'weighted_sum',
        inputs, weights, bias: raw.bias, actual, activation, metadata, input_indices: inputIndices,
        connected, padding, terms, products: terms.map(term => term.product), prefixes, dot,
        preactivationValue, reconstructed, absoluteError: finite(actual) && finite(reconstructed) ? Math.abs(actual - reconstructed) : null,
        complete: reconstructed !== null, counts, maxCursor: terms.length + 2 };
}

/** Cursor is pedagogical, independent of inference time: [0,N) terms, [N,N+1)
 * bias, [N+1,N+2) activation, N+2 output. Seeking has no mutable history.
 */
export function flowState(calculation, cursor = 0) {
    if (!calculation?.terms?.length) return null;
    const count = calculation.terms.length, end = count + 2;
    const position = finite(cursor) ? Math.max(0, Math.min(end, cursor)) : 0;
    const termIndex = Math.min(count - 1, Math.floor(position));
    const phase = position < count ? position - termIndex : 1;
    const done = position === end;
    const phaseName = done ? 'result' : position >= count + 1 ? 'activation' : position >= count ? 'bias'
        : phase < .35 ? 'input' : phase < .65 ? 'product' : phase < .85 ? 'accumulate' : 'accumulated';
    const completedCount = position >= count ? count : termIndex + (phase >= .85 ? 1 : 0);
    const biasApplied = position >= count + 1;
    const prefix = calculation.prefixes[completedCount];
    const partialSum = biasApplied ? calculation.preactivationValue : prefix;
    const statusByTerm = calculation.terms.map((term, index) => {
        if (['disconnected', 'padding', 'unrecorded'].includes(term.membership)) return term.membership;
        if (index < completedCount) return 'accumulated';
        if (position < count && index === termIndex) return phaseName;
        return 'pending';
    });
    return { cursor: position, maxCursor: end, termIndex, phase, phaseName, completedCount,
        partialSum, product: calculation.terms[termIndex].product, currentTerm: calculation.terms[termIndex],
        before: calculation.prefixes[termIndex], after: calculation.prefixes[termIndex + 1],
        biasApplied, biasPhase: Math.max(0, Math.min(1, position - count)),
        activationPhase: Math.max(0, Math.min(1, position - count - 1)),
        finalValue: calculation.reconstructed, actual: calculation.actual, done, statusByTerm, counts: calculation.counts };
}

function flattened(value, prefix = [], result = []) {
    if (Array.isArray(value)) value.forEach((item, index) => flattened(item, [...prefix, index], result));
    else result.push({ value: value === undefined ? null : value, index: prefix });
    return result;
}

function addNested(add, raw, inputs, weights, labels) {
    if (!Array.isArray(inputs) || !Array.isArray(weights)) return;
    const cells = flattened(inputs), weightCells = flattened(weights);
    // Equal flatten lengths alone cannot prove channel/kernel alignment.
    if (cells.length !== weightCells.length || cells.some((cell, i) => JSON.stringify(cell.index) !== JSON.stringify(weightCells[i].index))) return;
    add({ ...raw, inputs: cells.map(cell => cell.value), weights: weightCells.map(cell => cell.value),
        input_indices: labels ? cells.map(cell => labels(cell.index)) : cells.map(cell => cell.index) });
}

function softmaxCalculation(id, metadata, trackId) {
    const logits = metadata.logits;
    if (!Array.isArray(logits) || !logits.length || !logits.every(finite)) return null;
    const selected = metadata.selected_class;
    if (!Number.isInteger(selected) || selected < 0 || selected >= logits.length) return null;
    const max = Math.max(...logits), exponentials = logits.map(value => Math.exp(value - max));
    return { id, stage: 'softmax', title: `Softmax · ${metadata.class_labels?.[selected] ?? selected}`,
        operation: 'softmax', track_id: trackId, target: { class: selected },
        inputs: exponentials, weights: logits.map(() => 1), bias: 0,
        actual: scalar(metadata.probabilities?.[selected]), activation: 'softmax',
        input_indices: logits.map((_, index) => `class ${index}`),
        formula: `m = max(logits); p[${selected}] = exp(logit[${selected}] − m) / Σ exp(logit[j] − m)`,
        metadata: { ...metadata, max_logit: max, numerator: exponentials[selected],
            term_labels: logits.map((value, index) => `exp(logit[${index}]=${value} − m=${max})`) } };
}

function liveCalculations(snapshot, trackId, frame) {
    if (!snapshot || !Number.isInteger(snapshot.frame) || (frame != null && snapshot.frame !== frame)) return [];
    const latest = new Map();
    const events = Object.values(snapshot.nodes || {}).filter(event => event?.frame === snapshot.frame
        && (!event.job_id || !snapshot.job_id || event.job_id === snapshot.job_id))
        .sort((left, right) => (left.seq || 0) - (right.seq || 0));
    for (const event of events) {
        for (const raw of event.calculations || []) {
            const isYolo = raw.stage === 'yolo' && raw.track_id == null;
            if (!isYolo && !same(raw.track_id, trackId)) continue;
            if ((raw.frame != null && raw.frame !== snapshot.frame)
                || (raw.job_id && snapshot.job_id && raw.job_id !== snapshot.job_id)) continue;
            const context = { source: 'live', job_id: snapshot.job_id ?? event.job_id ?? null,
                frame: snapshot.frame, seq: event.seq ?? null, node: event.node ?? null,
                time_seconds: event.time_seconds ?? snapshot.time_seconds ?? null,
                track_id: isYolo ? null : raw.track_id, captured_at_ms: event.captured_at_ms ?? null };
            const calculation = createCalculation(raw, context);
            if (!calculation) continue;
            // Delete first so arrival order reflects the latest occurrence of a repeated id.
            latest.delete(calculation.id);
            latest.set(calculation.id, calculation);
            if (raw.stage === 'classifier') {
                const softmax = softmaxCalculation(`${raw.id}.softmax`, raw.metadata || {}, raw.track_id);
                if (softmax) latest.set(softmax.id, createCalculation(softmax, context));
            }
        }
    }
    return [...latest.values()].filter(Boolean);
}

function recordMatches(record, frame, trackId, needsTrack) {
    if (!record) return false;
    if (frame != null && record.source?.frame != null && frame !== record.source.frame) return false;
    if (needsTrack && (trackId == null || (record.source?.track_id != null && !same(trackId, record.source.track_id)))) return false;
    return true;
}

function recordContext(record, frame, trackId) {
    return { ...record.source, source: 'record', source_note: record.source?.note ?? null,
        job_id: record.source?.job_id ?? null, frame: record.source?.frame ?? frame ?? null,
        track_id: record.source?.track_id ?? trackId ?? null };
}

function savedYolo(frontend, add) {
    const example = frontend.first_conv?.example;
    if (!example) return;
    addNested(add, { id: 'yolo.conv0', stage: 'yolo', title: 'YOLO · RGB 3×3 卷积 → BN → SiLU',
        operation: 'convolution', track_id: null,
        target: { channel: example.output_channel, position: example.position }, bias: scalar(example.bias),
        actual: scalar(example.actual_output), activation: 'batch_norm_silu',
        formula: 'z = Σ RGB,kernel(patch × W) + b; a = SiLU(γ(z − μ)/√(variance + ε) + β)',
        metadata: { bn: example.bn, input_origin: example.input_origin,
            recorded_conv_output: example.conv_output, operand_decimals: 6,
            actual_kind: 'independent_stem_forward_hook' } }, example.patch, example.weights,
    ([channel, y, x]) => `${['R', 'G', 'B'][channel]}[${y},${x}]`);
}

function savedLstm(data, add, trackId) {
    const rounds = data.lstm?.rounds || [], stats = data.normalization?.lstm;
    const mean = at(stats?.mean, 0, 0), std = at(stats?.std, 0, 0);
    const observed = at(data.input?.values, 0, 0), normalized = at(rounds[0]?.input?.values, 0, 0);
    if (finite(observed) && finite(mean) && finite(std) && std > 0 && finite(normalized)) {
        add({ id: `lstm.normalize.track${trackId}`, stage: 'normalization', title: '历史骨架 · joint 0 x 归一化',
            operation: 'normalization', target: { timestep: 0, joint: 0, coordinate: 'x' },
            inputs: [observed, mean], weights: [1 / std, -1 / std], bias: 0, actual: normalized,
            input_indices: ['原始坐标 x', 'mean'], formula: stats.formula || '(x − mean) / std',
            metadata: { mean, std, statistics_source: data.normalization?.mode } });
    }
    const gates = data.lstm?.gates, step = gates?.steps?.at(-1);
    if (step && Array.isArray(step.input_values) && Array.isArray(step.previous_hidden)) {
        const inputs = [...step.input_values, ...step.previous_hidden];
        const indices = [...step.input_values.map((_, index) => `x[t,${index}]`),
            ...step.previous_hidden.map((_, index) => `h[t−1,${index}]`)];
        (gates.gate_order || ['i', 'f', 'g', 'o']).forEach((gate, index) => {
            if (!Array.isArray(gates.input_weights?.[index]) || !Array.isArray(gates.recurrent_weights?.[index])) return;
            add({ id: `lstm.gate.${gate}.track${trackId}`, stage: 'decoder', title: `LSTM · ${gate} 门 · t=${step.t}`,
                operation: 'lstm_gate', target: { gate, unit: gates.unit, layer: gates.layer, timestep: step.t },
                inputs, weights: [...gates.input_weights[index], ...gates.recurrent_weights[index]],
                bias: scalar(gates.bias?.[index]), actual: null, preactivation: scalar(step.preactivation?.[index]),
                activation: gate === 'g' ? 'tanh' : 'sigmoid', input_indices: indices,
                formula: `${gate} = ${gate === 'g' ? 'tanh' : 'sigmoid'}(Σ W_ih x_t + Σ W_hh h_(t−1) + b_ih + b_hh)`,
                metadata: { round: gates.round, recorded_gate: step[gate],
                    actual_kind: 'reconstructed_gate_validated_against_final_state',
                    validation_max_error: gates.validation_max_error } });
        });
        const unit = gates.unit ?? 0, layer = (gates.layer ?? 1) - 1;
        const hooked = rounds[(gates.round ?? 1) - 1]?.layers?.[layer];
        if ([step.c_previous, step.f, step.g, step.i].every(finite)) {
            add({ id: `lstm.cell.track${trackId}`, stage: 'decoder', title: 'LSTM · 更新记忆 cₜ', operation: 'lstm_cell',
                target: { unit, timestep: step.t }, inputs: [step.c_previous, step.g], weights: [step.f, step.i],
                bias: 0, actual: scalar(hooked?.c?.[unit]), input_indices: ['旧记忆 cₜ₋₁ × fₜ', '候选记忆 gₜ × iₜ'],
                formula: 'cₜ = cₜ₋₁ × fₜ + gₜ × iₜ', metadata: { reconstructed_cell: step.c } });
        }
        if ([step.c, step.o].every(finite)) {
            add({ id: `lstm.hidden.track${trackId}`, stage: 'decoder', title: 'LSTM · 输出隐藏状态 hₜ', operation: 'lstm_hidden',
                target: { unit, timestep: step.t }, inputs: [Math.tanh(step.c)], weights: [step.o],
                bias: 0, actual: scalar(hooked?.h?.[unit]), input_indices: [`tanh(cₜ=${step.c})`],
                formula: 'hₜ = tanh(cₜ) × oₜ', metadata: { cell: step.c, reconstructed_hidden: step.h } });
        }
    }
    rounds.forEach((round, index) => {
        const number = round.index ?? index + 1, decoder = round.decoder_example;
        if (decoder) add({ id: `lstm.decoder.round${number}.track${trackId}`, stage: 'decoder', title: `LSTM · 第 ${number} 轮全连接解码`,
            operation: 'linear', target: { round: number, output_index: decoder.output_index ?? 0 },
            inputs: decoder.input_values, weights: decoder.weights, bias: scalar(decoder.bias), actual: scalar(decoder.actual),
            formula: decoder.formula, metadata: { round: number, complete_sum: decoder.complete_sum } });
        const forecast = at(round.forecast?.values, 0, 0), actual = at(round.forecast_poses, 0, 0, 0);
        if ([forecast, mean, std, actual].every(finite)) add({ id: `lstm.denormalize.round${number}.track${trackId}`,
            stage: 'denormalization', title: `未来骨架 · 第 ${number} 轮恢复像素坐标`, operation: 'denormalization',
            target: { round: number, forecast_frame: 0, joint: 0, coordinate: 'x' }, inputs: [forecast, mean],
            weights: [std, 1], bias: 0, actual, input_indices: ['预测归一化坐标', 'mean'],
            formula: 'forecast_pixel = forecast_normalized × std + mean', metadata: { mean, std, round: number } });
    });
}

function savedStgcn(data, add, trackId) {
    const graph = data.stgcn;
    if (!graph) return;
    (graph.blocks || []).forEach((block, index) => {
        const number = block.index ?? index + 1, arithmetic = block.arithmetic || {};
        const spatial = arithmetic.spatial, temporal = arithmetic.temporal, residual = arithmetic.residual;
        if (spatial) {
            const adjacency = flattened(spatial.adjacency).map(cell => cell.value);
            addNested(add, { id: `stgcn.spatial.block${number}.track${trackId}`, stage: 'spatial', title: `ST-GCN ${number} · 空间邻域聚合`,
                operation: 'graph_aggregation', target: { block: number, timestep: spatial.timestep, joint: spatial.output_joint, channel: spatial.output_channel },
                bias: 0, actual: scalar(spatial.actual), connected: adjacency.map((value,i) => {const base=graph.adjacency?.[Math.floor(i/17)]?.[i%17]?.[spatial.output_joint];return finite(base)?base!==0:value!==0;}),
                formula: spatial.formula || 'Σ partition,source_joint(projected × effective_A)', metadata: { block: number, term_axes: spatial.term_axes,
                    target_indices:[spatial.output_channel,spatial.timestep,spatial.output_joint],graph_slice:recordedGraphSlice(graph,block,spatial) } },
            spatial.projected_values, spatial.adjacency, ([partition, joint]) => `partition ${partition} · joint ${joint}`);
        }
        if (temporal) {
            const cells = flattened(temporal.input_values);
            const length = block.input_shape?.[2];
            const padding = cells.map(cell => {
                const time = temporal.input_timesteps?.[cell.index[1]];
                return finite(time) && finite(length) && (time < 0 || time >= length);
            });
            addNested(add, { id: `stgcn.temporal.block${number}.track${trackId}`, stage: 'temporal', title: `ST-GCN ${number} · 时间卷积`,
                operation: 'temporal_convolution', target: { block: number, timestep: temporal.timestep, joint: temporal.output_joint, channel: temporal.output_channel },
                bias: scalar(temporal.bias), actual: scalar(temporal.actual), padding,
                formula: temporal.formula || 'Σ channel,kernel(W × x) + b', metadata: { block: number, kernel_size: temporal.kernel_size,
                    input_timesteps: temporal.input_timesteps, term_axes: temporal.term_axes } },
            temporal.input_values, temporal.weights, ([channel, kernel]) => `channel ${channel} · t=${temporal.input_timesteps?.[kernel] ?? '?'} · kernel ${kernel}`);
        }
        if (residual) add({ id: `stgcn.residual.block${number}.track${trackId}`, stage: 'temporal', title: `ST-GCN ${number} · 残差相加 → ReLU`,
            operation: 'residual_add', target: { block: number }, inputs: [scalar(residual.temporal), scalar(residual.residual)],
            weights: [1, 1], bias: 0, actual: scalar(residual.actual), activation: 'relu',
            connected: [true, residual.mode !== 'none'], input_indices: ['时间卷积 + BN 输出', '残差分支'],
            formula: residual.formula, metadata: { block: number, residual_mode: residual.mode } });
    });
    const pool = graph.pool_arithmetic;
    if (pool?.values && Number.isInteger(pool.count) && pool.count > 0) {
        const cells = flattened(pool.values);
        if (cells.length === pool.count) add({ id: `stgcn.pool.track${trackId}`, stage: 'pool', title: '全局池化 · 全部时间 × 关节点',
            operation: 'mean_pool', target: { channel: pool.channel }, inputs: cells.map(cell => cell.value),
            weights: cells.map(() => 1 / pool.count), bias: 0, actual: scalar(pool.actual),
            input_indices: cells.map(cell => `time ${cell.index[0]} · joint ${cell.index[1]}`),
            formula: pool.formula || 'pool = Σ time,joint(x) / (T × V)', metadata: { divisor: pool.count, input_shape: pool.input_shape } });
    }
    (graph.head_weights || []).forEach((weights, index) => {
        add({ id: `stgcn.classifier.class${index}.track${trackId}`, stage: 'classifier', title: `分类全连接 · ${graph.labels?.[index] ?? index}`,
            operation: 'linear', target: { class: index }, inputs: graph.pool, weights, bias: scalar(graph.head_bias?.[index]),
            actual: scalar(graph.logits?.[index]), formula: `logit[${index}] = Σ channel(pool × W[${index}]) + b[${index}]`,
            metadata: { class_labels: graph.labels, selected_class: index } });
    });
    const softmax = softmaxCalculation(`stgcn.softmax.track${trackId}`, { logits: graph.logits,
        probabilities: graph.probabilities, class_labels: graph.labels, selected_class: graph.class_id }, trackId);
    if (softmax) add(softmax);
}

/** A supplied live snapshot is an exclusive source, even if it is empty. */
export function collectCalculations({ snapshot, data, frontend, trackId, frame } = {}) {
    if (snapshot !== undefined && snapshot !== null) return liveCalculations(snapshot, trackId, frame);
    const result = [];
    if (recordMatches(frontend, frame, trackId, false)) {
        const context = recordContext(frontend, frame, null);
        savedYolo(frontend, raw => {
            const calculation = createCalculation(raw, context);
            if (calculation) result.push(calculation);
        });
    }
    if (recordMatches(data, frame, trackId, true)) {
        const context = recordContext(data, frame, trackId);
        const add = raw => {
            const calculation = createCalculation(raw, context);
            if (calculation) result.push(calculation);
        };
        savedLstm(data, add, context.track_id);
        savedStgcn(data, add, context.track_id);
    }
    return result;
}
import {recordedGraphSlice} from './spatialGraph.js';
