const finite = value => typeof value === 'number' && Number.isFinite(value);
const sigmoid = value => 1 / (1 + Math.exp(-value));

function batchNormalized(value, bn) {
    const denominator = finite(bn?.running_var) && finite(bn?.epsilon) ? bn.running_var + bn.epsilon : null;
    if (!bn || !finite(denominator) || denominator <= 0
        || ![bn.gamma, bn.beta, bn.running_mean].every(finite)) return null;
    return bn.gamma * (value - bn.running_mean) / Math.sqrt(denominator) + bn.beta;
}

export function activationValue(value, activation = 'identity', metadata = {}) {
    if (!finite(value)) return null;
    if (activation === 'identity') return value;
    if (activation === 'relu') return Math.max(0, value);
    if (activation === 'tanh') return Math.tanh(value);
    if (activation === 'sigmoid') return sigmoid(value);
    if (activation === 'silu') return value * sigmoid(value);
    if (activation === 'softmax') return finite(metadata.numerator) && value > 0 ? metadata.numerator / value : null;
    if (activation === 'batch_norm') return batchNormalized(value, metadata.bn);
    if (activation === 'batch_norm_silu') {
        const normalized = batchNormalized(value, metadata.bn);
        return normalized === null ? null : normalized * sigmoid(normalized);
    }
    return null;
}

function membershipOf({ recorded, connected, padding, product }) {
    return !recorded ? 'unrecorded' : !connected ? 'disconnected' : padding ? 'padding'
        : product === 0 ? 'zeroContribution' : 'connected';
}

function denseTerms(raw) {
    const requested = Array.isArray(raw.term_shape) && raw.term_shape.length
        ? raw.term_shape : [raw.inputs.length, 1, 1];
    const extents = requested.map(extent => Number.isInteger(extent) && extent > 0 ? extent : null);
    if (extents.some(extent => extent === null)
        || extents.reduce((product, extent) => product * extent, 1) !== raw.inputs.length) return null;
    const axes = Array.isArray(raw.term_axes) && raw.term_axes.length === extents.length
        ? raw.term_axes : ['in_channel', 'kernel_y', 'kernel_x'];
    const terms = [];
    for (let index = 0; index < raw.inputs.length; index += 1) {
        let remainder = index;
        const inputIndex = new Array(extents.length);
        for (let axis = extents.length - 1; axis >= 0; axis -= 1) {
            inputIndex[axis] = remainder % extents[axis];
            remainder = Math.floor(remainder / extents[axis]);
        }
        const input = raw.inputs[index], weight = raw.weights[index];
        const recorded = finite(input) && finite(weight);
        const connected = raw.connected?.[index] !== false;
        const padding = raw.padding?.[index] === true;
        const product = recorded ? input * weight : null;
        terms.push({
            index: inputIndex, inputIndex, ordinal: index, input, weight, product, recorded, connected, padding,
            membership: membershipOf({ recorded, connected, padding, product }),
            label: inputIndex.map((value, axis) => `${axes[axis] ?? `axis ${axis}`} ${value}`).join(' · '),
        });
    }
    return terms;
}

function legacyTerms(raw) {
    return raw.inputs.map((input, index) => {
        const weight = raw.weights[index];
        const connected = raw.connected?.[index] !== false;
        const padding = raw.padding?.[index] === true;
        const recorded = finite(input) && finite(weight);
        const product = recorded ? input * weight : null;
        return {
            index, inputIndex: raw.input_indices?.[index] ?? index, input, weight, product,
            connected, padding, recorded,
            membership: membershipOf({ recorded, connected, padding, product }),
            label: raw.metadata?.term_labels?.[index] ?? String(raw.input_indices?.[index] ?? index),
        };
    });
}

/** Expand dense detail rows or legacy rows into complete Kahan-prefixed terms. */
export function normalizeCalculation(raw) {
    if (!raw || !Array.isArray(raw.inputs) || !raw.inputs.length
        || !Array.isArray(raw.weights) || raw.inputs.length !== raw.weights.length) return null;
    const activation = raw.activation || 'identity';
    const terms = raw.operation === 'dense_conv' ? denseTerms(raw) : legacyTerms(raw);
    if (!terms) return null;
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
    const bias = finite(raw.bias) ? raw.bias : null;
    const preactivationValue = finite(dot) && finite(bias) ? dot + bias : null;
    const reconstructed = activationValue(preactivationValue, activation, raw.metadata || {});
    const actual = finite(raw.actual) ? raw.actual : null;
    const counts = terms.reduce((result, term) => {
        result.total += 1;
        result[term.membership] += 1;
        return result;
    }, { total: 0, connected: 0, disconnected: 0, zeroContribution: 0, padding: 0, unrecorded: 0 });
    return {
        ...raw, terms, prefixes, dot, bias, preactivationValue, reconstructed, actual,
        absoluteError: finite(actual) && finite(reconstructed) ? Math.abs(actual - reconstructed) : null,
        counts, maxCursor: terms.length + 2,
    };
}

/** Cursor is pedagogical: [0,N) terms, [N,N+1) bias, [N+1,N+2) activation, N+2 output. */
export function flowState(calculation, cursor = 0) {
    if (!calculation?.terms?.length) return null;
    const count = calculation.terms.length, end = count + 2;
    const position = finite(cursor) ? Math.max(0, Math.min(end, cursor)) : 0;
    const termIndex = Math.min(count - 1, Math.floor(position));
    const phase = position < count ? position - termIndex : 1;
    const done = position === end;
    const phaseName = done ? 'result' : position >= count + 1 ? 'activation' : position >= count ? 'bias'
        : phase < 0.35 ? 'input' : phase < 0.65 ? 'product' : phase < 0.85 ? 'accumulate' : 'accumulated';
    const completedCount = position >= count ? count : termIndex + (phase >= 0.85 ? 1 : 0);
    const biasApplied = position >= count;
    return {
        cursor: position, maxCursor: end, termIndex, phase, phaseName, completedCount,
        partialSum: biasApplied ? calculation.preactivationValue : calculation.prefixes[completedCount],
        product: calculation.terms[termIndex].product, currentTerm: calculation.terms[termIndex],
        before: calculation.prefixes[termIndex], after: calculation.prefixes[termIndex + 1],
        biasApplied, biasPhase: Math.max(0, Math.min(1, position - count)),
        activationPhase: Math.max(0, Math.min(1, position - count - 1)),
        finalValue: calculation.reconstructed, actual: calculation.actual, done, counts: calculation.counts,
    };
}
