/** Numerical helpers shared by the arithmetic animation and its tests. */
export function clamp01(value) {
    return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

export function weightedTerms(inputs, weights, bias = 0) {
    if (!Array.isArray(inputs) || !Array.isArray(weights) || !inputs.length || inputs.length !== weights.length || !Number.isFinite(bias)) return null;
    const products = [];
    const prefixes = [0];
    let inputPeak = 0;
    let weightPeak = 0;
    for (let index = 0; index < inputs.length; index += 1) {
        if (!Number.isFinite(inputs[index]) || !Number.isFinite(weights[index])) return null;
        const product = inputs[index] * weights[index];
        const sum = prefixes[index] + product;
        if (!Number.isFinite(product) || !Number.isFinite(sum)) return null;
        products.push(product);
        prefixes.push(sum);
        inputPeak = Math.max(inputPeak, Math.abs(inputs[index]));
        weightPeak = Math.max(weightPeak, Math.abs(weights[index]));
    }
    const dot = prefixes.at(-1);
    if (!Number.isFinite(dot + bias)) return null;
    return { products, prefixes, dot, total: dot + bias, inputPeak: inputPeak || 1, weightPeak: weightPeak || 1 };
}

/** Long vectors linger on their first three real terms, then visit the rest before bias/output. */
export function accumulationFrame(progress, count) {
    const p = clamp01(progress);
    if (!Number.isInteger(count) || count < 1) return { index: 0, phase: 0, completed: 0, biasPhase: 0, outputPhase: 0 };
    const position = count > 9
        ? p < .45 ? p / .45 * 3 : 3 + (p - .45) / .43 * (count - 3)
        : p / .88 * count;
    const index = Math.min(count - 1, Math.floor(position));
    const phase = p >= .88 ? 1 : position - index;
    return {
        index,
        phase,
        completed: Math.min(count, index + (phase >= .78 ? 1 : 0)),
        biasPhase: clamp01((p - .88) / .07),
        // Keep the result visible before a parent timeline rolls into its next segment.
        outputPhase: clamp01((p - .95) / .02),
    };
}

export function activationValue(value, kind) {
    if (!Number.isFinite(value)) return null;
    if (kind === 'relu') return Math.max(0, value);
    if (kind === 'tanh') return Math.tanh(value);
    // Numerically stable on both tails, including real saturated gate values.
    return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

export function displayNumber(value, digits = 4) {
    if (!Number.isFinite(value)) return '—';
    if (value !== 0 && (Math.abs(value) < .0001 || Math.abs(value) >= 100000)) return value.toExponential(2);
    const result = value.toFixed(digits);
    return /^-0\.0+$/.test(result) ? result.slice(1) : result;
}
