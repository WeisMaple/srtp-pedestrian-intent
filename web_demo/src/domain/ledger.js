export const TERM_FILTERS = [
    { value: 'all', label: '全部项' },
    { value: 'disconnected', label: '无连接' },
    { value: 'zero', label: '贡献为零' },
    { value: 'unknown', label: '未采集' },
];
const STATUS = { disconnected: '无连接', padding: '零填充', unrecorded: '未采集',
    accumulated: '已累计', current: '当前项', pending: '待累计' };

export function filterTerms(terms, filter) {
    if (filter === 'disconnected') return terms.filter(term => term.connected === false);
    if (filter === 'zero') return terms.filter(term => term.recorded && term.connected && term.product === 0);
    if (filter === 'unknown') return terms.filter(term => !term.recorded);
    return terms;
}

/** Dense rows keep their coordinate array in `index`; seek/key by `ordinal`. */
export function termOrdinal(term) {
    if (Number.isInteger(term?.ordinal)) return term.ordinal;
    return Number.isInteger(term?.index) ? term.index : 0;
}

export function termStatus(term, index, state) {
    if (['disconnected', 'padding', 'unrecorded'].includes(term.membership)) return term.membership;
    if (index < (state?.completedCount || 0)) return 'accumulated';
    if (index === state?.termIndex && !state?.done) return 'current';
    return 'pending';
}

export function formatValue(value) {
    if (!Number.isFinite(value)) return '未采集';
    if (value === 0) return '0';
    return Math.abs(value) < 0.001 || Math.abs(value) >= 100000
        ? value.toExponential(3) : String(Number(value.toPrecision(6)));
}

export function statusLabel(status) {
    return STATUS[status] || status;
}
