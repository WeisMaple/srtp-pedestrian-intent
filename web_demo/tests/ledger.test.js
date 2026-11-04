import test from 'node:test';
import assert from 'node:assert/strict';
import { TERM_FILTERS, filterTerms, formatValue, termOrdinal, termStatus } from '../src/domain/ledger.js';

const terms = [
    { index: 0, connected: true, recorded: true, product: 2, membership: 'connected' },
    { index: 1, connected: false, recorded: true, product: 0, membership: 'disconnected' },
    { index: 2, connected: true, recorded: true, product: 0, membership: 'zeroContribution' },
    { index: 3, connected: true, recorded: false, product: null, membership: 'unrecorded' },
];

test('filters distinguish disconnected, zero and unrecorded terms', () => {
    assert.equal(filterTerms(terms, 'disconnected').length, 1);
    assert.equal(filterTerms(terms, 'zero').length, 1);
    assert.equal(filterTerms(terms, 'unknown').length, 1);
    assert.equal(filterTerms(terms, 'all').length, 4);
    assert.deepEqual(TERM_FILTERS.map(item => item.value), ['all', 'disconnected', 'zero', 'unknown']);
});

test('term status follows the shared cursor', () => {
    const state = { completedCount: 2, termIndex: 2, done: false };
    assert.equal(termStatus(terms[0], 0, state), 'accumulated');
    assert.equal(termStatus(terms[2], 2, state), 'current');
    assert.equal(termStatus(terms[3], 3, state), 'unrecorded');
});

test('formatValue never turns missing values into zero', () => {
    assert.equal(formatValue(null), '未采集');
    assert.equal(formatValue(0), '0');
    assert.equal(formatValue(0.1234567), '0.123457');
});

test('dense rows resolve ordinal rather than the coordinate index', () => {
    const dense = { index: [0, 0, 1], ordinal: 3, connected: true, recorded: true, product: 4, membership: 'connected' };
    const legacy = { index: 2, connected: true, recorded: true, product: 4, membership: 'connected' };
    assert.equal(termOrdinal(dense), 3);
    assert.equal(termOrdinal(legacy), 2);
    const state = { completedCount: 2, termIndex: 3, done: false };
    assert.equal(termStatus(dense, termOrdinal(dense), state), 'current');
});
