import test from 'node:test';
import assert from 'node:assert/strict';
import { PAGES, hashForPage, pageFromHash } from '../src/app/page.js';

test('known hashes map to pages', () => {
    assert.deepEqual(PAGES, ['workbench', 'corridor', 'camera', 'results']);
    assert.equal(pageFromHash('#/workbench'), 'workbench');
    assert.equal(pageFromHash('#/corridor'), 'corridor');
    assert.equal(pageFromHash('#/results'), 'results');
    assert.equal(pageFromHash('#/camera'), 'camera');
});

test('unknown or empty hashes fall back to workbench', () => {
    assert.equal(pageFromHash(''), 'workbench');
    assert.equal(pageFromHash('#/unknown'), 'workbench');
    assert.equal(pageFromHash(null), 'workbench');
});

test('hashForPage always produces a known route', () => {
    assert.equal(hashForPage('corridor'), '#/corridor');
    assert.equal(hashForPage('results'), '#/results');
    assert.equal(hashForPage('nope'), '#/workbench');
});
