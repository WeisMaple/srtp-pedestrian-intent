# 蓝图长廊 · 前端重写实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写整个网页为「蓝图长廊」：工作台、六步三维长廊、两处计算放大镜、数值台与结果页；骨架识别与 ID 绑定清楚可见，所有卷积阶段都能查看真实代表算式。

**Architecture:** 单一 Three.js 长廊场景 + DOM 叠层标签；六步共享一个全局游标；实时快照与后端详细缓存驱动所有数值；纯数据适配放在 `domain/`，场景只绘制与拾取。视觉规范来自 `docs/superpowers/specs/2026-11-04-blueprint-corridor-design.md`。

**Tech Stack:** React 19、Vite 6、Three.js 0.186、原生 CSS、`node --test`。无新增依赖。

## Global Constraints

- 1920×1080 投影为基准；1440 宽与竖屏平板可用；手机只保证可查看。
- 全站深色蓝图风格：背景 `#061225 → #081A31`，结构线 `#E8F1FF`，焦点 `#6FD6FF`，当前项 `#FFB454`，负值 `#FF8A7A`，未采集 `#5F86AB`。
- 三维场景内最多 3 个信息卡；密集数值只进底部数值台。
- 任何显示数值必须来自 `/api/jobs/{id}/live`、记录接口或 `/api/model`；缺失显示「未采集」，不补零。
- 保持现有上传、分析、恢复、下载流程；服务仍为本地 `127.0.0.1`。
- 不新增 UI 框架、图表库或远程字体；中文使用系统无衬线。
- 自动讲解受真实数据门控：没有该步数据时停在等待状态。
- 播放时间是讲解速度，不是推理耗时；界面必须标明。
- 保持 React 19 / Vite 6 / Three.js 0.186；不修改 `AGENTS.md`。
- 测试命令：`cd web_demo && npm test && npm run build`；浏览器检查用 Playwright Chromium。
- 每个任务结束提交一次，使用描述性祈使句。

---

### Task 1: 应用外壳、蓝图 token 与哈希路由

**Files:**
- Modify: `web_demo/src/main.jsx`
- Create: `web_demo/src/App.jsx`
- Create: `web_demo/src/app/page.js`
- Create: `web_demo/src/components/AppShell.jsx`
- Create: `web_demo/src/styles/tokens.css`
- Create: `web_demo/src/styles/base.css`
- Create: `web_demo/tests/page.test.js`

**Interfaces:**
- Produces: `PAGES = ['workbench', 'corridor', 'results']`；`pageFromHash(hash)`；`hashForPage(page)`。
- Produces: `AppShell({page, onNavigate, status, children})`。
- Produces: CSS 变量 `--bg-0`, `--bg-1`, `--line`, `--focus`, `--current`, `--positive`, `--negative`, `--muted`, `--font-sans`, `--font-mono`, `--text-1`, `--text-2`, `--step-title`, `--body`, `--label`, `--value`。

- [ ] **Step 1: 写失败的路由测试**

Create `web_demo/tests/page.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { PAGES, hashForPage, pageFromHash } from '../src/app/page.js';

test('known hashes map to pages', () => {
    assert.deepEqual(PAGES, ['workbench', 'corridor', 'results']);
    assert.equal(pageFromHash('#/workbench'), 'workbench');
    assert.equal(pageFromHash('#/corridor'), 'corridor');
    assert.equal(pageFromHash('#/results'), 'results');
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/page.test.js`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `src/app/page.js`.

- [ ] **Step 3: 实现路由、token 与外壳**

Create `web_demo/src/app/page.js`:

```javascript
export const PAGES = ['workbench', 'corridor', 'results'];
const DEFAULT_PAGE = 'workbench';

export function pageFromHash(hash) {
    const value = String(hash || '').replace(/^#\/?/, '').split('/')[0];
    return PAGES.includes(value) ? value : DEFAULT_PAGE;
}

export function hashForPage(page) {
    return '#/' + (PAGES.includes(page) ? page : DEFAULT_PAGE);
}
```

Create `web_demo/src/styles/tokens.css`:

```css
:root {
    --bg-0: #061225;
    --bg-1: #081a31;
    --bg-2: #0b2340;
    --line: #e8f1ff;
    --line-soft: rgba(232, 241, 255, 0.22);
    --line-dim: rgba(232, 241, 255, 0.09);
    --focus: #6fd6ff;
    --focus-soft: rgba(111, 214, 255, 0.16);
    --current: #ffb454;
    --current-soft: rgba(255, 180, 84, 0.16);
    --positive: #a8e9ff;
    --negative: #ff8a7a;
    --muted: #5f86ab;
    --danger: #ff6b6b;
    --text-1: #e8f1ff;
    --text-2: #8fb3d4;
    --font-sans: "PingFang SC", "Noto Sans SC", system-ui, -apple-system, sans-serif;
    --font-mono: "SFMono-Regular", Menlo, Consolas, monospace;
    --step-title: clamp(22px, 1.8vw, 30px);
    --body: clamp(14px, 0.95vw, 17px);
    --label: clamp(11px, 0.7vw, 13px);
    --value: clamp(13px, 0.9vw, 17px);
    --page-pad: clamp(12px, 1.4vw, 24px);
    --rail-width: clamp(240px, 18vw, 320px);
    --top-height: 52px;
    color-scheme: dark;
}
```

Create `web_demo/src/styles/base.css`:

```css
* { box-sizing: border-box; }
html, body, #root { height: 100%; margin: 0; }
body {
    background: radial-gradient(1200px 700px at 50% -10%, #123557 0%, var(--bg-1) 48%, var(--bg-0) 100%);
    color: var(--text-1);
    font-family: var(--font-sans);
    font-size: var(--body);
    line-height: 1.55;
    overflow: hidden;
}
button, input, select { font: inherit; color: inherit; }
button { cursor: pointer; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.app-shell { height: 100%; display: flex; flex-direction: column; }
.app-main { flex: 1; min-height: 0; position: relative; }
.app-top {
    height: var(--top-height); display: flex; align-items: center; gap: 18px;
    padding: 0 var(--page-pad); border-bottom: 1px solid var(--line-dim);
    background: rgba(6, 18, 37, 0.82); backdrop-filter: blur(10px); z-index: 20;
}
.app-brand { font-weight: 600; letter-spacing: 0.18em; font-size: 13px; }
.app-brand span { color: var(--focus); }
.app-nav { display: flex; gap: 6px; }
.app-nav button { border: 1px solid transparent; background: none; color: var(--text-2); padding: 6px 12px; border-radius: 4px; font-size: 13px; }
.app-nav button[aria-current="page"] { color: var(--focus); border-color: var(--focus); background: var(--focus-soft); }
.app-status { margin-left: auto; display: flex; align-items: center; gap: 8px; color: var(--text-2); font: 12px var(--font-mono); }
.app-status i { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
.app-status.ready i { background: var(--focus); box-shadow: 0 0 8px var(--focus); }
.mono { font-family: var(--font-mono); }
.muted { color: var(--text-2); }
.blueprint-panel { background: linear-gradient(180deg, rgba(11, 35, 64, 0.92), rgba(6, 18, 37, 0.92)); border: 1px solid var(--line-soft); border-radius: 6px; }
.blueprint-button { border: 1px solid var(--focus); background: var(--focus-soft); color: var(--focus); padding: 8px 14px; border-radius: 4px; font-size: 13px; }
.blueprint-button[disabled] { opacity: 0.4; cursor: not-allowed; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
```

Create `web_demo/src/components/AppShell.jsx`:

```jsx
import { PAGES } from '../app/page.js';
const LABELS = { workbench: '工作台', corridor: '长廊', results: '结果' };

export default function AppShell({ page, onNavigate, status, children }) {
    return <div className="app-shell">
        <header className="app-top">
            <div className="app-brand">看见推理 <span>·</span> 行人意图</div>
            <nav className="app-nav" aria-label="页面">
                {PAGES.map(value => <button key={value} aria-current={page === value ? 'page' : undefined}
                    onClick={() => onNavigate(value)}>{LABELS[value]}</button>)}
            </nav>
            <div className={`app-status ${status?.ready ? 'ready' : ''}`}><i />{status?.label || '本地模型'}</div>
        </header>
        <main className="app-main">{children}</main>
    </div>;
}
```

Replace `web_demo/src/App.jsx` with a temporary shell that will be completed in Task 9:

```jsx
import { useEffect, useState } from 'react';
import AppShell from './components/AppShell';
import { hashForPage, pageFromHash } from './app/page';

export default function App() {
    const [page, setPage] = useState(() => pageFromHash(window.location.hash));
    useEffect(() => {
        const onChange = () => setPage(pageFromHash(window.location.hash));
        window.addEventListener('hashchange', onChange);
        return () => window.removeEventListener('hashchange', onChange);
    }, []);
    const navigate = next => { window.location.hash = hashForPage(next); setPage(pageFromHash(hashForPage(next))); };
    return <AppShell page={page} onNavigate={navigate} status={{ ready: false, label: '本地模型' }}>
        <div className="page-placeholder">正在加载 {page}…</div>
    </AppShell>;
}
```

Modify `web_demo/src/main.jsx`:

```jsx
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/tokens.css';
import './styles/base.css';

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);
```

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/page.test.js && npm run build`

Expected: 3 routing tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/main.jsx web_demo/src/App.jsx web_demo/src/app/page.js web_demo/src/components/AppShell.jsx web_demo/src/styles/tokens.css web_demo/src/styles/base.css web_demo/tests/page.test.js
git commit -m "Add blueprint app shell and hash routing"
```

---

### Task 2: 模型目录客户端与实时详细状态

**Files:**
- Modify: `web_demo/src/api.js`
- Create: `web_demo/src/hooks/useModelStructure.js`
- Create: `web_demo/src/domain/model.js`
- Create: `web_demo/tests/model.test.js`
- Modify: `web_demo/src/liveComputation.js`
- Modify: `web_demo/tests/liveComputation.test.js`

**Interfaces:**
- Produces: `modelUrl()`；`yoloConvUrl(jobId, frame, module)`。
- Produces: `normalizeModelStructure(raw) -> {yoloModules, yoloById, stgcnBlocks, stgcnById, classes, labels, versions}`。
- Produces: `useModelStructure() -> {structure, loading, error, retry}`。
- Produces: `detailForModule(snapshot, module)`、`detailForBlock(snapshot, block)`、`detailCoverage(snapshot)`、`mergeDetail(snapshot, payload)`。
- Produces: `reduceLiveEvent` 的 snapshot 保留 `details`，computation 事件合并详细行。

- [ ] **Step 1: 写模型归一化与详细缓存测试**

Create `web_demo/tests/model.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { detailCoverage, detailForBlock, detailForModule, normalizeModelStructure } from '../src/domain/model.js';

const raw = {
    yolo: { modules: [
        { index: 0, type: 'Conv', from: [-1], representative: { kernel: [3, 3], in_channels: 3, out_channels: 32 } },
        { index: 1, type: 'C3k2', from: [0], representative: null },
    ] },
    stgcn: { blocks: [{ index: 1, out_channels: 64 }, { index: 2, out_channels: 64 }], classes: ['a', 'b', 'c'], labels: ['其他', '不过街', '过街'] },
    versions: { stgcn_epoch: 121 },
};

test('normalizes module and block lookups', () => {
    const model = normalizeModelStructure(raw);
    assert.equal(model.yoloById.get(0).type, 'Conv');
    assert.equal(model.stgcnBlocks.length, 2);
    assert.deepEqual(model.labels, ['其他', '不过街', '过街']);
    assert.equal(model.versions.stgcn_epoch, 121);
});

test('reads latest detail rows from a live snapshot', () => {
    const snapshot = { details: { yolo: { 4: { id: 'yolo.conv.4', module: 4 } }, stgcn: { 2: [{ id: 's' }] } } };
    assert.equal(detailForModule(snapshot, 4).id, 'yolo.conv.4');
    assert.equal(detailForModule(snapshot, 5), null);
    assert.equal(detailForBlock(snapshot, 2).length, 1);
    assert.deepEqual(detailCoverage(snapshot), { yolo: [4], stgcn: [2] });
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/model.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现模型客户端与详细缓存**

Modify `web_demo/src/api.js`:

```javascript
export const modelUrl = () => '/api/model';
export const yoloConvUrl = (id, frame, module) =>
    jobUrl(id, `/yolo-conv?frame=${encodeURIComponent(frame)}&module=${encodeURIComponent(module)}`);
```

Create `web_demo/src/domain/model.js`:

```javascript
const EMPTY = { yolo: {}, stgcn: {} };

export function normalizeModelStructure(raw) {
    const modules = raw?.yolo?.modules || [];
    const blocks = raw?.stgcn?.blocks || [];
    return {
        yoloModules: modules,
        yoloById: new Map(modules.map(module => [module.index, module])),
        stgcnBlocks: blocks,
        stgcnById: new Map(blocks.map(block => [block.index, block])),
        classes: raw?.stgcn?.classes || ['irrelevant', 'not crossing', 'crossing'],
        labels: raw?.stgcn?.labels || ['其他', '不过街', '过街'],
        versions: raw?.versions || {},
    };
}

export function detailForModule(snapshot, module) {
    return snapshot?.details?.yolo?.[module] || null;
}

export function detailForBlock(snapshot, block) {
    return snapshot?.details?.stgcn?.[block] || [];
}

export function detailCoverage(snapshot) {
    const details = snapshot?.details || EMPTY;
    return {
        yolo: Object.keys(details.yolo || {}).map(Number).sort((a, b) => a - b),
        stgcn: Object.keys(details.stgcn || {}).map(Number).sort((a, b) => a - b),
    };
}

export function mergeDetail(snapshot, payload) {
    const rows = payload?.calculations || [];
    if (!rows.length || snapshot?.frame !== payload?.frame) return snapshot?.details || EMPTY;
    const details = {
        yolo: { ...(snapshot?.details?.yolo || {}) },
        stgcn: { ...(snapshot?.details?.stgcn || {}) },
    };
    for (const row of rows) {
        if (!row?.metadata?.detail) continue;
        if (row.stage === 'yolo') {
            details.yolo[row.module] = row;
        } else {
            const block = row.metadata.detail_block;
            if (block == null) continue;
            details.stgcn[block] = [...(details.stgcn[block] || []), row].slice(-12);
        }
    }
    return details;
}
```

Create `web_demo/src/hooks/useModelStructure.js`:

```javascript
import { useCallback, useEffect, useState } from 'react';
import { modelUrl, request } from '../api';
import { normalizeModelStructure } from '../domain/model';

export function useModelStructure() {
    const [raw, setRaw] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const load = useCallback(async () => {
        setLoading(true); setError('');
        try { setRaw(await request(modelUrl())); }
        catch (cause) { setError(cause.message || '模型结构未就绪。'); }
        finally { setLoading(false); }
    }, []);
    useEffect(() => { load(); }, [load]);
    return { structure: raw ? normalizeModelStructure(raw) : null, loading, error, retry: load };
}
```

Modify `web_demo/src/liveComputation.js`:
- Import `mergeDetail` from `./domain/model.js`.
- In `createLiveState`, add `details: { yolo: {}, stgcn: {} }`.
- In the `snapshot` branch, set `details: payload.details || { yolo: {}, stgcn: {} }`.
- In the computation branch, set `details: mergeDetail(state.snapshot, payload)` inside the new snapshot.

Add to `web_demo/tests/liveComputation.test.js`:

```javascript
import { mergeDetail } from '../src/domain/model.js';

test('detail rows merge into a live snapshot without replacing other modules', () => {
    const snapshot = { frame: 3, details: { yolo: { 0: { id: 'yolo.conv.0' } }, stgcn: {} } };
    const payload = { frame: 3, calculations: [
        { stage: 'yolo', module: 4, id: 'yolo.conv.4', metadata: { detail: true } },
        { stage: 'spatial', id: 'stgcn.spatial.block1.track1', metadata: { detail: true, detail_block: 1 } },
    ] };
    const details = mergeDetail(snapshot, payload);
    assert.equal(details.yolo[0].id, 'yolo.conv.0');
    assert.equal(details.yolo[4].id, 'yolo.conv.4');
    assert.equal(details.stgcn[1].length, 1);
    assert.deepEqual(mergeDetail(snapshot, { frame: 4, calculations: payload.calculations }), snapshot.details);
});
```

- [ ] **Step 4: 运行测试**

Run: `cd web_demo && node --test tests/model.test.js tests/liveComputation.test.js`

Expected: model tests and existing live tests PASS.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/api.js web_demo/src/hooks/useModelStructure.js web_demo/src/domain/model.js web_demo/src/liveComputation.js web_demo/tests/model.test.js web_demo/tests/liveComputation.test.js
git commit -m "Add model catalog client and live detail cache"
```

---

### Task 3: 稠密计算、姿态与时间轴领域逻辑

**Files:**
- Create: `web_demo/src/domain/calculation.js`
- Create: `web_demo/src/domain/pose.js`
- Create: `web_demo/src/domain/timeline.js`
- Create: `web_demo/tests/domain-calculation.test.js`
- Create: `web_demo/tests/domain-pose.test.js`
- Create: `web_demo/tests/domain-timeline.test.js`

**Interfaces:**
- Produces: `normalizeCalculation(raw)`、`flowState(calculation, cursor)`、`activationValue(value, activation, metadata)`。
- Produces: `COCO_EDGES`、`asPose(value)`、`normalizePose(pose, size, frame)`、`poseBounds(poses)`。
- Produces: `STEPS`、`beatsForStep(step)`、`stepReady(step, snapshot, job)`、`nextBeat(state, availableBeats)`。

- [ ] **Step 1: 写领域测试**

Create `web_demo/tests/domain-calculation.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { flowState, normalizeCalculation } from '../src/domain/calculation.js';

const dense = {
    id: 'yolo.conv.0', stage: 'yolo', operation: 'dense_conv', module: 0,
    term_axes: ['in_channel', 'kernel_y', 'kernel_x'], term_shape: [1, 1, 2],
    inputs: [2, 3], weights: [4, 5], bias: 1, actual: 11,
    activation: 'identity', fused: true, metadata: { complete_sum: true, detail: true },
};

test('dense convolution expands every term and accumulates completely', () => {
    const calculation = normalizeCalculation(dense);
    assert.equal(calculation.terms.length, 2);
    assert.deepEqual(calculation.terms[0].index, [0, 0, 0]);
    assert.equal(calculation.terms[0].product, 8);
    assert.equal(calculation.terms[1].product, 15);
    assert.equal(calculation.dot, 23);
    assert.equal(calculation.preactivationValue, 24);
    assert.equal(calculation.reconstructed, 24);
    assert.equal(calculation.absoluteError, 13);
});

test('cursor phases cover terms, bias, activation and result', () => {
    const calculation = normalizeCalculation({ ...dense, actual: 24 });
    assert.equal(flowState(calculation, 0).phaseName, 'input');
    assert.equal(flowState(calculation, 0.9).phaseName, 'accumulated');
    assert.equal(flowState(calculation, 2).phaseName, 'bias');
    assert.equal(flowState(calculation, 3).phaseName, 'activation');
    assert.equal(flowState(calculation, 4).phaseName, 'result');
    assert.equal(flowState(calculation, 2).partialSum, 24);
});

test('missing operands stay missing and never become zero', () => {
    const calculation = normalizeCalculation({ ...dense, inputs: [2, null], actual: null });
    assert.equal(calculation.terms[1].recorded, false);
    assert.equal(calculation.terms[1].product, null);
    assert.equal(calculation.terms[1].membership, 'unrecorded');
    assert.equal(calculation.dot, null);
});
```

Create `web_demo/tests/domain-pose.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { asPose, normalizePose, poseBounds } from '../src/domain/pose.js';

test('flat 51 values become 17 x 3 points', () => {
    const flat = Array.from({ length: 51 }, (_, index) => index);
    const pose = asPose(flat);
    assert.equal(pose.length, 17);
    assert.deepEqual(pose[1], [3, 4, 5]);
});

test('normalization centers and flips image y', () => {
    const pose = [[0, 0, 1], [10, 0, 1], [0, 10, 1]];
    const result = normalizePose(pose, 10);
    assert.ok(Math.abs(result[0][0] + 5) < 1e-9);
    assert.ok(Math.abs(result[2][1] + 5) < 1e-9);
});

test('bounds ignore low-confidence points', () => {
    const points = [[[0, 0, 0.9], [10, 10, 0.1]]];
    assert.deepEqual(poseBounds(points), { minX: 0, maxX: 0, minY: 0, maxY: 0 });
});
```

Create `web_demo/tests/domain-timeline.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { STEPS, beatsForStep, nextBeat, stepReady } from '../src/domain/timeline.js';

test('six steps and their beats are stable', () => {
    assert.equal(STEPS.length, 6);
    assert.deepEqual(STEPS.map(step => step.title), ['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图']);
    assert.ok(beatsForStep(2).length >= 2);
});

test('step readiness is data-gated', () => {
    const empty = { tracks: [], nodes: {}, details: { yolo: {}, stgcn: {} } };
    assert.equal(stepReady(1, empty, { metadata: {} }), true);
    assert.equal(stepReady(2, empty, { metadata: {} }), false);
    const posed = { tracks: [{ track_id: 1, keypoints: [[1, 2, 3]] }], nodes: {}, details: { yolo: {}, stgcn: {} } };
    assert.equal(stepReady(2, posed, { metadata: {} }), true);
    assert.equal(stepReady(4, posed, { metadata: {} }), false);
    assert.equal(stepReady(6, { ...posed, nodes: { 'stgcn.logits': { frame: 1 } } }, { metadata: {} }), true);
});

test('nextBeat advances and stops at the end', () => {
    assert.equal(nextBeat({ step: 1, beat: 0, paused: false, following: true }, 3), 1);
    assert.equal(nextBeat({ step: 6, beat: 2, paused: false, following: true }, 3), null);
    assert.equal(nextBeat({ step: 1, beat: 0, paused: true, following: true }, 3), null);
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/domain-calculation.test.js tests/domain-pose.test.js tests/domain-timeline.test.js`

Expected: FAIL with missing modules.

- [ ] **Step 3: 实现领域逻辑**

Create `web_demo/src/domain/calculation.js`:

```javascript
const ACTIVATIONS = new Set(['identity', 'relu', 'tanh', 'sigmoid', 'silu', 'batch_norm_silu', 'softmax']);
const finite = value => typeof value === 'number' && Number.isFinite(value);

export function activationValue(value, activation = 'identity', metadata = {}) {
    if (!finite(value)) return null;
    if (activation === 'identity') return value;
    if (activation === 'relu') return Math.max(0, value);
    if (activation === 'tanh') return Math.tanh(value);
    if (activation === 'sigmoid') return 1 / (1 + Math.exp(-value));
    if (activation === 'silu') return value / (1 + Math.exp(-value));
    if (activation === 'softmax') return finite(metadata.numerator) && value > 0 ? metadata.numerator / value : null;
    if (activation === 'batch_norm_silu') {
        const bn = metadata.bn;
        if (!bn || ![bn.gamma, bn.beta, bn.running_mean, bn.running_var, bn.epsilon].every(finite)) return null;
        const normalized = bn.gamma * (value - bn.running_mean) / Math.sqrt(bn.running_var + bn.epsilon) + bn.beta;
        return normalized / (1 + Math.exp(-normalized));
    }
    return null;
}

function denseTerms(raw) {
    const shape = raw.term_shape || [raw.inputs.length, 1, 1];
    const axes = raw.term_axes || ['in_channel', 'kernel_y', 'kernel_x'];
    const terms = [];
    let index = 0;
    const visit = (prefix, axis) => {
        const size = shape[axis] || 1;
        for (let i = 0; i < size; i += 1) {
            if (axis === shape.length - 1) {
                const input = raw.inputs[index];
                const weight = raw.weights[index];
                terms.push({
                    index, inputIndex: [...prefix, i], input, weight,
                    recorded: finite(input) && finite(weight),
                    product: finite(input) && finite(weight) ? input * weight : null,
                    connected: true, padding: false,
                    label: axes.map((name, at) => `${name} ${[...prefix, i][at]}`).join(' · '),
                });
                index += 1;
            } else {
                visit([...prefix, i], axis + 1);
            }
        }
    };
    visit([], 0);
    return terms;
}

function legacyTerms(raw) {
    return raw.inputs.map((input, index) => {
        const weight = raw.weights[index];
        const connected = raw.connected?.[index] !== false;
        const padding = raw.padding?.[index] === true;
        const recorded = finite(input) && finite(weight);
        const product = recorded ? input * weight : null;
        const membership = !recorded ? 'unrecorded' : !connected ? 'disconnected' : padding ? 'padding'
            : product === 0 ? 'zeroContribution' : 'connected';
        return { index, inputIndex: raw.input_indices?.[index] ?? index, input, weight, product,
            connected, padding, recorded, membership,
            label: raw.metadata?.term_labels?.[index] ?? String(raw.input_indices?.[index] ?? index) };
    });
}

export function normalizeCalculation(raw) {
    if (!raw || !Array.isArray(raw.inputs) || !Array.isArray(raw.weights) || raw.inputs.length !== raw.weights.length) return null;
    const activation = ACTIVATIONS.has(raw.activation) ? raw.activation : 'identity';
    const terms = raw.operation === 'dense_conv' ? denseTerms(raw) : legacyTerms(raw);
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
    const bias = raw.bias;
    const preactivationValue = finite(dot) && finite(bias) ? dot + bias : null;
    const reconstructed = activationValue(preactivationValue, activation, raw.metadata || {});
    const actual = finite(raw.actual) ? raw.actual : null;
    const counts = terms.reduce((result, term) => {
        result.total += 1;
        result[term.membership || (term.recorded ? 'connected' : 'unrecorded')] += 1;
        return result;
    }, { total: 0, connected: 0, disconnected: 0, zeroContribution: 0, padding: 0, unrecorded: 0 });
    return { ...raw, terms, prefixes, dot, preactivationValue, reconstructed, actual,
        absoluteError: finite(actual) && finite(reconstructed) ? Math.abs(actual - reconstructed) : null,
        counts, maxCursor: terms.length + 2 };
}

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
    const biasApplied = position >= count + 1;
    return { cursor: position, maxCursor: end, termIndex, phase, phaseName, completedCount,
        partialSum: biasApplied ? calculation.preactivationValue : calculation.prefixes[completedCount],
        product: calculation.terms[termIndex].product, currentTerm: calculation.terms[termIndex],
        before: calculation.prefixes[termIndex], after: calculation.prefixes[termIndex + 1],
        biasApplied, biasPhase: Math.max(0, Math.min(1, position - count)),
        activationPhase: Math.max(0, Math.min(1, position - count - 1)),
        finalValue: calculation.reconstructed, actual: calculation.actual, done, counts: calculation.counts };
}
```

Create `web_demo/src/domain/pose.js`:

```javascript
export const COCO_EDGES = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
export const JOINT_NAMES = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'];
const finite = value => typeof value === 'number' && Number.isFinite(value);

export function validPoint(point) {
    return Array.isArray(point) && finite(point[0]) && finite(point[1]) && (point.length < 3 || point[2] > 0.05);
}

export function asPose(value) {
    const source = value?.keypoints || value?.pose || value;
    if (!Array.isArray(source)) return null;
    if (source.length === 51 && !Array.isArray(source[0])) {
        return Array.from({ length: 17 }, (_, index) => source.slice(index * 3, index * 3 + 3));
    }
    return source.length === 17 && source.every(point => Array.isArray(point) && finite(point[0]) && finite(point[1]))
        ? source : null;
}

export function poseBounds(poses) {
    const points = (poses || []).flatMap(pose => asPose(pose) || []).filter(validPoint);
    if (!points.length) return null;
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

export function normalizePose(pose, size = 3.6, frame = null) {
    const points = asPose(pose);
    if (!points) return null;
    const bounds = frame || poseBounds([points]);
    if (!bounds) return null;
    const midX = (bounds.minX + bounds.maxX) / 2;
    const midY = (bounds.minY + bounds.maxY) / 2;
    const scale = size / Math.max(bounds.maxY - bounds.minY, bounds.maxX - bounds.minX, 1e-6);
    return points.map(point => [(point[0] - midX) * scale, -(point[1] - midY) * scale,
        finite(point[2]) ? point[2] : null]);
}
```

Create `web_demo/src/domain/timeline.js`:

```javascript
export const STEPS = [
    { step: 1, title: '视频帧', kicker: 'FRAME', beats: ['读取源帧', 'Letterbox / RGB', '送入模型'] },
    { step: 2, title: 'YOLO 骨架', kicker: 'POSE', beats: ['检测行人', '定位 17 关节', '写入置信度'] },
    { step: 3, title: 'ID 绑定', kicker: 'TRACK', beats: ['检测关联', '累积历史', '等待 50 帧'] },
    { step: 4, title: 'LSTM 预测', kicker: 'MEMORY', beats: ['编码历史', '五轮预测', '还原坐标'] },
    { step: 5, title: 'ST-GCN', kicker: 'GRAPH', beats: ['空间邻居', '时间卷积', '残差输出'] },
    { step: 6, title: '意图', kicker: 'CLASS', beats: ['全局池化', '分类头', 'Softmax'] },
];

export function beatsForStep(step) {
    return STEPS.find(item => item.step === step)?.beats || [];
}

export function stepReady(step, snapshot = {}, job = {}) {
    const tracks = snapshot?.tracks || [];
    const nodes = snapshot?.nodes || {};
    if (step === 1) return Boolean(job?.metadata);
    if (step === 2) return tracks.length > 0;
    if (step === 3) return tracks.some(track => (track.history_length || 0) > 0 || track.keypoints);
    if (step === 4) return Boolean(nodes['lstm.1']) || tracks.some(track => (track.history_length || 0) >= 50);
    if (step === 5) return Object.keys(nodes).some(node => node.startsWith('stgcn.'));
    if (step === 6) return Boolean(nodes['stgcn.logits']) || tracks.some(track => Array.isArray(track.probabilities));
    return false;
}

export function nextBeat(state, availableBeats) {
    if (state?.paused) return null;
    const beat = Number.isFinite(state?.beat) ? state.beat : 0;
    return beat + 1 < availableBeats ? beat + 1 : null;
}
```

- [ ] **Step 4: 运行测试**

Run: `cd web_demo && node --test tests/domain-calculation.test.js tests/domain-pose.test.js tests/domain-timeline.test.js`

Expected: all domain tests PASS.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/domain/calculation.js web_demo/src/domain/pose.js web_demo/src/domain/timeline.js web_demo/tests/domain-calculation.test.js web_demo/tests/domain-pose.test.js web_demo/tests/domain-timeline.test.js
git commit -m "Add blueprint domain adapters for calculations, poses and timeline"
```

---

### Task 4: 长廊布局、相机与场景底座

**Files:**
- Create: `web_demo/src/scene/layout.js`
- Create: `web_demo/src/scene/camera.js`
- Create: `web_demo/src/scene/materials.js`
- Create: `web_demo/src/scene/corridorScene.js`
- Create: `web_demo/src/components/CorridorStage.jsx`
- Create: `web_demo/src/styles/corridor.css`
- Create: `web_demo/tests/layout.test.js`

**Interfaces:**
- Produces: `BAYS`、`bayForStep(step)`、`cameraForStep(step, {focus})`。
- Produces: `createCorridorScene(host, {onPick, onError}) -> {setStep, focus, reset, append, dispose}`。
- Produces: `CorridorStage({step, onPick, onError, children})`。

- [ ] **Step 1: 写布局/相机测试**

Create `web_demo/tests/layout.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { BAYS, bayForStep, cameraForStep } from '../src/scene/layout.js';

test('six bays are finite and ordered left to right', () => {
    assert.equal(BAYS.length, 6);
    assert.deepEqual(BAYS.map(bay => bay.step), [1, 2, 3, 4, 5, 6]);
    for (let index = 1; index < BAYS.length; index += 1) assert.ok(BAYS[index].position[0] > BAYS[index - 1].position[0]);
    assert.ok(BAYS.every(bay => bay.position.every(Number.isFinite)));
});

test('camera keeps a readable distance and looks at the bay', () => {
    const camera = cameraForStep(3);
    const distance = Math.hypot(camera.position[0] - camera.target[0], camera.position[1] - camera.target[1], camera.position[2] - camera.target[2]);
    assert.ok(distance > 8 && distance < 30);
    assert.equal(camera.target[0], bayForStep(3).position[0]);
});

test('unknown steps fall back to the first bay', () => {
    assert.equal(bayForStep(99).step, 1);
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/layout.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现布局、相机、材质与场景底座**

Create `web_demo/src/scene/layout.js`:

```javascript
export const BAYS = Array.from({ length: 6 }, (_, index) => ({
    step: index + 1,
    position: [(index - 2.5) * 7, 0, (index - 2.5) * 1.2],
    label: ['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图'][index],
}));

export function bayForStep(step) {
    return BAYS.find(bay => bay.step === step) || BAYS[0];
}

export function cameraForStep(step, { focus = null } = {}) {
    const bay = bayForStep(step);
    const target = [bay.position[0], 2.1, bay.position[2]];
    if (focus) return { target: focus, position: [focus[0] + 5.5, focus[1] + 4.2, focus[2] + 12] };
    return { target, position: [target[0] + 1.5, 8.2, target[2] + 16] };
}
```

Create `web_demo/src/scene/materials.js`:

```javascript
import * as THREE from 'three';

export const COLORS = {
    background: 0x061225, grid: 0x17456b, line: 0xe8f1ff, focus: 0x6fd6ff,
    current: 0xffb454, positive: 0xa8e9ff, negative: 0xff8a7a, muted: 0x5f86ab, danger: 0xff6b6b,
};

export function lineMaterial(color = COLORS.line, opacity = 0.5) {
    return new THREE.LineBasicMaterial({ color, transparent: true, opacity });
}

export function glowMaterial(color = COLORS.focus, opacity = 0.5) {
    return new THREE.MeshBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false });
}

export function pointMaterial(color = COLORS.focus, size = 0.08) {
    return new THREE.PointsMaterial({ color, size, transparent: true, opacity: 0.9, sizeAttenuation: true });
}
```

Create `web_demo/src/scene/camera.js`:

```javascript
import * as THREE from 'three';

export function createCameraRig(host) {
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 300);
    let motion = null;
    function setAspect(width, height) {
        camera.aspect = Math.max(1, width) / Math.max(1, height);
        camera.updateProjectionMatrix();
    }
    function moveTo(target, position, smooth = true) {
        if (!smooth) { camera.position.set(...position); motion = null; return; }
        motion = { target: new THREE.Vector3(...target), position: new THREE.Vector3(...position) };
    }
    function update() {
        if (!motion) return false;
        camera.position.lerp(motion.position, 0.085);
        if (camera.position.distanceTo(motion.position) < 0.02) {
            camera.position.copy(motion.position); motion = null;
        }
        return true;
    }
    return { camera, setAspect, moveTo, update, lookAt: target => camera.lookAt(...target) };
}
```

Create `web_demo/src/scene/corridorScene.js`:

```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BAYS, cameraForStep } from './layout.js';
import { COLORS, lineMaterial, glowMaterial } from './materials.js';

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

export function createCorridorScene(host, { onPick, onError } = {}) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);
    scene.fog = new THREE.Fog(COLORS.background, 60, 150);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.className = 'corridor-canvas';
    renderer.domElement.setAttribute('aria-label', '六步神经网络长廊。拖动旋转，滚轮缩放，点击对象检查计算。');
    renderer.domElement.setAttribute('role', 'img');
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 300);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.minDistance = 4; controls.maxDistance = 80; controls.maxPolarAngle = Math.PI * 0.86;
    controls.enabled = false;
    scene.add(new THREE.HemisphereLight(0xcdeaff, 0x061225, 1.6));
    const sun = new THREE.DirectionalLight(0x9fe8ff, 1.4); sun.position.set(-8, 16, 12); scene.add(sun);
    const floor = new THREE.GridHelper(140, 70, 0x17456b, 0x0c2b47);
    floor.position.y = -3.2; floor.material.transparent = true; floor.material.opacity = 0.4; scene.add(floor);
    const root = new THREE.Group(); scene.add(root);
    const bays = new Map();
    const pickables = [];
    let step = 1, motion = null, disposed = false, frame = 0, lastTime = performance.now();

    function makeLabel(text, position, size = 0.32, color = COLORS.line) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 512; canvas.height = 64;
        context.font = '500 32px "SFMono-Regular", Menlo, monospace';
        context.textAlign = 'center'; context.textBaseline = 'middle';
        context.fillStyle = '#' + new THREE.Color(color).getHexString();
        context.fillText(text, 256, 32);
        const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
        sprite.position.set(...position); sprite.scale.set(5 * size * 2, 0.62 * size * 2, 1);
        sprite.renderOrder = 4; root.add(sprite); return sprite;
    }

    function bayGroup(bay) {
        const group = new THREE.Group(); group.position.set(...bay.position); root.add(group);
        const plate = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 3.4), glowMaterial(COLORS.focus, 0.035));
        plate.position.set(0, 1.9, 0); plate.rotation.x = -Math.PI / 2; group.add(plate);
        const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(5.8, 4.2, 3.2)), lineMaterial(COLORS.line, 0.28));
        box.position.set(0, 1.4, 0); group.add(box);
        makeLabel('0' + bay.step + ' / ' + bay.label, [0, 4.3, 0], 0.3);
        group.userData.bay = bay; pickables.push(box); return group;
    }
    BAYS.forEach(bay => bays.set(bay.step, bayGroup(bay)));

    function frameCamera(animate = true) {
        const pose = cameraForStep(step);
        if (!animate) { camera.position.set(...pose.position); camera.lookAt(...pose.target); motion = null; return; }
        motion = { target: V(...pose.target), position: V(...pose.position) };
    }
    function updateCamera() {
        if (!motion) return;
        camera.position.lerp(motion.position, 0.08);
        const look = camera.position.clone().add(motion.target.clone().sub(camera.position).normalize().multiplyScalar(10));
        camera.lookAt(look);
        if (camera.position.distanceTo(motion.position) < 0.03) {
            camera.position.copy(motion.position); camera.lookAt(motion.target); motion = null;
        }
    }
    function resize() {
        const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
        renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
    }
    const observer = new ResizeObserver(resize); observer.observe(host); resize();

    function select(event) {
        if (motion) return;
        const rect = renderer.domElement.getBoundingClientRect();
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
            -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
        const hit = raycaster.intersectObjects(pickables, false)[0];
        if (!hit) return;
        const bay = hit.object.parent?.userData?.bay;
        if (bay) onPick?.({ kind: 'bay', step: bay.step });
    }
    renderer.domElement.addEventListener('pointerup', select);
    renderer.domElement.addEventListener('webglcontextlost', event => {
        event.preventDefault(); onError?.('三维渲染上下文已中断，数值台仍可使用。');
    });

    function animate(now) {
        if (disposed) return;
        frame = requestAnimationFrame(animate);
        lastTime = now; updateCamera(); controls.update(); renderer.render(scene, camera);
    }
    frameCamera(false); frame = requestAnimationFrame(animate);

    return {
        setStep(next) { step = next; controls.enabled = false; frameCamera(true); },
        focus(target) { controls.enabled = true; camera.position.set(target[0] + 5.5, target[1] + 4.2, target[2] + 12); camera.lookAt(...target); motion = null; },
        reset() { controls.enabled = false; frameCamera(true); },
        append(object) { root.add(object); },
        dispose() {
            disposed = true; cancelAnimationFrame(frame); observer.disconnect();
            renderer.domElement.removeEventListener('pointerup', select); controls.dispose();
            renderer.dispose(); renderer.domElement.remove();
        },
    };
}
```

Create `web_demo/src/components/CorridorStage.jsx`:

```jsx
import { useEffect, useRef, useState } from 'react';
import { createCorridorScene } from '../scene/corridorScene';

export default function CorridorStage({ step, onPick, onError, children }) {
    const host = useRef(null);
    const engine = useRef(null);
    const [error, setError] = useState('');
    useEffect(() => {
        try {
            engine.current = createCorridorScene(host.current, {
                onPick, onError: message => { setError(message); onError?.(message); },
            });
        } catch (cause) { setError(cause.message || 'WebGL 不可用'); }
        return () => { engine.current?.dispose(); engine.current = null; };
    }, []);
    useEffect(() => { engine.current?.setStep(step); }, [step]);
    if (error) return <div className="corridor-fallback" role="alert">
        <strong>三维画布暂不可用</strong><p>{error}</p>{children}
    </div>;
    return <div className="corridor-stage" ref={host} />;
}
```

Create `web_demo/src/styles/corridor.css`:

```css
.corridor-page { height: 100%; display: grid; grid-template-columns: var(--rail-width) 1fr; overflow: hidden; }
.corridor-stage { position: relative; height: 100%; min-height: 0; }
.corridor-canvas { display: block; width: 100%; height: 100%; }
.corridor-fallback { padding: 24px; display: grid; gap: 8px; align-content: center; }
.corridor-main { display: grid; grid-template-rows: 1fr auto auto; min-width: 0; }
.corridor-stage-wrap { position: relative; min-height: 0; }
.stage-overlay { position: absolute; inset: 0; pointer-events: none; }
.stage-overlay > * { pointer-events: auto; }
```

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/layout.test.js && npm run build`

Expected: layout tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/scene/layout.js web_demo/src/scene/camera.js web_demo/src/scene/materials.js web_demo/src/scene/corridorScene.js web_demo/src/components/CorridorStage.jsx web_demo/src/styles/corridor.css web_demo/tests/layout.test.js
git commit -m "Add blueprint corridor layout and Three.js stage"
```

---

### Task 5: 步骤 ①②③ — 源帧、检测骨架、ID 绑定

**Files:**
- Create: `web_demo/src/scene/skeletonLayer.js`
- Create: `web_demo/src/scene/idRibbon.js`
- Create: `web_demo/src/domain/tracks.js`
- Create: `web_demo/src/components/ContextRail.jsx`
- Create: `web_demo/tests/tracks.test.js`
- Modify: `web_demo/src/scene/corridorScene.js`（新增 ID 带）
- Modify: `web_demo/src/components/CorridorStage.jsx`（接收 `track`）

**Interfaces:**
- Produces: `historyPoses(track)`、`historyGaps(track)`、`trackIsWarm(track)`。
- Produces: `buildSkeletonGroup(pose, {color,size,frame})`、`updateSkeletonGroup(group, pose, options)`。
- Produces: `buildRibbonGroup(track, {maxFrames,spacing})`。
- Produces: `ContextRail({job,tracks,track,onSelect})`。
- Produces: `createCorridorScene(...).setTrack(track)` 与 `CorridorStage({step, track, ...})`。

- [ ] **Step 1: 写历史姿态测试**

Create `web_demo/tests/tracks.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { historyGaps, historyPoses, trackIsWarm } from '../src/domain/tracks.js';

test('history keeps real poses and marks missing ones as gaps', () => {
    const track = { history: [[[1, 2, 3]], null, [[4, 5, 6]]], history_length: 3 };
    const poses = historyPoses(track);
    assert.equal(poses.length, 3);
    assert.equal(poses[1], null);
    assert.deepEqual(historyGaps(track), [1]);
});

test('warm-up is explicit before fifty observations', () => {
    assert.equal(trackIsWarm({ history_length: 49 }), false);
    assert.equal(trackIsWarm({ history_length: 50 }), true);
});
```

Create `web_demo/src/domain/tracks.js`:

```javascript
export function historyPoses(track) {
    const history = track?.history;
    if (Array.isArray(history)) return history.slice(-50);
    return track?.keypoints ? [track.keypoints] : [];
}

export function historyGaps(track) {
    return historyPoses(track).reduce((gaps, pose, index) => {
        if (!pose) gaps.push(index);
        return gaps;
    }, []);
}

export function trackIsWarm(track) {
    return (track?.history_length || 0) >= 50;
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/tracks.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现姿态与 ID 图层**

Create `web_demo/src/scene/skeletonLayer.js`:

```javascript
import * as THREE from 'three';
import { COCO_EDGES, normalizePose } from '../domain/pose';
import { COLORS, lineMaterial } from './materials';

const JOINTS = 17;

export function buildSkeletonGroup(pose, { color = COLORS.focus, size = 2.4, frame = null } = {}) {
    const group = new THREE.Group();
    const joints = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color, size: 0.09, transparent: true, opacity: 0.95 }));
    const jointGeometry = new THREE.BufferGeometry();
    jointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(JOINTS * 3), 3));
    joints.geometry = jointGeometry;
    const bones = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial(color, 0.8));
    bones.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(COCO_EDGES.length * 6), 3));
    group.add(joints, bones);
    updateSkeletonGroup(group, pose, { size, frame });
    return group;
}

export function updateSkeletonGroup(group, pose, { size = 2.4, frame = null } = {}) {
    const points = normalizePose(pose, size, frame);
    const [joints, bones] = group.children;
    if (!points) { joints.visible = false; bones.visible = false; return; }
    joints.visible = true; bones.visible = true;
    const jointArray = joints.geometry.attributes.position.array;
    const boneArray = bones.geometry.attributes.position.array;
    for (let index = 0; index < JOINTS; index += 1) {
        const point = points[index] || [0, 0, 0];
        jointArray.set([point[0], point[1], 0], index * 3);
    }
    COCO_EDGES.forEach(([a, b], index) => {
        const left = points[a] || [0, 0, 0], right = points[b] || [0, 0, 0];
        boneArray.set([left[0], left[1], 0, right[0], right[1], 0], index * 6);
    });
    joints.geometry.attributes.position.needsUpdate = true;
    bones.geometry.attributes.position.needsUpdate = true;
    joints.geometry.computeBoundingSphere(); bones.geometry.computeBoundingSphere();
}
```

Create `web_demo/src/scene/idRibbon.js`:

```javascript
import * as THREE from 'three';
import { historyPoses } from '../domain/tracks';
import { buildSkeletonGroup } from './skeletonLayer';
import { COLORS, lineMaterial } from './materials';

export function buildRibbonGroup(track, { maxFrames = 50, spacing = 0.62 } = {}) {
    const group = new THREE.Group();
    const poses = historyPoses(track).slice(-maxFrames);
    poses.forEach((pose, index) => {
        if (!pose) {
            group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(index * spacing, -0.8, 0), new THREE.Vector3(index * spacing, 0.8, 0)]),
                lineMaterial(COLORS.muted, 0.4)));
            return;
        }
        const skeleton = buildSkeletonGroup(pose, { color: COLORS.focus, size: 1.5 });
        skeleton.position.set(index * spacing, 0, 0); group.add(skeleton);
    });
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -1.3, 0), new THREE.Vector3(Math.max(1, poses.length - 1) * spacing, -1.3, 0)]),
        lineMaterial(COLORS.current, 0.5)));
    return group;
}
```

Create `web_demo/src/components/ContextRail.jsx`:

```jsx
import { CLASS_LABELS, formatTime, jobUrl } from '../api';
import { COCO_EDGES, asPose } from '../domain/pose';
import { historyPoses, trackIsWarm } from '../domain/tracks';

function MiniSkeleton({ pose }) {
    const points = asPose(pose);
    if (!points) return <div className="rail-skeleton empty">暂无真实骨架</div>;
    return <svg className="rail-skeleton" viewBox="0 0 100 120" aria-label="所选行人骨架">
        {COCO_EDGES.map(([a, b], index) => <line key={index} x1={points[a][0] / 12} y1={points[a][1] / 12}
            x2={points[b][0] / 12} y2={points[b][1] / 12} stroke="currentColor" strokeWidth="1.6"
            vectorEffect="non-scaling-stroke" />)}
        {points.map((point, index) => <circle key={index} cx={point[0] / 12} cy={point[1] / 12} r="1.5" fill="currentColor" />)}
    </svg>;
}

export default function ContextRail({ job, tracks, track, onSelect }) {
    const history = historyPoses(track);
    const warm = trackIsWarm(track);
    return <aside className="context-rail">
        <div className="rail-heading"><span>01 / VIDEO CONTEXT</span><b>视频与行人</b></div>
        <div className="rail-video">
            {job ? <video src={jobUrl(job.id, '/files/source')} poster={jobUrl(job.id, '/files/poster')}
                muted playsInline preload="metadata" /> : <div className="rail-empty">等待视频</div>}
            <div className="rail-meta mono">{job ? `${job.metadata?.frame_count || '—'} 帧 · ${formatTime(job.metadata?.duration)}` : '—'}</div>
        </div>
        <div className="rail-person">
            <MiniSkeleton pose={track?.keypoints} />
            <div>
                <span className="rail-id mono">{track ? `ID ${String(track.track_id).padStart(2, '0')}` : '无行人'}</span>
                <strong>{track ? (track.class_id != null ? CLASS_LABELS[track.class_id] : warm ? '等待预测' : '积累历史中') : '等待检测'}</strong>
                <small className="mono">历史 {track?.history_length || history.length} / 50 · 17 关节</small>
            </div>
        </div>
        <div className="rail-scores">
            {(track?.probabilities || [null, null, null]).map((value, index) => <div key={index}>
                <span>{CLASS_LABELS[index]}</span><i><em style={{ width: `${Math.max(0, value || 0) * 100}%` }} /></i>
                <b className="mono">{typeof value === 'number' ? value.toFixed(3) : '—'}</b>
            </div>)}
        </div>
        <div className="rail-people">{tracks.map(person => <button key={person.track_id}
            aria-pressed={person.track_id === track?.track_id} onClick={() => onSelect(person.track_id)}>
            ID {String(person.track_id).padStart(2, '0')}
        </button>)}</div>
        <p className="rail-note">ID 为分析时 ByteTrack 的真实归属；匹配代价未记录，因此不显示虚构分数。</p>
    </aside>;
}
```

Append to `corridor.css`:

```css
.context-rail { border-right: 1px solid var(--line-dim); padding: 14px; display: flex; flex-direction: column; gap: 12px; background: rgba(6,18,37,.6); overflow: auto; }
.rail-heading span { font: 11px var(--font-mono); color: var(--muted); letter-spacing: .14em; }
.rail-heading b { display: block; font-size: 16px; }
.rail-video video { width: 100%; border: 1px solid var(--line-soft); border-radius: 4px; aspect-ratio: 16/9; object-fit: cover; background: #000; }
.rail-meta { margin-top: 4px; font-size: 11px; color: var(--text-2); }
.rail-person { display: flex; gap: 10px; align-items: center; }
.rail-skeleton { width: 78px; height: 94px; color: var(--focus); }
.rail-skeleton.empty { display: grid; place-items: center; font-size: 10px; color: var(--muted); }
.rail-id { display: block; color: var(--current); font-size: 12px; }
.rail-person strong { display: block; font-size: 18px; font-weight: 500; }
.rail-person small { color: var(--text-2); font-size: 11px; }
.rail-scores { display: grid; gap: 6px; }
.rail-scores > div { display: grid; grid-template-columns: 48px 1fr 46px; gap: 6px; align-items: center; font-size: 12px; }
.rail-scores i { height: 4px; background: var(--line-dim); border-radius: 2px; overflow: hidden; }
.rail-scores em { display: block; height: 100%; background: var(--focus); }
.rail-people { display: flex; gap: 6px; flex-wrap: wrap; }
.rail-people button { border: 1px solid var(--line-soft); background: none; color: var(--text-2); padding: 4px 8px; border-radius: 3px; font: 12px var(--font-mono); }
.rail-people button[aria-pressed="true"] { color: var(--current); border-color: var(--current); }
.rail-note { margin: 0; color: var(--muted); font-size: 11px; line-height: 1.5; }
```

- [ ] **Step 3b: 把 ID 带接入场景**

Modify `web_demo/src/scene/corridorScene.js`:

```javascript
import { buildRibbonGroup } from './idRibbon.js';
```

Add after the bay loop:

```javascript
    const ribbon = new THREE.Group();
    ribbon.position.set(...BAYS[2].position).add(V(0, 1.2, 0));
    root.add(ribbon);
    let trackId = null;
    function setTrack(track) {
        if (String(track?.track_id ?? '') === String(trackId ?? '')) return;
        trackId = track?.track_id ?? null;
        while (ribbon.children.length) {
            const child = ribbon.children.pop();
            child.traverse?.(object => {
                object.geometry?.dispose?.();
                object.material?.dispose?.();
            });
        }
        if (track) ribbon.add(buildRibbonGroup(track));
    }
```

Add `setTrack` to the returned object. Update `CorridorStage.jsx`:

```jsx
export default function CorridorStage({ step, track, onPick, onError, children }) {
    // ...existing host/engine setup...
    useEffect(() => { engine.current?.setTrack(track); }, [track]);
    // ...
}
```

In the Task 9 `Corridor.jsx` stage call, pass `track={selectedTrack}`.

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/tracks.test.js tests/domain-pose.test.js && npm run build`

Expected: tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/scene/skeletonLayer.js web_demo/src/scene/idRibbon.js web_demo/src/domain/tracks.js web_demo/src/components/ContextRail.jsx web_demo/src/styles/corridor.css web_demo/tests/tracks.test.js
git commit -m "Add source frame, skeleton and ID ribbon layers"
```

---

### Task 6: 步骤 ④⑤⑥ — LSTM、ST-GCN 与意图

**Files:**
- Create: `web_demo/src/domain/liveLayers.js`
- Create: `web_demo/src/scene/memoryLayer.js`
- Create: `web_demo/src/scene/featureLayer.js`
- Create: `web_demo/src/scene/graphLayer.js`
- Create: `web_demo/src/scene/intentLayer.js`
- Create: `web_demo/tests/liveLayers.test.js`

**Interfaces:**
- Produces: `liveLayerModel(snapshot, trackId, structure) -> {frame, track, features, memory, graph, intent}`。
- Produces: `buildFeatureLayer/updateFeatureLayer`、`buildMemoryLayer/updateMemoryLayer`、`buildGraphLayer/updateGraphLayer`、`buildIntentLayer/updateIntentLayer`。

- [ ] **Step 1: 写纯数据图层测试**

Create `web_demo/tests/liveLayers.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { liveLayerModel } from '../src/domain/liveLayers.js';

const structure = { yoloModules: [{ index: 0 }], stgcnBlocks: [{ index: 1 }] };
const snapshot = {
    frame: 7,
    nodes: {
        'yolo.0': { frame: 7, tensors: { output: { values: [[[[1, 2], [3, 4]]]], shape: [1, 1, 2, 2], indices: [[0], [0], [0, 1], [0, 1]] } } },
        'stgcn.logits': { frame: 7, tensors: { output: { values: [1, 2, 3], shape: [3], indices: [[0, 1, 2]] } } },
    },
    tracks: [{ track_id: 2, history_length: 50, probabilities: [0.1, 0.2, 0.7], class_id: 2 }],
    details: { yolo: {}, stgcn: {} },
};

test('live layers select the chosen track and real probabilities', () => {
    const model = liveLayerModel(snapshot, 2, structure);
    assert.equal(model.frame, 7);
    assert.equal(model.intent.probabilities.length, 3);
    assert.equal(model.intent.classId, 2);
    assert.ok(model.features.length >= 1);
});

test('missing probabilities stay null instead of zero', () => {
    const model = liveLayerModel({ ...snapshot, tracks: [{ track_id: 2 }] }, 2, structure);
    assert.equal(model.intent.probabilities, null);
});
```

Create `web_demo/src/domain/liveLayers.js`:

```javascript
export function liveLayerModel(snapshot, trackId, structure) {
    const track = (snapshot?.tracks || []).find(item => String(item.track_id) === String(trackId)) || null;
    const nodes = snapshot?.nodes || {};
    const features = (structure?.yoloModules || []).map(module => ({
        module: module.index,
        output: nodes[`yolo.${module.index}`]?.tensors?.output || null,
        detail: snapshot?.details?.yolo?.[module.index] || null,
    }));
    const memory = {
        rounds: Object.keys(nodes).filter(node => /^lstm\.\d+$/.test(node)).sort(),
        history: nodes.history?.tensors?.history || null,
        future: nodes.future?.tensors?.future || null,
    };
    const graph = {
        blocks: (structure?.stgcnBlocks || []).map(block => ({
            block: block.index,
            output: nodes[`stgcn.${block.index - 1}.output`]?.tensors?.output || null,
            details: snapshot?.details?.stgcn?.[block.index - 1] || [],
        })),
    };
    return { frame: snapshot?.frame ?? -1, track, features, memory, graph,
        intent: { probabilities: Array.isArray(track?.probabilities) ? track.probabilities : null,
            classId: Number.isInteger(track?.class_id) ? track.class_id : null } };
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/liveLayers.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现场景图层模块**

Create `web_demo/src/scene/featureLayer.js`:

```javascript
import * as THREE from 'three';
import { COLORS, lineMaterial, pointMaterial } from './materials';

export function buildFeatureLayer(count = 23) {
    const group = new THREE.Group();
    const points = [];
    Array.from({ length: count }, (_, index) => {
        const x = (index % 8 - 3.5) * 1.5, y = 1.2 + Math.floor(index / 8) * 1.4, z = index * 0.12;
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.7), new THREE.MeshBasicMaterial({
            color: COLORS.focus, transparent: true, opacity: 0.06, side: THREE.DoubleSide }));
        mesh.position.set(x, y, z); mesh.rotation.x = -Math.PI / 2; group.add(mesh);
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1.1, 0.7)),
            lineMaterial(COLORS.focus, 0.35));
        outline.position.copy(mesh.position); outline.rotation.copy(mesh.rotation); group.add(outline); points.push(mesh);
    });
    const dots = new THREE.Points(new THREE.BufferGeometry(), pointMaterial(COLORS.focus, 0.06));
    group.add(dots);
    return { group, points, dots };
}

export function updateFeatureLayer(layer, model) {
    model.features.forEach((feature, index) => {
        const value = feature.output?.values?.[0]?.[0]?.[0]?.[0];
        const strength = Number.isFinite(value) ? Math.min(1, Math.abs(value)) : 0;
        const mesh = layer.points[index];
        if (mesh) mesh.material.opacity = 0.05 + strength * 0.25;
    });
}
```

Create `web_demo/src/scene/memoryLayer.js`:

```javascript
import * as THREE from 'three';
import { COLORS } from './materials';

export function buildMemoryLayer() {
    const group = new THREE.Group(); const bars = [];
    for (let index = 0; index < 5; index += 1) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.2, 0.6),
            new THREE.MeshBasicMaterial({ color: COLORS.focus, transparent: true, opacity: 0.2 }));
        bar.position.set(index * 1.2, 0, 0); group.add(bar); bars.push(bar);
    }
    return { group, bars };
}

export function updateMemoryLayer(layer, memory) {
    layer.bars.forEach((bar, index) => {
        const active = index < (memory.rounds?.length || 0);
        bar.material.opacity = active ? 0.55 : 0.14;
        bar.scale.y = active ? 1.4 : 0.8;
    });
}
```

Create `web_demo/src/scene/graphLayer.js`:

```javascript
import * as THREE from 'three';
import { COCO_EDGES, normalizePose } from '../domain/pose';
import { COLORS, lineMaterial, pointMaterial } from './materials';

export function buildGraphLayer() {
    const group = new THREE.Group();
    const points = new THREE.Points(new THREE.BufferGeometry(), pointMaterial(COLORS.focus, 0.11));
    const bones = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial(COLORS.focus, 0.6));
    group.add(points, bones); return { group, points, bones };
}

export function updateGraphLayer(layer, pose) {
    const points = normalizePose(pose, 4.2);
    if (!points) { layer.points.visible = false; layer.bones.visible = false; return; }
    layer.points.visible = true; layer.bones.visible = true;
    layer.points.geometry.setAttribute('position', new THREE.Float32BufferAttribute(
        points.flatMap(point => [point[0], point[1], 0]), 3));
    const segments = [];
    COCO_EDGES.forEach(([a, b]) => segments.push(points[a][0], points[a][1], 0, points[b][0], points[b][1], 0));
    layer.bones.geometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
    layer.points.geometry.computeBoundingSphere(); layer.bones.geometry.computeBoundingSphere();
}
```

Create `web_demo/src/scene/intentLayer.js`:

```javascript
import * as THREE from 'three';
import { COLORS } from './materials';

export function buildIntentLayer(labels) {
    const group = new THREE.Group();
    const bars = labels.map((label, index) => {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7),
            new THREE.MeshBasicMaterial({ color: index === 2 ? COLORS.focus : COLORS.muted, transparent: true, opacity: 0.65 }));
        bar.position.set((index - 1) * 1.4, 0, 0); group.add(bar); return bar;
    });
    return { group, bars };
}

export function updateIntentLayer(layer, probabilities) {
    layer.bars.forEach((bar, index) => {
        const value = Array.isArray(probabilities) ? probabilities[index] : 0;
        const height = Math.max(0.04, (Number.isFinite(value) ? value : 0) * 6);
        bar.scale.y = height; bar.position.y = height / 2 - 2;
        bar.material.opacity = Array.isArray(probabilities) ? 0.35 + 0.65 * Math.min(1, value * 2) : 0.2;
    });
}
```

- [ ] **Step 3b: 把 LSTM / ST-GCN / 意图图层接入场景**

Modify `web_demo/src/scene/corridorScene.js`:

```javascript
import { buildFeatureLayer, updateFeatureLayer } from './featureLayer.js';
import { buildMemoryLayer, updateMemoryLayer } from './memoryLayer.js';
import { buildGraphLayer, updateGraphLayer } from './graphLayer.js';
import { buildIntentLayer, updateIntentLayer } from './intentLayer.js';
```

Add after the bay loop:

```javascript
    const feature = buildFeatureLayer(23);
    feature.group.position.set(...BAYS[1].position).add(V(0, 1.5, 0)); root.add(feature.group);
    const memory = buildMemoryLayer();
    memory.group.position.set(...BAYS[3].position).add(V(0, 1.2, 0)); root.add(memory.group);
    const graph = buildGraphLayer();
    graph.group.position.set(...BAYS[4].position).add(V(0, 1.4, 0)); root.add(graph.group);
    const intent = buildIntentLayer(['其他', '不过街', '过街']);
    intent.group.position.set(...BAYS[5].position).add(V(0, 1, 0)); root.add(intent.group);
```

Add to the returned object:

```javascript
        setLive(model) {
            updateFeatureLayer(feature, model);
            updateMemoryLayer(memory, model.memory);
            updateGraphLayer(graph, model.track?.keypoints || null);
            updateIntentLayer(intent, model.intent.probabilities);
        },
```

Modify `web_demo/src/components/CorridorStage.jsx`:

```jsx
import { liveLayerModel } from '../domain/liveLayers';

export default function CorridorStage({ step, track, live, trackId, structure, onPick, onError, children }) {
    // ...existing host/engine setup...
    useEffect(() => {
        if (!live || !engine.current) return;
        engine.current.setLive(liveLayerModel(live, trackId, structure));
    }, [live, trackId, structure]);
    // ...
}
```

The Task 9 `Corridor.jsx` stage call already passes `live`, `trackId`, and `structure`. Add `track={selectedTrack}`.

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/liveLayers.test.js && npm run build`

Expected: tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/domain/liveLayers.js web_demo/src/scene/memoryLayer.js web_demo/src/scene/featureLayer.js web_demo/src/scene/graphLayer.js web_demo/src/scene/intentLayer.js web_demo/tests/liveLayers.test.js
git commit -m "Add live LSTM, graph and intent corridor layers"
```

---

### Task 7: 全局游标、自动讲解与播放控制

**Files:**
- Create: `web_demo/src/app/useCorridor.js`
- Create: `web_demo/src/hooks/useReducedMotion.js`
- Create: `web_demo/src/components/TransportDeck.jsx`
- Create: `web_demo/src/components/StepNav.jsx`
- Create: `web_demo/tests/corridorState.test.js`
- Modify: `web_demo/src/styles/corridor.css`

**Interfaces:**
- Produces: `createCorridorState()`、`corridorReducer(state, action)`、`advanceNarration(state, {snapshot, job})`。
- Produces: `useCorridor({job, live, trackId}) -> {state, actions}`；动作 `selectStep`, `selectTrack`, `togglePause`, `seekTerm`, `next`, `previous`, `selectModule`, `selectBlock`, `followLatest`。
- Produces: `TransportDeck`、`StepNav`。

- [ ] **Step 1: 写游标/门控测试**

Create `web_demo/tests/corridorState.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { advanceNarration, corridorReducer, createCorridorState } from '../src/app/useCorridor.js';

test('selecting a step pauses auto narration and resets the beat', () => {
    const state = createCorridorState();
    const next = corridorReducer(state, { type: 'step', step: 4 });
    assert.equal(next.step, 4);
    assert.equal(next.beat, 0);
    assert.equal(next.paused, true);
});

test('narration advances only when the step is ready', () => {
    const state = { ...createCorridorState(), step: 2, ready: false };
    assert.equal(advanceNarration(state, { snapshot: {}, job: {} }).beat, 0);
    const ready = { ...state, ready: true };
    assert.equal(advanceNarration(ready, { snapshot: {}, job: {} }).beat, 1);
});

test('pause freezes all cursor actions that advance time', () => {
    const state = corridorReducer(createCorridorState(), { type: 'pause' });
    assert.equal(state.paused, true);
    assert.equal(corridorReducer(state, { type: 'next' }).beat, state.beat);
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/corridorState.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现游标与播放控制**

Create `web_demo/src/app/useCorridor.js`:

```javascript
import { useEffect, useMemo, useReducer, useRef } from 'react';
import { STEPS, beatsForStep, stepReady } from '../domain/timeline';

export function createCorridorState() {
    return { step: 1, beat: 0, paused: false, following: true, selectedTrackId: null,
        module: 0, block: 1, termIndex: 0, lens: null, ready: true };
}

export function corridorReducer(state, action) {
    switch (action.type) {
        case 'step': return { ...state, step: action.step, beat: 0, paused: true, following: false, lens: null };
        case 'beat': return { ...state, beat: action.beat };
        case 'pause': return { ...state, paused: true, following: false };
        case 'play': return { ...state, paused: false, following: true };
        case 'ready': return { ...state, ready: Boolean(action.ready) };
        case 'track': return { ...state, selectedTrackId: action.trackId, termIndex: 0 };
        case 'module': return { ...state, module: action.module, termIndex: 0 };
        case 'block': return { ...state, block: action.block, termIndex: 0 };
        case 'term': return { ...state, termIndex: action.index };
        case 'next': return state.paused ? state : advanceNarration(state, action);
        case 'beat-advance': return advanceNarration(state, action);
        default: return state;
    }
}

export function advanceNarration(state, { snapshot, job } = {}) {
    if (state.paused || !state.ready) return state;
    const available = beatsForStep(state.step).length;
    const beat = state.beat + 1;
    if (beat >= available) {
        const nextStep = Math.min(STEPS.length, state.step + 1);
        return { ...state, step: nextStep, beat: 0, ready: stepReady(nextStep, snapshot, job) };
    }
    return { ...state, beat };
}

export function useCorridor({ job, live, trackId }) {
    const [state, dispatch] = useReducer(corridorReducer, undefined, createCorridorState);
    const stateRef = useRef(state); stateRef.current = state;
    useEffect(() => {
        if (trackId != null && stateRef.current.selectedTrackId !== trackId) dispatch({ type: 'track', trackId });
    }, [trackId]);
    useEffect(() => { dispatch({ type: 'ready', ready: stepReady(state.step, live, job) }); }, [state.step, live, job]);
    useEffect(() => {
        if (state.paused) return undefined;
        const timer = window.setTimeout(() => dispatch({ type: 'beat-advance', snapshot: live, job }), 2200);
        return () => window.clearTimeout(timer);
    }, [state.step, state.beat, state.paused, live, job]);
    const actions = useMemo(() => ({
        selectStep: step => dispatch({ type: 'step', step }),
        selectTrack: id => dispatch({ type: 'track', trackId: id }),
        selectModule: module => dispatch({ type: 'module', module }),
        selectBlock: block => dispatch({ type: 'block', block }),
        seekTerm: index => dispatch({ type: 'term', index }),
        seekBeat: beat => dispatch({ type: 'beat', beat }),
        togglePause: () => dispatch({ type: stateRef.current.paused ? 'play' : 'pause' }),
        followLatest: () => dispatch({ type: 'play' }),
        next: () => dispatch({ type: 'next', snapshot: live, job }),
        previous: () => dispatch({ type: 'term', index: Math.max(0, stateRef.current.termIndex - 1) }),
    }), [live, job]);
    return { state, actions };
}
```

Create `web_demo/src/components/StepNav.jsx`:

```jsx
import { STEPS } from '../domain/timeline';

export default function StepNav({ state, actions }) {
    return <nav className="step-nav" aria-label="六步讲解">
        {STEPS.map(item => <button key={item.step} aria-current={state.step === item.step ? 'step' : undefined}
            onClick={() => actions.selectStep(item.step)}>
            <span>{String(item.step).padStart(2, '0')}</span>
            <strong>{item.title}</strong>
            <small>{item.kicker}</small>
        </button>)}
    </nav>;
}
```

Create `web_demo/src/components/TransportDeck.jsx`:

```jsx
export default function TransportDeck({ state, actions, currentBeat, totalBeats, stepTitle }) {
    return <div className="transport-deck">
        <button className="transport-play" aria-label={state.paused ? '播放讲解' : '暂停讲解'}
            onClick={actions.togglePause}>{state.paused ? '▶' : 'Ⅱ'}</button>
        <div className="transport-progress">
            <div><strong>{stepTitle}</strong><span className="mono">{currentBeat + 1} / {totalBeats}</span></div>
            <input aria-label="讲解进度" type="range" min="0" max={Math.max(0, totalBeats - 1)}
                value={Math.min(currentBeat, Math.max(0, totalBeats - 1))}
                onChange={event => actions.seekBeat?.(Number(event.target.value))} />
        </div>
        <button className="transport-next" onClick={actions.next}>下一步 →</button>
        <button className="transport-follow" aria-pressed={state.following} onClick={actions.followLatest}>跟随最新</button>
    </div>;
}
```

Create `web_demo/src/hooks/useReducedMotion.js`:

```javascript
import { useEffect, useState } from 'react';
export function useReducedMotion() {
    const query = '(prefers-reduced-motion: reduce)';
    const [reduced, setReduced] = useState(() => window.matchMedia?.(query).matches || false);
    useEffect(() => {
        const media = window.matchMedia?.(query);
        if (!media) return undefined;
        const onChange = event => setReduced(event.matches);
        media.addEventListener?.('change', onChange);
        return () => media.removeEventListener?.('change', onChange);
    }, []);
    return reduced;
}
```

Append to `corridor.css`:

```css
.step-nav { position: absolute; top: 12px; left: 12px; right: 12px; display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; }
.step-nav button { border: 1px solid var(--line-dim); background: rgba(6,18,37,.72); color: var(--text-2); text-align: left; padding: 7px 9px; border-radius: 4px; }
.step-nav button[aria-current="step"] { border-color: var(--focus); color: var(--focus); background: var(--focus-soft); }
.step-nav span { display: block; font: 10px var(--font-mono); color: var(--muted); }
.step-nav strong { display: block; font-size: 12px; font-weight: 500; }
.step-nav small { display: block; font: 9px var(--font-mono); color: var(--muted); letter-spacing: .1em; }
.transport-deck { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-top: 1px solid var(--line-dim); background: rgba(6,18,37,.8); }
.transport-play { width: 36px; height: 36px; border-radius: 50%; border: 1px solid var(--focus); background: var(--focus-soft); color: var(--focus); }
.transport-progress { flex: 1; display: grid; gap: 4px; }
.transport-progress > div { display: flex; justify-content: space-between; font-size: 12px; }
.transport-progress input { width: 100%; accent-color: var(--focus); }
.transport-next, .transport-follow { border: 1px solid var(--line-soft); background: none; color: var(--text-2); padding: 6px 10px; border-radius: 4px; font-size: 12px; }
.transport-follow[aria-pressed="true"] { color: var(--current); border-color: var(--current); }
```

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/corridorState.test.js && npm run build`

Expected: tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/app/useCorridor.js web_demo/src/hooks/useReducedMotion.js web_demo/src/components/TransportDeck.jsx web_demo/src/components/StepNav.jsx web_demo/src/styles/corridor.css web_demo/tests/corridorState.test.js
git commit -m "Add global corridor cursor and data-gated narration"
```

---

### Task 8: 数值台与两处放大镜

**Files:**
- Create: `web_demo/src/domain/ledger.js`
- Create: `web_demo/src/components/NumericLedger.jsx`
- Create: `web_demo/src/components/LensPanel.jsx`
- Create: `web_demo/src/styles/ledger.css`
- Create: `web_demo/tests/ledger.test.js`

**Interfaces:**
- Produces: `TERM_FILTERS`、`filterTerms(terms, filter)`、`termStatus(term, index, state)`、`formatValue(value)`、`statusLabel(status)`。
- Produces: `NumericLedger`、`LensPanel`。

- [ ] **Step 1: 写参与项表测试**

Create `web_demo/tests/ledger.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { TERM_FILTERS, filterTerms, formatValue, termStatus } from '../src/domain/ledger.js';

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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/ledger.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现数值台与放大镜**

Create `web_demo/src/domain/ledger.js`:

```javascript
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
```

Create `web_demo/src/components/NumericLedger.jsx`:

```jsx
import { TERM_FILTERS, formatValue, statusLabel, termStatus } from '../domain/ledger';

export default function NumericLedger({ calculation, state, terms, filter, page, onPage, onFilter, onSeek }) {
    if (!calculation) return <div className="ledger-empty">等待可检查的真实计算。</div>;
    const pageSize = 8;
    const maxPage = Math.max(0, Math.ceil(terms.length / pageSize) - 1);
    const visible = terms.slice(page * pageSize, page * pageSize + pageSize);
    return <section className="numeric-ledger" aria-label="参与项数值">
        <div className="ledger-head">
            <div><span>PARTICIPATION / 针对当前目标</span><h3>{calculation.title}</h3></div>
            <p className="mono">{calculation.counts.connected} 有连接 · {calculation.counts.disconnected} 无连接 · {calculation.counts.zeroContribution} 贡献为零{calculation.counts.unrecorded ? ` · ${calculation.counts.unrecorded} 未采集` : ''}</p>
        </div>
        <div className="ledger-tools">
            <div>{TERM_FILTERS.map(item => <button key={item.value} aria-pressed={filter === item.value}
                onClick={() => onFilter(item.value)}>{item.label}</button>)}</div>
            <div><button aria-label="上一页" disabled={page === 0} onClick={() => onPage(page - 1)}>←</button>
                <span className="mono">{page + 1} / {maxPage + 1}</span>
                <button aria-label="下一页" disabled={page === maxPage} onClick={() => onPage(page + 1)}>→</button></div>
        </div>
        <div className="ledger-table" role="table" aria-label="完整参与项">
            <div role="row" className="ledger-row head"><span>来源 / 索引</span><span>输入</span><span>权重</span><span>乘积</span><span>作用</span></div>
            {visible.map(term => {
                const status = termStatus(term, term.index, state);
                return <button role="row" key={term.index} className={`ledger-row ${status}`} onClick={() => onSeek(term.index + 0.5)}>
                    <span><b>#{term.index + 1}</b> {term.label}</span>
                    <span>{formatValue(term.input)}</span>
                    <span>{formatValue(term.weight)}</span>
                    <span>{formatValue(term.product)}</span>
                    <span>{statusLabel(status)}</span>
                </button>;
            })}
        </div>
    </section>;
}
```

Create `web_demo/src/components/LensPanel.jsx`:

```jsx
import NumericLedger from './NumericLedger';
import { formatValue } from '../domain/ledger';

export default function LensPanel({ open, calculation, state, terms, filter, page, onFilter, onPage, onSeek, onClose }) {
    if (!open) return null;
    return <aside className="lens-panel blueprint-panel" role="dialog" aria-label="计算放大镜">
        <header><div><span>CALCULATION LENS</span><strong>{calculation?.title || '等待选择对象'}</strong></div>
            <button onClick={onClose} aria-label="关闭放大镜">×</button></header>
        <div className="lens-equation mono">
            {state?.currentTerm ? `${formatValue(state.currentTerm.input)} × ${formatValue(state.currentTerm.weight)} = ${formatValue(state.currentTerm.product)}` : '—'}
        </div>
        <NumericLedger calculation={calculation} state={state} terms={terms} filter={filter}
            page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />
    </aside>;
}
```

Create `web_demo/src/styles/ledger.css`:

```css
.numeric-ledger { border-top: 1px solid var(--line-dim); padding: 12px 16px; overflow: auto; max-height: 34vh; }
.ledger-head { display: flex; justify-content: space-between; gap: 16px; align-items: end; }
.ledger-head span { font: 11px var(--font-mono); color: var(--muted); letter-spacing: .12em; }
.ledger-head h3 { margin: 2px 0 0; font-size: 16px; font-weight: 500; }
.ledger-head p { margin: 0; color: var(--text-2); font-size: 11px; }
.ledger-tools { display: flex; justify-content: space-between; gap: 12px; margin: 10px 0; }
.ledger-tools button { border: 1px solid var(--line-soft); background: none; color: var(--text-2); padding: 3px 8px; border-radius: 3px; font-size: 11px; }
.ledger-tools button[aria-pressed="true"] { color: var(--current); border-color: var(--current); }
.ledger-table { display: grid; gap: 2px; }
.ledger-row { display: grid; grid-template-columns: 2.2fr 1fr 1fr 1fr .9fr; gap: 8px; align-items: center; text-align: left; border: 0; background: rgba(232,241,255,.03); color: var(--text-1); padding: 5px 8px; font: 12px var(--font-mono); }
.ledger-row.head { color: var(--muted); background: none; font-size: 10px; letter-spacing: .08em; }
.ledger-row:not(.head):hover { background: var(--focus-soft); }
.ledger-row.current { border-left: 2px solid var(--current); }
.ledger-row.accumulated { color: var(--focus); }
.ledger-row.disconnected, .ledger-row.unrecorded, .ledger-row.padding { color: var(--muted); }
.ledger-empty { padding: 16px; color: var(--muted); }
.lens-panel { position: absolute; right: 16px; bottom: 16px; width: min(560px, 46vw); max-height: 54vh; z-index: 10; overflow: hidden; }
.lens-panel header { display: flex; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid var(--line-dim); }
.lens-panel header span { display: block; font: 10px var(--font-mono); color: var(--muted); letter-spacing: .12em; }
.lens-panel header strong { font-size: 14px; }
.lens-panel header button { background: none; border: 0; color: var(--text-2); font-size: 20px; }
.lens-equation { padding: 8px 12px; color: var(--current); font-size: 12px; }
```

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/ledger.test.js && npm run build`

Expected: tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/domain/ledger.js web_demo/src/components/NumericLedger.jsx web_demo/src/components/LensPanel.jsx web_demo/src/styles/ledger.css web_demo/tests/ledger.test.js
git commit -m "Add numeric ledger and calculation lens"
```

---

### Task 9: 工作台、长廊页与结果页

**Files:**
- Create: `web_demo/src/domain/workbench.js`
- Create: `web_demo/src/components/Workbench.jsx`
- Create: `web_demo/src/components/Corridor.jsx`
- Create: `web_demo/src/components/Results.jsx`
- Create: `web_demo/src/components/EmptyState.jsx`
- Create: `web_demo/src/styles/workbench.css`
- Modify: `web_demo/src/App.jsx`
- Create: `web_demo/tests/workbench.test.js`

**Interfaces:**
- Produces: `formatStatus(job)`、`uploadHint(file, modelsReady)`。
- Produces: `Workbench`、`Corridor`、`Results`、`EmptyState`。

- [ ] **Step 1: 写工作台状态测试**

Create `web_demo/tests/workbench.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { formatStatus, uploadHint } from '../src/domain/workbench.js';

test('status labels distinguish ready, busy and terminal jobs', () => {
    assert.equal(formatStatus({ status: 'ready' }), '等待开始');
    assert.equal(formatStatus({ status: 'running', progress: 0.4 }), '正在分析 40%');
    assert.equal(formatStatus({ status: 'completed' }), '分析完成');
    assert.equal(formatStatus({ status: 'failed', error: '坏了' }), '分析失败 · 坏了');
});

test('upload hint reflects file and model readiness', () => {
    assert.equal(uploadHint(null, true), '拖入一段视频，或点击选择文件');
    assert.equal(uploadHint(null, false), '模型未就绪，请先检查权重与 ffmpeg');
    assert.equal(uploadHint({ name: 'a.mp4' }, true), 'a.mp4 · 可以开始分析');
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd web_demo && node --test tests/workbench.test.js`

Expected: FAIL with missing module.

- [ ] **Step 3: 实现页面组件与 App 组合**

Create `web_demo/src/domain/workbench.js`:

```javascript
export function formatStatus(job) {
    if (!job) return '等待视频';
    if (job.status === 'running') return `正在分析 ${Math.round((job.progress || 0) * 100)}%`;
    if (job.status === 'loading') return '正在加载模型';
    if (job.status === 'encoding') return '正在编码结果';
    if (job.status === 'completed') return '分析完成';
    if (job.status === 'failed') return `分析失败 · ${job.error || '未知错误'}`;
    if (job.status === 'cancelled') return '已停止分析';
    return '等待开始';
}

export function uploadHint(file, modelsReady) {
    if (!modelsReady) return '模型未就绪，请先检查权重与 ffmpeg';
    if (!file) return '拖入一段视频，或点击选择文件';
    return `${file.name} · 可以开始分析`;
}
```

Create `web_demo/src/components/Workbench.jsx`:

```jsx
import { useRef, useState } from 'react';
import { formatTime, jobUrl } from '../api';
import { formatStatus, uploadHint } from '../domain/workbench';

export default function Workbench({ job, health, error, pending, onUpload, onStart, onCancel, onOpenCorridor, onOpenResults }) {
    const input = useRef(null);
    const [dragging, setDragging] = useState(false);
    const busy = ['loading', 'running', 'encoding'].includes(job?.status);
    const ready = Boolean(health?.models_ready && health?.encoder_ready);
    const choose = file => { if (file && !busy) onUpload(file); };
    return <section className="workbench">
        <div className="workbench-copy">
            <span className="mono workbench-kicker">DEEP LEARNING COURSE PROJECT / 2026</span>
            <h1>看见一次预测的发生</h1>
            <p>把一个真实行人的骨架送进模型，沿着六步长廊看清检测、绑定、预测与意图。</p>
            <div className="workbench-actions">
                <input ref={input} type="file" accept=".mp4,.mov,.avi" hidden
                    onChange={event => choose(event.target.files?.[0])} />
                <button className="blueprint-button primary" onClick={() => input.current?.click()}
                    disabled={busy || !ready}>选择视频</button>
                {job?.status === 'ready' && <button className="blueprint-button" onClick={onStart} disabled={!ready}>开始分析</button>}
                {busy && <button className="blueprint-button" onClick={onCancel}>停止分析</button>}
                {job && <button className="blueprint-button" onClick={onOpenCorridor}>进入长廊</button>}
                {job?.status === 'completed' && <button className="blueprint-button" onClick={onOpenResults}>查看结果</button>}
            </div>
            <div className={`drop-zone ${dragging ? 'dragging' : ''}`}
                onDragOver={event => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files?.[0]); }}>
                <strong>{uploadHint(job ? { name: job.filename } : null, ready)}</strong>
                <span className="mono">{job ? `${job.metadata?.frame_count || '—'} 帧 · ${formatTime(job.metadata?.duration)}` : 'MP4 / MOV / AVI'}</span>
            </div>
            {error && <p className="workbench-error" role="alert">{error}</p>}
            {job && <div className="workbench-job">
                <span className="mono">{job.id.slice(0, 8)}</span>
                <strong>{formatStatus(job)}</strong>
                <div className="job-progress"><i style={{ transform: `scaleX(${job.progress || 0})` }} /></div>
            </div>}
        </div>
        <div className="workbench-light-table blueprint-panel">
            {job ? <video src={jobUrl(job.id, '/files/source')} poster={jobUrl(job.id, '/files/poster')}
                muted playsInline preload="metadata" /> : <div className="light-table-empty">
                <span className="mono">LIGHT TABLE</span><b>源视频将在这里展开</b>
                <small>真实骨架、ID 与分数不会在这里伪造</small>
            </div>}
            <div className="light-table-caption mono">{job
                ? `${job.metadata?.width || '—'}×${job.metadata?.height || '—'} · ${job.metadata?.fps || '—'} fps`
                : '等待视频'}</div>
        </div>
        <div className="workbench-steps">
            {['视频帧', 'YOLO 骨架', 'ID 绑定', 'LSTM 预测', 'ST-GCN', '意图'].map((label, index) =>
                <div key={label}><span className="mono">{String(index + 1).padStart(2, '0')}</span><b>{label}</b><i /></div>)}
        </div>
    </section>;
}
```

Create `web_demo/src/components/Corridor.jsx`:

```jsx
import CorridorStage from './CorridorStage';
import ContextRail from './ContextRail';
import StepNav from './StepNav';
import TransportDeck from './TransportDeck';
import NumericLedger from './NumericLedger';
import LensPanel from './LensPanel';
import { beatsForStep, STEPS } from '../domain/timeline';
import { flowState } from '../domain/calculation';

export default function Corridor({ job, live, structure, state, actions, selectedTrack, tracks, onSelectTrack,
    calculation, terms, filter, page, onFilter, onPage, onSeek, lensOpen, onCloseLens }) {
    const beats = beatsForStep(state.step);
    const stepTitle = STEPS.find(item => item.step === state.step)?.title || '视频帧';
    const flow = calculation ? flowState(calculation, state.termIndex) : null;
    return <section className="corridor-page">
        <ContextRail job={job} tracks={tracks} track={selectedTrack} onSelect={onSelectTrack} />
        <div className="corridor-main">
            <div className="corridor-stage-wrap">
                <CorridorStage step={state.step} track={selectedTrack} live={live} trackId={selectedTrack?.track_id}
                    structure={structure} onPick={packet => { if (packet?.kind === 'bay') actions.selectStep(packet.step); }} />
                <div className="stage-overlay">
                    <StepNav state={state} actions={actions} />
                    <LensPanel open={lensOpen} calculation={calculation} state={flow} terms={terms}
                        filter={filter} page={page} onFilter={onFilter} onPage={onPage}
                        onSeek={index => { onSeek(index); actions.seekTerm(index); }} onClose={onCloseLens} />
                </div>
            </div>
            <TransportDeck state={state} actions={actions} currentBeat={state.beat}
                totalBeats={beats.length} stepTitle={stepTitle} />
            {!lensOpen && <NumericLedger calculation={calculation} state={flow} terms={terms}
                filter={filter} page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />}
        </div>
    </section>;
}
```

Create `web_demo/src/components/Results.jsx`:

```jsx
import { jobUrl } from '../api';

export default function Results({ job, onOpenCorridor }) {
    if (!job) return <div className="results-empty">没有可显示的任务。</div>;
    const metrics = job.metrics || {};
    return <section className="results-page">
        <header><span className="mono">RESULT / 一次真实推理</span><h1>结果与下载</h1>
            <button className="blueprint-button" onClick={onOpenCorridor}>回到长廊</button></header>
        <div className="results-grid">
            <div className="results-video blueprint-panel">
                {job.outputs?.video ? <video src={jobUrl(job.id, '/files/video')} controls playsInline /> : <p>结果视频尚未生成。</p>}
            </div>
            <aside className="results-metrics blueprint-panel">
                <div><span>处理帧</span><b>{metrics.frames ?? job.processed_frames ?? '—'}</b></div>
                <div><span>有效观测</span><b>{metrics.pose_observations ?? '—'}</b></div>
                <div><span>预测更新</span><b>{metrics.classification_updates ?? '—'}</b></div>
                <div><span>处理 FPS</span><b>{Number.isFinite(metrics.processing_fps) ? metrics.processing_fps.toFixed(2) : '—'}</b></div>
                <div className="downloads">
                    <a href={jobUrl(job.id, '/files/video')}>标注 MP4</a>
                    <a href={jobUrl(job.id, '/files/data')}>逐帧 JSONL</a>
                    <a href={jobUrl(job.id, '/files/bundle')}>完整 ZIP</a>
                </div>
            </aside>
        </div>
    </section>;
}
```

Create `web_demo/src/components/EmptyState.jsx`:

```jsx
export default function EmptyState({ title, children }) {
    return <div className="empty-state blueprint-panel"><strong>{title}</strong><p>{children}</p></div>;
}
```

Create `web_demo/src/styles/workbench.css`:

```css
.workbench { height: 100%; display: grid; grid-template-columns: minmax(320px, .9fr) 1.1fr; grid-template-rows: 1fr auto; gap: 24px; padding: 40px var(--page-pad); overflow: auto; }
.workbench-copy { align-self: center; max-width: 560px; }
.workbench-kicker { color: var(--focus); font-size: 11px; letter-spacing: .16em; }
.workbench h1 { font-size: clamp(34px, 3.4vw, 52px); font-weight: 500; margin: 12px 0 10px; letter-spacing: .02em; }
.workbench-copy > p { color: var(--text-2); max-width: 460px; }
.workbench-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 20px 0; }
.blueprint-button.primary { background: var(--focus); color: var(--bg-0); }
.drop-zone { border: 1px dashed var(--line-soft); border-radius: 6px; padding: 22px; display: grid; gap: 4px; color: var(--text-2); }
.drop-zone.dragging { border-color: var(--focus); background: var(--focus-soft); }
.drop-zone strong { color: var(--text-1); font-weight: 500; }
.workbench-light-table { align-self: center; overflow: hidden; }
.workbench-light-table video { width: 100%; aspect-ratio: 16/9; object-fit: contain; background: #000; display: block; }
.light-table-empty { aspect-ratio: 16/9; display: grid; place-content: center; gap: 6px; text-align: center; color: var(--text-2); border-bottom: 1px solid var(--line-dim); }
.light-table-empty span { color: var(--focus); font-size: 11px; letter-spacing: .18em; }
.light-table-caption { padding: 8px 10px; color: var(--muted); font-size: 11px; }
.workbench-steps { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
.workbench-steps > div { border-top: 1px solid var(--line-soft); padding-top: 8px; display: grid; gap: 3px; }
.workbench-steps span { color: var(--focus); font-size: 10px; }
.workbench-steps b { font-size: 12px; font-weight: 500; }
.workbench-steps i { height: 1px; background: linear-gradient(90deg, var(--line-soft), transparent); }
.workbench-error { color: var(--danger); }
.job-progress { height: 3px; background: var(--line-dim); border-radius: 2px; overflow: hidden; margin-top: 6px; }
.job-progress i { display: block; height: 100%; background: var(--focus); transform-origin: left; }
.results-page { padding: 32px var(--page-pad); overflow: auto; height: 100%; }
.results-page header { display: flex; align-items: end; gap: 16px; margin-bottom: 20px; }
.results-page header span { color: var(--focus); font-size: 11px; letter-spacing: .16em; }
.results-page h1 { margin: 0; font-size: 32px; font-weight: 500; }
.results-page header button { margin-left: auto; }
.results-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; }
.results-video video { width: 100%; display: block; background: #000; }
.results-metrics { padding: 16px; display: grid; gap: 12px; align-content: start; }
.results-metrics > div { display: flex; justify-content: space-between; border-bottom: 1px solid var(--line-dim); padding-bottom: 8px; }
.results-metrics span { color: var(--text-2); }
.results-metrics b { font-family: var(--font-mono); font-weight: 500; }
.results-metrics .downloads { display: grid; gap: 6px; border: 0; }
.downloads a { color: var(--focus); }
```

Replace `web_demo/src/App.jsx` with the composition:

```jsx
import { useEffect, useMemo, useState } from 'react';
import AppShell from './components/AppShell';
import Workbench from './components/Workbench';
import Corridor from './components/Corridor';
import Results from './components/Results';
import { hashForPage, pageFromHash } from './app/page';
import { useJob } from './hooks/useJob';
import { useLiveComputation } from './hooks/useLiveComputation';
import { useModelStructure } from './hooks/useModelStructure';
import { useCorridor } from './app/useCorridor';
import { normalizeCalculation } from './domain/calculation';
import { filterTerms } from './domain/ledger';

function latestCalculation(snapshot, step, structure, moduleIndex, blockIndex) {
    if (!snapshot || !structure) return null;
    const rows = Object.values(snapshot.nodes || {}).flatMap(node => node.calculations || []);
    if (step === 2) return snapshot.details?.yolo?.[moduleIndex] || rows.find(row => row.stage === 'yolo') || null;
    if (step === 5) return (snapshot.details?.stgcn?.[blockIndex] || [])[0] || rows.find(row => row.stage === 'spatial') || null;
    if (step === 6) return rows.find(row => row.stage === 'classifier') || null;
    return null;
}

export default function App() {
    const [page, setPage] = useState(() => pageFromHash(window.location.hash));
    const jobState = useJob();
    const live = useLiveComputation(jobState.job);
    const model = useModelStructure();
    const [selectedTrackId, setSelectedTrackId] = useState(null);
    const [filter, setFilter] = useState('all');
    const [pageIndex, setPageIndex] = useState(0);
    const [lensOpen, setLensOpen] = useState(false);
    const tracks = live.snapshot?.tracks?.length ? live.snapshot.tracks : (jobState.job?.tracks || []);
    const selectedTrack = tracks.find(track => String(track.track_id) === String(selectedTrackId)) || tracks[0] || null;
    const { state, actions } = useCorridor({ job: jobState.job, live: live.snapshot, trackId: selectedTrack?.track_id });
    useEffect(() => { if (selectedTrackId == null && tracks.length) setSelectedTrackId(tracks[0].track_id); }, [tracks, selectedTrackId]);
    useEffect(() => {
        const onChange = () => setPage(pageFromHash(window.location.hash));
        window.addEventListener('hashchange', onChange);
        return () => window.removeEventListener('hashchange', onChange);
    }, []);
    const navigate = next => { window.location.hash = hashForPage(next); setPage(pageFromHash(hashForPage(next))); };
    const raw = useMemo(() => latestCalculation(live.snapshot, state.step, model.structure, state.module, state.block),
        [live.snapshot, state.step, state.module, state.block, model.structure]);
    const calculation = useMemo(() => normalizeCalculation(raw), [raw]);
    const terms = useMemo(() => calculation ? filterTerms(calculation.terms, filter) : [], [calculation, filter]);
    const status = { ready: Boolean(jobState.health?.models_ready), label: jobState.health?.model_name || '本地模型' };
    return <AppShell page={page} onNavigate={navigate} status={status}>
        {page === 'workbench' && <Workbench job={jobState.job} health={jobState.health} error={jobState.error}
            pending={jobState.pending} onUpload={jobState.upload} onStart={jobState.start} onCancel={jobState.cancel}
            onOpenCorridor={() => navigate('corridor')} onOpenResults={() => navigate('results')} />}
        {page === 'corridor' && <Corridor job={jobState.job} live={live.snapshot} structure={model.structure}
            state={state} actions={actions} selectedTrack={selectedTrack} tracks={tracks}
            onSelectTrack={id => { setSelectedTrackId(id); actions.selectTrack(id); }}
            calculation={calculation} terms={terms} filter={filter} page={pageIndex}
            onFilter={value => { setFilter(value); setPageIndex(0); }} onPage={setPageIndex}
            onSeek={index => { actions.seekTerm(index); setLensOpen(true); }} lensOpen={lensOpen}
            onCloseLens={() => setLensOpen(false)} />}
        {page === 'results' && <Results job={jobState.job} onOpenCorridor={() => navigate('corridor')} />}
    </AppShell>;
}
```

Import `workbench.css` and `ledger.css` from `App.jsx` or `main.jsx`:

```javascript
import './styles/workbench.css';
import './styles/ledger.css';
```

- [ ] **Step 4: 运行测试与构建**

Run: `cd web_demo && node --test tests/workbench.test.js && npm run build`

Expected: workbench tests PASS; build succeeds.

- [ ] **Step 5: 提交**

```bash
git add web_demo/src/domain/workbench.js web_demo/src/components/Workbench.jsx web_demo/src/components/Corridor.jsx web_demo/src/components/Results.jsx web_demo/src/components/EmptyState.jsx web_demo/src/App.jsx web_demo/src/styles/workbench.css web_demo/tests/workbench.test.js
git commit -m "Rewrite workbench, corridor and results pages"
```

---

### Task 10: 收口、清理、文档与真实浏览器验证

**Files:**
- Modify: `web_demo/index.html`
- Modify: `docs/WEB_DEMO.md`
- Delete obsolete frontend files (list below)
- Create: `docs/superpowers/reports/2026-11-04-blueprint-browser-check.md`

**Interfaces:**
- Produces: 可运行的 `npm run build` 产物；Playwright 验证记录；删除旧视图后的干净前端。

- [ ] **Step 1: 清理旧视图与样式**

After the new pages build and pass tests, delete the obsolete files with `git rm`:

```bash
git rm web_demo/src/components/ProcessPage.jsx \
  web_demo/src/components/InputPanel.jsx \
  web_demo/src/components/VideoPanel.jsx \
  web_demo/src/components/LiveComputationPanel.jsx \
  web_demo/src/components/NeuralScene3D.jsx \
  web_demo/src/components/CalculationWorkbench.jsx \
  web_demo/src/components/CalculationScene3D.jsx \
  web_demo/src/components/PipelineViews.jsx \
  web_demo/src/components/NetworkViews.jsx \
  web_demo/src/components/ArithmeticTheater.jsx \
  web_demo/src/components/NetworkPrimitives.jsx \
  web_demo/src/components/OperatorTabs.jsx \
  web_demo/src/components/PipelineStrip.jsx \
  web_demo/src/components/TrackPanel.jsx \
  web_demo/src/components/Header.jsx \
  web_demo/src/components/Icon.jsx \
  web_demo/src/components/Skeleton.jsx \
  web_demo/src/components/GraphStoryControls.jsx
git rm web_demo/src/three/NeuralScene.js \
  web_demo/src/three/CalculationScene.js \
  web_demo/src/three/NetworkTrace.js \
  web_demo/src/three/GraphMessages.js \
  web_demo/src/three/networkTraceData.js \
  web_demo/src/three/sceneData.js
git rm web_demo/src/calculationFlow.js \
  web_demo/src/calculation-scene.css \
  web_demo/src/calculation-workbench.css \
  web_demo/src/arithmetic.css \
  web_demo/src/network.css \
  web_demo/src/networkTimeline.js \
  web_demo/src/spatialGraph.js \
  web_demo/src/styles.css \
  web_demo/src/three-scene.css \
  web_demo/src/live-workbench.css \
  web_demo/src/workbench.css
git rm web_demo/tests/arithmeticMath.test.js \
  web_demo/tests/calculationFlow.test.js \
  web_demo/tests/calculationPlayback.test.js \
  web_demo/tests/networkTimeline.test.js \
  web_demo/tests/networkTraceData.test.js \
  web_demo/tests/sceneData.test.js \
  web_demo/tests/spatialGraph.test.js
```

Keep `web_demo/src/api.js`, `web_demo/src/liveComputation.js`, `web_demo/src/hooks/useJob.js`, `web_demo/src/hooks/useLiveComputation.js`, and `web_demo/tests/liveComputation.test.js`. If any deleted file is still imported, the build will fail; fix the import rather than keeping the old file.

- [ ] **Step 2: 更新页面标题与文档**

Modify `web_demo/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>看见推理 · 行人意图蓝图长廊</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

Update `docs/WEB_DEMO.md`:
- Replace the old 8-chapter/3D-lab description with the new six-step corridor, workbench, results, and lens/ledger.
- Keep the model honesty constraints, upload limits, ffmpeg, local-only, and validation commands.
- Add the detail coverage/budget section from the data-foundation plan.
- Add the new browser check results after running them.

- [ ] **Step 3: 运行完整自动化测试**

Run:

```bash
python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v
cd web_demo
npm test
npm run build
```

Expected: Python tests PASS except the known real-weight skips; all frontend tests PASS; production build succeeds. Record exact counts.

- [ ] **Step 4: 真实浏览器验证**

Start the app with the final weights and a real test video:

```bash
python -s run_web_demo.py --port 8765 --model-dir weights/final
```

Use Playwright Chromium at 1920×1080:

1. Workbench: upload a real short video, start analysis, confirm progress and preview.
2. Corridor: confirm the six steps appear; the skeleton and ID in step ②/③ match the live snapshot; no fabricated values.
3. Wait for 50 observations; confirm LSTM/ST-GCN/意图 steps populate.
4. Click a module in step ② and a joint in step ⑤; confirm the lens shows complete terms and the actual output.
5. Pause and confirm the numbers freeze while analysis continues; click 「跟随最新」 and confirm it resumes.
6. Check 1440×900 and tablet portrait; check reduced-motion; simulate WebGL loss if possible.
7. Save screenshots for each step (outside Git or under `artifacts/`, which is ignored).

Create `docs/superpowers/reports/2026-11-04-blueprint-browser-check.md` with:
- The exact video file, model directory, browser, and viewport.
- Pass/fail for each numbered check.
- Screenshot filenames and locations.
- Any console errors.
- The measured FPS/overhead if recorded.
- A statement of what was not verified (e.g., a specific GPU or real classroom projector).

- [ ] **Step 5: 最终提交**

```bash
git add web_demo/index.html docs/WEB_DEMO.md docs/superpowers/reports/2026-11-04-blueprint-browser-check.md
git add -u web_demo
git commit -m "Complete blueprint corridor rewrite and verification"
```

---

## Self-review

- Spec coverage: six-step corridor (Tasks 4–7), YOLO skeleton/ID binding (Task 5), all convolution stages via live details and lens/ledger (Tasks 2–3, 6, 8), blueprint visual system (Tasks 1, 4, 9), workbench/results (Task 9), cleanup/docs/browser checks (Task 10).
- Placeholders: none; code steps contain concrete code; shell steps contain exact commands.
- Type consistency: `detail_block`, `module`, `block`, `termIndex`, `flowState`, `normalizeCalculation`, `filterTerms`, `structure.yoloModules`, `structure.stgcnBlocks` are used consistently across tasks.
- Known gap to watch during execution: the first implementation uses module 0 and block 1 as defaults; the lens must expose selectors for all 23 modules and 10 blocks before final acceptance. Add the selectors in Task 8/9:
  - In `LensPanel`, render `<select aria-label="选择模块" value={state.module} onChange={...}>` for `structure.yoloModules` when `calculation.stage === 'yolo'`.
  - Render a block selector for `structure.stgcnBlocks` when `calculation.stage` is `spatial` or `temporal`.
  - Wire `actions.selectModule`/`actions.selectBlock`; both are defined in Task 7/9.
- Known integration gap: the first pass draws the skeleton and graph at the bay positions but does not yet attach per-joint picking; the lens opens from the numeric ledger. Before final acceptance, add raycast picking for skeleton joints and feature planes, mapping hits to `onPick({kind:'joint', joint})` or `{kind:'module', module}`. This is required by the spec's click-to-inspect interaction.
- Known integration gap: step ① uses the `ContextRail` video as the source frame; the 3D lightbox plane for the source frame is not included in Task 4/5. Before final acceptance, add a `videoPlane.js` with a `THREE.VideoTexture` at `BAYS[0]` and pass `jobUrl(job.id, '/files/source')` from `Corridor`. Do not fabricate a frame if the video cannot load; show the labeled lightbox instead.
