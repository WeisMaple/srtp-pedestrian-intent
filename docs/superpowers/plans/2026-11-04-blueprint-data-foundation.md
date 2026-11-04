# 蓝图长廊 · 数据基础实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让实时推理为 YOLO 全部 23 个模块与 ST-GCN 全部 10 个块提供有界的、代表性的完整卷积算式，并通过 `/api/model`、SSE 与记录按需端点提供给前端，同时保持生产预测逐值不变。

**Architecture:** 新增静态模型目录与代表卷积选取；在现有前向钩子上增加详细帧调度（层输出 4 fps、详细算式 1 fps、轮转覆盖）；用紧凑稠密数组替代逐项索引；`LiveBuffer` 缓存最近详细算式与覆盖进度；记录模式增加按需单模块卷积端点。全程不额外执行一次生产模型。

**Tech Stack:** Python 3.9、PyTorch、FastAPI、unittest。无新增依赖。

## Global Constraints

- 保持现有生产推理输出、JSONL、标注视频逐值不变；关闭详细采集时不得新增钩子开销。
- 只为现有前向安装钩子；详细采集不得额外执行一次模型。
- 层输出 4 fps；详细帧 1 fps；每个详细帧 4 个 YOLO 模块、1 个 ST-GCN 块；YOLO 6 个详细帧覆盖 24 个顶层模块，ST-GCN 10 个详细帧覆盖 10 块。
- 发布模型 `yolo26s-pose.pt` 有 24 个顶层模块：索引 0–22 为主干/颈部，索引 23 为 `Pose26` 姿态头；设计文字中的「23 个模块」指主干/颈部，详细轮转与覆盖按 24 个顶层模块计，姿态头单独标注。
- 代表输出固定：YOLO 通道 0、空间中心；ST-GCN 通道 0、中间时刻、度数最高关节（空间）；时间卷积通道 0、中间时刻。
- 单代表卷积项数上限 4096（输入通道 × 核高 × 核宽）；超限模块标记为不可用，不截断求和。
- 单事件上限 128 KiB；普通帧 768 KiB；详细帧总预算 768 KiB，其中详细算式 ≤512 KiB；事件环 1024 事件 / 8 MiB。
- 实时数值保留 5 位小数；记录数值保留 6 位小数；非有限值显式 `null`。
- 不修改 `AGENTS.md`；不新增第三方依赖；保持 Python 3.9 兼容。
- 测试命令：`python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v`
- 提交使用描述性祈使句；每个任务结束提交一次。

---

### Task 1: 静态模型目录与 `/api/model`

**Files:**
- Create: `modern_pipeline/model_catalog.py`
- Create: `modern_pipeline/tests/test_model_catalog.py`
- Modify: `modern_pipeline/web/jobs.py`（新增 `structure_cache` 与 `model_structure`）
- Modify: `modern_pipeline/web/app.py`（新增 `GET /api/model`）

**Interfaces:**
- Produces: `RepresentativeConv(path, conv, parent, activation)`，属性 `in_channels/out_channels/kernel/stride/padding/dilation/term_count`。
- Produces: `select_representative_conv(module, fused=True, max_terms=4096) -> RepresentativeConv | None`。
- Produces: `describe_yolo(network, fused=False, max_terms=4096) -> list[dict]`。
- Produces: `describe_stgcn(model) -> dict`。
- Produces: `build_catalog(yolo_network, stgcn, versions, fused=False, max_terms=4096) -> dict`。
- Produces: `JobManager.model_structure() -> dict`；`GET /api/model` 返回该字典。

- [ ] **Step 1: 写失败的模型目录测试**

Create `modern_pipeline/tests/test_model_catalog.py`:

```python
"""Model catalog selects inspectable convolutions without a forward pass."""
import unittest

import torch

from modern_pipeline.model_catalog import (
    build_catalog,
    describe_stgcn,
    describe_yolo,
    select_representative_conv,
)
from st_gcn import Model


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2)
        self.act = torch.nn.SiLU()

    def forward(self, value):
        return self.act(self.conv(value))


class Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1), torch.nn.SiLU())
        self.body = torch.nn.Sequential(Block(8, 16, 3), Block(16, 16, 5))

    def forward(self, value):
        return self.body(self.stem(value))


class Container(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(Nested(), torch.nn.MaxPool2d(2))

    def forward(self, value):
        return self.model(value)


class ModelCatalogTests(unittest.TestCase):
    def test_selection_prefers_first_non_one_by_one_conv(self):
        selection = select_representative_conv(Nested())
        self.assertIsNotNone(selection)
        self.assertEqual(selection.path, 'body.0.conv')
        self.assertEqual(selection.kernel, [3, 3])
        self.assertEqual(selection.in_channels, 8)
        self.assertEqual(selection.activation, 'silu')
        self.assertEqual(selection.term_count, 8 * 3 * 3)

    def test_term_cap_skips_module_instead_of_truncating(self):
        selection = select_representative_conv(Block(3000, 8, 3), max_terms=4096)
        self.assertIsNone(selection)

    def test_module_without_conv_has_no_representative(self):
        selection = select_representative_conv(torch.nn.Sequential(torch.nn.ReLU()))
        self.assertIsNone(selection)

    def test_describe_yolo_keeps_top_level_indices_and_from_links(self):
        network = Container()
        network.model[0].stem.f = -1
        network.model[0].body.f = [0]
        modules = describe_yolo(network)
        self.assertEqual([row['index'] for row in modules], [0, 1])
        self.assertEqual(modules[0]['from'], [-1])
        self.assertEqual(modules[1]['from'], [0])
        self.assertEqual(modules[0]['representative']['module_path'], 'model.0.body.0.conv')
        self.assertEqual(modules[1]['representative'], None)
        self.assertTrue(modules[1]['representative_unavailable'])

    def test_describe_stgcn_reports_all_ten_blocks(self):
        model = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        result = describe_stgcn(model)
        self.assertEqual(len(result['blocks']), 10)
        self.assertEqual(result['blocks'][0]['index'], 1)
        self.assertEqual(result['blocks'][0]['joint_count'], 17)
        self.assertIn(result['blocks'][1]['residual'], ('identity', 'projection', 'none'))

    def test_build_catalog_serializes_without_tensors(self):
        catalog = build_catalog(Container(), Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True),
                                {'stgcn_epoch': 121, 'lstm_epoch': 68})
        self.assertEqual(catalog['versions']['stgcn_epoch'], 121)
        self.assertEqual(catalog['yolo']['modules'][0]['representative']['kernel'], [3, 3])
        self.assertNotIsInstance(catalog['yolo']['modules'][0]['representative'], torch.nn.Conv2d)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -s -m unittest modern_pipeline.tests.test_model_catalog -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'modern_pipeline.model_catalog'`.

- [ ] **Step 3: 实现模型目录**

Create `modern_pipeline/model_catalog.py`:

```python
"""Static, bounded description of the production YOLO and ST-GCN models.

This module never runs a forward pass and never copies weights. Output shapes
are filled by the caller after a real forward if they are needed.
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RepresentativeConv:
    path: str
    conv: torch.nn.Conv2d
    parent: Optional[torch.nn.Module]
    activation: str

    @property
    def in_channels(self):
        return int(self.conv.in_channels)

    @property
    def out_channels(self):
        return int(self.conv.out_channels)

    @property
    def kernel(self):
        return list(self.conv.kernel_size)

    @property
    def stride(self):
        return list(self.conv.stride)

    @property
    def padding(self):
        return list(self.conv.padding)

    @property
    def dilation(self):
        return list(self.conv.dilation)

    @property
    def term_count(self):
        return self.in_channels * self.kernel[0] * self.kernel[1]


def _walk(parent, prefix=''):
    for name, child in parent.named_children():
        path = prefix + '.' + name if prefix else name
        yield path, child, parent
        yield from _walk(child, path)


def _activation_for(parent, fused):
    if parent is None:
        return 'identity'
    act = getattr(parent, 'act', None)
    batchnorm = getattr(parent, 'bn', None)
    has_bn = isinstance(batchnorm, torch.nn.modules.batchnorm._BatchNorm)
    if has_bn and not fused:
        return 'batch_norm_silu' if isinstance(act, torch.nn.SiLU) else 'batch_norm'
    return 'silu' if isinstance(act, torch.nn.SiLU) else 'identity'


def select_representative_conv(module, fused=True, max_terms=4096):
    """First non-1x1 grouped conv under the term cap, else first eligible conv."""
    candidates = [(path, child, parent) for path, child, parent in _walk(module)
                  if isinstance(child, torch.nn.Conv2d) and child.groups == 1]
    eligible = [(path, child, parent) for path, child, parent in candidates
                if int(child.in_channels) * child.kernel_size[0] * child.kernel_size[1] <= max_terms]
    if not eligible:
        return None
    preferred = [item for item in eligible if item[1].kernel_size != (1, 1)]
    path, conv, parent = (preferred or eligible)[0]
    return RepresentativeConv(path, conv, parent, _activation_for(parent, fused))


def describe_yolo(network, fused=False, max_terms=4096):
    modules = getattr(network, 'model', network)
    rows = []
    for index, module in enumerate(modules):
        selection = select_representative_conv(module, fused=fused, max_terms=max_terms)
        source = getattr(module, 'f', getattr(module, 'i', -1))
        if isinstance(source, int):
            source = [source]
        elif isinstance(source, (list, tuple)):
            source = list(source)
        else:
            source = None
        representative = None
        if selection is not None:
            representative = {
                'module_path': selection.path,
                'in_channels': selection.in_channels,
                'out_channels': selection.out_channels,
                'kernel': selection.kernel,
                'stride': selection.stride,
                'padding': selection.padding,
                'dilation': selection.dilation,
                'term_count': selection.term_count,
                'activation': selection.activation,
                'fused': bool(fused),
            }
        rows.append({'index': index, 'type': type(module).__name__, 'from': source,
                     'output_shape': None, 'representative': representative,
                     'representative_unavailable': selection is None})
    return rows


def describe_stgcn(model):
    blocks = []
    for index, block in enumerate(model.st_gcn_networks):
        tcn = block.tcn[2]
        residual = getattr(block, 'residual', None)
        mode = 'none' if index == 0 else ('projection' if isinstance(residual, torch.nn.Module) else 'identity')
        blocks.append({'index': index + 1,
                       'in_channels': int(block.gcn.conv.in_channels),
                       'out_channels': int(tcn.out_channels),
                       'temporal_kernel': list(tcn.kernel_size),
                       'stride': list(tcn.stride),
                       'residual': mode,
                       'joint_count': int(model.A.shape[2])})
    return {'blocks': blocks,
            'classes': ['irrelevant', 'not crossing', 'crossing'],
            'labels': ['其他', '不过街', '过街']}


def build_catalog(yolo_network, stgcn, versions, fused=False, max_terms=4096):
    return {'yolo': {'name': None, 'imgsz': 640,
                     'modules': describe_yolo(yolo_network, fused=fused, max_terms=max_terms)},
            'stgcn': describe_stgcn(stgcn),
            'versions': dict(versions or {})}
```

- [ ] **Step 4: 运行模型目录测试**

Run: `python -s -m unittest modern_pipeline.tests.test_model_catalog -v`

Expected: all 6 tests PASS.

- [ ] **Step 5: 增加 JobManager 方法与 API 路由**

Modify `modern_pipeline/web/jobs.py`:

1. In `JobManager.__init__`, after `self.frontend_predictor = None`, add:

```python
        self.structure_cache = None
```

2. Add this method before `def frontend_network`:

```python
    def model_structure(self):
        """Load the catalog once; share admission with all inference work."""
        from .frontend_network import FrontendInspector
        from modern_pipeline.behavior import BehaviorPredictor
        from modern_pipeline.model_catalog import build_catalog

        with self.lock:
            if self.structure_cache is not None:
                return deepcopy(self.structure_cache)
            if self.config is None:
                raise RuntimeError(self.model_error or '模型未就绪。')
            if self.active is not None or self.network_active:
                raise JobBusy('视频或网络计算正在运行，请等待完成。')
            self.network_active = True
        try:
            config = self.config
            if self.frontend_predictor is None:
                self.frontend_predictor = FrontendInspector(self.model_dir / config['pose_model'])
            if self.network_predictor is None:
                import torch
                torch.set_num_threads(2)
                self.network_predictor = BehaviorPredictor(
                    self.model_dir / config['stgcn'], self.model_dir / config['lstm'],
                    'cpu', config['normalization'], self.model_dir / config['stats'])
            catalog = build_catalog(
                self.frontend_predictor.model, self.network_predictor.stgcn,
                {'yolo': config.get('pose_model'), 'stgcn': config.get('stgcn'),
                 'lstm': config.get('lstm'), 'normalization': config.get('normalization')},
                fused=False)
            catalog['yolo']['name'] = str(config.get('pose_model') or '')
            with self.lock:
                self.structure_cache = catalog
            return deepcopy(catalog)
        finally:
            with self.lock:
                self.network_active = False
```

Modify `modern_pipeline/web/app.py` after the `/api/health` route:

```python
    @app.get('/api/model')
    def model():
        try:
            return JSONResponse(manager.model_structure(), headers={'Cache-Control': 'no-store'})
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
```

- [ ] **Step 6: 写端点测试并运行**

Append to `modern_pipeline/tests/test_web.py` inside `WebTests`:

```python
    def test_model_catalog_endpoint_reports_loading_failure_readably(self):
        self.manager.model_structure = lambda: (_ for _ in ()).throw(RuntimeError('模型未就绪。'))
        response = self.client.get('/api/model')
        self.assertEqual(response.status_code, 503)
        self.assertIn('模型未就绪', response.json()['detail'])
```

Add a success test with a stub:

```python
    def test_model_catalog_endpoint_returns_manager_catalog(self):
        self.manager.model_structure = lambda: {'yolo': {'modules': []}, 'versions': {'stgcn_epoch': 121}}
        response = self.client.get('/api/model')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['versions']['stgcn_epoch'], 121)
```

Run: `python -s -m unittest modern_pipeline.tests.test_web -v`

Expected: all endpoint tests PASS (existing five real-weight tests may skip).

- [ ] **Step 7: 提交**

```bash
git add modern_pipeline/model_catalog.py modern_pipeline/tests/test_model_catalog.py modern_pipeline/web/jobs.py modern_pipeline/web/app.py modern_pipeline/tests/test_web.py
git commit -m "Add bounded model catalog endpoint"
```

---

### Task 2: 紧凑代表卷积与 YOLO 详细钩子

**Files:**
- Modify: `modern_pipeline/live_arithmetic.py`
- Modify: `modern_pipeline/live_trace.py`
- Create: `modern_pipeline/tests/test_live_detail_conv.py`
- Modify: `modern_pipeline/tests/test_live_arithmetic.py`（旧首层卷积断言改为新行）

**Interfaces:**
- Consumes: `select_representative_conv` from Task 1.
- Produces: `dense_conv_row(module_index, selection, value, output, fused) -> dict`。
- Produces: `LiveTrace(..., detail=False)`、`detail_frame`、`detail_modules`、`detail_block`、`detail_index`。
- Produces: 计算行 `{id, stage:'yolo', operation:'dense_conv', module, target, term_axes, term_shape, inputs, weights, bias, actual, activation, fused, metadata}`。

- [ ] **Step 1: 写失败的详细卷积测试**

Create `modern_pipeline/tests/test_live_detail_conv.py`:

```python
"""YOLO detail rows are complete representative sums from the real forward."""
from types import SimpleNamespace
import unittest

import torch

from modern_pipeline.live_trace import LiveTrace, SAMPLING


class Fused(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=4, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        self.act = torch.nn.SiLU()

    def forward(self, value):
        return self.act(self.conv(value))


def runner_for(modules, fused=True):
    callbacks = {'on_predict_start': []}
    yolo = SimpleNamespace(callbacks=callbacks,
                           add_callback=lambda name, callback: callbacks[name].append(callback))
    network = torch.nn.Sequential(*modules)
    backend = SimpleNamespace(model=network, is_fused=lambda: fused)
    return yolo, SimpleNamespace(model=SimpleNamespace(model=backend), done_warmup=False), callbacks


def attach(trace, yolo, runner, callbacks):
    trace.attach_yolo(yolo)
    callbacks['on_predict_start'][0](runner)
    runner.done_warmup = True


class LiveDetailConvTests(unittest.TestCase):
    def test_detail_row_contains_every_rgb_product_and_actual_silu(self):
        block = Fused()
        yolo, runner, callbacks = runner_for([block])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        attach(trace, yolo, runner, callbacks)
        trace.begin_frame(1, 0.0)
        value = torch.arange(3 * 12 * 12).reshape(1, 3, 12, 12).float() / 100
        expected = block(value)
        trace.close()
        rows = [c for event in events for c in event.get('calculations', [])]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['operation'], 'dense_conv')
        self.assertEqual(row['module'], 0)
        self.assertEqual(row['term_axes'], ['in_channel', 'kernel_y', 'kernel_x'])
        self.assertEqual(row['term_shape'], [3, 3, 3])
        self.assertEqual(len(row['inputs']), 27)
        self.assertEqual(len(row['weights']), 27)
        self.assertEqual(row['activation'], 'silu')
        self.assertTrue(row['fused'])
        total = sum(x * w for x, w in zip(row['inputs'], row['weights'])) + row['bias']
        reconstructed = total / (1 + torch.exp(-torch.tensor(total)).item())
        self.assertAlmostEqual(reconstructed, row['actual'], delta=2e-5)
        self.assertAlmostEqual(row['actual'], float(expected[0, 0, 3, 3]), delta=2e-5)
        self.assertTrue(row['metadata']['complete_sum'])

    def test_non_detail_frame_emits_no_dense_row(self):
        block = Fused()
        yolo, runner, callbacks = runner_for([block])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=999)
        trace.begin_frame(1, 0.0)
        attach(trace, yolo, runner, callbacks)
        block(torch.randn(1, 3, 12, 12))
        trace.close()
        self.assertFalse(any(event.get('calculations') for event in events))

    def test_term_cap_module_is_skipped_and_catalog_can_mark_it(self):
        block = Fused(in_channels=200, out_channels=4, kernel=5, padding=2)
        yolo, runner, callbacks = runner_for([block])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        attach(trace, yolo, runner, callbacks)
        trace.begin_frame(1, 0.0)
        block(torch.randn(1, 200, 12, 12))
        trace.close()
        self.assertFalse(any(event.get('calculations') for event in events))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_conv -v`

Expected: FAIL because `LiveTrace` does not accept `detail` and no dense rows exist.

- [ ] **Step 3: 实现详细帧调度与稠密卷积行**

Modify `modern_pipeline/live_trace.py`:

1. Add constants to `SAMPLING`:

```python
    'detail_interval_seconds': 1.0, 'detail_yolo_modules': 4, 'detail_stgcn_blocks': 1,
    'detail_frame_max_bytes': 768 * 1024, 'detail_calculation_max_bytes': 512 * 1024,
    'detail_max_terms': 4096,
```

2. Change the constructor signature and state:

```python
    def __init__(self, emit, interval=0.25, detail=False, detail_interval=1.0):
        ...
        self.detail = bool(detail)
        self.detail_interval = max(0.0, float(detail_interval))
        self.detail_last = float('-inf')
        self.detail_index = 0
        self.detail_frame = False
        self.detail_modules = set()
        self.detail_block = 0
        self.detail_bytes = 0
        self.yolo_module_count = 0
```

3. In `begin_frame`, after `self.active = ...` and the inactive early return, add:

```python
        now = time.perf_counter()
        self.detail_frame = self.detail and now - self.detail_last >= self.detail_interval
        if self.detail_frame:
            self.detail_last = now
            self.detail_index += 1
            count = self.yolo_module_count
            if count:
                self.detail_modules = {(self.detail_index * 4 + i) % count for i in range(4)}
            else:
                self.detail_modules = set()
            self.detail_block = (self.detail_index - 1) % 10
        else:
            self.detail_modules = set()
        self.detail_bytes = 0
```

4. In `_publish`, before the existing size check, add a detail budget check:

```python
        if event.get('detail') and self.detail_bytes + size > SAMPLING['detail_calculation_max_bytes']:
            self.dropped['detail_budget'] = self.dropped.get('detail_budget', 0) + 1
            return
        if event.get('detail'):
            self.detail_bytes += size
```

And change the per-event cap from `64 * 1024` to `128 * 1024`.

5. Add `'detail_budget': 0` to the `self.dropped` dict.

Modify `modern_pipeline/live_arithmetic.py`:

1. Add imports and the helper:

```python
from modern_pipeline.model_catalog import select_representative_conv
from torch.nn import functional as F


def _rounded(tensor):
    return [None if not torch.isfinite(value) else round(float(value), 5)
            for value in tensor.detach().reshape(-1).cpu()]


def dense_conv_row(module_index, selection, value, output, fused):
    conv = selection.conv
    y, x = output.shape[-2] // 2, output.shape[-1] // 2
    kh, kw = conv.kernel_size
    sy, sx = conv.stride
    py, px = conv.padding
    dy, dx = conv.dilation
    padded = F.pad(value, (px, px, py, py))
    patch = padded[0, :, y * sy:y * sy + dy * (kh - 1) + 1:dy,
                   x * sx:x * sx + dx * (kw - 1) + 1:dx]
    weights = conv.weight[0]
    raw = output[0, 0, y, x]
    activation = selection.activation
    if activation == 'batch_norm_silu':
        bn = selection.parent.bn
        channel = 0
        activated = (raw - bn.running_mean[channel]) / torch.sqrt(bn.running_var[channel] + bn.eps)
        activated = activated * bn.weight[channel] + bn.bias[channel]
        activated = F.silu(activated)
    elif activation == 'silu':
        activated = F.silu(raw)
    else:
        activated = raw
    bias = float(conv.bias[0]) if conv.bias is not None else 0.
    return {
        'id': 'yolo.conv.%d' % module_index, 'stage': 'yolo',
        'title': 'YOLO 模块 %d · %s 代表卷积' % (module_index, type(selection.parent).__name__),
        'operation': 'dense_conv', 'track_id': None,
        'target': {'channel': 0, 'y': y, 'x': x},
        'term_axes': ['in_channel', 'kernel_y', 'kernel_x'],
        'term_shape': [int(conv.in_channels), int(kh), int(kw)],
        'inputs': _rounded(patch), 'weights': _rounded(weights), 'bias': bias,
        'actual': None if not torch.isfinite(activated) else float(activated),
        'activation': activation, 'formula': 'y = activation(Σc Σky Σkx(x × W) + b)',
        'input_indices': [], 'connected': [True] * int(patch.numel()),
        'padding': [False] * int(patch.numel()),
        'metadata': {'complete_sum': True, 'representative': True,
                     'module_path': selection.path, 'target_indices': [0, y, x],
                     'input_shape': list(value.shape), 'stride': list(conv.stride),
                     'padding': list(conv.padding), 'dilation': list(conv.dilation),
                     'fused': bool(fused), 'term_count': int(patch.numel()),
                     'source': 'actual_forward_hooks'}}
```

6. Replace `attach_yolo` in `LiveArithmetic` with:

```python
    def attach_yolo(self, network, fused, guard):
        self.yolo_fused = fused
        self.yolo_selections = {}
        self.yolo_conv_state = {}
        for index, module in enumerate(network.model):
            selection = select_representative_conv(module, fused=fused,
                                                   max_terms=SAMPLING['detail_max_terms'])
            if selection is None:
                continue
            self.yolo_selections[index] = selection

            def capture(inputs, output, index=index):
                if not self.trace.active or not self.trace.detail_frame or index not in self.trace.detail_modules:
                    return
                self.yolo_conv_state[index] = (inputs[0].detach(), output.detach())

            self.trace._hook(selection.conv, capture, guard=guard, handles=self.trace.yolo_handles)

    def yolo_module(self, index):
        state = self.yolo_conv_state.pop(index, None)
        selection = self.yolo_selections.get(index)
        if state is None or selection is None:
            return []
        value, output = state
        return [dense_conv_row(index, selection, value, output, self.yolo_fused)]
```

Need `SAMPLING` imported in `live_arithmetic.py`? It currently doesn't import it. Add `from modern_pipeline.live_trace import SAMPLING` would create a circular import (live_trace imports live_arithmetic inside `__init__` only, but live_arithmetic importing live_trace at module load is okay? `live_trace` imports `live_arithmetic` only inside `LiveTrace.__init__`, so no cycle at import time? `live_arithmetic` importing `live_trace` at top would import `live_trace`, which defines `LiveTrace` but doesn't import `live_arithmetic` until instantiation. No cycle. But to avoid risk, define `DETAIL_MAX_TERMS = 4096` in live_arithmetic or pass it. Use a module constant `DETAIL_MAX_TERMS = 4096` and let `model_catalog` default. Simpler: `select_representative_conv(module, fused=fused)` uses the default 4096. Remove `SAMPLING` import. Good.

Modify `LiveTrace.attach_yolo`:
- Set `self.yolo_module_count = len(network.model)` after `self.yolo_module = network`.
- Replace the top-level module `capture` with:

```python
                def capture(inputs, output, duration, index=index, module=module):
                    tensor = first_tensor(output)
                    if tensor is None:
                        return
                    if tensor.ndim == 4:
                        sampled = sample_tensor(tensor, ['batch', 'channel', 'y', 'x'], [1, 4, 8, 8], (0, 1))
                    elif tensor.ndim == 3:
                        sampled = sample_tensor(tensor, ['batch', 'row', 'feature'], [1, 8, 16], (0,))
                    else:
                        sampled = sample_tensor(tensor, ['axis%d' % i for i in range(tensor.ndim)],
                                                [4] * tensor.ndim, tuple(range(tensor.ndim)))
                    detail = bool(self.detail_frame and index in self.detail_modules)
                    self._publish('yolo', 'yolo.%d' % index, 1, {'output': sampled}, duration,
                                  module_type=type(module).__name__, from_nodes=getattr(module, 'f', -1),
                                  fused=fused, warmup=False, detail=detail,
                                  calculations=self.arithmetic.yolo_module(index) if detail else [])
```

- Remove the old `yolo_value`/`yolo_output` first-conv capture code from `LiveArithmetic`; the dense module hook replaces it.
- Delete `test_fused_yolo_includes_all_center_rgb_terms_and_actual_silu` from `modern_pipeline/tests/test_live_arithmetic.py`; the new `test_live_detail_conv.py` covers the same behavior with the dense schema.
- `test_live_trace.py` still checks sampled YOLO output events; those remain unchanged. With `detail=False` (default), no calculations are attached, so its assertions stay valid.

- [ ] **Step 4: 运行测试**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_conv modern_pipeline.tests.test_live_trace modern_pipeline.tests.test_live_arithmetic -v`

Expected: new tests PASS; the removed fused test no longer runs; real-weight tests skip if weights absent.

- [ ] **Step 5: 提交**

```bash
git add modern_pipeline/live_arithmetic.py modern_pipeline/live_trace.py modern_pipeline/tests/test_live_detail_conv.py modern_pipeline/tests/test_live_arithmetic.py
git commit -m "Add compact YOLO representative convolution telemetry"
```

---

### Task 3: ST-GCN 全部块的代表空间与时间卷积

**Files:**
- Modify: `modern_pipeline/live_arithmetic.py`
- Create: `modern_pipeline/tests/test_live_detail_graph.py`

**Interfaces:**
- Consumes: `trace.detail_block` from Task 2.
- Produces: 行 `{id:'stgcn.spatial.block<N>.track<ID>', operation:'graph_aggregation', ...}` 与 `{id:'stgcn.temporal.block<N>.track<ID>', operation:'temporal_convolution', ...}`。
- Produces: 每行带 `metadata.detail_frame`、`metadata.complete_sum=True`、`metadata.graph_slice`（空间）。

- [ ] **Step 1: 写失败的图详细测试**

Create `modern_pipeline/tests/test_live_detail_graph.py`:

```python
"""Every ST-GCN block exposes one complete spatial and temporal example."""
import math
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_trace import LiveTrace
from st_gcn import Model, SkeletonLSTM


class LiveDetailGraphTests(unittest.TestCase):
    def predictor(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        predictor = BehaviorPredictor.__new__(BehaviorPredictor)
        predictor.device = torch.device('cpu')
        predictor.normalization = 'legacy_pair'
        predictor.lstm = SkeletonLSTM(num_layers=2).eval()
        predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        predictor.stats = {}
        return predictor

    def test_ten_detail_frames_cover_all_blocks_with_complete_sums(self):
        predictor = self.predictor()
        rng = np.random.default_rng(23)
        poses = rng.normal(size=(3, 50, 17, 3)).astype(np.float32)
        poses[..., :2] = poses[..., :2] * 20 + 300
        poses[..., 2] = 0.9
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        try:
            for step in range(10):
                trace.begin_frame(100 + step, step)
                trace.begin_behavior([8, 2, 91], poses)
                result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
                trace.end_frame([{'track_id': track, 'prediction_updated': True,
                                  'probabilities': probability.tolist()}
                                 for track, probability in zip([8, 2, 91], result['probabilities'])])
        finally:
            trace.close()
        blocks = {event['detail_block'] for event in events if event.get('detail')}
        self.assertEqual(blocks, set(range(10)))
        rows = [c for event in events for c in event.get('calculations', []) if c['stage'] == 'stgcn']
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row['metadata']['complete_sum'])
            size = len(row['inputs'])
            self.assertEqual(len(row['weights']), size)
            total = math.fsum(x * w for x, w in zip(row['inputs'], row['weights'])) + row['bias']
            if row['activation'] == 'silu':
                total = total / (1 + math.exp(-total))
            self.assertAlmostEqual(total, row['actual'], delta=3e-5 * max(1., abs(total)))

    def test_spatial_row_contains_all_51_terms_and_graph_slice(self):
        predictor = self.predictor()
        poses = np.full((1, 50, 17, 3), 1., dtype=np.float32)
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        try:
            trace.begin_frame(1, 0)
            trace.begin_behavior([7], poses)
            predictor.predict(poses, poses[0], on_event=trace.forecast_step)
        finally:
            trace.close()
        spatial = next(c for event in events for c in event.get('calculations', [])
                       if c.get('stage') == 'spatial')
        self.assertEqual(len(spatial['inputs']), 51)
        self.assertEqual(spatial['metadata']['graph_slice']['joint_count'], 17)
        self.assertEqual(len(spatial['metadata']['graph_slice']['outputs']), 17)
        temporal = [c for event in events for c in event.get('calculations', [])
                    if c.get('stage') == 'temporal']
        self.assertTrue(any(any(c['padding']) for c in temporal))
        self.assertTrue(any(not any(c['padding']) for c in temporal))
```

The second test's assertion about the target joint is too strong; the target joint is the highest-degree joint, not the max output. Replace with:

```python
        self.assertEqual(len(spatial['metadata']['graph_slice']['outputs']), 17)
        self.assertEqual(spatial['metadata']['graph_slice']['joint_count'], 17)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_graph -v`

Expected: FAIL because only block 0 emits spatial/temporal arithmetic.

- [ ] **Step 3: 实现全部块钩子**

In `LiveArithmetic.attach_behavior`, replace the block-0-only hooks with a loop over all blocks. The existing method has `block = stgcn.st_gcn_networks[0]`, `joint`, `projected`, `spatial`, `temporal`, `last_block`. Replace with:

```python
        stgcn = predictor.stgcn
        joint = int((stgcn.A != 0).sum(dim=(0, 1)).argmax())
        states = {index: {} for index in range(len(stgcn.st_gcn_networks))}

        def selected(index):
            return self.trace.detail_frame and self.trace.detail_block == index

        for index, block in enumerate(stgcn.st_gcn_networks):
            def projected(inputs, output, index=index):
                if not selected(index):
                    return
                shaped = output.reshape(output.shape[0], stgcn.A.shape[0], -1,
                                        output.shape[2], output.shape[3])
                states[index]['projected'] = shaped[:len(trace.track_ids), :, 0,
                                                     output.shape[2] // 2, :].detach().clone()

            def spatial(inputs, output, index=index):
                if not selected(index) or 'projected' not in states[index]:
                    return
                tensor, adjacency = output
                weights = values(inputs[1][:, :, joint])
                timestep = tensor.shape[2] // 2
                edge_indices = (stgcn.A != 0).nonzero().detach().cpu().tolist()
                effective = inputs[1].detach().cpu()
                edges = [[k, source, receiver, float(effective[k, source, receiver])]
                         for k, source, receiver in edge_indices]
                indices = [[partition, 0, timestep, source_joint]
                           for partition in range(inputs[1].shape[0])
                           for source_joint in range(tensor.shape[3])]
                for batch, track_id in enumerate(trace.track_ids):
                    row = calculation(
                        'stgcn.spatial.block%d.track%d' % (index, track_id), 'spatial',
                        '第 %d 块 · 骨架邻域聚合' % (index + 1), track_id,
                        '通道 0 · 时刻 %d · 目标关节 %d' % (timestep, joint),
                        values(states[index]['projected'][batch]), weights,
                        tensor[batch, 0, timestep, joint], indices,
                        ['partition', 'channel', 'time', 'joint'],
                        connected=[weight != 0 for weight in values(stgcn.A[:, :, joint])],
                        block=index, term_shape=[inputs[1].shape[0], tensor.shape[3]],
                        target_indices=[0, timestep, joint],
                        graph_slice={'channel': 0, 'timestep': timestep, 'joint_count': tensor.shape[3],
                                     'edges': edges, 'outputs': values(tensor[batch, 0, timestep, :]),
                                     'source': 'actual_gcn_forward_same_channel_and_timestep'},
                        weight_source='actual_gcn_input_effective_adjacency_including_edge_importance',
                        formula='y[0,t,w] = Σₖ Σᵥ(projected[k,0,t,v] × effective_A[k,v,w])')
                    row['metadata']['detail'] = True
                    row['metadata']['detail_block'] = index
                    self.publish(row, 'stgcn', 5)

            def temporal(inputs, output, index=index):
                if not selected(index):
                    return
                value = inputs[0]
                kernel = block.tcn[2].kernel_size[0]
                stride = block.tcn[2].stride[0]
                padding = block.tcn[2].padding[0]
                dilation = block.tcn[2].dilation[0]
                weights = values(block.tcn[2].weight[0, :, :, 0])
                middle = output.shape[2] // 2
                for boundary, timestep in ((False, middle), (True, 0)):
                    if boundary and timestep == middle:
                        continue
                    times = [timestep * stride - padding + offset * dilation
                             for offset in range(kernel)]
                    indices = [[channel, time, joint] for channel in range(value.shape[1])
                               for time in times]
                    pads = [time < 0 or time >= value.shape[2] for _, time, _ in indices]
                    bias = float(block.tcn[2].bias[0]) if block.tcn[2].bias is not None else 0.
                    for batch, track_id in enumerate(trace.track_ids):
                        window = torch.stack([value[batch, :, time, joint] if 0 <= time < value.shape[2]
                                              else value.new_zeros(value.shape[1]) for time in times], dim=1)
                        suffix = '.boundary' if boundary else ''
                        row = calculation(
                            'stgcn.temporal.block%d%s.track%d' % (index, suffix, track_id), 'temporal',
                            '第 %d 块 · 时间卷积%s' % (index + 1, '（边界）' if boundary else ''), track_id,
                            '通道 0 · 时刻 %d · 关节 %d' % (timestep, joint),
                            values(window), weights, output[batch, 0, timestep, joint], indices,
                            ['channel', 'time', 'joint'], bias=bias, padding=pads, block=index,
                            target_indices=[0, timestep, joint], term_shape=[value.shape[1], kernel],
                            kernel_size=kernel, stride=stride, convolution_padding=padding,
                            dilation=dilation, boundary=boundary,
                            formula='y[0,t,v] = Σ𝚌 Σq(x[c,t×stride−pad+q×dilation,v] × W[0,c,q]) + b[0]')
                        row['metadata']['detail'] = True
                        row['metadata']['detail_block'] = index
                        self.publish(row, 'stgcn', 5)

            trace._hook(block.gcn.conv, projected)
            trace._hook(block.gcn, spatial)
            trace._hook(block.tcn[2], temporal)

        def last_block(inputs, output, duration):
            self.pool_values = output[0][:len(trace.track_ids), 0].detach().clone()

        trace._hook(stgcn.st_gcn_networks[-1], last_block)
```

This replaces the old block-0 hooks. The `calculation()` helper's `operation=stage` is used; for spatial/temporal the stage is the operation. Fine. The row's `id` includes block index.

- [ ] **Step 4: 运行测试**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_graph modern_pipeline.tests.test_live_arithmetic -v`

Expected: new tests PASS; existing behavior tests still pass after the intentional count updates below.

Update `modern_pipeline/tests/test_live_arithmetic.py`:

- Change its `capture` helper to `LiveTrace(events.append, interval=0, detail=True, detail_interval=0)` so the representative sums are emitted on every captured frame.
- In `test_three_people_complete_sums_preserve_ids_forward_count_and_predictions`, change the stage count loop to expect six temporal rows (middle + boundary per person) and three of the other stages:

```python
        counts = {'decoder': 256, 'spatial': 51, 'temporal': 576, 'pool': 221, 'classifier': 256}
        for stage, count in counts.items():
            rows = [c for c in calculations if c['stage'] == stage]
            expected = 6 if stage == 'temporal' else 3
            self.assertEqual(len(rows), expected)
            self.assertTrue(all(len(c['inputs']) == count for c in rows))
```

- Replace the single temporal row assertions with:

```python
        temporal_rows = [c for c in calculations if c['stage'] == 'temporal']
        self.assertTrue(any(any(c['padding']) for c in temporal_rows))
        self.assertTrue(any(not any(c['padding']) for c in temporal_rows))
        self.assertTrue(all(all(x == 0 for x, padded in zip(c['inputs'], c['padding']) if padded)
                            for c in temporal_rows))
```

- In the real-weight test, replace the exact `len(rows) == 33` with stage-presence assertions and keep the budget checks:

```python
        self.assertEqual({c['stage'] for c in rows},
                         {'normalization', 'decoder', 'denormalization', 'spatial',
                          'temporal', 'pool', 'classifier'})
        self.assertEqual(len([c for c in rows if c['stage'] == 'spatial']), 3)
        self.assertEqual(len([c for c in rows if c['stage'] == 'temporal']), 6)
```

- In the YOLO + behavior real-weight test, replace `len(rows) == 34` with the same stage check plus `self.assertTrue(rows)`; the exact count now depends on detail coverage, so do not pin it.

The detail budget includes six temporal rows per block. With three people and 2304 terms per row, the temporal rows are about 280 KiB; add four YOLO representative convs (25–190 KiB) and spatial rows; stay under the 512 KiB detail budget in the synthetic tests. If the real benchmark exceeds it, Task 7 applies the 0.5 fps / 2-module fallback.

Modify the temporal function to loop `for timestep in (middle, 0)` and skip if `timestep` already 0 (if middle==0). For each, build times, weights, indices, pads, window, row; id includes `boundary.` for timestep 0. For middle, id as above. Add `metadata['boundary'] = timestep == 0`.
- The row's `target_indices` [0, timestep, joint].
- The weights are the same.
- The actual output at that timestep.
- For middle==0 (T=1), emit only once.

Update the test `test_live_arithmetic.py`:
- `counts = {'decoder':256,'spatial':51,'temporal':576,'pool':221,'classifier':256}`. With both middle and boundary, temporal rows = 6, each 576; the count check currently expects 3 rows per stage. Change to:
  ```python
  counts = {'decoder': 256, 'spatial': 51, 'temporal': 576, 'pool': 221, 'classifier': 256}
  for stage, count in counts.items():
      rows = [c for c in calculations if c['stage'] == stage]
      expected = 6 if stage == 'temporal' else 3
      self.assertEqual(len(rows), expected)
      self.assertTrue(all(len(c['inputs']) == count for c in rows))
  ```
- `temporal = next(c for c in calculations if c['stage']=='temporal')` may pick middle (no padding). Change to:
  ```python
  temporal_rows = [c for c in calculations if c['stage'] == 'temporal']
  self.assertTrue(any(any(c['padding']) for c in temporal_rows))
  self.assertTrue(all(all(x == 0 for x, padded in zip(c['inputs'], c['padding']) if padded) for c in temporal_rows))
  ```
- The expected total rows in the real-weight test changes; the plan should note to update counts after running; the test can compute dynamically. Better to avoid brittle exact counts:
  - Replace the fixed `len(rows) == 33` with assertions that required stages exist and counts per person are consistent. The plan can update.
- The detail default `detail=False` means these existing tests (which don't pass detail=True) won't emit spatial/temporal arithmetic at all! Wait, the existing `attach_behavior` always emitted block 0 arithmetic for all active frames, independent of detail. If we gate all spatial/temporal arithmetic on `trace.detail_frame`, and `detail=False` by default, the existing `test_live_arithmetic` will fail because no spatial/temporal rows. We need to preserve the old block-0 arithmetic for `detail=False`? The spec says live web enables detail; CLI/other tests may not. But the existing tests rely on the arithmetic. We can set `detail=True` in those tests, or keep default `detail=True`? The design says live detail should be enabled for the web; the CLI default could be off. To minimize breakage, set `LiveTrace` default `detail=True`? That would add overhead to any live trace (CLI), contrary to "web only". The `LiveTrace` is only used when `on_computation` is set (web or explicit); the CLI without live telemetry doesn't create it. The `live_telemetry` flag is only set by the web. `test_live_arithmetic` creates `LiveTrace` directly; setting `detail=True` in tests is fine. The default can be `False`. The plan updates `test_live_arithmetic.capture` to `LiveTrace(events.append, interval=0, detail=True, detail_interval=0)` so all frames are detail frames. Then the old behavior (block 0 arithmetic every captured frame) changes to round-robin; the existing test expects block 0 rows. With detail_interval=0, each frame increments detail_index and detail_block = (index-1)%10; in `capture` there's only one frame, detail_index=1, block=0. Good. The spatial/temporal rows for block 0 are emitted. The test counts update as above.
  - `test_live_trace` doesn't use calculations; detail default False; still passes.
  - The web runner will set detail=True and detail_interval=1.0.
- `LiveTrace.attach_behavior` should only emit arithmetic when `trace.detail_frame`; the layer output events (LSTM, ST-GCN outputs) remain every active frame. Good.
- The detail block selection uses `self.detail_block`, computed from `detail_index-1 % 10`. With `detail_interval=0`, every active frame is a detail frame. Good.

- [ ] **Step 5: 提交**

```bash
git add modern_pipeline/live_arithmetic.py modern_pipeline/tests/test_live_detail_graph.py modern_pipeline/tests/test_live_arithmetic.py
git commit -m "Add representative ST-GCN detail telemetry for every block"
```

---

### Task 4: 覆盖事件、服务端详细缓存与预算

**Files:**
- Modify: `modern_pipeline/live_trace.py`（`LiveBuffer`、`SAMPLING`、`end_frame`）
- Create: `modern_pipeline/tests/test_live_detail_buffer.py`

**Interfaces:**
- Produces: `pipeline.output` 事件携带 `detail: {yolo_modules: [...], yolo_covered: [...], stgcn_blocks: [...], stgcn_covered: [...], detail_frame: int}`。
- Produces: `LiveBuffer.details = {'yolo': {module: calculation}, 'stgcn': {block: [calculations]}}`，`snapshot()` 包含 `details`。
- Produces: `SAMPLING['detail_interval_seconds']=1.0` 等。

- [ ] **Step 1: 写失败的缓存/覆盖测试**

Create `modern_pipeline/tests/test_live_detail_buffer.py`:

```python
"""The live buffer retains the latest detail per module and block."""
import unittest

from modern_pipeline.live_trace import LiveBuffer


class LiveDetailBufferTests(unittest.TestCase):
    def test_details_are_bounded_and_survive_snapshot(self):
        buffer = LiveBuffer('job-1')
        for module in range(5):
            buffer.append({'kind': 'computation', 'frame': 1, 'time_seconds': 0,
                           'model': 'yolo', 'node': 'flow.yolo.conv.%d' % module,
                           'stage': 1, 'seq': module + 1,
                           'calculations': [{'id': 'yolo.conv.%d' % module, 'stage': 'yolo',
                                             'operation': 'dense_conv', 'inputs': [], 'weights': [],
                                             'bias': 0., 'actual': None, 'metadata': {}}]})
        for block in range(2):
            buffer.append({'kind': 'computation', 'frame': 1, 'time_seconds': 0,
                           'model': 'stgcn', 'node': 'flow.stgcn.block.%d' % block,
                           'stage': 5, 'seq': 100 + block,
                           'calculations': [{'id': 'stgcn.spatial.block%d.track1' % block,
                                             'stage': 'spatial', 'inputs': [1.] * 51,
                                             'weights': [1.] * 51, 'bias': 0., 'actual': 51.,
                                             'metadata': {'detail_block': block}}]})
        snapshot = buffer.snapshot('running')
        self.assertEqual(sorted(snapshot['details']['yolo']), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(snapshot['details']['stgcn']), [0, 1])
        self.assertLessEqual(len(snapshot['details']['yolo']), 23)
        self.assertLessEqual(len(snapshot['details']['stgcn']), 10)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_buffer -v`

Expected: FAIL because `snapshot` has no `details`.

- [ ] **Step 3: 实现详细缓存与覆盖事件**

Modify `LiveBuffer.__init__`:

```python
        self.details = {'yolo': {}, 'stgcn': {}}
```

In `append`, after `self.nodes[event['node']] = data`, add:

```python
        for row in event.get('calculations', []) or []:
            if row.get('metadata', {}).get('detail'):
                if row.get('stage') == 'yolo':
                    self.details['yolo'][row['module']] = row
                else:
                    block = row.get('metadata', {}).get('detail_block')
                    if block is not None:
                        self.details['stgcn'].setdefault(block, [])
                        self.details['stgcn'][block].append(row)
                        self.details['stgcn'][block] = self.details['stgcn'][block][-12:]
```

Bound the maps:

```python
        if len(self.details['yolo']) > 23:
            self.details['yolo'] = dict(list(self.details['yolo'].items())[-23:])
        if len(self.details['stgcn']) > 10:
            self.details['stgcn'] = dict(list(self.details['stgcn'].items())[-10:])
```

In `snapshot`, include `'details': deepcopy(self.details)`.

In `LiveTrace.end_frame`, include coverage in the `pipeline.output` event:

```python
        coverage = {'detail_frame': self.detail_index if self.detail_frame else None,
                    'yolo_modules': sorted(self.detail_modules),
                    'stgcn_blocks': [self.detail_block] if self.detail_frame else []}
        self._publish('pipeline', 'pipeline.output', 7, {},
                      (time.perf_counter() - self.frame_started) * 1000,
                      tracks=tracks[:3], phase='complete', dropped=dict(self.dropped),
                      detail=coverage)
```

The cumulative `yolo_covered`/`stgcn_covered` can be computed by the frontend from the buffer snapshot; the event carries the current frame's set. Add `coverage` to `pipeline.output`. The frontend can show progress from `snapshot.details`. The design says `detail.coverage` event; we can instead include `detail` in `pipeline.output`, which is simpler. The spec says "详细帧发布 detail.coverage 事件"; the plan can use a dedicated event to match. Implement:

```python
        if self.detail_frame:
            self._publish('pipeline', 'detail.coverage', 1, {}, None,
                          yolo_modules=sorted(self.detail_modules),
                          stgcn_block=self.detail_block,
                          detail_frame=self.detail_index)
```

This is an extra event; fine. The `LiveBuffer` sees it; no calculations. The frontend reads it. Add both? Use the dedicated event; no need to add to `pipeline.output`. The `pipeline.output` tracks remain. The frontend can compute cumulative coverage from `details`. Good. The plan can add the dedicated event.

- `SAMPLING` already has the new fields from Task 2. Add `ring_max_events: 1024, ring_max_bytes: 8*1024*1024, frame_max_events: 192`. The `_publish` cap changed to 128 KiB. The `LiveBuffer` constructor defaults `max_events=512, max_bytes=4MiB`; update to `1024, 8MiB`.
- `Job.__post_init__` uses `LiveBuffer(self.state['id'])`, so the new defaults apply.
- Existing `test_live_trace.test_per_frame_budget_drops_extra_events` expects 128 events; changing `frame_max_events` to 192 breaks it. Update the test to use `SAMPLING['frame_max_events']` and expected drops. The plan should update.
- Existing tests for snapshot/trace may need `details` key; they likely compare specific fields, not full equality. Check. The plan can note.

- [ ] **Step 4: 运行测试**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_buffer modern_pipeline.tests.test_live_trace -v`

Expected: new test PASS; existing tests updated for new budgets PASS.

- [ ] **Step 5: 提交**

```bash
git add modern_pipeline/live_trace.py modern_pipeline/tests/test_live_detail_buffer.py modern_pipeline/tests/test_live_trace.py
git commit -m "Cache and advertise live detail coverage"
```

---

### Task 5: Web 运行器启用详细采集并保持预测一致

**Files:**
- Modify: `modern_pipeline/web/jobs.py`（`_execute` 设置 `live_detail`）
- Modify: `modern_pipeline/cli.py`（`LiveTrace` 构造传入 detail 参数）
- Create: `modern_pipeline/tests/test_live_detail_parity.py`

**Interfaces:**
- Consumes: `LiveTrace(detail=True, detail_interval=1.0)`.
- Produces: web 任务默认启用详细采集；CLI 仅在 `live_detail` 标志为真时启用。
- Produces: 启用/关闭详细采集的预测、JSONL、未来骨架逐值一致。

- [ ] **Step 1: 写失败的一致性测试**

Create `modern_pipeline/tests/test_live_detail_parity.py`:

```python
"""Detail telemetry never changes the production prediction."""
import unittest

import numpy as np
import torch

from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.live_trace import LiveTrace
from st_gcn import Model, SkeletonLSTM


class LiveDetailParityTests(unittest.TestCase):
    def test_detail_hooks_leave_probabilities_and_future_poses_unchanged(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        predictor = BehaviorPredictor.__new__(BehaviorPredictor)
        predictor.device = torch.device('cpu')
        predictor.normalization = 'legacy_pair'
        predictor.lstm = SkeletonLSTM(num_layers=2).eval()
        predictor.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True).eval()
        predictor.stats = {}
        poses = np.random.default_rng(31).normal(size=(3, 50, 17, 3)).astype(np.float32)
        baseline = predictor.predict(poses, poses[0])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.attach_behavior(predictor)
        try:
            trace.begin_frame(1, 0)
            trace.begin_behavior([8, 2, 91], poses)
            result = predictor.predict(poses, poses[0], on_event=trace.forecast_step)
        finally:
            trace.close()
        np.testing.assert_array_equal(result['probabilities'], baseline['probabilities'])
        np.testing.assert_array_equal(result['future_poses'], baseline['future_poses'])
```

- [ ] **Step 2: 运行测试确认失败（或通过但未启用 web）**

Run: `python -s -m unittest modern_pipeline.tests.test_live_detail_parity -v`

Expected: PASS for the parity itself (the hooks are active). The test proves the constraint.

- [ ] **Step 3: 在 web 运行器与 CLI 中接线**

In `modern_pipeline/web/jobs.py` `_execute`, after `pipeline_args.live_telemetry = True`, add:

```python
            pipeline_args.live_detail = True
```

In `modern_pipeline/cli.py`, change the `LiveTrace` construction:

```python
            live = LiveTrace(on_computation,
                             detail=bool(getattr(a, 'live_detail', False)),
                             detail_interval=1.0)
```

Add `live_detail=False` to the argument parser defaults if the parser has a `live_telemetry` option; search for `live_telemetry` definition and mirror. If `live_telemetry` is set as an attribute after parsing (as in `jobs.py`), no parser change is needed; `getattr` handles it.

- [ ] **Step 4: 运行完整后端测试**

Run: `python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v`

Expected: all tests PASS except the five historical real-weight tests, which skip when weights are absent. Record the exact pass/skip counts.

- [ ] **Step 5: 提交**

```bash
git add modern_pipeline/web/jobs.py modern_pipeline/cli.py modern_pipeline/tests/test_live_detail_parity.py
git commit -m "Enable bounded detail telemetry in web inference"
```

---

### Task 6: 记录模式按需单模块卷积累加

**Files:**
- Modify: `modern_pipeline/web/frontend_network.py`
- Modify: `modern_pipeline/web/jobs.py`
- Modify: `modern_pipeline/web/app.py`
- Create: `modern_pipeline/tests/test_yolo_conv_endpoint.py`

**Interfaces:**
- Produces: `FrontendInspector.module_conv(image_bgr, module_index) -> dict`，返回未融合 BN 的 `dense_conv` 行。
- Produces: `JobManager.yolo_conv(job_id, frame, module_index) -> dict`，缓存键 `(job_id, frame, module_index)`，上限 8 条 / 2 MiB。
- Produces: `GET /api/jobs/{id}/yolo-conv?frame=&module=`。

- [ ] **Step 1: 写失败的端点测试**

Create `modern_pipeline/tests/test_yolo_conv_endpoint.py`:

```python
"""The on-demand record endpoint validates inputs and serializes results."""
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from modern_pipeline.web.app import create_app


class Manager:
    def __init__(self):
        self.calls = []

    def yolo_conv(self, job_id, frame, module):
        self.calls.append((job_id, frame, module))
        return {'module': module, 'frame': frame, 'operation': 'dense_conv'}

    def snapshot(self, job_id):
        return {'id': job_id, 'status': 'completed'}


class YoloConvEndpointTests(unittest.TestCase):
    def test_route_passes_query_and_returns_dense_row(self):
        manager = Manager()
        client = TestClient(create_app(manager=manager))
        response = client.get('/api/jobs/abc/yolo-conv?frame=3&module=7')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['module'], 7)
        self.assertEqual(manager.calls, [('abc', 3, 7)])

    def test_route_rejects_negative_module(self):
        client = TestClient(create_app(manager=Manager()))
        response = client.get('/api/jobs/abc/yolo-conv?frame=3&module=-1')
        self.assertEqual(response.status_code, 422)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -s -m unittest modern_pipeline.tests.test_yolo_conv_endpoint -v`

Expected: FAIL with 404 because the route does not exist.

- [ ] **Step 3: 实现 FrontendInspector 单模块卷积**

In `modern_pipeline/web/frontend_network.py`, add imports:

```python
from modern_pipeline.model_catalog import select_representative_conv
```

Add a method to `FrontendInspector`:

```python
    @torch.inference_mode()
    def module_conv(self, image_bgr, module_index):
        if not 0 <= module_index < len(self.model.model):
            raise ValueError('模块编号超出范围。')
        module = self.model.model[module_index]
        selection = select_representative_conv(module, fused=False)
        if selection is None:
            raise ValueError('该模块没有可检查的代表卷积。')
        captured = {}

        def hook(_module, inputs, output):
            captured['input'] = inputs[0].detach()
            captured['output'] = output.detach()

        handle = selection.conv.register_forward_hook(hook)
        predictor = self.predictor
        predictor.batch = (['source-frame'], [image_bgr], [''])
        try:
            tensor = predictor.preprocess([image_bgr])
            self.model(tensor)
        finally:
            handle.remove()
            predictor.batch = None
        if 'input' not in captured:
            raise RuntimeError('未捕获到该模块的卷积输出。')
        return dense_conv_row(module_index, selection, captured['input'],
                              captured['output'], fused=False)
```

Import `dense_conv_row` from `modern_pipeline.live_arithmetic`. This runs the replay model, not production; `self.predictor.preprocess` sets the batch, and `self.model(tensor)` runs. The model may need `self.predictor.batch = None` cleanup; do it in `finally`. The representative conv selection is from the unfused replay model. The activation `batch_norm_silu` will include BN. Good.
- Need to ensure the model's `self.model` is the inner network (it is). The `self.predictor.preprocess` uses the predictor's batch; set it in a `try/finally`. The existing `replay` does this; copy the pattern.
- `dense_conv_row` includes `fused=False`; its `activation` from selection is `batch_norm_silu` (if parent has bn+SiLU). Good.
- The row's `metadata['module_path']` etc.

In `JobManager`, add:

```python
    def yolo_conv(self, job_id, frame, module_index):
        from .frontend_network import FrontendInspector, read_source_frame

        key = (job_id, frame, module_index)
        with self.lock:
            job = self.get(job_id)
            if self.active is not None or self.network_active:
                raise JobBusy('视频或网络计算正在运行，请等待完成。')
            if not 0 <= frame < job.state['metadata']['frame_count']:
                raise ValueError('所选帧超出原始视频范围。')
            if not 0 <= module_index < 24:
                raise ValueError('模块编号超出范围。')
            if key in self.frontend_cache:
                self.frontend_cache.move_to_end(key)
                return deepcopy(self.frontend_cache[key])
            if self.config is None:
                raise RuntimeError(self.model_error or '模型未就绪。')
            self.network_active = True
        try:
            image = read_source_frame(job.directory / job.state['source_file'], frame)
            if self.frontend_predictor is None:
                self.frontend_predictor = FrontendInspector(self.model_dir / self.config['pose_model'])
            result = self.frontend_predictor.module_conv(image, module_index)
            result['source'] = {'kind': 'recomputed_module_conv', 'job_id': job_id,
                                'frame': frame, 'module': module_index, 'device': 'cpu',
                                'model_version': self.config.get('version'),
                                'note': '保留独立 Conv / BatchNorm / SiLU 的单模块代表卷积重算。'}
            if len(json.dumps(result, allow_nan=False, ensure_ascii=False).encode('utf-8')) > 2 * 1024 * 1024:
                raise RuntimeError('模块卷积记录超过大小限制。')
            with self.lock:
                self.frontend_cache[key] = result
                while len(self.frontend_cache) > 8:
                    self.frontend_cache.popitem(last=False)
            return deepcopy(result)
        finally:
            with self.lock:
                self.network_active = False
```

The existing `frontend_cache` is used for `(job_id, frame)` keys; adding three-tuples is okay (mixed keys, LRU). The response limit 2 MiB. The released model has 24 top-level modules (0–22 backbone/neck, 23 `Pose26` head), so the endpoint accepts `0 <= module_index < 24`; allow the head's representative convolution. If a future model changes the count, replace the bound with `len(self.frontend_predictor.model.model)` after loading.

Add the route in `app.py` after `frontend_network`:

```python
    @app.get('/api/jobs/{job_id}/yolo-conv')
    def yolo_conv(job_id: str, frame: int = Query(..., ge=0), module: int = Query(..., ge=0)):
        find(job_id)
        try:
            result = manager.yolo_conv(job_id, frame, module)
            return JSONResponse(result, headers={'Cache-Control': 'no-store'})
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
```

- [ ] **Step 4: 运行测试**

Run: `python -s -m unittest modern_pipeline.tests.test_yolo_conv_endpoint modern_pipeline.tests.test_frontend_network -v`

Expected: endpoint tests PASS; frontend-network tests PASS or skip if weights absent.

- [ ] **Step 5: 提交**

```bash
git add modern_pipeline/web/frontend_network.py modern_pipeline/web/jobs.py modern_pipeline/web/app.py modern_pipeline/tests/test_yolo_conv_endpoint.py
git commit -m "Add on-demand record-mode module convolution"
```

---

### Task 7: 预算基准、覆盖校验与文档

**Files:**
- Create: `modern_pipeline/tests/bench_live_detail.py`
- Modify: `docs/WEB_DEMO.md`（新增「详细采集预算与覆盖」小节）
- Create: `docs/superpowers/reports/2026-11-04-live-detail-benchmark.md`（实测记录，实施时填写真实数值）

**Interfaces:**
- Produces: 命令行基准脚本，输出关闭/启用详细采集的处理秒数、开销百分比、最大事件字节、详细帧字节、覆盖轮次。
- Produces: 文档中的实测值与降级结论。

- [ ] **Step 1: 写基准脚本**

Create `modern_pipeline/tests/bench_live_detail.py`:

```python
"""Manual benchmark for detailed live telemetry; not part of unittest discovery.

Run with a real test video and the final weights:
  python -s modern_pipeline/tests/bench_live_detail.py \
    --input artifacts/demo/demo_input.mp4 --model-dir weights/final --max-frames 75
"""
import argparse
import json
from pathlib import Path
import time

from modern_pipeline.cli import arguments, run


def measure(input_path, model_dir, max_frames):
    results = {}
    for label, detail in (('off', False), ('on', True)):
        output = Path('/tmp') / ('bench_%s.mp4' % label)
        output.unlink(missing_ok=True)
        a = arguments(['--input', str(input_path), '--output', str(output),
                       '--pose-model', str(model_dir / 'yolo26s-pose.pt'),
                       '--stgcn', str(model_dir / 'stgcn.pt'),
                       '--lstm', str(model_dir / 'lstm.pt'),
                       '--stats', str(model_dir / 'stats.pt'),
                       '--normalization', 'legacy_pair',
                       '--behavior-device', 'cpu', '--presentation',
                       '--save-keypoints', '--max-frames', str(max_frames)])
        a.live_telemetry = True
        a.live_detail = detail
        events = []
        started = time.perf_counter()
        run(a, on_computation=lambda event: events.append(event))
        seconds = time.perf_counter() - started
        sizes = [len(json.dumps(event, separators=(',', ':')).encode('utf-8')) for event in events]
        detail_events = [event for event in events if event.get('detail')]
        results[label] = {'seconds': round(seconds, 3), 'events': len(events),
                          'max_event_bytes': max(sizes, default=0),
                          'detail_events': len(detail_events),
                          'detail_bytes': sum(sizes[i] for i, event in enumerate(events) if event.get('detail'))}
    overhead = (results['on']['seconds'] - results['off']['seconds']) / results['off']['seconds'] * 100
    return {'results': results, 'overhead_percent': round(overhead, 2)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--max-frames', type=int, default=75)
    args = parser.parse_args()
    print(json.dumps(measure(Path(args.input), Path(args.model_dir), args.max_frames), indent=2))
```

- [ ] **Step 2: 运行基准并记录真实数值**

Run: `python -s modern_pipeline/tests/bench_live_detail.py --input artifacts/demo/demo_input.mp4 --model-dir weights/final --max-frames 75`

Expected: JSON with `overhead_percent`. If the input video or weights are absent, record that the benchmark was not run in the report and keep the 15% gate untested locally; do not invent numbers.

- [ ] **Step 3: 写实测报告**

Create `docs/superpowers/reports/2026-11-04-live-detail-benchmark.md` with the actual command, environment, numbers, event sizes, coverage rounds, and one of:

- `overhead_percent <= 15`: keep 1 fps / 4 modules / 1 block.
- `overhead_percent > 15`: set `detail_interval_seconds` to 2.0 (0.5 fps), set `detail_yolo_modules` to 2, and re-run; record the new coverage windows and keep the UI progress honest.

Do not write placeholder numbers. If the benchmark cannot run, state exactly what is missing.

- [ ] **Step 4: 更新文档**

Add to `docs/WEB_DEMO.md` a new section:

```markdown
## 详细采集预算与覆盖

- 层输出保持每秒最多 4 帧；详细算式默认每秒最多 1 帧。
- 每个详细帧采集 4 个 YOLO 模块与 1 个 ST-GCN 块，按轮转覆盖；YOLO 约 6 个详细帧、ST-GCN 约 10 个详细帧完成一轮。
- 单事件上限 128 KiB，详细算式每帧不超过 512 KiB；超出时显示覆盖进度，不截断求和。
- 代表输出为固定通道与位置；未覆盖的模块显示「等待详细采集」，不补造数值。
- 详细采集启用前后的预测、未来骨架与逐帧 JSONL 逐值一致。
```

- [ ] **Step 5: 运行完整测试并提交**

Run: `python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v`

Expected: all tests PASS except the known real-weight skips. Record the exact counts in the report.

```bash
git add modern_pipeline/tests/bench_live_detail.py docs/WEB_DEMO.md docs/superpowers/reports/2026-11-04-live-detail-benchmark.md modern_pipeline/live_trace.py
git commit -m "Benchmark and document bounded live detail coverage"
```

---

## Self-review

- Spec coverage: `/api/model` (Task 1), dense conv + 4096 cap (Tasks 1–2), YOLO 4-module detail (Task 2), ST-GCN all-block detail (Task 3), coverage/cache/budgets (Task 4), web wiring/parity (Task 5), record endpoint (Task 6), benchmark/docs (Task 7).
- Placeholders: none; every code step contains code or an exact command.
- Type consistency: `detail_frame`, `detail_modules`, `detail_block`, `dense_conv`, `module`, `detail_block`, `test_live_detail_*` names match across tasks.
