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


class BatchNormed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, value):
        return self.bn(self.conv(value))


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
    def setUp(self):
        # Five-decimal operands make the reconstruction tolerance dependent on
        # the random convolution init; seed it so the check is reproducible.
        torch.manual_seed(36)

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
        for key in ('complete_sum', 'detail', 'module_path', 'target_indices',
                    'input_shape', 'stride', 'padding', 'dilation'):
            self.assertIn(key, row['metadata'])
        self.assertTrue(row['metadata']['complete_sum'])
        self.assertTrue(row['metadata']['detail'])

    def test_non_detail_frame_emits_no_dense_row(self):
        block = Fused()
        yolo, runner, callbacks = runner_for([block])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=999)
        attach(trace, yolo, runner, callbacks)
        self.assertTrue(trace.begin_frame(1, 0.0))
        self.assertTrue(trace.detail_frame)
        self.assertTrue(trace.begin_frame(2, 0.1))
        self.assertFalse(trace.detail_frame)
        self.assertEqual(trace.detail_modules, set())
        block(torch.randn(1, 3, 12, 12))
        trace.close()
        self.assertFalse(any(event.get('calculations') for event in events))

    def test_detail_off_installs_no_representative_conv_hooks(self):
        block = Fused()
        yolo, runner, callbacks = runner_for([block])
        events = []
        trace = LiveTrace(events.append, interval=0)
        self.assertFalse(trace.detail)
        attach(trace, yolo, runner, callbacks)
        self.assertFalse(block.conv._forward_hooks)
        self.assertFalse(block.conv._forward_pre_hooks)
        self.assertEqual(len(block._forward_hooks), 1)
        self.assertEqual(len(block._forward_pre_hooks), 1)
        trace.begin_frame(1, 0.0)
        block(torch.randn(1, 3, 12, 12))
        self.assertFalse(any(event.get('calculations') for event in events))
        trace.close()
        self.assertFalse(block._forward_hooks)
        self.assertFalse(block._forward_pre_hooks)

    def test_six_detail_frames_cover_all_twenty_four_modules(self):
        yolo, runner, callbacks = runner_for([torch.nn.Identity() for _ in range(24)])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        # Production order: the first frame starts before on_predict_start.
        self.assertTrue(trace.begin_frame(0, 0.0))
        self.assertEqual(trace.detail_modules, set())
        attach(trace, yolo, runner, callbacks)
        self.assertEqual(trace.detail_modules, {4, 5, 6, 7})
        seen = set(trace.detail_modules)
        for step in range(1, 6):
            self.assertTrue(trace.begin_frame(step, float(step)))
            self.assertTrue(trace.detail_frame)
            seen.update(trace.detail_modules)
        trace.close()
        self.assertEqual(seen, set(range(24)))

    def test_receptive_field_padding_marks_zero_operands(self):
        block = Fused(in_channels=1, out_channels=1, kernel=3, stride=1, padding=1)
        yolo, runner, callbacks = runner_for([block])
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        attach(trace, yolo, runner, callbacks)
        trace.begin_frame(1, 0.0)
        block(torch.ones(1, 1, 1, 1))
        trace.close()
        row = [c for event in events for c in event.get('calculations', [])][0]
        self.assertEqual(row['term_shape'], [1, 3, 3])
        self.assertEqual(row['padding'], [True, True, True, True, False, True, True, True, True])
        self.assertEqual(row['inputs'], [0., 0., 0., 0., 1., 0., 0., 0., 0.])
        self.assertTrue(all(row['connected']))

    def test_batch_norm_activation_applies_affine_without_silu(self):
        block = BatchNormed().eval()
        with torch.no_grad():
            block.bn.running_mean.fill_(0.2)
            block.bn.running_var.fill_(1.5)
            block.bn.weight.fill_(1.3)
            block.bn.bias.fill_(-0.4)
        yolo, runner, callbacks = runner_for([block], fused=False)
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        attach(trace, yolo, runner, callbacks)
        trace.begin_frame(1, 0.0)
        value = torch.arange(3 * 12 * 12).reshape(1, 3, 12, 12).float() / 100
        expected = block(value)
        trace.close()
        row = [c for event in events for c in event.get('calculations', [])][0]
        self.assertEqual(row['activation'], 'batch_norm')
        self.assertFalse(row['fused'])
        self.assertAlmostEqual(row['actual'], float(expected[0, 0, 3, 3]), delta=2e-5)

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
