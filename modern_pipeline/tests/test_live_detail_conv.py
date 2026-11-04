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
