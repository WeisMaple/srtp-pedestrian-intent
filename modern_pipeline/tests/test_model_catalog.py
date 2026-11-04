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
        network.model[0].f = -1
        network.model[1].f = [0]
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


if __name__ == '__main__':
    unittest.main()
