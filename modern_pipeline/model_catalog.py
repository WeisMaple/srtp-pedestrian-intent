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
    prefix = 'model.' if modules is not network else ''
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
                'module_path': prefix + str(index) + '.' + selection.path,
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
