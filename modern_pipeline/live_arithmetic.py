"""Complete, bounded scalar examples from existing live forwards.

The spatial/temporal indexing follows web/network.py's inspected sums, but this
module has no web dependency and never calls a network. Only representative
operands are copied. Calculation inputs, weights, scalar bias/actual, and graph
edges are serialized to five decimals with non-finite values as null; pool
contributions are scaled operands. Probability provenance and normalization
statistics retain their observed precision.
"""
import math

import torch
from torch.nn import functional as F

from modern_pipeline.model_catalog import select_representative_conv

# Each ST-GCN block occupies two detail frames: an odd YOLO subframe with
# spatial + middle rows and an even boundary subframe with spatial + middle +
# boundary rows. Shallow blocks can carry every boundary track; deep blocks
# (more than 576 temporal terms) emit one rotating boundary track so the frame
# stays under 512 KiB detail / 768 KiB total even when YOLO layer events share
# it. Three 20-frame block cycles cover all boundary tracks. The Task 7
# benchmark remains the production measurement for combined frames.
DETAIL_BOUNDARY_TERM_LIMIT = 4608
DETAIL_DEEP_BOUNDARY_TERM_LIMIT = 576
DETAIL_DEEP_BOUNDARY_TRACKS = 1


def _rounded(tensor):
    return [None if not torch.isfinite(value) else round(float(value), 5)
            for value in tensor.detach().reshape(-1).cpu()]


def _scalar(value):
    if value is None:
        return None
    value = float(value)
    return round(value, 5) if math.isfinite(value) else None


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
    height, width = value.shape[-2], value.shape[-1]
    padding = []
    for _channel in range(int(conv.in_channels)):
        for ky in range(kh):
            for kx in range(kw):
                iy = y * sy - py + ky * dy
                ix = x * sx - px + kx * dx
                padding.append(iy < 0 or iy >= height or ix < 0 or ix >= width)
    weights = conv.weight[0]
    raw = output[0, 0, y, x]
    activation = selection.activation
    if activation in ('batch_norm', 'batch_norm_silu'):
        bn = selection.parent.bn
        channel = 0
        activated = (raw - bn.running_mean[channel]) / torch.sqrt(bn.running_var[channel] + bn.eps)
        activated = activated * bn.weight[channel] + bn.bias[channel]
        if activation == 'batch_norm_silu':
            activated = F.silu(activated)
    elif activation == 'silu':
        activated = F.silu(raw)
    else:
        activated = raw
    if conv.bias is None:
        bias = 0.
    elif not torch.isfinite(conv.bias[0]):
        bias = None
    else:
        bias = round(float(conv.bias[0]), 5)
    return {
        'id': 'yolo.conv.%d' % module_index, 'stage': 'yolo',
        'module': int(module_index), 'fused': bool(fused),
        'title': 'YOLO 模块 %d · %s 代表卷积' % (module_index, type(selection.parent).__name__),
        'operation': 'dense_conv', 'track_id': None,
        'target': {'channel': 0, 'y': y, 'x': x},
        'term_axes': ['in_channel', 'kernel_y', 'kernel_x'],
        'term_shape': [int(conv.in_channels), int(kh), int(kw)],
        'inputs': _rounded(patch), 'weights': _rounded(weights), 'bias': bias,
        'actual': None if not torch.isfinite(activated) else round(float(activated), 5),
        'activation': activation, 'formula': 'y = activation(Σc Σky Σkx(x × W) + b)',
        'input_indices': [], 'connected': [True] * int(patch.numel()),
        'padding': padding,
        'metadata': {'complete_sum': True, 'representative': True, 'detail': True,
                     'module_path': selection.path, 'target_indices': [0, y, x],
                     'input_shape': list(value.shape), 'stride': list(conv.stride),
                     'padding': list(conv.padding), 'dilation': list(conv.dilation),
                     'fused': bool(fused), 'term_count': int(patch.numel()),
                     'source': 'actual_forward_hooks'}}


def values(tensor):
    return [_scalar(value) for value in tensor.detach().reshape(-1).cpu()]


def calculation(identifier, stage, title, track_id, target, inputs, weights,
                actual, indices, axes, bias=0., connected=None, padding=None,
                activation='identity', operation=None, **metadata):
    inputs = [_scalar(value) for value in inputs]
    weights = [_scalar(value) for value in weights]
    size = len(inputs)
    return {'id': identifier, 'stage': stage, 'title': title,
            'operation': stage if operation is None else operation,
            'track_id': track_id, 'target': target,
            'inputs': inputs, 'weights': weights, 'bias': _scalar(bias),
            'actual': _scalar(actual), 'activation': activation,
            'formula': metadata.pop('formula', 'y = Σᵢ(xᵢ × wᵢ) + b'),
            'input_indices': indices,
            'connected': [True] * size if connected is None else connected,
            'padding': [False] * size if padding is None else padding,
            'metadata': {'complete_sum': True, 'representative': True,
                         'input_axes': axes, 'source': 'actual_forward_hooks', **metadata}}


class LiveArithmetic:
    def __init__(self, trace):
        self.trace = trace
        self.predictor = None
        self.yolo_fused = False
        self.yolo_conv_state = {}
        self.reset()

    def reset(self):
        self.statistics = []
        self.pool_values = None
        self.forecast_values = []
        self.classifiers = []
        self.yolo_conv_state = {}

    def publish(self, row, model, stage):
        if not self.trace.detail_frame:
            return
        metadata = row['metadata']
        detail = {}
        if metadata.get('detail'):
            metadata.setdefault('detail_frame', self.trace.detail_index)
            detail = {'detail': True, 'detail_block': metadata.get('detail_block')}
        self.trace._publish(model, 'flow.' + row['id'], stage, {}, None,
                            calculations=[row], representative=True, **detail,
                            timing_note='Operands observed during the running forward; pedagogical playback has no inference timing meaning.')

    def begin_behavior(self, histories):
        self.statistics = []
        if (not self.trace.active or not self.trace.detail_frame or histories is None
                or self.predictor is None or self.predictor.lstm is None):
            return
        predictor = self.predictor
        poses = torch.as_tensor(histories[:len(self.trace.track_ids)], dtype=torch.float32,
                                device=predictor.device)
        if predictor.normalization == 'training_stats':
            mean, std = predictor.stats['lstm_mean'], predictor.stats['lstm_std']
            for batch in range(len(poses)):
                self.statistics.append((float(poses[batch, 0, 0, 0]), float(mean[0, 0]), float(std[0, 0]),
                                        'saved_training_statistics_used_by_predictor'))
        else:
            # These statistics are recomputed by the identical production recipe;
            # the normalized result is independently captured at the LSTM input.
            mean = poses.mean(dim=1, keepdim=True)
            std = poses.std(dim=1, correction=0, keepdim=True) + predictor.EPSILON
            for batch in range(len(poses)):
                self.statistics.append((float(poses[batch, 0, 0, 0]), float(mean[batch, 0, 0, 0]),
                                        float(std[batch, 0, 0, 0]), 'recomputed_from_actual_history_with_production_recipe'))

    def normalization(self, sequence):
        for batch, (raw, mean, std, source) in enumerate(self.statistics):
            track_id = self.trace.track_ids[batch]
            row = calculation('lstm.normalize.track%d' % track_id, 'normalization',
                              '历史坐标标准化', track_id, '第 1 帧 · 关节 0 · x 标准化坐标',
                              [raw, mean], [1. / std, -1. / std], sequence[batch, 0, 0],
                              [[0, 0, 0], [0, 0, 0]], ['time', 'joint', 'coordinate'],
                              mean=mean, std=std, statistics_source=source,
                              input_names=['原始 x 坐标', 'x 坐标均值'], target_indices=[0, 0, 0],
                              formula='z = (x − mean) / std',
                              actual_source='first_round_lstm_input', normalization=self.predictor.normalization)
            self.publish(row, 'lstm', 4)

    def decoder(self, module, inputs, output, duration):
        self.forecast_values = values(output[:len(self.trace.track_ids), 0])
        if self.trace.round != 1:
            return
        weights = values(module.weight[0])
        bias = float(module.bias[0]) if module.bias is not None else 0.
        for batch, track_id in enumerate(self.trace.track_ids):
            row = calculation('lstm.decoder.round1.track%d' % track_id, 'decoder', 'LSTM 隐状态解码',
                              track_id, '预测第 1 帧 · 关节 0 · x 标准化坐标',
                              values(inputs[0][batch]), weights, output[batch, 0],
                              [[i] for i in range(len(weights))], ['hidden_unit'], bias=bias,
                              round=1, output_index=0, target_indices=[0, 0, 0],
                              formula='forecast[0,0,x] = Σᵤ(h[u] × W[0,u]) + b[0]')
            self.publish(row, 'lstm', 4)

    def denormalization(self, event):
        if not self.statistics or len(self.forecast_values) != len(self.statistics):
            return
        step = event['step']
        for batch, (_, mean, std, source) in enumerate(self.statistics):
            track_id = self.trace.track_ids[batch]
            row = calculation('lstm.denormalize.round%d.track%d' % (step, track_id),
                              'denormalization', '预测坐标还原', track_id,
                              '预测第 %d 帧 · 关节 0 · x 像素坐标' % ((step - 1) * 10 + 1),
                              [self.forecast_values[batch], mean], [std, 1.],
                              event['poses'][batch, 0, 0, 0], [[0, 0, 0], [0, 0, 0]],
                              ['time', 'joint', 'coordinate'], round=step, mean=mean, std=std,
                              statistics_source=source, input_names=['预测的标准化 x', 'x 坐标均值'],
                              target_indices=[(step - 1) * 10, 0, 0],
                              formula='x_pixels = forecast_normalized × std + mean',
                              actual_source='production_forecast_step_event')
            self.publish(row, 'lstm', 4)

    def attach_behavior(self, predictor):
        if not self.trace.detail:
            return
        self.predictor = predictor
        trace = self.trace
        if predictor.lstm is not None:
            trace._hook(predictor.lstm.fc, lambda inputs, output, duration:
                        self.decoder(predictor.lstm.fc, inputs, output, duration))
        stgcn = predictor.stgcn
        joint = int((stgcn.A != 0).sum(dim=(0, 1)).argmax())
        states = {index: {} for index in range(len(stgcn.st_gcn_networks))}

        def selected(index):
            return self.trace.detail_frame and self.trace.detail_block == index

        for index, block in enumerate(stgcn.st_gcn_networks):
            def projected(inputs, output, duration, index=index):
                if not selected(index):
                    return
                shaped = output.reshape(output.shape[0], stgcn.A.shape[0], -1,
                                        output.shape[2], output.shape[3])
                states[index]['projected'] = shaped[:len(trace.track_ids), :, 0,
                                                     output.shape[2] // 2, :].detach().clone()

            def spatial(inputs, output, duration, index=index):
                if not selected(index) or 'projected' not in states[index]:
                    return
                tensor, adjacency = output
                weights = values(inputs[1][:, :, joint])
                timestep = tensor.shape[2] // 2
                edge_indices = (stgcn.A != 0).nonzero().detach().cpu().tolist()
                effective = inputs[1].detach().cpu()
                edges = [[k, source, receiver, _scalar(effective[k, source, receiver])]
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
                        operation='graph_aggregation',
                        formula='y[0,t,w] = Σₖ Σᵥ(projected[k,0,t,v] × effective_A[k,v,w])')
                    row['metadata']['detail'] = True
                    row['metadata']['detail_block'] = index
                    self.publish(row, 'stgcn', 5)
                states[index].pop('projected', None)

            def temporal(inputs, output, duration, index=index, block=block):
                if not selected(index):
                    return
                value = inputs[0]
                kernel = block.tcn[2].kernel_size[0]
                stride = block.tcn[2].stride[0]
                padding = block.tcn[2].padding[0]
                dilation = block.tcn[2].dilation[0]
                weights = values(block.tcn[2].weight[0, :, :, 0])
                middle = output.shape[2] // 2
                terms = value.shape[1] * kernel
                if not self.trace.detail_boundary:
                    boundary_tracks = 0
                elif terms > DETAIL_DEEP_BOUNDARY_TERM_LIMIT:
                    boundary_tracks = min(len(trace.track_ids), DETAIL_DEEP_BOUNDARY_TRACKS)
                else:
                    boundary_tracks = min(len(trace.track_ids),
                                          DETAIL_BOUNDARY_TERM_LIMIT // terms)
                tracks = list(enumerate(trace.track_ids))
                for boundary, timestep in ((False, middle), (True, 0)):
                    if boundary and timestep == middle:
                        continue
                    if boundary and boundary_tracks:
                        shift = (self.trace.detail_index // 2) % len(trace.track_ids)
                        chosen = (tracks[shift:] + tracks[:shift])[:boundary_tracks]
                    elif boundary:
                        chosen = []
                    else:
                        chosen = tracks
                    times = [timestep * stride - padding + offset * dilation
                             for offset in range(kernel)]
                    indices = [[channel, time, joint] for channel in range(value.shape[1])
                               for time in times]
                    pads = [time < 0 or time >= value.shape[2] for _, time, _ in indices]
                    bias = float(block.tcn[2].bias[0]) if block.tcn[2].bias is not None else 0.
                    for batch, track_id in chosen:
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
                            operation='temporal_convolution',
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

    def pool(self, tensor):
        if self.pool_values is None:
            return
        times, joints = self.pool_values.shape[1:]
        count = times * joints
        # A five-decimal uniform weight cannot sum to one; express the mean as
        # the sum of per-term scaled contributions instead, so every operand is
        # exactly five decimals and the weights are exactly one.
        weights = [1.] * count
        for batch, track_id in enumerate(self.trace.track_ids):
            row = calculation('stgcn.pool.track%d' % track_id, 'pool', '全局平均池化', track_id,
                              '第 10 块输出 · 通道 0 的平均值',
                              [None if value is None else value / count
                               for value in values(self.pool_values[batch])],
                              weights, tensor[batch, 0, 0, 0],
                              [[0, time, joint] for time in range(times) for joint in range(joints)],
                              ['channel', 'time', 'joint'], target_indices=[0],
                              term_shape=[times, joints], divisor=count,
                              weight_rounding='operands_scaled_by_divisor',
                              formula='pool[0] = Σt Σv((x[0,t,v] / (T×V)) × 1)')
            self.publish(row, 'stgcn', 5)
        self.pool_values = None

    def classifier(self, module, inputs, output):
        self.classifiers = []
        for batch, track_id in enumerate(self.trace.track_ids):
            logits = values(output[batch])
            operands = values(inputs[0][batch, :, 0, 0])
            finite = [(index, value) for index, value in enumerate(logits) if value is not None]
            if finite:
                selected, actual = max(finite, key=lambda item: item[1])
                weights = values(module.weight[selected, :, 0, 0])
                bias = float(module.bias[selected]) if module.bias is not None else 0.
                target = '类别 %d 的 logit' % selected
                targets = [selected]
            else:
                selected = actual = bias = None
                weights = [None] * len(operands)
                target = '类别 logit（全部非有限）'
                targets = []
            row = calculation('stgcn.classifier.track%d' % track_id, 'classifier', '意图分类输出', track_id,
                              target, operands, weights, actual,
                              [[i] for i in range(len(operands))], ['channel'],
                              bias=bias, target_indices=targets, selected_class=selected,
                              logits=logits, class_labels=['irrelevant', 'not crossing', 'crossing'],
                              formula='logit[class] = Σc(pool[c] × W[class,c]) + b[class]')
            if not finite:
                row['metadata']['complete_sum'] = False
                row['metadata']['nonfinite_logits'] = True
            self.classifiers.append(row)

    def finish(self, tracks):
        # Softmax is functional, outside the classifier module. The CLI's final
        # records provide independently observed probabilities without replay.
        by_id = {track['track_id']: track for track in tracks}
        for row in self.classifiers:
            track = by_id.get(row['track_id'], {})
            if track.get('prediction_updated') and track.get('probabilities') is not None:
                row['metadata']['probabilities'] = list(track['probabilities'])
                row['metadata']['probabilities_source'] = 'production_prediction_result'
            self.publish(row, 'stgcn', 5)
        self.reset()

    def attach_yolo(self, network, fused, guard):
        self.yolo_fused = fused
        self.yolo_conv_state = {}
        for index, module in enumerate(network.model):
            selection = select_representative_conv(module, fused=fused)
            if selection is None:
                continue

            def capture(inputs, output, duration, index=index, selection=selection):
                if not self.trace.active or not self.trace.detail_frame or index not in self.trace.detail_modules:
                    return
                # The row must be built before the parent's in-place activation
                # mutates the convolution output.
                self.yolo_conv_state[index] = dense_conv_row(
                    index, selection, inputs[0], output, self.yolo_fused)

            self.trace._hook(selection.conv, capture, guard=guard, handles=self.trace.yolo_handles)

    def yolo_module(self, index):
        row = self.yolo_conv_state.pop(index, None)
        return [] if row is None else [row]
