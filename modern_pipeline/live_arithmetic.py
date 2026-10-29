"""Complete, bounded scalar examples from existing live forwards.

The spatial/temporal indexing follows web/network.py's inspected sums, but this
module has no web dependency and never calls a network. Only representative
operands are copied. All floating values retain the observed tensor precision.
"""


def values(tensor):
    return tensor.detach().reshape(-1).cpu().tolist()


def calculation(identifier, stage, title, track_id, target, inputs, weights,
                actual, indices, axes, bias=0., connected=None, padding=None,
                activation='identity', **metadata):
    size = len(inputs)
    return {'id': identifier, 'stage': stage, 'title': title,
            'operation': stage, 'track_id': track_id, 'target': target,
            'inputs': inputs, 'weights': weights, 'bias': float(bias),
            'actual': float(actual), 'activation': activation,
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
        self.reset()

    def reset(self):
        self.statistics = []
        self.projected = None
        self.pool_values = None
        self.forecast_values = []
        self.classifiers = []
        self.yolo_value = None

    def publish(self, row, model, stage):
        self.trace._publish(model, 'flow.' + row['id'], stage, {}, None,
                            calculations=[row], representative=True,
                            timing_note='Operands observed during the running forward; pedagogical playback has no inference timing meaning.')

    def begin_behavior(self, histories):
        self.statistics = []
        if not self.trace.active or histories is None or self.predictor is None or self.predictor.lstm is None:
            return
        import torch
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
        self.predictor = predictor
        trace = self.trace
        if predictor.lstm is not None:
            trace._hook(predictor.lstm.fc, lambda inputs, output, duration:
                        self.decoder(predictor.lstm.fc, inputs, output, duration))
        stgcn = predictor.stgcn
        block = stgcn.st_gcn_networks[0]
        joint = int((stgcn.A != 0).sum(dim=(0, 1)).argmax())

        def projected(inputs, output, duration):
            shaped = output.reshape(output.shape[0], stgcn.A.shape[0], -1, output.shape[2], output.shape[3])
            self.projected = shaped[:len(trace.track_ids), :, 0, output.shape[2] // 2, :].detach().clone()

        def spatial(inputs, output, duration):
            tensor, adjacency = output
            if self.projected is None:
                return
            weights = values(inputs[1][:, :, joint])  # Already A * edge_importance.
            timestep = tensor.shape[2] // 2
            # Keep original topology even when a learned coefficient is zero.
            # Sparse [partition, source, receiver, effective coefficient] entries
            # let the UI inspect every receiver without another model forward.
            edge_indices = (stgcn.A != 0).nonzero().detach().cpu().tolist()
            effective = inputs[1].detach().cpu()
            edges = [[k, source, receiver, float(effective[k, source, receiver])]
                     for k, source, receiver in edge_indices]
            indices = [[partition, 0, timestep, source_joint]
                       for partition in range(inputs[1].shape[0]) for source_joint in range(tensor.shape[3])]
            for batch, track_id in enumerate(trace.track_ids):
                row = calculation('stgcn.spatial.block0.track%d' % track_id, 'spatial', '第 1 块 · 骨架邻域聚合',
                                  track_id, '通道 0 · 时刻 %d · 目标关节 %d' % (timestep, joint),
                                  values(self.projected[batch]), weights, tensor[batch, 0, timestep, joint],
                                  indices, ['partition', 'channel', 'time', 'joint'],
                                  connected=[weight != 0 for weight in values(stgcn.A[:, :, joint])], block=0,
                                  term_shape=[inputs[1].shape[0], tensor.shape[3]],
                                  target_indices=[0, timestep, joint],
                                  graph_slice={'channel': 0, 'timestep': timestep, 'joint_count': tensor.shape[3],
                                               'edges': edges, 'outputs': values(tensor[batch, 0, timestep, :]),
                                               'source': 'actual_gcn_forward_same_channel_and_timestep'},
                                  weight_source='actual_gcn_input_effective_adjacency_including_edge_importance',
                                  formula='y[0,t,w] = Σₖ Σᵥ(projected[k,0,t,v] × effective_A[k,v,w])')
                self.publish(row, 'stgcn', 5)
            self.projected = None

        temporal_module = block.tcn[2]

        def temporal(inputs, output, duration):
            import torch
            value = inputs[0]
            # Boundary output deliberately exposes real zero padding.
            timestep = 0
            kernel = temporal_module.kernel_size[0]
            times = [timestep * temporal_module.stride[0] - temporal_module.padding[0]
                     + offset * temporal_module.dilation[0] for offset in range(kernel)]
            weights = values(temporal_module.weight[0, :, :, 0])
            indices = [[channel, time, joint] for channel in range(value.shape[1]) for time in times]
            padding = [time < 0 or time >= value.shape[2] for _, time, _ in indices]
            bias = float(temporal_module.bias[0]) if temporal_module.bias is not None else 0.
            for batch, track_id in enumerate(trace.track_ids):
                window = torch.stack([value[batch, :, time, joint] if 0 <= time < value.shape[2]
                                      else value.new_zeros(value.shape[1]) for time in times], dim=1)
                row = calculation('stgcn.temporal.block0.track%d' % track_id, 'temporal', '第 1 块 · 时间卷积',
                                  track_id, '通道 0 · 时刻 0 · 关节 %d' % joint,
                                  values(window), weights, output[batch, 0, timestep, joint], indices,
                                  ['channel', 'time', 'joint'], bias=bias, padding=padding,
                                  block=0, target_indices=[0, timestep, joint], term_shape=[value.shape[1], kernel],
                                  kernel_size=kernel, stride=temporal_module.stride[0],
                                  convolution_padding=temporal_module.padding[0], dilation=temporal_module.dilation[0],
                                  formula='y[0,t,v] = Σ𝚌 Σq(x[c,t×stride−pad+q×dilation,v] × W[0,c,q]) + b[0]')
                self.publish(row, 'stgcn', 5)

        def last_block(inputs, output, duration):
            self.pool_values = output[0][:len(trace.track_ids), 0].detach().clone()

        trace._hook(block.gcn.conv, projected)
        trace._hook(block.gcn, spatial)
        trace._hook(temporal_module, temporal)
        trace._hook(stgcn.st_gcn_networks[-1], last_block)

    def pool(self, tensor):
        if self.pool_values is None:
            return
        times, joints = self.pool_values.shape[1:]
        count = times * joints
        for batch, track_id in enumerate(self.trace.track_ids):
            row = calculation('stgcn.pool.track%d' % track_id, 'pool', '全局平均池化', track_id,
                              '第 10 块输出 · 通道 0 的平均值', values(self.pool_values[batch]),
                              [1. / count] * count, tensor[batch, 0, 0, 0],
                              [[0, time, joint] for time in range(times) for joint in range(joints)],
                              ['channel', 'time', 'joint'], target_indices=[0],
                              term_shape=[times, joints], divisor=count,
                              formula='pool[0] = Σt Σv(x[0,t,v] × 1/(T×V))')
            self.publish(row, 'stgcn', 5)
        self.pool_values = None

    def classifier(self, module, inputs, output):
        self.classifiers = []
        for batch, track_id in enumerate(self.trace.track_ids):
            logits = values(output[batch])
            selected = max(range(len(logits)), key=logits.__getitem__)
            weights = values(module.weight[selected, :, 0, 0])
            bias = float(module.bias[selected]) if module.bias is not None else 0.
            row = calculation('stgcn.classifier.track%d' % track_id, 'classifier', '意图分类输出', track_id,
                              '类别 %d 的 logit' % selected, values(inputs[0][batch, :, 0, 0]),
                              weights, logits[selected], [[i] for i in range(len(weights))], ['channel'],
                              bias=bias, target_indices=[selected], selected_class=selected,
                              logits=logits, class_labels=['irrelevant', 'not crossing', 'crossing'],
                              formula='logit[class] = Σc(pool[c] × W[class,c]) + b[class]')
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
        import torch
        outer = network.model[0]
        direct = isinstance(outer, torch.nn.Conv2d)
        module = outer if direct else getattr(outer, 'conv', None)
        activation = 'identity' if direct else ('silu' if isinstance(getattr(outer, 'act', None), torch.nn.SiLU) else None)
        if (not isinstance(module, torch.nn.Conv2d) or module.groups != 1
                or module.in_channels != 3 or module.kernel_size != (3, 3)
                or activation is None or (not direct and hasattr(outer, 'bn'))):
            return

        def capture(inputs, output, duration):
            value = inputs[0]
            target_y, target_x = output.shape[2] // 2, output.shape[3] // 2
            indices = [[channel, target_y * module.stride[0] - module.padding[0] + y * module.dilation[0],
                        target_x * module.stride[1] - module.padding[1] + x * module.dilation[1]]
                       for channel in range(3) for y in range(3) for x in range(3)]
            padded = [y < 0 or y >= value.shape[2] or x < 0 or x >= value.shape[3]
                      for channel, y, x in indices]
            operands = [0. if pad else float(value[0, channel, y, x])
                        for (channel, y, x), pad in zip(indices, padded)]
            bias = float(module.bias[0]) if module.bias is not None else 0.
            self.yolo_value = calculation('yolo.conv0', 'yolo', 'YOLO 首层卷积与激活', None,
                                          '输出通道 0 · 位置 (%d, %d)' % (target_y, target_x), operands, values(module.weight[0]),
                                          output[0, 0, target_y, target_x], indices, ['RGB_channel', 'y', 'x'],
                                          bias=bias, padding=padded, activation=activation,
                                          target_indices=[0, target_y, target_x], term_shape=[3, 3, 3], fused=fused,
                                          input_shape=list(value.shape), stride=list(module.stride),
                                          convolution_padding=list(module.padding),
                                          coordinate_system='resized_letterboxed_rgb_normalized_0_to_1',
                                          formula='y = SiLU(Σc Σdy Σdx(x[c,y+dy,x+dx] × W[0,c,dy,dx]) + b[0])'
                                          if activation == 'silu' else 'y = Σᵢ(xᵢ × wᵢ) + b')
            self.yolo_value['preactivation'] = float(output[0, 0, target_y, target_x])

        self.trace._hook(module, capture, guard=guard, handles=self.trace.yolo_handles)

    def yolo_output(self, output):
        row, self.yolo_value = self.yolo_value, None
        if row is None:
            return []
        channel, y, x = row['metadata']['target_indices']
        row['actual'] = float(output[0, channel, y, x])
        return [row]
