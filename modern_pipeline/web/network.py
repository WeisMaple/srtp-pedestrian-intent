"""Read-only hooks with bounded arithmetic examples from the production models.

The hooks run around BehaviorPredictor.predict, not a second implementation of
the forecasting or classification algorithm. LSTM layer 1 gates use the real
weights and are checked against hooked h_n/c_n. Convolution, decoder, residual,
normalization and pooling examples retain independent actual PyTorch outputs.
"""
import json
import math

import torch
from torch.nn import functional as F

from modern_pipeline.cli import LABELS
from modern_pipeline.history import HistoryStore


def _values(tensor):
    return tensor.detach().cpu().tolist()


def _statistics(tensor):
    value = tensor.detach().float()
    return {'min': float(value.min()), 'max': float(value.max()),
            'mean': float(value.mean()), 'std': float(value.std(correction=0))}


def _matrix(tensor):
    return {'shape': list(tensor.shape), 'values': _values(tensor), **_statistics(tensor)}


def _activation(tensor):
    """A clearly identified slice, with statistics over the complete tensor."""
    middle = tensor.shape[2] // 2
    return {'shape': list(tensor.shape), 'values': _values(tensor[0, :8, middle, :]),
            'sampling': 'batch 0; first 8 channels; middle timestep; all 17 joints',
            'timestep': middle, **_statistics(tensor)}


def _sum_check(products, actual, bias=0.):
    """Sum every displayed product, retaining the independently hooked value."""
    total = math.fsum(_values(products.reshape(-1))) + bias
    actual = float(actual)
    return {'sum': total, 'actual': actual, 'absolute_error': abs(total - actual)}


def _spatial_arithmetic(module, inputs, output, projected, joint):
    """All K x V terms of one actual einsum, and one complete 1x1 projection."""
    value, adjacency = inputs
    timestep = output.shape[2] // 2
    weights = adjacency[:, :, joint]
    products = projected * weights
    flat_index = int(products.abs().argmax())
    partition, source_joint = divmod(flat_index, products.shape[1])
    projection_channel = partition * output.shape[1]
    projection_input = value[0, :, timestep, source_joint]
    projection_weights = module.conv.weight[projection_channel, :, 0, 0]
    projection_products = projection_input * projection_weights
    bias = float(module.conv.bias[projection_channel]) if module.conv.bias is not None else 0.
    projection = {
        'partition': partition, 'source_joint': source_joint,
        'input_values': _values(projection_input), 'weights': _values(projection_weights),
        'products': _values(projection_products), 'bias': bias,
        **_sum_check(projection_products, projected[partition, source_joint], bias),
        'formula': 'projected[k,c,t,v] = sum_input_channel(W[k,c,i] * x[i,t,v]) + bias[k,c]',
        'sampling': 'all input channels for the largest absolute contributing graph term'}
    return {'timestep': timestep, 'output_channel': 0, 'output_joint': joint,
            'projected_values': _values(projected), 'adjacency': _values(weights),
            'products': _values(products), 'projection': projection,
            **_sum_check(products, output[0, 0, timestep, joint]),
            'formula': 'spatial[c,t,w] = sum_partition_k,sum_source_joint_v(projected[k,c,t,v] * effective_A[k,v,w])',
            'term_shape': list(products.shape), 'term_axes': ['partition', 'source_joint'],
            'complete_sum': True}


def _temporal_arithmetic(module, inputs, output, joint):
    """All input-channel x kernel terms, including true convolution padding."""
    value = inputs[0]
    timestep = output.shape[2] // 2
    kernel, stride = module.kernel_size[0], module.stride[0]
    padding, dilation = module.padding[0], module.dilation[0]
    times = [timestep * stride - padding + offset * dilation for offset in range(kernel)]
    window = torch.stack([value[0, :, time, joint] if 0 <= time < value.shape[2]
                          else value.new_zeros(value.shape[1]) for time in times], dim=1)
    weights = module.weight[0, :, :, 0]
    products = window * weights
    bias = float(module.bias[0]) if module.bias is not None else 0.
    return {'timestep': timestep, 'output_channel': 0, 'output_joint': joint,
            'input_timesteps': times, 'input_values': _values(window), 'weights': _values(weights),
            'products': _values(products), 'input_channels': value.shape[1],
            'kernel_size': kernel, 'stride': stride, 'padding': padding, 'dilation': dilation,
            'bias': bias, **_sum_check(products, output[0, 0, timestep, joint], bias),
            'formula': 'conv[o,t,v] = sum_input_channel_c,sum_kernel_offset_q(W[o,c,q] * x[c,t*stride-padding+q*dilation,v]) + bias[o]',
            'term_axes': ['input_channel', 'kernel_offset'], 'complete_sum': True,
            'padding_note': 'out-of-range input timesteps contribute zero, as in the actual Conv2d'}


@torch.inference_mode()
def inspect_lstm_gates(recurrent, sequence, unit=0):
    """Reconstruct layer 1 for one sequence; keep all units in the recurrence.

    Reporting unit 0 does NOT mean running an isolated scalar recurrence:
    W_hh uses all hidden units at every step, exactly as the trained LSTM does.
    """
    if sequence.ndim != 3 or sequence.shape[0] != 1:
        raise ValueError('Gate inspection requires one [1,T,input_size] sequence')
    if not 0 <= unit < recurrent.hidden_size or recurrent.bidirectional or recurrent.proj_size:
        raise ValueError('Gate inspection requires a valid unit in a unidirectional unprojected LSTM')
    hidden = sequence.new_zeros(1, recurrent.hidden_size)
    cell = hidden.clone()
    steps = []
    weight_input = recurrent.weight_ih_l0
    weight_hidden = recurrent.weight_hh_l0
    bias = recurrent.bias_ih_l0 + recurrent.bias_hh_l0
    selected = torch.arange(4, device=sequence.device) * recurrent.hidden_size + unit
    for t in range(sequence.shape[1]):
        previous_hidden = _values(hidden[0])
        input_dot = F.linear(sequence[:, t], weight_input)
        recurrent_dot = F.linear(hidden, weight_hidden)
        preactivation = input_dot + recurrent_dot + bias
        i, f, g, o = preactivation.chunk(4, dim=1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        previous = float(cell[0, unit])
        cell = f * cell + i * g
        hidden = o * cell.tanh()
        steps.append({'t': t + 1, 'i': float(i[0, unit]), 'f': float(f[0, unit]),
                      'g': float(g[0, unit]), 'o': float(o[0, unit]),
                      'c_previous': previous, 'c': float(cell[0, unit]), 'h': float(hidden[0, unit]),
                      'input_values': _values(sequence[0, t]), 'previous_hidden': previous_hidden,
                      'input_dot': _values(input_dot[0, selected]),
                      'recurrent_dot': _values(recurrent_dot[0, selected]),
                      'bias': _values(bias[selected]),
                      'preactivation': _values(preactivation[0, selected])})
    return {'layer': 1, 'round': 1, 'unit': unit, 'gate_order': ['i', 'f', 'g', 'o'],
            'steps': steps, 'hidden_final': _values(hidden[0]), 'cell_final': _values(cell[0]),
            'input_weights': _values(weight_input[selected]),
            'recurrent_weights': _values(weight_hidden[selected]), 'bias': _values(bias[selected]),
            'formulas': {'z': 'z = W_ih x_t + W_hh h_(t-1) + b_ih + b_hh',
                         'i': 'i_t = sigmoid(z_i)', 'f': 'f_t = sigmoid(z_f)',
                         'g': 'g_t = tanh(z_g)', 'o': 'o_t = sigmoid(z_o)',
                         'c': 'c_t = f_t * c_(t-1) + i_t * g_t',
                         'h': 'h_t = o_t * tanh(c_t)'},
            'initial_state': 'h_0 = c_0 = 0; reset at the start of every forecast round',
            'method': 'Reconstructed with real layer-1 weights and all hidden units; unit 0 displayed'}


def load_reference(path, max_gap=2, ttl=30):
    """Replay the CLI reference choice from original JSONL keypoints, in order.

    The trace ring may have discarded the first eligible person. The JSONL
    contains every recorded person's full-precision (possibly filtered) pose,
    so it can reproduce the CLI's first accepted 50-observation history.
    """
    store = HistoryStore(max_gap=max_gap, ttl=ttl)
    with path.open(encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            frame = record['frame']
            for track in record['tracks']:
                if not track.get('pose_accepted'):
                    continue
                if 'keypoints' not in track:
                    raise ValueError('原始记录缺少关键点，无法恢复固定参考序列。')
                state = store.update(track['track_id'], frame, track['keypoints'])
                if state.ready:
                    return state.as_array(), {
                        'reference_source': 'first eligible 50-observation history replayed from original render.jsonl keypoints',
                        'reference_frame': frame, 'reference_track_id': track['track_id']}
            store.expire(frame)
    raise ValueError('原始记录中没有可恢复的 50 帧固定参考序列。')


@torch.inference_mode()
def inspect_model(predictor, history, reference=None):
    """Return actual model activations for one saved, rounded pose history."""
    history = predictor._validated_array(history, (50, 17, 3), 'history')
    poses = torch.from_numpy(history[None]).to(predictor.device)
    hooks, rounds = [], []
    stgcn = predictor.stgcn
    result = {'input': _matrix(poses[0].reshape(50, 51)),
              'normalization': {'mode': predictor.normalization},
              'lstm': None,
              'stgcn': {'adjacency': _values(stgcn.A), 'edges': [list(edge) for edge in stgcn.graph.edge],
                        'graph_layout': 'yolov8', 'graph_strategy': 'spatial', 'graph_center': stgcn.graph.center,
                        'partition_labels': ['root', 'root + inward', 'outward'],
                        'blocks': [], 'labels': list(LABELS),
                        'parameter_count': sum(p.numel() for p in stgcn.parameters()),
                        'head_weights': _values(stgcn.fcn.weight[:, :, 0, 0]),
                        'head_bias': _values(stgcn.fcn.bias)}}
    result['input']['channels'] = ['x', 'y', 'confidence']
    result['input']['column_order'] = 'joint 0 x,y,confidence; joint 1 x,y,confidence; ... joint 16'

    def register(module, callback, before=False):
        hooks.append(module.register_forward_pre_hook(callback) if before else module.register_forward_hook(callback))

    def recurrent_hook(module, inputs, output):
        sequence, (hidden, cell) = output
        one = {'index': len(rounds) + 1, 'input': _matrix(inputs[0][0]),
               'layers': [{'index': i + 1, 'h': _values(hidden[i, 0, :16]),
                           'c': _values(cell[i, 0, :16]), **_statistics(hidden[i, 0])}
                          for i in range(module.num_layers)],
               'sequence': {'shape': list(sequence[0].shape), 'values': _values(sequence[0, :, :8]),
                            'sampling': 'all 50 timesteps; first 8 units of final recurrent layer'},
               'layer_sampling': 'h_n and c_n at timestep 50; first 16 units; statistics over all h_n units'}
        if not rounds:
            gates = inspect_lstm_gates(module, inputs[0])
            error = max(float((hidden[0, 0] - torch.tensor(gates['hidden_final'], device=hidden.device)).abs().max()),
                        float((cell[0, 0] - torch.tensor(gates['cell_final'], device=cell.device)).abs().max()))
            gates['validation_max_error'] = error
            gates['validation'] = 'max absolute difference against actual PyTorch layer-1 h_n and c_n'
            if error > 1e-4:
                raise RuntimeError('LSTM gate reconstruction did not match the actual recurrent output')
            result['lstm']['gates'] = gates
        rounds.append(one)

    def forecast_hook(module, inputs, output):
        rounds[-1]['forecast'] = _matrix(output[0].reshape(10, 51))

    def decoder_hook(module, inputs, output):
        values, weights = inputs[0][0], module.weight[0]
        products = values * weights
        bias = float(module.bias[0]) if module.bias is not None else 0.
        rounds[-1]['decoder_example'] = {
            'input_values': _values(values), 'weights': _values(weights), 'products': _values(products),
            'bias': bias, **_sum_check(products, output[0, 0], bias),
            'output_index': 0, 'forecast_frame': 0, 'joint': 0, 'coordinate': 'x',
            'formula': 'forecast_normalized[0,0,x] = sum_hidden_unit(final_layer_h[unit] * decoder_weight[0,unit]) + bias[0]',
            'sampling': 'first forecast frame, joint 0 x; every final recurrent-layer hidden unit',
            'complete_sum': True}

    def observe(event):
        if event['kind'] == 'forecast_step':
            rounds[-1]['forecast_poses'] = event['poses'][0].tolist()

    def classifier_input(module, inputs):
        tensor = inputs[0]
        result['stgcn']['input'] = _matrix(tensor[0, :, :, :, 0].permute(1, 2, 0).reshape(50, 51))
        result['stgcn']['input_shape'] = list(tensor.shape)

    def normalized_hook(module, inputs, output):
        result['stgcn']['data_bn'] = _matrix(output[0].transpose(0, 1))
        timestep = output.shape[2] // 2
        value = float(inputs[0][0, 0, timestep])
        mean, variance = float(module.running_mean[0]), float(module.running_var[0])
        weight, bias = float(module.weight[0]), float(module.bias[0])
        normalized = (value - mean) / math.sqrt(variance + module.eps)
        affine = normalized * weight + bias
        actual = float(output[0, 0, timestep])
        result['stgcn']['data_bn_example'] = {
            'feature': 0, 'joint': 0, 'coordinate': 'x', 'timestep': timestep,
            'input_value': value, 'running_mean': mean, 'running_variance': variance,
            'epsilon': module.eps, 'weight': weight, 'bias': bias, 'normalized': normalized,
            'sum': affine, 'actual': actual, 'absolute_error': abs(affine - actual),
            'formula': 'output = weight * (input - running_mean) / sqrt(running_variance + epsilon) + bias',
            'sampling': 'batch/person 0; joint 0 x; middle input timestep; saved evaluation statistics'}

    def pool_hook(module, inputs):
        result['stgcn']['pool'] = _values(inputs[0][0, :, 0, 0])
        arithmetic = result['stgcn']['pool_arithmetic']
        arithmetic['actual'] = float(inputs[0][0, arithmetic['channel'], 0, 0])
        arithmetic['absolute_error'] = abs(arithmetic['mean'] - arithmetic['actual'])

    def logits_hook(module, inputs, output):
        result['stgcn']['logits'] = _values(output[0])

    try:
        if predictor.lstm is not None:
            lstm = predictor.lstm
            if predictor.normalization == 'training_stats':
                mean, std = predictor.stats['lstm_mean'], predictor.stats['lstm_std']
                formula = '(pose - saved_training_mean) / saved_training_std'
            else:
                mean = poses.mean(dim=1, keepdim=True)
                std = poses.std(dim=1, correction=0, keepdim=True) + predictor.EPSILON
                formula = '(pose - per_joint_50_observation_mean) / (population_std + 1e-8)'
            result['normalization']['lstm'] = {'mean': _values(mean.reshape(17, 3)),
                                              'std': _values(std.reshape(17, 3)), 'formula': formula}
            result['lstm'] = {'layers': lstm.num_layers, 'hidden_size': lstm.hidden_size,
                              'input_size': lstm.lstm.input_size, 'output_frames': 10,
                              'rounds': rounds, 'parameter_count': sum(p.numel() for p in lstm.parameters()),
                              'decoder': {'input_features': lstm.fc.in_features, 'output_features': lstm.fc.out_features,
                                          'formula': 'forecast = Linear(final_layer_h_at_t50).reshape(10,17,3)'},
                              'recursion': 'drop earliest 10 frames; append 10 predicted frames; repeat five times'}
            register(lstm.lstm, recurrent_hook)
            register(lstm.fc, decoder_hook)
            register(lstm, forecast_hook)
        register(stgcn, classifier_input, before=True)
        register(stgcn.data_bn, normalized_hook)
        register(stgcn.fcn, pool_hook, before=True)
        register(stgcn, logits_hook)
        for i, block in enumerate(stgcn.st_gcn_networks):
            # The highest-degree joint ensures the example shows neighbor mixing.
            joint = int((stgcn.A != 0).sum(dim=(0, 1)).argmax())
            observed = {}
            details = {'index': i + 1, 'in_channels': block.gcn.conv.in_channels,
                       'out_channels': block.tcn[2].out_channels, 'stride': block.tcn[2].stride[0],
                       'temporal_kernel': list(block.tcn[2].kernel_size),
                       'residual': 'none' if i == 0 else ('projection' if isinstance(block.residual, torch.nn.Module) else 'identity'),
                       'effective_adjacency': _values(stgcn.A * stgcn.edge_importance[i]),
                       'edge_importance': _values(torch.as_tensor(stgcn.edge_importance[i])),
                       'arithmetic': {'sampling': {'batch': 0, 'output_channel': 0, 'output_joint': joint,
                                                   'selection': 'channel 0; highest-degree graph joint; middle output timestep for each operation; all terms for each selected sum'}},
                       'formula': 'ReLU(TemporalConv(BN/ReLU(SpatialGraphConv(x, A * importance))) + residual(x))'}
            result['stgcn']['blocks'].append(details)

            def block_input(module, inputs, record=details, state=observed, selected_joint=joint):
                record['input_shape'] = list(inputs[0].shape)
                state['residual'] = (float(inputs[0][0, 0, inputs[0].shape[2] // 2, selected_joint])
                                     if record['residual'] == 'identity' else 0.)

            def projection_hook(module, inputs, output, state=observed, partitions=stgcn.A.shape[0]):
                # Keep only K x V values; no complete activation tensor is retained.
                shaped = output.reshape(output.shape[0], partitions, -1, output.shape[2], output.shape[3])
                state['projected'] = shaped[0, :, 0, output.shape[2] // 2, :].detach().clone()

            def spatial_hook(module, inputs, output, record=details, state=observed, selected_joint=joint):
                record['spatial'] = _activation(output[0])
                arithmetic = _spatial_arithmetic(module, inputs, output[0], state.pop('projected'), selected_joint)
                record['arithmetic']['spatial'] = arithmetic
                record['arithmetic']['sampling']['spatial_timestep'] = arithmetic['timestep']

            def convolution_hook(module, inputs, output, record=details, selected_joint=joint):
                arithmetic = _temporal_arithmetic(module, inputs, output, selected_joint)
                record['arithmetic']['temporal'] = arithmetic
                record['arithmetic']['sampling']['temporal_timestep'] = arithmetic['timestep']

            def temporal_hook(module, inputs, output, record=details, state=observed, selected_joint=joint):
                record['temporal'] = _activation(output)
                state['temporal'] = float(output[0, 0, output.shape[2] // 2, selected_joint])

            def residual_hook(module, inputs, output, state=observed, selected_joint=joint):
                state['residual'] = float(output[0, 0, output.shape[2] // 2, selected_joint])

            def block_output(module, inputs, output, record=details, state=observed, selected_joint=joint):
                record['output_shape'] = list(output[0].shape)
                record['output'] = _activation(output[0])
                total = state['temporal'] + state['residual']
                actual = float(output[0][0, 0, output[0].shape[2] // 2, selected_joint])
                record['arithmetic']['residual'] = {
                    'mode': record['residual'], 'temporal': state['temporal'], 'residual': state['residual'],
                    'sum': total, 'relu': max(total, 0.), 'actual': actual,
                    'absolute_error': abs(max(total, 0.) - actual),
                    'formula': 'output = max(temporal_after_batch_norm + residual, 0)'}
                if record['index'] == len(stgcn.st_gcn_networks):
                    values = output[0][0, 0]
                    total = math.fsum(_values(values.reshape(-1)))
                    result['stgcn']['pool_arithmetic'] = {
                        'channel': 0, 'input_shape': list(values.shape), 'values': _values(values),
                        'count': values.numel(), 'sum': total, 'mean': total / values.numel(),
                        'formula': 'pool[channel] = sum_time,sum_joint(final_block[channel,t,v]) / (T * V)',
                        'sampling': 'batch/person 0, channel 0; every final-block time and joint position',
                        'complete_sum': True}

            register(block, block_input, before=True)
            register(block.gcn.conv, projection_hook)
            register(block.gcn, spatial_hook)
            register(block.tcn[2], convolution_hook)
            register(block.tcn, temporal_hook)
            if isinstance(block.residual, torch.nn.Module):
                register(block.residual, residual_hook)
            register(block, block_output)
        prediction = predictor.predict(history[None], reference, on_event=observe)
    finally:
        for handle in hooks:
            handle.remove()

    result['future'] = {'shape': list(prediction['future_poses'][0].shape),
                        'values': prediction['future_poses'][0].tolist()}
    result['stgcn']['probabilities'] = prediction['probabilities'][0].tolist()
    result['stgcn']['class_id'] = int(prediction['class_ids'][0])
    result['stgcn']['pool_formula'] = 'mean over all final-block time and joint positions; 256 features'
    result['stgcn']['head_formula'] = 'logits = W @ pool + b; probabilities = softmax(logits)'
    if predictor.normalization == 'legacy_pair':
        pair = torch.stack((torch.as_tensor(reference, device=predictor.device),
                            torch.as_tensor(prediction['future_poses'][0], device=predictor.device)))
        mean = pair.mean(dim=0)
        std = pair.std(dim=0, correction=1)
        std = torch.where(std == 0, torch.full_like(std, 1e-6), std).clamp_min(predictor.EPSILON)
        result['normalization']['stgcn'] = {
            'formula': '(forecast - pair_mean(reference, forecast)) / pair_sample_std; zero std -> 1e-6',
            'reference': _matrix(pair[0].reshape(50, 51)),
            'mean': _matrix(mean.reshape(50, 51)), 'std': _matrix(std.reshape(50, 51)),
            'reference_source': 'caller supplied fixed reference'}
    else:
        result['normalization']['stgcn'] = {
            'formula': '(forecast - saved_training_mean) / saved_training_std',
            'reference_source': 'saved training statistics; no fixed person reference',
            'mean': _matrix(predictor.stats['stgcn_mean'][0, :, :, :, 0].permute(1, 2, 0).reshape(50, 51)),
            'std': _matrix(predictor.stats['stgcn_std'][0, :, :, :, 0].permute(1, 2, 0).reshape(50, 51))}
    return result
