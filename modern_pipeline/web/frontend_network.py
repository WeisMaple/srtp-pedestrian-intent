"""Bounded, actual YOLO26 activations replayed from a selected source frame.

The production Ultralytics preprocessing, model and postprocessing are used.
Conv/BN fusion is disabled in this separate replay model so the trained batch
normalization parameters remain inspectable. No tracker is created or replayed.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def read_source_frame(path, frame):
    """Decode one zero-based original frame, without retaining the video."""
    if frame < 0:
        raise ValueError('视频帧编号必须非负。')
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError('无法打开任务的原始视频。')
        if frame >= int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError('所选帧超出原始视频范围。')
        if frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise ValueError('无法定位所选原始视频帧。')
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError('无法读取所选原始视频帧。')
        return image
    finally:
        capture.release()


def _values(value):
    return value.detach().float().cpu().numpy().round(6).tolist()


def feature_map(tensor, channels=6, side=16):
    """Keep true native samples, never render an invented activation texture."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or channels < 1 or side < 1:
        raise ValueError('Feature inspection requires one [1,C,H,W] tensor')
    value = tensor.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError('模型激活包含非有限数值。')
    height, width = value.shape[-2:]
    rows = torch.linspace(0, height - 1, min(side, height), device=value.device).round().long()
    columns = torch.linspace(0, width - 1, min(side, width), device=value.device).round().long()
    selected = value[0, :channels].index_select(1, rows).index_select(2, columns)
    return {'shape': list(value.shape), 'channels': list(range(selected.shape[0])),
            'values': _values(selected), 'sample_y': rows.tolist(), 'sample_x': columns.tolist(),
            'sampling': 'batch 0; first channels; exact native positions on a uniform grid; values rounded to 6 decimals',
            'min': float(value.min()), 'max': float(value.max()),
            'mean': float(value.mean()), 'std': float(value.std(correction=0))}


def convolution_example(block, tensor, output, position=None, channel=0):
    """Expose every multiply for one RGB convolution and verify actual SiLU."""
    conv, bn = block.conv, block.bn
    if conv.groups != 1 or conv.in_channels != 3 or not 0 <= channel < conv.out_channels:
        raise ValueError('The inspected stem must be a standard RGB convolution')
    y, x = position or (output.shape[-2] // 2, output.shape[-1] // 2)
    if not (0 <= y < output.shape[-2] and 0 <= x < output.shape[-1]):
        raise ValueError('Convolution position is outside its output')
    kh, kw = conv.kernel_size
    sy, sx = conv.stride
    py, px = conv.padding
    dy, dx = conv.dilation
    padded = F.pad(tensor, (px, px, py, py))
    patch = padded[0, :, y * sy:y * sy + dy * (kh - 1) + 1:dy,
                   x * sx:x * sx + dx * (kw - 1) + 1:dx]
    weights = conv.weight[channel]
    products = patch * weights
    product_sum = products.sum()
    bias = conv.bias[channel] if conv.bias is not None else product_sum.new_zeros(())
    convolution = product_sum + bias
    normalized = (convolution - bn.running_mean[channel]) / torch.sqrt(bn.running_var[channel] + bn.eps)
    normalized = normalized * bn.weight[channel] + bn.bias[channel]
    activation = F.silu(normalized)
    actual = output[0, channel, y, x]
    error = float((activation - actual).abs())
    if error > 1e-4:
        raise RuntimeError('卷积重建与实际 YOLO 激活不一致。')
    return {'output_channel': channel, 'position': [y, x], 'input_origin': [y * sy - py, x * sx - px],
            'patch': _values(patch), 'weights': _values(weights), 'products': _values(products),
            'sum': float(product_sum), 'bias': float(bias), 'conv_output': float(convolution),
            'bn': {'gamma': float(bn.weight[channel]), 'beta': float(bn.bias[channel]),
                   'running_mean': float(bn.running_mean[channel]), 'running_var': float(bn.running_var[channel]),
                   'epsilon': bn.eps, 'output': float(normalized)},
            'activation_output': float(activation), 'actual_output': float(actual),
            'validation_max_error': error}


def _image_url(image_bgr):
    height, width = image_bgr.shape[:2]
    if max(height, width) > 480:
        ratio = 480 / max(height, width)
        image_bgr = cv2.resize(image_bgr, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError('无法编码输入图像预览。')
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')


class FrontendInspector:
    """One retained CPU model; callers serialize it with video/model inference."""

    def __init__(self, weights):
        from modern_pipeline.bootstrap import configure
        configure()
        from ultralytics import YOLO
        from ultralytics.models.yolo.pose.predict import PosePredictor
        from ultralytics.nn.autobackend import AutoBackend

        weights = Path(weights)
        if not weights.is_file():
            raise OSError('缺少已经校验的 YOLO 姿态权重。')
        torch.set_num_threads(2)
        loaded = YOLO(str(weights), task='pose')
        self.predictor = PosePredictor(overrides={
            'task': 'pose', 'imgsz': 640, 'conf': 0.1, 'classes': [0], 'max_det': 100,
            'rect': True, 'device': 'cpu', 'quantize': 32, 'verbose': False, 'save': False})
        # The real AutoBackend is used, but retaining BN makes its arithmetic
        # visible. This is a separate replay model, never the tracker's model.
        self.predictor.model = AutoBackend(model=loaded.model, device=torch.device('cpu'),
                                           fp16=False, fuse=False, verbose=False,
                                           end2end=self.predictor.args.nms is False)
        self.predictor.device = torch.device('cpu')
        self.predictor.imgsz = [640, 640]
        self.model = self.predictor.model.model.eval()
        self.name = weights.name
        if not hasattr(self.model.model[0], 'bn') or type(self.model.model[-1]).__name__ != 'Pose26':
            raise ValueError('此计算视图需要保留 BatchNorm 的 YOLO26 Pose 模型。')

    @torch.inference_mode()
    def replay(self, image_bgr):
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ValueError('YOLO replay requires a uint8 BGR source frame')
        torch.set_num_threads(2)
        predictor = self.predictor
        predictor.batch = (['source-frame'], [image_bgr], [''])
        tensor = predictor.preprocess([image_bgr])
        rgb = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        result = {
            'model': {'name': self.name, 'parameter_count': sum(p.numel() for p in self.model.parameters()),
                      'module_count': len(self.model.model), 'fusion': False},
            'input': {'original_shape': list(image_bgr.shape), 'tensor_shape': list(tensor.shape),
                      'image_data_url': _image_url(image_bgr), 'tensor_image_data_url': _image_url(rgb[:, :, ::-1]),
                      'color_order': 'RGB', 'normalization': 'Ultralytics LetterBox → BGR to RGB → BCHW float32 / 255',
                      'image_sampling': 'JPEG preview, longest edge at most 480 px; tensor values are not JPEG-derived'},
            'layers': [], 'heads': [], 'first_conv': None,
            'postprocess': {'end2end': bool(self.model.model[-1].end2end), 'threshold': 0.1,
                            'max_detections': 100, 'coordinate_space': 'original source-frame pixels',
                            'method': 'Ultralytics PosePredictor.postprocess; production head selection, confidence filtering, NMS when enabled, source-coordinate scaling'},
        }
        handles = []

        def stem_hook(module, inputs, output):
            y, center_x = output.shape[-2] // 2, output.shape[-1] // 2
            start = max(0, min(center_x - 4, output.shape[-1] - 9))
            result['first_conv'] = {
                'kernel_shape': list(module.conv.weight.shape), 'stride': list(module.conv.stride),
                'padding': list(module.conv.padding), 'kernels': _values(module.conv.weight[:6]),
                'input_channels': ['R', 'G', 'B'],
                'example': convolution_example(module, inputs[0], output, (y, center_x)),
                'steps': [convolution_example(module, inputs[0], output, (y, x))
                          for x in range(start, min(start + 9, output.shape[-1]))],
                'formulas': {'convolution': 'z = sum over RGB and kernel positions (patch * weight) + bias',
                             'batch_norm': 'b = gamma * (z - running_mean) / sqrt(running_variance + epsilon) + beta',
                             'silu': 'a = b * sigmoid(b)'},
                'validation': 'Each reconstructed scalar is checked against the actual hooked stem output'}

        try:
            handles.append(self.model.model[0].register_forward_hook(stem_hook))
            for index, module in enumerate(self.model.model[:-1]):
                def layer_hook(block, inputs, output, number=index):
                    result['layers'].append({'index': number, 'type': type(block).__name__, 'from': block.f,
                                             'section': 'backbone' if number <= 10 else 'neck', **feature_map(output)})
                handles.append(module.register_forward_hook(layer_hook))
            head = self.model.model[-1]
            prefix = 'one2one_' if head.end2end else ''
            for branch, attribute in [('box', 'cv2'), ('class', 'cv3'), ('keypoints', 'cv4_kpts')]:
                for scale, module in enumerate(getattr(head, prefix + attribute)):
                    def head_hook(block, inputs, output, kind=branch, level=scale, name=prefix + attribute):
                        result['heads'].append({'name': name + '.' + str(level), 'branch': kind,
                                                'scale': level, 'stride': int(head.stride[level]), **feature_map(output)})
                    handles.append(module.register_forward_hook(head_hook))
            raw = predictor.model(tensor)
            predictions = predictor.postprocess(raw, tensor, [image_bgr])[0]
            result['detections'] = [
                {'bbox': _values(box), 'confidence': float(score), 'class_id': int(cls), 'keypoints': _values(pose)}
                for box, score, cls, pose in zip(predictions.boxes.xyxy, predictions.boxes.conf,
                                                 predictions.boxes.cls, predictions.keypoints.data)]
        finally:
            for handle in handles:
                handle.remove()
            predictor.batch = None
        return result
