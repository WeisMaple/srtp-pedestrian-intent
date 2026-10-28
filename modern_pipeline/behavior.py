"""Batched PyTorch behavior prediction for COCO 17-joint pixel coordinates.

This module intentionally avoids importing the legacy video/DeepSORT modules.
The default reproduces the recovered two-sample normalization recipe.
"""
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch
from st_gcn import Model, SkeletonLSTM

@torch.inference_mode()
def forecast_with_statistics(lstm, poses, mean, std):
    """Shared training-feature/runtime forecast, using saved training statistics."""
    series=(poses-mean)/std
    for _ in range(5):
        with torch.backends.cudnn.flags(benchmark=False,deterministic=True,allow_tf32=False):
            forecast=lstm(series.reshape(len(poses),50,51))
        series=torch.cat((series[:,10:],forecast),dim=1)
    return series*std+mean


class BehaviorPredictor:
    """Predict intent from B x 50 x 17 x 3 histories.

    ``legacy_pair`` needs a fixed 50-frame reference. ``training_stats``
    instead needs a small statistics .pt file, without full dataset tensors.
    Passing no LSTM selects classification of the observed history.
    """
    WINDOW = 50
    JOINTS = 17
    CHANNELS = 3
    EPSILON = 1e-8

    def __init__(self, stgcn_path: Union[str, Path],
                 lstm_path: Optional[Union[str, Path]] = None,
                 device: str = 'cuda:0', normalization: str = 'legacy_pair',
                 stats_path: Optional[Union[str, Path]] = None):
        if normalization not in {'legacy_pair', 'training_stats'}:
            raise ValueError('normalization must be legacy_pair or training_stats')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use device="cpu"')
        self.normalization = normalization
        self.stgcn = Model(3, 3, {'layout': 'yolov8', 'strategy': 'spatial'}, True)
        self.stgcn.load_state_dict(self._load_weights(stgcn_path))
        self.stgcn.to(self.device).eval()
        self.lstm = None
        if lstm_path is not None:
            self.lstm = SkeletonLSTM(input_size=51, hidden_size=256, num_layers=10,
                                     output_size=51, output_frames=10)
            self.lstm.load_state_dict(self._load_weights(lstm_path))
            self.lstm.to(self.device).eval()
        self.stats = {}
        if normalization == 'training_stats':
            if stats_path is None:
                raise ValueError('training_stats normalization requires stats_path')
            stats_file = Path(stats_path)
            if not stats_file.is_file():
                raise FileNotFoundError(stats_file)
            loaded = torch.load(stats_file, map_location='cpu', weights_only=True)
            if not isinstance(loaded, dict):
                raise ValueError('Statistics file must contain a dictionary')
            shapes = {'stgcn_mean': (1, 3, 50, 17, 1),
                      'stgcn_std': (1, 3, 50, 17, 1),
                      'lstm_mean': (17, 3), 'lstm_std': (17, 3)}
            required = ['stgcn_mean', 'stgcn_std']
            if self.lstm is not None:
                required += ['lstm_mean', 'lstm_std']
            for key in required:
                if key not in loaded:
                    raise ValueError('Missing statistics key: ' + key)
                tensor = torch.as_tensor(loaded[key], dtype=torch.float32)
                if tuple(tensor.shape) != shapes[key]:
                    raise ValueError(f'{key} must have shape {shapes[key]}, got {tuple(tensor.shape)}')
                if not torch.isfinite(tensor).all():
                    raise ValueError(key + ' must contain finite values')
                if key.endswith('_std'):
                    if (tensor < 0).any():
                        raise ValueError(key + ' cannot contain negative standard deviations')
                    tensor = tensor.clamp_min(self.EPSILON)
                self.stats[key] = tensor.to(self.device)

    @staticmethod
    def _load_weights(path):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        state = torch.load(source, map_location='cpu', weights_only=True)
        if not isinstance(state, dict):
            raise ValueError('Model weights must be a state_dict: ' + str(source))
        return state

    @staticmethod
    def _validated_array(value, shape, name):
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f'{name} must have shape {shape}, got {array.shape}')
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(name + ' must contain numeric values')
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(name + ' must contain finite values')
        return array

    @torch.inference_mode()
    def predict(self, histories: np.ndarray,
                reference: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Return class_ids[B], probabilities[B,3], raw future_poses and horizon.

        Forecast mode recursively predicts five blocks of 10 future frames;
        classification-only mode returns the observed poses and horizon 0.
        """
        array = np.asarray(histories)
        if array.ndim != 4 or tuple(array.shape[1:]) != (50, 17, 3):
            raise ValueError('histories must have shape [B,50,17,3]')
        array = self._validated_array(array, array.shape, 'histories')
        ref = None
        if reference is not None:
            ref_array = self._validated_array(reference, (50, 17, 3), 'reference')
            ref = torch.from_numpy(ref_array).to(self.device)
        if self.normalization == 'legacy_pair' and ref is None:
            raise ValueError('legacy_pair normalization requires a fixed reference [50,17,3]')
        batch = len(array)
        if batch == 0:
            return {'class_ids': np.empty(0, dtype=np.int64),
                    'probabilities': np.empty((0, 3), dtype=np.float32),
                    'future_poses': array.copy(),
                    'horizon_frames': 50 if self.lstm is not None else 0}
        poses = torch.from_numpy(array).to(self.device)
        if self.lstm is not None:
            if self.normalization == 'training_stats':
                mean = self.stats['lstm_mean']
                std = self.stats['lstm_std']
            else:
                mean = poses.mean(dim=1, keepdim=True)
                std = poses.std(dim=1, correction=0, keepdim=True) + self.EPSILON
            poses = forecast_with_statistics(self.lstm,poses,mean,std)
        if not torch.isfinite(poses).all():
            raise RuntimeError('Behavior forecast contains non-finite values')
        if self.normalization == 'training_stats':
            inputs = poses.permute(0, 3, 1, 2).unsqueeze(-1)
            inputs = (inputs - self.stats['stgcn_mean']) / self.stats['stgcn_std']
        else:
            # Each target keeps its own pair: never fit statistics across people.
            pairs = torch.stack((ref.expand(batch, -1, -1, -1), poses), dim=1)
            mean = pairs.mean(dim=1, keepdim=True)
            std = pairs.std(dim=1, correction=1, keepdim=True)
            std = torch.where(std == 0, torch.full_like(std, 1e-6), std)
            std = std.clamp_min(self.EPSILON)
            normalized = ((pairs - mean) / std)[:, 1]
            inputs = normalized.permute(0, 3, 1, 2).unsqueeze(-1)
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            probabilities = self.stgcn(inputs).softmax(dim=1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError('Behavior classifier contains non-finite values')
        return {'class_ids': probabilities.argmax(dim=1).cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'future_poses': poses.cpu().numpy(),
                'horizon_frames': 50 if self.lstm is not None else 0}


