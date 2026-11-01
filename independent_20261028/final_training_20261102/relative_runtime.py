"""Experimental relative-pose bundle; original network definitions unchanged."""
import numpy as np
import torch
from modern_pipeline.behavior import BehaviorPredictor
from forecast_relative import forecast_relative

class RelativeBehaviorPredictor(BehaviorPredictor):
    @torch.inference_mode()
    def predict(self,histories,reference=None,on_event=None):
        if self.lstm is None:
            return super().predict(histories,reference,on_event)
        if self.normalization!='training_stats':
            raise ValueError('Relative pose weights require training_stats')
        a=np.asarray(histories)
        if a.ndim!=4 or tuple(a.shape[1:])!=(50,17,3):
            raise ValueError('history must be [B,50,17,3]')
        a=self._validated_array(a,a.shape,'histories')
        if len(a)==0:
            return dict(class_ids=np.empty(0,dtype=np.int64),probabilities=np.empty((0,3),dtype=np.float32),future_poses=a.copy(),horizon_frames=50)
        x=torch.from_numpy(a).to(self.device)
        poses=forecast_relative(self.lstm,x,self.stats['lstm_std'])
        inputs=(poses.permute(0,3,1,2).unsqueeze(-1)-self.stats['stgcn_mean'])/self.stats['stgcn_std']
        with torch.backends.cudnn.flags(benchmark=False,deterministic=True,allow_tf32=False):
            probs=self.stgcn(inputs).softmax(1)
        if not torch.isfinite(probs).all() or not torch.isfinite(poses).all():
            raise RuntimeError('Nonfinite prediction')
        return dict(class_ids=probs.argmax(1).cpu().numpy(),probabilities=probs.cpu().numpy(),future_poses=poses.cpu().numpy(),horizon_frames=50)
