"""Decode unchanged SkeletonLSTM outputs as last-observation-relative changes.

Weights trained with absolute-coordinate targets are NOT compatible with this
representation. The fixed standard deviations must come from training data.
"""
import torch

def forecast_relative(model, history, std, blocks=5):
    if history.ndim!=4 or tuple(history.shape[1:])!=(50,17,3):
        raise ValueError('history must have shape [B,50,17,3]')
    if tuple(std.shape)!=(17,3) or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('std must be positive finite training statistics [17,3]')
    if not isinstance(blocks,int) or blocks<1:
        raise ValueError('blocks must be a positive integer')
    series=history;result=[]
    for _ in range(blocks):
        anchor=series[:,-1:]
        delta=model(((series-anchor)/std).reshape(len(history),50,51))
        future=delta*std+anchor
        result.append(future)
        series=torch.cat((series[:,10:],future),dim=1)
    return torch.cat(result,dim=1)
