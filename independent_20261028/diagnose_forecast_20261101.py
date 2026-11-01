"""Evaluate all five recursive forecast blocks against later test observations."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,runpy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from st_gcn import SkeletonLSTM
from modern_pipeline.behavior import forecast_with_statistics
# Load the same deterministic sequence builder without invoking its main routine.
ns=runpy.run_path(str(ROOT/'independent_20261028/retrain_paired_20261101.py'),run_name='paired_helpers')
O=ns['OUT'];assert (O/'complete.json').exists()
pairs=ns['pose_pairs'](ns['SPLITS']['test'],horizon=50)
st=torch.load(O/'lstm/stats.pt',weights_only=True)
model=SkeletonLSTM().cuda().eval();model.load_state_dict(torch.load(O/'lstm/best.pt',map_location='cuda:0',weights_only=True))
errors=torch.zeros(5,dtype=torch.float64);persistence=torch.zeros(5,dtype=torch.float64);n=0
outside=0;confidence_count=0
with torch.inference_mode():
    for x,y in zip(pairs['lx'].split(64),pairs['ly'].split(64)):
        pred=forecast_with_statistics(model,x.cuda(),st['lstm_mean'].cuda(),st['lstm_std'].cuda()).cpu()
        truth=(y-st['lstm_mean'])/st['lstm_std'];norm=(pred-st['lstm_mean'])/st['lstm_std']
        base=((x[:,-1:]-st['lstm_mean'])/st['lstm_std']).expand_as(truth)
        for k in range(5):
            sl=slice(k*10,(k+1)*10)
            errors[k]+=((norm[:,sl,:,:2]-truth[:,sl,:,:2])**2).mean().item()*len(x)
            persistence[k]+=((base[:,sl,:,:2]-truth[:,sl,:,:2])**2).mean().item()*len(x)
        outside+=((pred[...,2]<0)|(pred[...,2]>1)).sum().item();confidence_count+=pred[...,2].numel();n+=len(x)
result=dict(samples=n,target='later YOLO observations on the same tracker segment; no fitting',metric='XY squared error standardized by training-only LSTM statistics',
    lstm_mse_by_10step_block=(errors/n).tolist(),persistence_mse_by_10step_block=(persistence/n).tolist(),
    forecast_confidence_outside_unit_interval_fraction=outside/confidence_count)
(O/'forecast_horizon_diagnostic.json').write_text(json.dumps(result,indent=2),'utf8');print(json.dumps(result))
