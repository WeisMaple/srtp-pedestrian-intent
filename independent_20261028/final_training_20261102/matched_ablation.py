import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,runpy,json,torch,numpy as np
from pathlib import Path
R=Path(r'F:\Artificial_Intelligence\srtp');sys.path.insert(0,str(R));O=Path(r'D:\SRTP_release_20261102\work\matched_ablation');O.mkdir(exist_ok=True)
ns=runpy.run_path(str(R/'independent_20261028/retrain_paired_20261101.py'),run_name='helpers')
g=ns['train_classifier'].__globals__;g['OUT']=O
# Preserve the original controlled training loop, changing only to sqrt weights.
import inspect
src=inspect.getsource(ns['train_classifier']).replace("weights=(counts.sum()/(3*counts)).to(DEVICE)","weights=counts.rsqrt().to(DEVICE)")
exec(src,g)
d=torch.load(r'D:\SRTP_lstm_retrain_20261101\train_val.pt',weights_only=False);f=torch.load(r'D:\SRTP_lstm_retrain_20261101\features.pt',weights_only=False)['observed']
g['train_classifier']('observed',d,f)
from modern_pipeline.behavior import BehaviorPredictor
from independent_20261028.evaluate_internal_metrics import summarize
m=BehaviorPredictor(O/'observed/best.pt',device='cuda:0',normalization='training_stats',stats_path=r'D:\SRTP_lstm_retrain_20261101\observed\normalization.pt')
test=ns['causal_classification'](ns['SPLITS']['test']);p=np.concatenate([m.predict(x.numpy())['probabilities'] for x in test['x'].split(64)]).argmax(1);y=test['y'].numpy()
cm=np.bincount(3*y+p,minlength=9).reshape(3,3)
final=json.loads(Path(r'D:\SRTP_class_weights_20261102\sqrt_inverse\test_predictions.json').read_text())
assert [list(k) for k in test['keys']]==[r['key'] for r in final]
res=dict(observed=summarize(cm),forecast=json.loads(Path(r'D:\SRTP_class_weights_20261102\comparison.json').read_text())['results']['sqrt_inverse'],uncertainty=ns['paired_bootstrap'](test['keys'],y,p,np.array([r['prediction'] for r in final])),scope='same sqrt-inverse loss, matched sample split, separate branch training; selected by val MacroF1')
(O/'comparison.json').write_text(json.dumps(res,indent=2),'utf8');print(json.dumps(res))
