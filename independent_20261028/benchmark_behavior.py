import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,sys,time,json,numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modern_pipeline.behavior import BehaviorPredictor
OUT=pathlib.Path(r'D:\SRTP_independent_20261028');torch.set_num_threads(2)
d=torch.load(OUT/'modern/data.pt',weights_only=False);x=d['test']['x'][:20].numpy();ref=x[0];results={};predictions={}
for device in ['cpu','cuda:0']:
    model=BehaviorPredictor(OUT/'modern/stgcn/best.pt',OUT/'modern/lstm/best.pt',device,'legacy_pair')
    for i in range(3):model.predict(x[:1],ref)
    if device.startswith('cuda'):torch.cuda.synchronize()
    begin=time.perf_counter();outputs=[]
    for history in x:outputs.append(model.predict(history[None],ref)['probabilities'][0])
    if device.startswith('cuda'):torch.cuda.synchronize()
    seconds=time.perf_counter()-begin;results[device]=dict(histories=len(x),seconds=seconds,histories_per_second=len(x)/seconds,batch=1,warmup=3)
    predictions[device]=np.array(outputs);del model
results['max_probability_difference']=float(np.abs(predictions['cpu']-predictions['cuda:0']).max());results['class_agreement']=float((predictions['cpu'].argmax(1)==predictions['cuda:0'].argmax(1)).mean())
results['speed_ratio']=results['cuda:0']['histories_per_second']/results['cpu']['histories_per_second'];results['note']='Identical weights, 20 identical 50-pose histories, batch=1, five LSTM recursions and ST-GCN; isolates CPU versus CUDA behavior runtime, not YOLO speed.'
(OUT/'speed/behavior_equal_work.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
