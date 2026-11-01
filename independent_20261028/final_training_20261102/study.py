"""Frozen relative LSTM/features; controlled ST-GCN class-weight comparison."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,random,hashlib,shutil,runpy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(r'F:\Artificial_Intelligence\srtp');BASE=Path(r'D:\SRTP_relative_pipeline_20261101');OLD=Path(r'D:\SRTP_lstm_retrain_20261101');O=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(BASE)]
from st_gcn import Model
from relative_runtime import RelativeBehaviorPredictor
from independent_20261028.evaluate_internal_metrics import summarize
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
    p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf8')
def load(n):return json.loads((O/n).read_text('utf8'))
def seed():random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
def log(**kw):save('progress.json',kw);print(json.dumps(kw),flush=True)
def signature(m):
    h=hashlib.sha256()
    for v in m.state_dict().values():h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
data=torch.load(OLD/'train_val.pt',weights_only=False);features=torch.load(BASE/'features.pt',weights_only=False)
counts=torch.bincount(data['train']['y'],minlength=3).float();N=counts.sum()
weights={'inverse':N/(3*counts),'sqrt_inverse':counts.rsqrt(),'effective_0999':.001/(1-torch.pow(torch.tensor(.999),counts))}
for name in ['sqrt_inverse','effective_0999']:weights[name]*=N/(weights[name]*counts).sum()
frozen={str(p):sha(p) for p in [BASE/'lstm/best.pt',BASE/'features.pt',BASE/'forecast/normalization.pt',OLD/'train_val.pt',ROOT/'st_gcn.py']}
production={str(p):sha(p) for p in (ROOT/'weights/final').glob('*.pt')}
protocol=dict(seed=2026,epochs=125,batch_size=128,optimizer='SGD',lr=.001,momentum=0,weight_decay=0,
    class_counts=counts.tolist(),weights={k:v.tolist() for k,v in weights.items()},effective_beta=.999,
    unchanged='LSTM, network architecture, input features, video split, initialization, batch order, training schedule, normalization',
    checkpoint_selection='maximum validation MacroF1, tie minimum unweighted validation CE',
    method_selection='same criterion across all three selected checkpoints, frozen before test evaluation',
    baseline='reuse completed inverse-weight trial from previous round',
    test_policy='evaluate all three once after method selection; previously used test videos; no test-driven retraining',frozen_sha256=frozen)
if (O/'protocol.json').exists():assert load('protocol.json')==protocol
save('protocol.json',protocol);save('protected_weights.json',production)
dest=O/'inverse';dest.mkdir(exist_ok=True)
for name in ['best.pt','best.json','history.json','complete.json']:shutil.copy2(BASE/'forecast'/name,dest/name)
init_hash=json.loads((BASE/'forecast/complete.json').read_text())['initial_state_sha256']
for name in ['sqrt_inverse','effective_0999']:
    dest=O/name;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():continue
    seed();model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).cuda();assert signature(model)==init_hash
    optimizer=torch.optim.SGD(model.parameters(),lr=.001);w=weights[name].cuda()
    train=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features['train'],data['train']['y']),batch_size=128,shuffle=True)
    val=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features['val'],data['val']['y']),batch_size=128)
    best=(-1.,-float('inf'));history=[];start=time.perf_counter()
    for epoch in range(1,126):
        model.train();total=0.;denom=0.
        for x,y in train:
            x,y=x.cuda(),y.cuda();optimizer.zero_grad();out=model(x)
            losses=torch.nn.functional.cross_entropy(out,y,weight=w,reduction='none');loss=losses.sum()/w[y].sum()
            assert torch.isfinite(loss);loss.backward();optimizer.step();total+=losses.sum().item();denom+=w[y].sum().item()
        model.eval();pred=[];vloss=0.
        with torch.inference_mode():
            for x,y in val:
                out=model(x.cuda());pred.extend(out.argmax(1).cpu().tolist());vloss+=torch.nn.functional.cross_entropy(out,y.cuda(),reduction='sum').item()
        cm=np.bincount(3*data['val']['y'].numpy()+np.array(pred),minlength=9).reshape(3,3);met=summarize(cm)
        row=dict(epoch=epoch,train_weighted_loss=total/denom,val_loss=vloss/len(pred),accuracy=met['accuracy'],macro_f1=met['macro_f1']);history.append(row)
        score=(met['macro_f1'],-row['val_loss'])
        if score>best:
            best=score;torch.save(model.state_dict(),dest/'best.pt');save(name+'/best.json',dict(**row,metrics=met))
        if epoch%5==0 or epoch==1:
            save(name+'/history.json',history);log(stage='train',method=name,seconds=time.perf_counter()-start,**row)
    save(name+'/history.json',history);save(name+'/complete.json',dict(epochs=125,seconds=time.perf_counter()-start,initial_state_sha256=init_hash,weights=weights[name].tolist()))
    del model,optimizer;torch.cuda.empty_cache()

selected=max(weights,key=lambda n:(load(n+'/best.json')['macro_f1'],-load(n+'/best.json')['val_loss']))
selection=dict(method=selected,criterion=protocol['method_selection'],validation={n:load(n+'/best.json') for n in weights})
save('selection_before_test.json',selection);log(stage='selected_before_test',method=selected)
if not (O/'comparison.json').exists():
    ns=runpy.run_path(str(ROOT/'independent_20261028/retrain_paired_20261101.py'),run_name='class_weight_helpers')
    test=ns['causal_classification'](ns['SPLITS']['test']);truth=test['y'].numpy()
    prior=json.loads((BASE/'forecast/test_predictions.json').read_text())
    assert [list(k) for k in test['keys']]==[r['key'] for r in prior] and truth.tolist()==[r['truth'] for r in prior]
    results={};predictions={};checks={}
    for name in weights:
        model=RelativeBehaviorPredictor(O/name/'best.pt',BASE/'lstm/best.pt','cuda:0','training_stats',BASE/'forecast/normalization.pt')
        with torch.inference_mode():expected=model.stgcn(features['val'].cuda()).softmax(1).cpu().numpy()
        actual=np.concatenate([model.predict(x.numpy())['probabilities'] for x in data['val']['x'].split(64)])
        err=float(np.abs(expected-actual).max());assert err<1e-4
        cm=np.bincount(3*data['val']['y'].numpy()+actual.argmax(1),minlength=9).reshape(3,3);assert cm.tolist()==load(name+'/best.json')['metrics']['confusion_matrix']
        probs=np.concatenate([model.predict(x.numpy())['probabilities'] for x in test['x'].split(64)]);pred=probs.argmax(1)
        cm=np.bincount(3*truth+pred,minlength=9).reshape(3,3);results[name]=summarize(cm);predictions[name]=pred
        save(name+'/test_predictions.json',[dict(key=k,truth=int(y),prediction=int(p),probabilities=q.tolist()) for k,y,p,q in zip(test['keys'],truth,pred,probs)])
        checks[name]=dict(runtime_feature_probability_max_error=err,validation_checkpoint_verified=True)
        del model;torch.cuda.empty_cache()
    assert np.array_equal(predictions['inverse'],np.array([r['prediction'] for r in prior]))
    uncertainty={n:ns['paired_bootstrap'](test['keys'],truth,predictions['inverse'],predictions[n]) for n in ['sqrt_inverse','effective_0999']}
    save('comparison.json',dict(selected_by_validation=selected,samples=len(truth),test_videos=len(ns['SPLITS']['test']),eligible=test['eligible'],coverage=len(truth)/test['eligible'],results=results,uncertainty_vs_inverse=uncertainty))
    save('runtime_checks.json',checks)
assert all(sha(p)==h for p,h in frozen.items()) and all(sha(p)==h for p,h in production.items())
save('complete.json',dict(frozen_inputs_unchanged=True,production_weights_unchanged=True,method_selected_before_test=True))
log(stage='complete',result=load('comparison.json'))
