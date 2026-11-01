"""Train cross-fitted relative LSTM -> unchanged ST-GCN, then paired evaluation."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,random,hashlib,runpy,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(r'F:\Artificial_Intelligence\srtp');sys.path.insert(0,str(ROOT))
O=Path(__file__).resolve().parent;OLD=Path(r'D:\SRTP_lstm_retrain_20261101');DIAG=Path(r'D:\SRTP_lstm_diagnosis_20261101')
from st_gcn import Model,SkeletonLSTM
from modern_pipeline.behavior import BehaviorPredictor
from forecast_relative import forecast_relative
from relative_runtime import RelativeBehaviorPredictor
from independent_20261028.evaluate_internal_metrics import summarize
ns=runpy.run_path(str(ROOT/'independent_20261028/retrain_paired_20261101.py'),run_name='paired_helpers')
torch.set_num_threads(2)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
def save(n,d):
    p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False),'utf8')
def load(n):return json.loads((O/n).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed():random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
def log(**kw):save('progress.json',kw);print(json.dumps(kw),flush=True)
def valid(y):return (y[...,2]>=.2)&(y[...,0]>0)&(y[...,1]>0)
data=torch.load(OLD/'train_val.pt',weights_only=False)
splits=json.loads((OLD/'split.json').read_text());folds=json.loads((OLD/'forecast_folds.json').read_text())
protected={p.name:sha(p) for p in (ROOT/'weights/final').glob('*.pt')}
protocol=dict(seed=2026,architecture_sha256=sha(ROOT/'st_gcn.py'),test_policy='evaluate only after validation-selected classifier; previously used project test split',
    lstm='reuse validation-selected relative LSTM: 13 short epochs + checkpoint13 of recursive fine tuning',
    folds='same 3 video folds as previous paired study; fold-specific training-only std; fixed 13+13 epochs',
    training=dict(short_epochs=13,short_lr=.001,short_batch=128,rollout_epochs=13,rollout_lr=.0001,rollout_batch=64,clip=1,
        loss='valid-target XY MSE + 0.1 confidence MSE',weight_decay=0),
    classifier='unchanged ST-GCN, same init seed2026, SGD lr.001 batch128 epochs125, inverse frequency weighted CE, best validation MacroF1',
    observed_control='reuse previous observed checkpoint; unchanged data, initialization and training configuration',
    default_weights_changed=False)
if (O/'protocol.json').exists():assert load('protocol.json')==protocol
save('protocol.json',protocol);save('protected_weights.json',protected);save('split.json',splits);save('forecast_folds.json',folds)
assert all(not(set(splits[a])&set(splits[b])) for a,b in [('train','val'),('train','test'),('val','test')])
main=O/'lstm';main.mkdir(exist_ok=True)
shutil.copy2(DIAG/'anchored_rollout/best.pt',main/'best.pt')
shutil.copy2(OLD/'lstm/stats.pt',main/'stats.pt')
shutil.copy2(DIAG/'anchored_rollout/best.json',main/'best.json')

def fit_fold(i,short,long):
    dest=O/f'fold{i}';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return dest
    seed();std=short['lx'].std((0,1),correction=0).clamp_min(1e-6)
    torch.save(dict(lstm_std=std),dest/'stats.pt');std=std.cuda()
    model=SkeletonLSTM().cuda();history=[];begin=time.perf_counter()
    for stage,d,lr,bs,blocks in [('short',short,.001,128,1),('rollout',long,.0001,64,5)]:
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(d['lx'],d['ly']),batch_size=bs,shuffle=True)
        model.train()
        for epoch in range(1,14):
            total=0
            for x,y in loader:
                x,y=x.cuda(),y.cuda();opt.zero_grad()
                p=forecast_relative(model,x,std,blocks=blocks);err=((p-y)/std).square()
                loss=err[...,:2][valid(y)].mean()+.1*err[...,2].mean()
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(x)
            row=dict(stage=stage,epoch=epoch,loss=total/len(d['lx']));history.append(row)
            if epoch in [1,5,10,13]:log(fold=i,seconds=time.perf_counter()-begin,**row)
        torch.save(model.state_dict(),dest/(stage+'.pt'))
    torch.save(model.state_dict(),dest/'best.pt');save(f'fold{i}/history.json',history)
    fit=sorted(set(k[0] for k in short['lkeys'])|set(k[0] for k in long['lkeys']))
    assert not(set(fit)&set(folds[i]))
    save(f'fold{i}/complete.json',dict(fit_videos=fit,short_samples=len(short['lx']),long_samples=len(long['lx']),epochs=[13,13],seconds=time.perf_counter()-begin))
    del model,opt;torch.cuda.empty_cache();return dest

@torch.inference_mode()
def forecast(path,x):
    model=SkeletonLSTM().cuda().eval();model.load_state_dict(torch.load(path/'best.pt',weights_only=True))
    std=torch.load(path/'stats.pt',weights_only=True)['lstm_std'].cuda()
    out=torch.cat([forecast_relative(model,b.cuda(),std).cpu() for b in x.split(64)])
    assert out.shape==x.shape and torch.isfinite(out).all()
    del model;torch.cuda.empty_cache();return out

if not (O/'features.pt').exists():
    long=ns['pose_pairs'](splits['train'],horizon=50);tr=data['train'];pred=torch.empty_like(tr['x']);seen=torch.zeros(len(pred),dtype=torch.int)
    for i,held in enumerate(folds):
        held=set(held);a=[j for j,k in enumerate(tr['lkeys']) if k[0] not in held];b=[j for j,k in enumerate(long['lkeys']) if k[0] not in held];xi=[j for j,k in enumerate(tr['keys']) if k[0] in held]
        def subset(d,idx):return dict(lx=d['lx'][idx],ly=d['ly'][idx],lkeys=[d['lkeys'][j] for j in idx])
        dest=fit_fold(i,subset(tr,a),subset(long,b))
        if xi:pred[xi]=forecast(dest,tr['x'][xi]);seen[xi]+=1
    assert (seen==1).all();raw=dict(train=pred,val=forecast(main,data['val']['x']))
    mean=pred.mean((0,1));std=pred.std((0,1),correction=0).clamp_min(1e-6)
    st=torch.load(main/'stats.pt',weights_only=True)
    st.update(stgcn_mean=mean.T[None,:,None,:,None].expand(1,3,50,17,1).clone(),stgcn_std=std.T[None,:,None,:,None].expand(1,3,50,17,1).clone())
    (O/'forecast').mkdir(exist_ok=True);torch.save(st,O/'forecast/normalization.pt')
    features={s:((raw[s]-mean)/std).permute(0,3,1,2).unsqueeze(-1).contiguous() for s in ['train','val']}
    torch.save(features,O/'features.pt');torch.save(raw,O/'raw_features.pt')
    save('feature_integrity.json',dict(each_training_forecast_once=True,held_video_excluded=True,training_only_statistics=True,source_main_lstm_sha256=sha(main/'best.pt')))
features=torch.load(O/'features.pt',weights_only=False)
# Reuse exact classifier training function, changing only output directory.
g=ns['train_classifier'].__globals__;g['OUT']=O
ns['train_classifier']('forecast',data,features)
assert load('forecast/complete.json')['initial_state_sha256']==json.loads((OLD/'observed/complete.json').read_text())['initial_state_sha256']

if not (O/'comparison.json').exists():
    log(stage='evaluate_selected_checkpoint')
    model=RelativeBehaviorPredictor(O/'forecast/best.pt',main/'best.pt','cuda:0','training_stats',O/'forecast/normalization.pt')
    with torch.inference_mode():expected=model.stgcn(features['val'].cuda()).softmax(1).cpu().numpy()
    actual=np.concatenate([model.predict(x.numpy())['probabilities'] for x in data['val']['x'].split(64)])
    parity=float(np.abs(expected-actual).max());assert parity<1e-4
    cm=np.bincount(3*data['val']['y'].numpy()+actual.argmax(1),minlength=9).reshape(3,3)
    assert cm.tolist()==load('forecast/best.json')['metrics']['confusion_matrix']
    test=ns['causal_classification'](splits['test']);truth=test['y'].numpy()
    oldpred=json.loads((OLD/'observed/test_predictions.json').read_text())
    assert [list(k) for k in test['keys']]==[r['key'] for r in oldpred]
    assert truth.tolist()==[r['truth'] for r in oldpred]
    probs=np.concatenate([model.predict(x.numpy())['probabilities'] for x in test['x'].split(64)])
    pred=probs.argmax(1);cm=np.bincount(3*truth+pred,minlength=9).reshape(3,3);met=summarize(cm)
    save('forecast/test_predictions.json',[dict(key=k,truth=int(y),prediction=int(p),probabilities=q.tolist()) for k,y,p,q in zip(test['keys'],truth,pred,probs)])
    old=json.loads((OLD/'comparison.json').read_text());observed=old['branches']['observed']
    result=dict(samples=len(truth),test_videos=len(splits['test']),eligible=test['eligible'],coverage=len(truth)/test['eligible'],
        branches=dict(observed=observed,previous_lstm=old['branches']['forecast'],relative_lstm=met),
        delta_vs_observed=dict(accuracy=met['accuracy']-observed['accuracy'],macro_f1=met['macro_f1']-observed['macro_f1']),
        uncertainty_vs_observed=ns['paired_bootstrap'](test['keys'],truth,np.array([r['prediction'] for r in oldpred]),pred),
        scope='fixed video split, same1506 paired samples, validation-only selection; test used earlier in project')
    save('comparison.json',result);save('runtime_parity.json',dict(max_probability_difference=parity,selected_checkpoint_reloaded=True))
    torch.save(dict(x=test['x'][:64]),O/'benchmark_inputs.pt')
    del model;torch.cuda.empty_cache()

if not (O/'forecast_quality.json').exists():
    model=SkeletonLSTM().cuda().eval();model.load_state_dict(torch.load(main/'best.pt',weights_only=True));std=torch.load(main/'stats.pt',weights_only=True)['lstm_std'].cuda()
    quality={}
    with torch.inference_mode():
        for horizon in [10,50]:
            pairs=ns['pose_pairs'](splits['test'],horizon=horizon);tot=np.zeros(horizon//10);base=tot.copy();num=0
            for x,y in zip(pairs['lx'].split(64),pairs['ly'].split(64)):
                x,y=x.cuda(),y.cuda();p=forecast_relative(model,x,std,horizon//10)
                for k in range(horizon//10):
                    sl=slice(k*10,(k+1)*10)
                    tot[k]+=(((p[:,sl]-y[:,sl])/std)[...,:2].square().mean().item())*len(x)
                    base[k]+=(((x[:,-1:]-y[:,sl])/std)[...,:2].square().mean().item())*len(x)
                num+=len(x)
            quality[str(horizon)]=dict(samples=num,xy_mse_by_block=(tot/num).tolist(),persistence_by_block=(base/num).tolist(),mean_xy_mse=float(tot.mean()/num))
    save('forecast_quality.json',quality)
assert all(sha(ROOT/'weights/final'/n)==h for n,h in protected.items())
assert sha(ROOT/'st_gcn.py')==protocol['architecture_sha256']
save('complete.json',dict(architecture_unchanged=True,original_weights_unchanged=True,paired_test_keys_verified=True))
log(stage='complete',result=load('comparison.json'))
