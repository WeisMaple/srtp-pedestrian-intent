"""Video-disjoint, paired retraining of the unchanged LSTM and ST-GCN.

Build causal tracker histories, fit all normalization on training data, and
cross-fit LSTM forecasts by video before training the forecast classifier.
The saved bundle is directly compatible with BehaviorPredictor(training_stats).
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,csv,random,hashlib,collections,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from st_gcn import Model,SkeletonLSTM
from independent_20261028 import experiment as e
from independent_20261028.evaluate_internal_metrics import summarize
from modern_pipeline.behavior import BehaviorPredictor,forecast_with_statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(r'D:\SRTP_lstm_retrain_20261101'))
parser.add_argument('--stage',choices=['prepare','train','evaluate','all'],default='all')
args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
DEVICE='cuda:0';assert torch.cuda.is_available()
SEED=2026
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),'utf8')
def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def progress(stage,**kw):
    row=dict(stage=stage,**kw);save('progress.json',row);print(json.dumps(row),flush=True)
def signature(model):
    h=hashlib.sha256()
    for v in model.state_dict().values():h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

# Freeze this split before any new training; only original train/val were regrouped.
SPLITS=json.loads(Path(r'D:\SRTP_final_20261028\split.json').read_text())
assert set(SPLITS['test'])==set(e.SPLITS['test'])
assert set(SPLITS['train']+SPLITS['val'])==set(e.SPLITS['train']+e.SPLITS['val'])
assert not any(set(SPLITS[a])&set(SPLITS[b]) for a,b in [('train','val'),('train','test'),('val','test')])

def causal_classification(videos):
    old=e.SPLITS['test'];e.SPLITS['test']=videos
    try:d,_=e.causal_test('modern',skip_reference_first=False)
    finally:e.SPLITS['test']=old
    assert len(set(map(tuple,d['keys'])))==len(d['keys'])
    assert torch.isfinite(d['x']).all()
    return d

def pose_pairs(videos,horizon=10):
    """Future-pose supervision uses only detector track IDs, never GT histories."""
    xs=[];ys=[];meta=[]
    for video in videos:
        rows,poses=e.raw(video,'modern');groups=collections.defaultdict(list)
        for row,p in zip(rows,poses):
            if ((p[:,2]>=.2)&(p[:,0]>0)&(p[:,1]>0)).sum()>=5:
                groups[int(row[1])].append((int(row[0]),p))
        for tid,items in groups.items():
            items.sort(key=lambda z:z[0]);cuts=[0]+[i for i in range(1,len(items)) if items[i][0]-items[i-1][0]>2]+[len(items)]
            for a,b in zip(cuts,cuts[1:]):
                if b-a<50+horizon:continue
                ps=np.stack([r[1] for r in items[a:b]])
                for j in range(0,len(ps)-50-horizon+1,10):
                    xs.append(ps[j:j+50]);ys.append(ps[j+50:j+50+horizon])
                    meta.append((video,tid,items[a+j][0],items[a+j+49][0]))
    return dict(lx=torch.tensor(np.stack(xs),dtype=torch.float32),
                ly=torch.tensor(np.stack(ys),dtype=torch.float32),lkeys=meta)

def prepare():
    cache=OUT/'train_val.pt'
    if cache.exists():return torch.load(cache,weights_only=False)
    data={}
    for split in ['train','val']:
        data[split]=causal_classification(SPLITS[split]);data[split].update(pose_pairs(SPLITS[split]))
        d=data[split];assert torch.bincount(d['y'],minlength=3).min()>0
        progress('prepare',split=split,classifier_samples=len(d['x']),lstm_samples=len(d['lx']))
    summary={}
    for split,d in data.items():
        summary[split]=dict(videos=len(SPLITS[split]),samples=len(d['x']),classes=torch.bincount(d['y'],minlength=3).tolist(),
            lstm_samples=len(d['lx']),eligible=d['eligible'],unique_people=len(set((k[0],k[1]) for k in d['keys'])))
    save('dataset.json',summary);torch.save(data,cache)
    return data

def train_lstm(name,d,valid=None,epochs=100):
    dest=OUT/name;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return dest
    seed();mean=d['lx'].mean((0,1));std=d['lx'].std((0,1),correction=0).clamp_min(1e-6)
    torch.save(dict(lstm_mean=mean,lstm_std=std),dest/'stats.pt')
    def loader(a,shuffle):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(((a['lx']-mean)/std).reshape(-1,50,51),(a['ly']-mean)/std),batch_size=128,shuffle=shuffle)
    train=loader(d,True);val=loader(valid,False) if valid is not None else None
    model=SkeletonLSTM().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    history=[];best=float('inf');begin=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();total=0
        for x,y in train:
            opt.zero_grad();loss=torch.nn.functional.mse_loss(model(x.to(DEVICE)),y.to(DEVICE))
            assert torch.isfinite(loss);loss.backward();opt.step();total+=loss.item()*len(x)
        row=dict(epoch=epoch,train_mse=total/len(d['lx']))
        if val is not None:
            model.eval();total=0
            with torch.inference_mode():
                for x,y in val:total+=torch.nn.functional.mse_loss(model(x.to(DEVICE)),y.to(DEVICE)).item()*len(x)
            row['val_mse']=total/len(valid['lx'])
            if row['val_mse']<best:
                best=row['val_mse'];torch.save(model.state_dict(),dest/'best.pt');save(name+'/best.json',row)
        elif epoch==epochs:
            torch.save(model.state_dict(),dest/'best.pt');save(name+'/best.json',row)
        history.append(row)
        if epoch%5==0 or epoch==1:
            save(name+'/history.json',history);progress(name,**row,seconds=time.perf_counter()-begin)
    save(name+'/history.json',history)
    save(name+'/complete.json',dict(epochs=epochs,seconds=time.perf_counter()-begin,selection='minimum validation MSE' if valid is not None else 'fixed main-model selected epoch',fit_videos=sorted(set(k[0] for k in d['lkeys'])),architecture_sha256=sha(ROOT/'st_gcn.py')))
    del model,opt;torch.cuda.empty_cache()
    return dest

@torch.inference_mode()
def forecast(path,x):
    model=SkeletonLSTM().to(DEVICE).eval();model.load_state_dict(torch.load(path/'best.pt',map_location=DEVICE,weights_only=True))
    st=torch.load(path/'stats.pt',weights_only=True)
    result=torch.cat([forecast_with_statistics(model,b.to(DEVICE),st['lstm_mean'].to(DEVICE),st['lstm_std'].to(DEVICE)).cpu() for b in x.split(64)])
    assert result.shape==x.shape and torch.isfinite(result).all()
    del model;torch.cuda.empty_cache();return result

def build_features(data):
    if (OUT/'features.pt').exists():return torch.load(OUT/'features.pt',weights_only=False)
    main=train_lstm('lstm',data['train'],data['val']);epoch=json.loads((main/'best.json').read_text())['epoch']
    vids=sorted(SPLITS['train']);np.random.default_rng(SEED).shuffle(vids);folds=[sorted(vids[i::3]) for i in range(3)]
    save('forecast_folds.json',folds);tr=data['train'];pred=torch.empty_like(tr['x']);seen=torch.zeros(len(pred),dtype=torch.int)
    for i,held in enumerate(folds):
        held=set(held);li=[j for j,k in enumerate(tr['lkeys']) if k[0] not in held];xi=[j for j,k in enumerate(tr['keys']) if k[0] in held]
        fit=dict(lx=tr['lx'][li],ly=tr['ly'][li],lkeys=[tr['lkeys'][j] for j in li])
        assert not(held & {k[0] for k in fit['lkeys']})
        path=train_lstm('fold'+str(i),fit,epochs=epoch)
        if xi:pred[xi]=forecast(path,tr['x'][xi]);seen[xi]+=1
    assert (seen==1).all()
    futures=dict(train=pred,val=forecast(main,data['val']['x']))
    features={}
    for branch in ['observed','forecast']:
        raw={s:data[s]['x'] if branch=='observed' else futures[s] for s in ['train','val']}
        # Pool time and examples. Every temporal position uses the same joint scales.
        mean=raw['train'].mean((0,1));std=raw['train'].std((0,1),correction=0).clamp_min(1e-6)
        st=torch.load(main/'stats.pt',weights_only=True)
        st.update(stgcn_mean=mean.T[None,:,None,:,None].expand(1,3,50,17,1).clone(),
                  stgcn_std=std.T[None,:,None,:,None].expand(1,3,50,17,1).clone())
        dest=OUT/branch;dest.mkdir(exist_ok=True);torch.save(st,dest/'normalization.pt')
        features[branch]={s:((raw[s]-mean)/std).permute(0,3,1,2).unsqueeze(-1).contiguous() for s in ['train','val']}
    torch.save(features,OUT/'features.pt')
    save('feature_integrity.json',dict(oof_folds=3,each_training_forecast_assigned_once=True,fit_excludes_held_video=True,statistics_fit_on_train_only=True,
        normalization='per joint/channel, pooled across training examples and time; saved immutable stats reused by inference',forecast_function='modern_pipeline.behavior.forecast_with_statistics'))
    return features

def train_classifier(branch,data,features):
    dest=OUT/branch
    if (dest/'complete.json').exists():return
    seed();model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).to(DEVICE)
    init_hash=signature(model);opt=torch.optim.SGD(model.parameters(),lr=.001)
    counts=torch.bincount(data['train']['y'],minlength=3).float();weights=(counts.sum()/(3*counts)).to(DEVICE)
    train=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features['train'],data['train']['y']),batch_size=128,shuffle=True)
    val=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features['val'],data['val']['y']),batch_size=128)
    best=(-1.,-float('inf'));history=[];begin=time.perf_counter()
    for epoch in range(1,126):
        model.train();total=0.;denom=0.
        for x,y in train:
            x=x.to(DEVICE);y=y.to(DEVICE);opt.zero_grad();out=model(x)
            losses=torch.nn.functional.cross_entropy(out,y,weight=weights,reduction='none');loss=losses.sum()/weights[y].sum()
            assert torch.isfinite(loss);loss.backward();opt.step();total+=losses.sum().item();denom+=weights[y].sum().item()
        model.eval();pred=[];vloss=0.
        with torch.inference_mode():
            for x,y in val:
                out=model(x.to(DEVICE));pred.extend(out.argmax(1).cpu().tolist())
                vloss+=torch.nn.functional.cross_entropy(out,y.to(DEVICE),reduction='sum').item()
        cm=np.bincount(3*data['val']['y'].numpy()+np.array(pred),minlength=9).reshape(3,3);met=summarize(cm)
        row=dict(epoch=epoch,train_weighted_loss=total/denom,val_loss=vloss/len(pred),accuracy=met['accuracy'],macro_f1=met['macro_f1'])
        history.append(row);score=(met['macro_f1'],-row['val_loss'])
        if score>best:
            best=score;torch.save(model.state_dict(),dest/'best.pt');save(branch+'/best.json',dict(**row,metrics=met))
        if epoch%5==0 or epoch==1:
            save(branch+'/history.json',history);progress(branch,**row,seconds=time.perf_counter()-begin)
    save(branch+'/history.json',history)
    save(branch+'/complete.json',dict(epochs=125,seconds=time.perf_counter()-begin,initial_state_sha256=init_hash,
        selection='maximum validation Macro-F1, tie-break minimum unweighted validation cross-entropy',class_counts=counts.tolist(),class_weights=weights.cpu().tolist(),lr=.001,batch_size=128,seed=SEED))
    del model,opt;torch.cuda.empty_cache()

def paired_bootstrap(keys,truth,p0,p1):
    videos=sorted(SPLITS['test']);cms=[]
    for v in videos:
        ix=[i for i,k in enumerate(keys) if k[0]==v]
        cms.append([np.bincount(3*truth[ix]+p[ix],minlength=9).reshape(3,3) for p in [p0,p1]])
    cms=np.array(cms);rng=np.random.default_rng(2026);delta=[]
    for _ in range(2000):
        cm=cms[rng.integers(0,len(videos),len(videos))].sum(0)
        if cm[0].sum()==0:continue
        a,b=map(summarize,cm);delta.append([b['accuracy']-a['accuracy'],b['macro_f1']-a['macro_f1']])
    return dict(unit='video',replicates=len(delta),method='paired percentile bootstrap over all test videos',
        accuracy_delta_95ci=np.percentile(delta,[2.5,97.5],axis=0)[:,0].tolist(),macro_f1_delta_95ci=np.percentile(delta,[2.5,97.5],axis=0)[:,1].tolist())

def evaluate(data,features):
    d=causal_classification(SPLITS['test']);truth=d['y'].numpy();outputs={};predictions={};parity={}
    for branch in ['observed','forecast']:
        model=BehaviorPredictor(OUT/branch/'best.pt',OUT/'lstm/best.pt' if branch=='forecast' else None,DEVICE,'training_stats',OUT/branch/'normalization.pt')
        with torch.inference_mode():expected=model.stgcn(features[branch]['val'].to(DEVICE)).softmax(1).cpu().numpy()
        actual=np.concatenate([model.predict(x.numpy())['probabilities'] for x in data['val']['x'].split(64)])
        diff=float(np.abs(expected-actual).max());assert diff<1e-4,diff
        cm=np.bincount(3*data['val']['y'].numpy()+actual.argmax(1),minlength=9).reshape(3,3)
        selected=json.loads((OUT/branch/'best.json').read_text());assert cm.tolist()==selected['metrics']['confusion_matrix']
        parity[branch]=dict(max_probability_difference=diff,selected_checkpoint_reloaded=True)
        probs=np.concatenate([model.predict(x.numpy())['probabilities'] for x in d['x'].split(64)])
        pred=probs.argmax(1);cm=np.bincount(3*truth+pred,minlength=9).reshape(3,3)
        outputs[branch]=summarize(cm);predictions[branch]=pred
        save(branch+'/test_predictions.json',[dict(key=k,truth=int(y),prediction=int(p),probabilities=pp.tolist()) for k,y,p,pp in zip(d['keys'],truth,pred,probs)])
        # Warmed inference latency: same 64 histories, single-person calls, includes transfers.
        xb=d['x'][:64].numpy();lat=[]
        for _ in range(5):model.predict(xb[:1])
        for repeat in range(3):
            for x in xb:
                torch.cuda.synchronize();start=time.perf_counter();model.predict(x[None]);torch.cuda.synchronize();lat.append((time.perf_counter()-start)*1000)
        outputs[branch]['behavior_gpu_latency_ms']=dict(n=len(lat),median=float(np.median(lat)),p95=float(np.percentile(lat,95)),mean=float(np.mean(lat)))
        del model;torch.cuda.empty_cache()
    result=dict(protocol='paired retrained branches; video-disjoint test',test_videos=len(SPLITS['test']),test_samples=len(truth),eligible=d['eligible'],coverage=len(truth)/d['eligible'],
        classes=np.bincount(truth,minlength=3).tolist(),branches=outputs,
        delta=dict(accuracy=outputs['forecast']['accuracy']-outputs['observed']['accuracy'],macro_f1=outputs['forecast']['macro_f1']-outputs['observed']['macro_f1']),
        uncertainty=paired_bootstrap(d['keys'],truth,predictions['observed'],predictions['forecast']),
        scope='test videos disjoint from current fit; previously used in earlier project development; fixed seed 2026')
    save('runtime_parity.json',parity);save('comparison.json',result)
    # The 10-step forecast target is a later detector observation on the same causal track.
    pairs=pose_pairs(SPLITS['test']);st=torch.load(OUT/'lstm/stats.pt',weights_only=True)
    model=SkeletonLSTM().to(DEVICE).eval();model.load_state_dict(torch.load(OUT/'lstm/best.pt',map_location=DEVICE,weights_only=True))
    mse=0.;base=0.;xy=0.;bxy=0.;n=0
    with torch.inference_mode():
        for x,y in zip(pairs['lx'].split(128),pairs['ly'].split(128)):
            xn=(x-st['lstm_mean'])/st['lstm_std'];yn=(y-st['lstm_mean'])/st['lstm_std']
            pred=model(xn.reshape(-1,50,51).to(DEVICE)).cpu();persist=xn[:,-1:].expand_as(yn)
            mse+=((pred-yn)**2).mean().item()*len(x);base+=((persist-yn)**2).mean().item()*len(x)
            xy+=((pred[...,:2]-yn[...,:2])**2).mean().item()*len(x);bxy+=((persist[...,:2]-yn[...,:2])**2).mean().item()*len(x);n+=len(x)
    save('lstm_test_forecast.json',dict(samples=n,horizon=10,normalized_mse=mse/n,persistence_mse=base/n,xy_mse=xy/n,persistence_xy_mse=bxy/n,target='future YOLO keypoints; all test tracks'))
    save('complete.json',dict(status='complete',architecture_sha256=sha(ROOT/'st_gcn.py'),models_unchanged=True,default_weights_changed=False))
    print('COMPARISON '+json.dumps(result),flush=True)

if __name__=='__main__':
    protocol=dict(seed=SEED,device=torch.cuda.get_device_name(),torch=torch.__version__,architecture_sha256=sha(ROOT/'st_gcn.py'),
        structure='unchanged YOLO26s-Pose/ByteTrack, LSTM(10 layers,256 hidden,10-step output,5 recursive rounds), original 10-block ST-GCN',
        raw_cache=str(e.OUT/'modern_raw'),pose_filter=False,min_valid_joints=5,keypoint_conf=.2,max_frame_gap=2,
        split='fixed earlier class-aware regrouping of original train/val only; official test unchanged',
        class_matching='same-frame Hungarian IoU>=0.5 for labels only; detector-ID history is never repaired by GT',
        lstm_supervision='all valid causal detector tracks in fit videos, history50,target10,stride10; no GT identity repair',
        lstm=dict(epochs=100,lr=.001,optimizer='Adam',weight_decay=0,batch_size=128,selection='validation MSE'),
        stgcn=dict(epochs=125,lr=.001,optimizer='SGD',momentum=0,weight_decay=0,batch_size=128,loss='inverse training class-frequency weighted cross-entropy',selection='validation Macro-F1; tie unweighted validation CE'),
        forecast_training='3-fold out-of-video LSTM forecasts; fold duration selected by main LSTM validation; no test used',
        normalization='train-only fixed mean/std pooled over examples/time per joint/channel; branch-specific classifier stats; shared runtime forecast function',
        test_policy='evaluate once after both branches selected; test previously used in project, not unseen blind benchmark')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text('utf8'))==protocol
    save('protocol.json',protocol);save('split.json',SPLITS)
    if not (OUT/'protected_weights.json').exists():save('protected_weights.json',{p.name:sha(p) for p in (ROOT/'weights/final').glob('*.pt')})
    data=prepare()
    if args.stage in ['train','all']:
        features=build_features(data)
        for branch in ['observed','forecast']:train_classifier(branch,data,features[branch])
        hashes=[json.loads((OUT/b/'complete.json').read_text())['initial_state_sha256'] for b in ['observed','forecast']];assert len(set(hashes))==1
    if args.stage in ['evaluate','all']:
        features=torch.load(OUT/'features.pt',weights_only=False);evaluate(data,features)
    protected=json.loads((OUT/'protected_weights.json').read_text());assert all(sha(ROOT/'weights/final'/n)==h for n,h in protected.items())
