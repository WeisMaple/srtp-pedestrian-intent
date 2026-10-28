"""Video-group validation and forecast-domain training; original networks unchanged."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,sys,json,collections,time,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from st_gcn import Model,SkeletonLSTM
from modern_pipeline.pose_filter import PoseFilter
from modern_pipeline.behavior import BehaviorPredictor,forecast_with_statistics
from independent_20261028 import experiment as e
OUT=pathlib.Path(r'D:\SRTP_final_20261028');OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
original_raw=e.raw;official={k:list(v) for k,v in e.SPLITS.items()}
def filtered_raw(video,variant):
    rows,poses=original_raw(video,variant);f=PoseFilter();poses=poses.copy()
    for i,r in enumerate(rows):poses[i]=f.update(int(r[1]),int(r[0]),poses[i])
    return rows,poses
e.raw=filtered_raw
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def seed():torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)

def prepare():
    cache=OUT/'data.pt'
    if cache.exists():return torch.load(cache,weights_only=False)
    pool=sorted(official['train']+official['val']);source=torch.load(r'D:\SRTP_classroom_20261028\modern\data.pt',weights_only=False)
    people={v:[set(),set(),set()] for v in pool}
    for s in ['train','val']:
        for key,y in zip(source[s]['keys'],source[s]['y'].tolist()):people[key[0]][y].add(key[1])
    counts=np.array([[len(x) for x in people[v]] for v in pool]);presence=(counts>0).astype(int);rng=np.random.default_rng(2026)
    best=None
    for _ in range(10000):
        ix=rng.choice(len(pool),44,replace=False);vc=counts[ix].sum(0);pc=presence[ix].sum(0)
        if np.any(pc<3) or np.any(presence.sum(0)-pc<3):continue
        score=float(np.square((vc-.2*counts.sum(0))/np.maximum(1,.2*counts.sum(0))).sum())
        if best is None or score<best[0]:best=(score,ix)
    assert best is not None
    val=sorted(pool[i] for i in best[1]);train=sorted(set(pool)-set(val));splits=dict(train=train,val=val,test=official['test'])
    assert not(set(train)&set(val) or set(train+val)&set(official['test']))
    assert set(train+val)==set(official['train']+official['val'])
    save('split.json',splits);save('split_design.json',dict(seed=2026,criterion='20% per-class person counts with >=3 videos per class in validation; no model scores used',train_people=counts[[pool.index(v) for v in train]].sum(0).tolist(),validation_people=counts[best[1]].sum(0).tolist()))
    result={}
    for s in ['train','val']:
        # Actual predicted-ID histories; GT is used only for labels, never history repair.
        e.SPLITS['test']=splits[s];d,_=e.causal_test('modern',skip_reference_first=False)
        lx=[];ly=[];videos=[]
        for video in splits[s]:
            segments,_=e.sequences(video,'modern')
            for _,pid,tid,label,fs,ps in segments:
                for i in range(0,len(ps)-59,10):lx.append(ps[i:i+50]);ly.append(ps[i+50:i+60]);videos.append(video)
        d.update(lx=torch.tensor(np.array(lx)),ly=torch.tensor(np.array(ly)),lvideo=videos)
        assert torch.bincount(d['y'],minlength=3).min()>0
        result[s]=d
    e.SPLITS['test']=official['test'];result['splits']=splits
    save('dataset.json',{s:dict(samples=len(result[s]['x']),classes=torch.bincount(result[s]['y'],minlength=3).tolist(),lstm_samples=len(result[s]['lx'])) for s in ['train','val']})
    torch.save(result,cache);print('DATA_READY',json.loads((OUT/'dataset.json').read_text()),flush=True)
    return result

def fit_lstm(name,train,valid=None,epochs=100):
    dest=OUT/name;dest.mkdir(exist_ok=True);statsfile=dest/'stats.pt'
    if (dest/'complete.json').exists():return dest
    seed();lp=train['lx'].reshape(-1,17,3);mean=lp.mean(0);std=lp.std(0,correction=0).clamp_min(1e-8)
    torch.save(dict(lstm_mean=mean,lstm_std=std),statsfile)
    def loader(d,shuffle):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(((d['lx']-mean)/std).reshape(-1,50,51),(d['ly']-mean)/std),batch_size=128,shuffle=shuffle)
    dl=loader(train,True);vl=loader(valid,False) if valid is not None else None
    model=SkeletonLSTM().cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0);history=[];best=float('inf');begin=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0
        for x,y in dl:
            opt.zero_grad();loss=torch.nn.functional.mse_loss(model(x.cuda()),y.cuda());assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss)*len(x)
        record=dict(epoch=epoch+1,train_mse=total/len(train['lx']))
        if vl is not None:
            model.eval();total=0
            with torch.inference_mode():
                for x,y in vl:total+=float(torch.nn.functional.mse_loss(model(x.cuda()),y.cuda()))*len(x)
            record['val_mse']=total/len(valid['lx'])
            if record['val_mse']<best:best=record['val_mse'];torch.save(model.state_dict(),dest/'best.pt');save(name+'/best.json',record)
        elif epoch==epochs-1:torch.save(model.state_dict(),dest/'best.pt')
        history.append(record)
        if (epoch+1)%25==0:print(name,record,flush=True);save(name+'/history.json',history)
    save(name+'/history.json',history)
    save(name+'/complete.json',dict(seconds=time.perf_counter()-begin,epochs=epochs,selection='validation MSE' if valid is not None else 'epoch selected by main LSTM validation MSE, no held-out fold fitted'))
    return dest

@torch.inference_mode()
def forecast(path,x):
    model=SkeletonLSTM().cuda().eval();model.load_state_dict(torch.load(path/'best.pt',weights_only=True));stats=torch.load(path/'stats.pt',weights_only=True)
    chunks=[forecast_with_statistics(model,b.cuda(),stats['lstm_mean'].cuda(),stats['lstm_std'].cuda()).cpu() for b in x.split(64)]
    return torch.cat(chunks)

def classification_features(data):
    dest=OUT/'forecast_data.pt'
    if dest.exists():return torch.load(dest,weights_only=False)
    train=data['train'];videos=sorted(set(k[0] for k in train['keys'])|set(train['lvideo']));rng=np.random.default_rng(2026);rng.shuffle(videos)
    main=fit_lstm('lstm',train,data['val']);selected_epoch=json.loads((main/'best.json').read_text())['epoch']
    folds=[sorted(videos[i::3]) for i in range(3)];save('forecast_folds.json',folds);pred=torch.empty_like(train['x']);assigned=torch.zeros(len(pred),dtype=torch.int64)
    for i,held in enumerate(folds):
        held=set(held);li=[j for j,v in enumerate(train['lvideo']) if v not in held];xi=[j for j,k in enumerate(train['keys']) if k[0] in held]
        assert not({train['lvideo'][j] for j in li}&held)
        fold=dict(lx=train['lx'][li],ly=train['ly'][li]);path=fit_lstm('fold'+str(i),fold,epochs=selected_epoch)
        if xi:pred[xi]=forecast(path,train['x'][xi]);assigned[xi]+=1
    assert (assigned==1).all()
    valpred=forecast(main,data['val']['x'])
    x=pred.permute(0,3,1,2).unsqueeze(-1);vx=valpred.permute(0,3,1,2).unsqueeze(-1)
    stats=torch.load(main/'stats.pt',weights_only=True);stats.update(stgcn_mean=x.mean(0,keepdim=True),stgcn_std=x.std(0,keepdim=True).clamp_min(1e-6));torch.save(stats,OUT/'normalization.pt')
    result=dict(train=(x-stats['stgcn_mean'])/stats['stgcn_std'],val=(vx-stats['stgcn_mean'])/stats['stgcn_std']);torch.save(result,dest)
    save('forecast_integrity.json',dict(folds=3,each_training_sample_predicted_once=True,held_out_video_not_used_to_fit_its_lstm=True,statistics_test_free=True))
    return result

def fit_classifier(data,features):
    dest=OUT/'stgcn';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    seed();d=data['train'];y=d['y'];person_counts=collections.Counter((k[0],k[1]) for k in d['keys'])
    sample_weights=torch.tensor([1/person_counts[(k[0],k[1])] for k in d['keys']],dtype=torch.double)
    people_by_class=[set() for _ in range(3)]
    for k,c in zip(d['keys'],y.tolist()):people_by_class[c].add((k[0],k[1]))
    weights=torch.tensor([len(p) for p in people_by_class],dtype=torch.float32).rsqrt();weights/=weights.mean()
    sampler=torch.utils.data.WeightedRandomSampler(sample_weights,len(y),replacement=True)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features['train'],y),batch_size=128,sampler=sampler)
    vl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features['val'],data['val']['y']),batch_size=128)
    model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).cuda();opt=torch.optim.SGD(model.parameters(),lr=.001);criterion=torch.nn.CrossEntropyLoss(weight=weights.cuda())
    history=[];best=(-1.,-float('inf'));begin=time.perf_counter()
    for epoch in range(125):
        model.train();losses=0
        for x,yy in dl:
            opt.zero_grad();loss=criterion(model(x.cuda()),yy.cuda());assert torch.isfinite(loss);loss.backward();opt.step();losses+=float(loss)*len(x)
        model.eval();pred=[];vtotal=0
        with torch.inference_mode():
            for x,yy in vl:
                out=model(x.cuda());pred.extend(out.argmax(1).cpu().tolist());vtotal+=float(criterion(out,yy.cuda()))*len(x)
        met=e.metrics(data['val']['y'].tolist(),pred);record=dict(epoch=epoch+1,train_loss=losses/len(y),val_loss=vtotal/len(pred),**met);history.append(record)
        score=(met['macro_f1'],-record['val_loss'])
        if score>best:best=score;torch.save(model.state_dict(),dest/'best.pt');save('stgcn/best.json',record)
        if (epoch+1)%5==0:print('STGCN',record,flush=True);save('stgcn/history.json',history)
    torch.save(model.state_dict(),dest/'final.pt');save('stgcn/complete.json',dict(seconds=time.perf_counter()-begin,epochs=125,selection='validation Macro-F1, tie-break validation loss',class_weights=weights.tolist(),sampling='equal expected weight per person'))

def finish(data,features):
    model=BehaviorPredictor(OUT/'stgcn/best.pt',OUT/'lstm/best.pt','cuda:0','training_stats',OUT/'normalization.pt')
    # Verify the saved inference path against the classifier training feature path.
    with torch.inference_mode():expected=model.stgcn(features['val'][:16].cuda()).softmax(1).cpu().numpy()
    actual=model.predict(data['val']['x'][:16].numpy())['probabilities'];difference=float(np.abs(expected-actual).max());assert difference<1e-4,difference
    save('runtime_parity.json',dict(max_probability_difference=difference))
    e.SPLITS['test']=official['test'];d,_=e.causal_test('modern',skip_reference_first=False);pred=[]
    for x in d['x'].split(64):pred.extend(model.predict(x.numpy())['class_ids'].tolist())
    met=e.metrics(d['y'].tolist(),pred);met.update(coverage=len(pred)/d['eligible'],eligible=d['eligible']);save('test_metrics.json',met)
    save('test_predictions.json',[dict(key=k,truth=int(y),prediction=p) for k,y,p in zip(d['keys'],d['y'],pred)])
    save('model_config.json',dict(normalization='training_stats',pose_filter=True,task='predict',classify_every=1,behavior_device='cpu',stgcn='stgcn/best.pt',lstm='lstm/best.pt',stats='normalization.pt'))
    save('complete.json',dict(status='complete',test_used_for_selection=False,architecture_sha256=hashlib.sha256((ROOT/'st_gcn.py').read_bytes()).hexdigest()));print('FINAL',met,flush=True)

if __name__=='__main__':
    save('protocol.json',dict(seed=2026,networks='Original ST-GCN and 10-layer LSTM, unchanged',lstm_epochs=100,stgcn_epochs=125,normalization='training-only LSTM statistics; out-of-fold training forecast ST-GCN statistics; same runtime function',selection='LSTM validation MSE, ST-GCN validation Macro-F1',test_policy='official test unchanged; already used in prior development, not a new blind benchmark'))
    data=prepare();features=classification_features(data);fit_classifier(data,features);finish(data,features)
