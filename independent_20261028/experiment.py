"""Video-disjoint retraining and paired evaluation of unchanged paper networks."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib, sys, json, time, collections, xml.etree.ElementTree as ET, argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from st_gcn import Model, SkeletonLSTM
from modern_pipeline.behavior import BehaviorPredictor
OUT=pathlib.Path(r'D:\SRTP_independent_20261028'); OLD=pathlib.Path(r'D:\SRTP_reproduction_20261027')
torch.set_num_threads(2); torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
SPLITS={s:(ROOT/'data/split_ids/all_videos'/f'{s}.txt').read_text().split() for s in ['train','val','test']}
assert not(set(SPLITS['train'])&set(SPLITS['test']) or set(SPLITS['val'])&set(SPLITS['test']) or set(SPLITS['train'])&set(SPLITS['val']))
OUT.mkdir(exist_ok=True);(OUT/'split.json').write_text(json.dumps(SPLITS,indent=2))

def gt(video):
    labels={p.get('id'):{-1:0,0:1,1:2}[int(p.get('crossing'))] for p in ET.parse(ROOT/'data/annotations_attributes'/f'{video}_attributes.xml').getroot().findall('pedestrian')}
    frames=collections.defaultdict(list); pframes=collections.defaultdict(list)
    for tr in ET.parse(ROOT/'data/annotations'/f'{video}.xml').getroot().findall('track'):
        for b in tr.findall('box'):
            pid=b.findtext("attribute[@name='id']")
            if pid not in labels or not pid.endswith('b') or b.get('outside')=='1':continue
            f=int(b.get('frame'));frames[f].append((pid,labels[pid],np.array([float(b.get(k)) for k in ['xtl','ytl','xbr','ybr']])));pframes[pid].append(f)
    eligible={(video,pid,f):labels[pid] for pid,fs in pframes.items() for i,f in enumerate(sorted(set(fs))) if i>=49 and f%10==0}
    return frames,eligible

def raw(video,variant):
    if variant=='modern':
        with np.load(OUT/'modern_raw'/f'{video}.npz') as z:return z['rows'],z['poses']
    with np.load(OLD/'tracks'/f'{video}.npz') as z:boxes={(int(r[0]),int(r[1])):r for r in z['rows']}
    rows=[];poses=[]
    with np.load(OLD/'features'/f'{video}.npz') as z:
        for i,m in enumerate(json.loads(str(z['meta']))):
            for f,p in zip(z[f'f{i}'],z[f'p{i}']):
                key=(int(f),m['track_id'])
                if key in boxes:rows.append(boxes[key]);poses.append(p)
    return np.asarray(rows).reshape(-1,6),np.asarray(poses).reshape(-1,17,3)

def sequences(video,variant):
    frames,eligible=gt(video);rows,poses=raw(video,variant);groups=collections.defaultdict(list)
    for row,p in zip(rows,poses):
        valid=(p[:,2]>=.2)&(p[:,0]>0)&(p[:,1]>0)
        if valid.sum()>=5:groups[int(row[0])].append((int(row[1]),row[2:],p))
    seqs=collections.defaultdict(list)
    for f,items in groups.items():
        annotations=frames[f]
        if not annotations:continue
        a=np.array([it[1] for it in items]);b=np.array([g[2] for g in annotations]);inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
        iou=inter/(np.maximum(0,a[:,2:]-a[:,:2]).prod(1)[:,None]+np.maximum(0,b[:,2:]-b[:,:2]).prod(1)[None,:]-inter+1e-8)
        ri,ci=linear_sum_assignment(-iou)
        for r,c in zip(ri,ci):
            if iou[r,c]>=.5:
                pid,label,_=annotations[c];tid,_,pose=items[r];seqs[(pid,tid,label)].append((f,pose))
    segments=[]
    for (pid,tid,label),items in seqs.items():
        items.sort(key=lambda x:x[0]); starts=[0]+[i for i in range(1,len(items)) if items[i][0]-items[i-1][0]>2]+[len(items)]
        for a,b in zip(starts,starts[1:]):
            fs=np.array([x[0] for x in items[a:b]]);ps=np.array([x[1] for x in items[a:b]],dtype=np.float32)
            if len(ps)>=50:segments.append((video,pid,tid,label,fs,ps))
    return segments,eligible

def build(variant):
    dest=OUT/variant;dest.mkdir(exist_ok=True);data={};eligible={}
    for split,videos in SPLITS.items():
        segments=[];eligible[split]={}
        for video in videos:
            ss,ee=sequences(video,variant);segments.extend(ss);eligible[split].update(ee)
        x=[];y=[];keys=[];lx=[];ly=[]
        for video,pid,tid,label,fs,ps in segments:
            for i in range(49,len(ps)):
                if fs[i]%10==0 and (video,pid,int(fs[i])) in eligible[split]:x.append(ps[i-49:i+1]);y.append(label);keys.append((video,pid,int(fs[i])))
            for i in range(0,len(ps)-59,10):lx.append(ps[i:i+50]);ly.append(ps[i+50:i+60])
        data[split]=dict(x=torch.tensor(np.array(x),dtype=torch.float32),y=torch.tensor(y),keys=keys,lx=torch.tensor(np.array(lx),dtype=torch.float32),ly=torch.tensor(np.array(ly),dtype=torch.float32),eligible=len(eligible[split]))
        assert len(x)>0 and len(lx)>0,(variant,split)
    train=data['train'];cx=train['x'].permute(0,3,1,2).unsqueeze(-1)
    # Fit all statistics strictly on training histories, never validation/test poses.
    lp=train['lx'].reshape(-1,17,3)
    stats=dict(stgcn_mean=cx.mean(0,keepdim=True),stgcn_std=cx.std(0,keepdim=True).clamp_min(1e-6),lstm_mean=lp.mean(0),lstm_std=lp.std(0,correction=0).clamp_min(1e-8))
    torch.save(stats,dest/'normalization.pt');torch.save(data,dest/'data.pt')
    summary={s:dict(windows=len(d['x']),lstm_windows=len(d['lx']),eligible=d['eligible'],classes=torch.bincount(d['y'],minlength=3).tolist()) for s,d in data.items()}
    (dest/'dataset.json').write_text(json.dumps(summary,indent=2));print('DATA',variant,summary,flush=True)
    return data,stats

def tensors(d,stats,kind):
    if kind=='stgcn':return (d['x'].permute(0,3,1,2).unsqueeze(-1)-stats['stgcn_mean'])/stats['stgcn_std'],d['y']
    return ((d['lx']-stats['lstm_mean'])/stats['lstm_std']).reshape(-1,50,51),(d['ly']-stats['lstm_mean'])/stats['lstm_std']

def train(variant,data,stats,kind):
    dest=OUT/variant/kind;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    model=(Model(3,3,{'layout':'yolov8','strategy':'spatial'},True) if kind=='stgcn' else SkeletonLSTM()).cuda()
    opt=torch.optim.SGD(model.parameters(),lr=.001) if kind=='stgcn' else torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    loss_fn=torch.nn.CrossEntropyLoss() if kind=='stgcn' else torch.nn.MSELoss()
    loaders={s:torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors(data[s],stats,kind)),batch_size=128,shuffle=s=='train') for s in ['train','val']}
    epochs=125 if kind=='stgcn' else 100; best=float('inf');history=[];begin=time.perf_counter()
    for epoch in range(epochs):
        record={'epoch':epoch+1}
        for split,loader in loaders.items():
            model.train(split=='train');total=0.;count=0;correct=0
            with torch.set_grad_enabled(split=='train'):
                for x,y in loader:
                    x=x.cuda();y=y.cuda();out=model(x);loss=loss_fn(out,y);assert torch.isfinite(loss)
                    if split=='train':opt.zero_grad();loss.backward();opt.step()
                    total+=float(loss)*len(x);count+=len(x)
                    if kind=='stgcn':correct+=int((out.argmax(1)==y).sum())
            record[split+'_loss']=total/count
            if kind=='stgcn':record[split+'_accuracy']=correct/count
        if record['val_loss']<best:
            best=record['val_loss'];torch.save(model.state_dict(),dest/'best.pt');(dest/'best.json').write_text(json.dumps(record))
        history.append(record)
        if (epoch+1)%5==0:
            (dest/'history.json').write_text(json.dumps(history));print('TRAIN',variant,kind,record,flush=True)
    torch.save(model.state_dict(),dest/'final.pt');(dest/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,epochs=epochs,selected_by='minimum validation loss',seed=2026,weight_decay=0)))
    del model,opt;torch.cuda.empty_cache()

def metrics(truth,pred):
    cm=np.zeros((3,3),dtype=int)
    for t,p in zip(truth,pred):cm[t,p]+=1
    tp=cm.diagonal();f1=2*tp/np.maximum(1,cm.sum(0)+cm.sum(1))
    return dict(n=len(truth),accuracy=float(tp.sum()/max(1,len(truth))),macro_f1=float(f1.mean()),crossing_recall=float(tp[2]/max(1,cm[2].sum())),confusion_matrix=cm.tolist())

def causal_test(variant):
    # Labels are used only to score an already-built tracker history, never to
    # repair identity switches, reset histories, or supply missing observations.
    from modern_pipeline.history import HistoryStore
    xs=[];ys=[];keys=[];refs={};eligible_count=0
    for video in SPLITS['test']:
        frames,eligible=gt(video);eligible_count+=len(eligible);rows,poses=raw(video,variant)
        groups=collections.defaultdict(list)
        for row,pose in zip(rows,poses):groups[int(row[0])].append((row,pose))
        store=HistoryStore(max_gap=2,ttl=30)
        for f,items in sorted(groups.items()):
            ready=[]
            for row,pose in items:
                if ((pose[:,2]>=.2)&(pose[:,0]>0)&(pose[:,1]>0)).sum()<5:continue
                state=store.update(int(row[1]),f,pose)
                if state.ready:
                    if video not in refs:refs[video]=state.as_array().copy();continue
                    ready.append((row[2:],state.as_array().copy()))
            if f%10 or not ready or not frames[f]:continue
            a=np.array([r[0] for r in ready]);ann=frames[f];b=np.array([r[2] for r in ann]);inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
            iou=inter/(np.maximum(0,a[:,2:]-a[:,:2]).prod(1)[:,None]+np.maximum(0,b[:,2:]-b[:,:2]).prod(1)[None,:]-inter+1e-8)
            ri,ci=linear_sum_assignment(-iou)
            for r,c in zip(ri,ci):
                pid,label,_=ann[c];key=(video,pid,f)
                if iou[r,c]>=.5 and key in eligible:xs.append(ready[r][1]);ys.append(label);keys.append(key)
    return dict(x=torch.tensor(np.array(xs),dtype=torch.float32),y=torch.tensor(ys),keys=keys,eligible=eligible_count),refs

def evaluate(variant,data):
    dest=OUT/variant;d,refs=causal_test(variant);rows={};results={}
    for norm in ['legacy_pair','training_stats']:
        model=BehaviorPredictor(dest/'stgcn/best.pt',dest/'lstm/best.pt','cuda:0',norm,dest/'normalization.pt')
        predictions=[]
        for video in sorted(set(k[0] for k in d['keys'])):
            ids=[i for i,k in enumerate(d['keys']) if k[0]==video]
            for j in range(0,len(ids),64):
                ix=ids[j:j+64];out=model.predict(d['x'][ix].numpy(),refs[video]);predictions.extend(zip(ix,out['class_ids'].tolist()))
        predictions.sort();pred=[p for i,p in predictions];results[norm]=metrics(d['y'].tolist(),pred)
        rows[norm]=[dict(key=k,truth=int(y),prediction=p) for k,y,p in zip(d['keys'],d['y'],pred)]
        if norm=='training_stats':
            xx,yy=tensors(d,torch.load(dest/'normalization.pt',weights_only=True),'stgcn')
            pp=[]
            with torch.inference_mode():
                for x in xx.split(128):pp.extend(model.stgcn(x.cuda()).argmax(1).cpu().tolist())
            results['stgcn_observed']=metrics(yy.tolist(),pp)
        del model;torch.cuda.empty_cache()
    results['coverage']=len(d['x'])/d['eligible'];results['eligible']=d['eligible']
    (dest/'test_predictions.json').write_text(json.dumps(rows));(dest/'test_metrics.json').write_text(json.dumps(results,indent=2));print('TEST',variant,results,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['legacy','modern'],required=True);a=p.parse_args()
    data,stats=build(a.variant)
    for kind in ['stgcn','lstm']:train(a.variant,data,stats,kind)
    evaluate(a.variant,data)
