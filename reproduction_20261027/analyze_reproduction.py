import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg'
import pathlib,json,csv,sys,collections,hashlib
import numpy as np
import torch
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));ART=pathlib.Path(r'D:\SRTP_reproduction_20261027')
from st_gcn import Model,SkeletonLSTM
torch.set_num_threads(2)
for name in ['stgcn','lstm','lstm_wd0']:
    final=ART/'models'/name/'final_validation.json'
    if not final.exists():raise RuntimeError('Training incomplete: '+name)
results={'dataset':json.loads((ART/'dataset_summary.json').read_text()),'models':{},'protocol_limitations':json.loads((ART/'reproduction_config.json').read_text(encoding='utf-8-sig'))['limitations']}
for name in ['stgcn','lstm','lstm_wd0']:
    folder=ART/'models'/name
    results['models'][name]={k:json.loads((folder/f'{k}.json').read_text()) for k in ['pretrained_baseline','best_validation','final_validation'] if (folder/f'{k}.json').exists()}
    if name=='lstm_wd0':results['models'][name]['pretrained_baseline_shared_with_lstm']=True
    split=json.loads((folder/'split.json').read_text())
    results['models'][name]['split']={k:v for k,v in split.items() if not k.endswith('_indices')}
    if name=='stgcn':results['models'][name]['epoch_100']=json.loads((folder/'epoch_100_validation.json').read_text())
    if name=='stgcn':
        for key in ['pretrained_baseline','best_validation','final_validation','epoch_100']:
            r=results['models'][name][key];cm=np.asarray(r['confusion_matrix']);f1=[]
            for c in range(3):
                tp=cm[c,c];precision=tp/cm[:,c].sum() if cm[:,c].sum() else 0.;recall=tp/cm[c].sum() if cm[c].sum() else 0.
                f1.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
            r['macro_f1']=float(np.mean(f1));r['majority_baseline_accuracy']=float(cm.sum(1).max()/cm.sum())
    with (folder/'history.csv').open() as f:history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss_sum'] for r in history],label='Train');axes[0].plot(epochs,[r['validation_loss_sum'] for r in history],label='Validation');axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Sum of batch-mean losses (legacy code)');axes[0].legend();axes[0].grid(alpha=.2)
    if name=='stgcn':
        axes[1].plot(epochs,[r['train_accuracy'] for r in history],label='Train');axes[1].plot(epochs,[r['validation_accuracy'] for r in history],label='Validation');axes[1].axhline(.975,color='gray',linestyle='--',label='Paper ~97.5%');axes[1].axvline(100,color='#cccccc',linestyle=':');axes[1].set_ylabel('Accuracy');axes[1].set_ylim(0,1.02)
    else:
        axes[1].plot(epochs,[r['train_loss_mean'] for r in history],label='Train');axes[1].plot(epochs,[r['validation_loss_mean'] for r in history],label='Validation');axes[1].set_ylabel('Mean element MSE')
    axes[1].set_xlabel('Epoch');axes[1].legend();axes[1].grid(alpha=.2);fig.suptitle(name.upper()+' | recreated 80/20 window split | seed 2026')
    fig.savefig(ART/(name+'_training_curves.png'),dpi=160);plt.close(fig)

# LSTM comparison on the exact seeded validation windows.
lstm_data=torch.load(ART/'datasets/lstm.pt',map_location='cpu',weights_only=True)
lstm_split=json.loads((ART/'models/lstm/split.json').read_text());ids=lstm_split['validation_indices']
lstm=SkeletonLSTM().cuda().eval();lstm.load_state_dict(torch.load(ART/'models/lstm/final_state_dict.pt',map_location='cuda',weights_only=True))
results['LSTM_hold_last_comparison']={}
for variant in ['lstm','lstm_wd0']:
    check_split=json.loads((ART/'models'/variant/'split.json').read_text())
    assert check_split['validation_indices']==ids
    assert check_split['train_indices']==lstm_split['train_indices']
    lstm.load_state_dict(torch.load(ART/'models'/variant/'final_state_dict.pt',map_location='cuda',weights_only=True))
    model_error=0.;hold_error=0.;elements=0
    with torch.inference_mode():
        for start in range(0,len(ids),128):
            ix=ids[start:start+128];x=lstm_data['x'][ix].cuda();y=lstm_data['y'][ix].cuda();pred=lstm(x);last=x[:,-1].reshape(-1,1,17,3).expand(-1,10,-1,-1)
            model_error+=float(((pred-y)**2).sum());hold_error+=float(((last-y)**2).sum());elements+=y.numel()
    results['LSTM_hold_last_comparison'][variant]={'reference':'automatic detected poses, same normalized training space','validation_windows':len(ids),'LSTM_MSE_mean':model_error/elements,'hold_last_MSE_mean':hold_error/elements,'relative_MSE_reduction':1-model_error/hold_error}
results['LSTM_weight_decay_control']={'same_seed':True,'same_train_validation_indices':True,'same_learning_rate':0.001,'same_epochs':100,'source_code_weight_decay':0.0001,'paper_Adam_default_comparison':0.0,'diagnostic':json.loads((ART/'LSTM_cold_start_diagnostic.json').read_text())}
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for variant,label in [('lstm','Source code: weight_decay=1e-4'),('lstm_wd0','Adam default: weight_decay=0')]:
    with (ART/'models'/variant/'history.csv').open() as f:h=list(csv.DictReader(f))
    ax.plot([int(r['epoch']) for r in h],[float(r['validation_loss_mean']) for r in h],label=label)
ax.set_xlabel('Epoch');ax.set_ylabel('Mean validation MSE');ax.set_title('LSTM control | identical seed and data split');ax.legend();ax.grid(alpha=.2)
fig.savefig(ART/'lstm_weight_decay_comparison.png',dpi=160);plt.close(fig)

# Joint paper-style inference on fixed validation histories; no training modifications.
cls_data=torch.load(ART/'datasets/stgcn.pt',map_location='cpu',weights_only=True)
cls_split=json.loads((ART/'models/stgcn/split.json').read_text());vidx=cls_split['validation_indices']
classifier=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).cuda().eval();classifier.load_state_dict(torch.load(ART/'models/stgcn/final_state_dict.pt',map_location='cuda',weights_only=True))
sequences={};video_refs={}
for path in sorted((ART/'features').glob('video_*.npz')):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['meta']));candidates=[]
        for i,m in enumerate(meta):
            poses=z[f'p{i}'].copy();frames=z[f'f{i}'].copy();key=(m['video'],m['person_id'],m['track_id']);sequences[key]=poses
            if len(poses)>=50:candidates.append((int(frames[49]),int(m['track_id']),poses[:50]))
        if candidates:video_refs[path.stem]=min(candidates,key=lambda r:(r[0],r[1]))[2]
rows=[]
for variant in ['lstm','lstm_wd0']:
    lstm.load_state_dict(torch.load(ART/'models'/variant/'final_state_dict.pt',map_location='cuda',weights_only=True))
    with torch.inference_mode():
        for start in range(0,len(vidx),32):
            ix=vidx[start:start+32];metas=[cls_data['metadata'][i] for i in ix]
            histories=np.asarray([sequences[(m['video'],m['person_id'],m['track_id'])][m['start_pose']:m['start_pose']+50] for m in metas],dtype=np.float32)
            mu=histories.mean(1);sd=histories.std(1)+1e-8
            series=torch.tensor((histories-mu[:,None])/sd[:,None],device='cuda')
            for _ in range(5):series=torch.cat([series[:,10:],lstm(series.reshape(-1,50,51))],dim=1)
            future=series*torch.tensor(sd[:,None],device='cuda')+torch.tensor(mu[:,None],device='cuda')
            for mode in ['same_person_reference','matched_video_reference']:
                refs=np.stack([sequences[(m['video'],m['person_id'],m['track_id'])][:50] if mode=='same_person_reference' else video_refs[m['video']] for m in metas])
                pair=torch.stack([torch.tensor(refs,device='cuda'),future],dim=1).permute(0,1,4,2,3).unsqueeze(-1)
                mean=pair.mean(1,keepdim=True);std=pair.std(1,keepdim=True);std[std==0]=1e-6;normalized=(pair-mean)/std
                probs=classifier(normalized.reshape(-1,3,50,17,1)).reshape(len(metas),2,3).softmax(-1)[:,1].cpu().numpy()
                for m,p in zip(metas,probs):rows.append({'lstm_variant':variant,'mode':mode,'video':m['video'],'person_id':m['person_id'],'start_pose':m['start_pose'],'true_class':m['class'],'pred_class':int(p.argmax()),'confidence':float(p.max())})
with (ART/'joint_validation_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
results['joint_inference_diagnostic']={}
for variant in ['lstm','lstm_wd0']:
    for mode in ['same_person_reference','matched_video_reference']:
        selected=[r for r in rows if r['mode']==mode and r['lstm_variant']==variant];cm=np.zeros((3,3),dtype=int)
        for r in selected:cm[r['true_class'],r['pred_class']]+=1
        results['joint_inference_diagnostic'][variant+'/'+mode]={'windows':len(selected),'accuracy':float(np.trace(cm)/cm.sum()),'confusion_matrix':cm.tolist(),'note':'Reconstructed detected histories, original two-sample normalization; not a separate official end-to-end test'}
(ART/'reproduction_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('ANALYSIS_COMPLETE '+json.dumps(results,ensure_ascii=False),flush=True)
