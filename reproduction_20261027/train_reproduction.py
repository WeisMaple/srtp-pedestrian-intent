import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,sys,json,csv,time,argparse,random,collections
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,Subset
ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
ART=pathlib.Path(r'D:\SRTP_reproduction_20261027')
from st_gcn import Model,SkeletonLSTM
parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['stgcn','lstm'],required=True);parser.add_argument('--baseline-only',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--weight-decay',type=float,default=None);args=parser.parse_args()
seed=2026;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
if not torch.cuda.is_available():raise RuntimeError('CUDA required for requested GPU reproduction')
device=torch.device('cuda:0')
data=torch.load(ART/'datasets'/(args.model+'.pt'),map_location='cpu',weights_only=True)
x=data['x'];y=data['y'];n=len(x);train_n=int(.8*n)
permutation=torch.randperm(n);train_ids=permutation[:train_n];val_ids=permutation[train_n:]
if train_n<1 or len(val_ids)<1:raise RuntimeError('Empty 80/20 split')
run_name=args.model if args.weight_decay is None else args.model+'_wd'+format(args.weight_decay,'g').replace('.','p')
if args.weight_decay is not None and args.model!='lstm':raise RuntimeError('Weight decay comparison is only defined for LSTM')
target=ART/'models'/run_name;target.mkdir(parents=True,exist_ok=True)
train_keys={(data['metadata'][i]['video'],data['metadata'][i]['person_id']) for i in train_ids.tolist()}
val_keys={(data['metadata'][i]['video'],data['metadata'][i]['person_id']) for i in val_ids.tolist()}
train_starts=collections.defaultdict(list)
for i in train_ids.tolist():
    m=data['metadata'][i];train_starts[(m['video'],m['person_id'])].append(m['start_pose'])
overlap=sum(any(abs(t-data['metadata'][i]['start_pose'])<50 for t in train_starts[(data['metadata'][i]['video'],data['metadata'][i]['person_id'])]) for i in val_ids.tolist())
split={'seed':seed,'samples':n,'train_samples':len(train_ids),'validation_samples':len(val_ids),'shared_people':len(train_keys&val_keys),'validation_people':len(val_keys),'validation_windows_overlapping_training_history':overlap,'validation_history_overlap_fraction':overlap/len(val_ids),'train_indices':train_ids.tolist(),'validation_indices':val_ids.tolist()}
(target/'split.json').write_text(json.dumps(split,indent=2))
with (target/'sample_split.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['sample_index','split','video','person_id','track_id','class','start_pose','first_frame','last_frame'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    trainset=set(train_ids.tolist())
    for i,m in enumerate(data['metadata']):w.writerow({'sample_index':i,'split':'train' if i in trainset else 'validation',**{k:m[k] for k in fields[2:]}})
labels=torch.nn.functional.one_hot(y,3).float() if args.model=='stgcn' else y
full=TensorDataset(x,labels)
train_loader=DataLoader(Subset(full,train_ids),batch_size=128,shuffle=True,num_workers=0)
val_loader=DataLoader(Subset(full,val_ids),batch_size=128,shuffle=False,num_workers=0)
if args.model=='stgcn':
    model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    criterion=nn.CrossEntropyLoss();epochs=125;old_weight=ROOT/'Models/stgcn_gpu_pre.pth'
else:
    model=SkeletonLSTM(input_size=51,hidden_size=256,num_layers=10,output_size=51,output_frames=10).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4 if args.weight_decay is None else args.weight_decay)
    criterion=nn.MSELoss();epochs=100;old_weight=ROOT/'Models/lstm_gpu_pre.pth'

def evaluate():
    model.eval();total=0.;weighted=0.;count=0;correct=0;cm=torch.zeros(3,3,dtype=torch.int64)
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(device);yb=yb.to(device);out=model(xb);loss=criterion(out,yb)
            total+=float(loss);weighted+=float(loss)*len(xb);count+=len(xb)
            if args.model=='stgcn':
                truth=yb.argmax(1);pred=out.argmax(1);correct+=int((truth==pred).sum());cm+=torch.bincount((3*truth+pred).cpu(),minlength=9).reshape(3,3)
    result={'loss_sum':total,'loss_mean':weighted/count,'samples':count}
    if args.model=='stgcn':result.update(accuracy=correct/count,confusion_matrix=cm.tolist())
    return result

if args.baseline_only:
    model.load_state_dict(torch.load(old_weight,map_location=device,weights_only=True));r=evaluate();r['note']='Existing supplied weights, recreated global normalization and seeded 80/20 window split; original training overlap unknown'
    (target/'pretrained_baseline.json').write_text(json.dumps(r,indent=2));print('PRETRAINED '+args.model+' '+json.dumps(r),flush=True);raise SystemExit(0)

checkpoint=target/'checkpoint_latest.pt';history=[];start_epoch=0;best=-float('inf');elapsed_previous=0.
if checkpoint.exists():
    if not args.resume:raise RuntimeError('Checkpoint exists; use --resume to avoid overwriting a run')
    state=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    start_epoch=state['epoch'];history=state['history'];best=state['best'];elapsed_previous=state.get('elapsed_seconds',0.)
    torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
start=time.perf_counter()
config=json.loads((ART/'reproduction_config.json').read_text(encoding='utf-8-sig'))
(target/'run_configuration.json').write_text(json.dumps({'model':args.model,'run_name':run_name,'weight_decay':(1e-4 if args.weight_decay is None else args.weight_decay) if args.model=='lstm' else 0.,'config':config,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'fresh_initialization':not args.resume,'batch_size':128,'epochs':epochs},indent=2))
for epoch in range(start_epoch,epochs):
    begin=time.perf_counter();model.train();loss_sum=0.;loss_weighted=0.;count=0;correct=0
    for batch,(xb,yb) in enumerate(train_loader):
        xb=xb.to(device);yb=yb.to(device);optimizer.zero_grad();out=model(xb);loss=criterion(out,yb)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward();optimizer.step();loss_sum+=float(loss);loss_weighted+=float(loss)*len(xb);count+=len(xb)
        if args.model=='stgcn':correct+=int((out.argmax(1)==yb.argmax(1)).sum())
    val=evaluate();record={'epoch':epoch+1,'train_loss_sum':loss_sum,'train_loss_mean':loss_weighted/count,'validation_loss_sum':val['loss_sum'],'validation_loss_mean':val['loss_mean'],'epoch_seconds':time.perf_counter()-begin}
    if args.model=='stgcn':record.update(train_accuracy=correct/count,validation_accuracy=val['accuracy'])
    history.append(record)
    with (target/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(record));w.writeheader();w.writerows(history)
    score=val['accuracy'] if args.model=='stgcn' else -val['loss_mean']
    if score>best:
        best=score;torch.save(model.state_dict(),target/'best_state_dict.pt')
        (target/'best_validation.json').write_text(json.dumps({'epoch':epoch+1,**val},indent=2))
    progress={'model':args.model,'run_name':run_name,'epoch':epoch+1,'epochs':epochs,'last':record,'best_score':best,'elapsed_seconds':elapsed_previous+time.perf_counter()-start,'phase':'training'}
    (ART/'training_progress.json').write_text(json.dumps(progress,indent=2));print('EPOCH '+args.model+' '+json.dumps(record),flush=True)
    if (epoch+1)%5==0 or epoch==0 or epoch+1==epochs:
        state={'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'best':best,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'elapsed_seconds':progress['elapsed_seconds']}
        tmp=checkpoint.with_name('checkpoint_latest.tmp.pt');torch.save(state,tmp);tmp.replace(checkpoint)
    if args.model=='stgcn' and epoch+1==100:
        torch.save(model.state_dict(),target/'epoch_100_state_dict.pt');(target/'epoch_100_validation.json').write_text(json.dumps(val,indent=2))
torch.save(model.state_dict(),target/'final_state_dict.pt')
final=evaluate();final.update(epochs=epochs,elapsed_seconds=elapsed_previous+time.perf_counter()-start,normalization_saved_in=str(ART/'datasets'/(args.model+'.pt')))
(target/'final_validation.json').write_text(json.dumps(final,indent=2))
(ART/'training_progress.json').write_text(json.dumps({'model':args.model,'phase':'complete','epoch':epochs,'epochs':epochs,'result':final},indent=2))
print('TRAINING_COMPLETE '+args.model+' '+json.dumps(final),flush=True)
