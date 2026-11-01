import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,random,hashlib,runpy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(r'F:\Artificial_Intelligence\srtp');sys.path.insert(0,str(ROOT))
from st_gcn import SkeletonLSTM
O=Path(__file__).resolve().parent;OLD=Path(r'D:\SRTP_lstm_retrain_20261101')
torch.set_num_threads(2)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
def save(name,x): (O/name).write_text(json.dumps(x,indent=2),'utf8')
def log(x): print(json.dumps(x),flush=True)
def seed():
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
data=torch.load(OLD/'train_val.pt',weights_only=False)
st=torch.load(OLD/'lstm/stats.pt',weights_only=True)
mean,std=st['lstm_mean'].cuda(),st['lstm_std'].cuda()
tr,va=data['train'],data['val']
def valid(y):return (y[...,2]>=.2)&(y[...,0]>0)&(y[...,1]>0)
def metrics(p,y,x):
    err=((p-y)/std)**2;mask=valid(y)
    return dict(xy_mse=err[...,:2].mean().item(),confidence_mse=err[...,2].mean().item(),all_mse=err.mean().item(),
        reliable_xy_mse=err[...,:2][mask].mean().item(),pixel_rmse=(((p-y)[...,:2][mask]**2).mean().sqrt()).item())
@torch.inference_mode()
def predict(model,x,mode,horizon=10):
    pieces=[];series=x
    for k in range(horizon//10):
        anchor=mean if mode=='absolute' else series[:,-1:]
        p=model(((series-anchor)/std).reshape(-1,50,51))*std+anchor
        pieces.append(p);series=torch.cat((series[:,10:],p),1)
    return torch.cat(pieces,1)
@torch.inference_mode()
def evaluate(model,d,mode,horizon=10):
    ps=[]
    for x in d['lx'].split(64):ps.append(predict(model,x.cuda(),mode,horizon).cpu())
    p=torch.cat(ps).cuda();y=d['ly'].cuda();x=d['lx'].cuda()
    return metrics(p,y,x),p.cpu()
old=SkeletonLSTM().cuda().eval();old.load_state_dict(torch.load(OLD/'lstm/best.pt',weights_only=True))
diag={}
for split,d in [('train',tr),('val',va)]:
    x,y=d['lx'].cuda(),d['ly'].cuda();m,p=evaluate(old,d,'absolute')
    diag[split]=dict(samples=len(x),baseline_lstm=m,persistence=metrics(x[:,-1:].expand_as(y),y,x),
        unreliable_target_fraction=(~valid(y)).float().mean().item(),zero_xy_fraction=(y[...,:2]==0).float().mean().item())
    with torch.inference_mode():
        z=((x-mean)/std).reshape(-1,50,51)
        a=torch.cat([old(b) for b in z.split(64)])
        repeated=z[:,-1:].expand_as(z).contiguous()
        b=torch.cat([old(q) for q in repeated.split(64)])
        diag[split]['repeat_last_input_prediction_change_mse']=(a-b).square().mean().item()
save('baseline_diagnostics.json',diag);log(diag)
results={'absolute_existing':diag['val']['baseline_lstm'],'persistence':diag['val']['persistence']}
save('protocol.json',dict(seed=2026,epochs=100,learning_rate=.001,batch_size=128,gradient_clip=1,
    selection='minimum validation all-joint XY MSE',test_used=False,
    architecture_sha256=hashlib.sha256((ROOT/'st_gcn.py').read_bytes()).hexdigest(),
    variants=['absolute_xy','anchored_mse','anchored_reliable'],
    loss={'absolute_xy':'xy_mse + 0.1 * confidence_mse','anchored_mse':'all channels MSE',
    'anchored_reliable':'target-valid XY MSE + 0.1 * confidence MSE'},
    anchoring='subtract last observed 17x3 pose from every history and target; decode by adding anchor; reanchor each recursive block',
    stats='unchanged fixed training-only joint/channel standard deviations',
    network='original SkeletonLSTM; unchanged forward, layers and parameter shapes'))
for mode in ['absolute_xy','anchored_mse','anchored_reliable']:
    dest=O/mode;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        results[mode]=json.loads((dest/'complete.json').read_text());continue
    seed();model=SkeletonLSTM().cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tr['lx'],tr['ly']),batch_size=128,shuffle=True)
    hist=[];best=1e9;begin=time.perf_counter()
    for epoch in range(1,101):
        model.train();loss_sum=0
        for x,y in loader:
            x,y=x.cuda(),y.cuda();anchor=mean if mode=='absolute_xy' else x[:,-1:]
            z=((x-anchor)/std).reshape(-1,50,51);target=(y-anchor)/std
            opt.zero_grad();p=model(z);err=(p-target).square()
            if mode=='anchored_mse':loss=err.mean()
            elif mode=='absolute_xy':loss=err[...,:2].mean()+.1*err[...,2].mean()
            else:loss=err[...,:2][valid(y)].mean()+.1*err[...,2].mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_sum+=loss.item()*len(x)
        model.eval();m,_=evaluate(model,va,'absolute' if mode=='absolute_xy' else 'anchored')
        row=dict(epoch=epoch,train_loss=loss_sum/len(tr['lx']),**m);hist.append(row)
        if m['xy_mse']<best:
            best=m['xy_mse'];torch.save(model.state_dict(),dest/'best.pt');save(mode+'/best.json',row)
        if epoch%10==0:
            log(dict(mode=mode,seconds=time.perf_counter()-begin,**row));save(mode+'/history.json',hist)
    b=json.loads((dest/'best.json').read_text());b['seconds']=time.perf_counter()-begin
    save(mode+'/complete.json',b);results[mode]=b
    del model,opt;torch.cuda.empty_cache()
save('validation_comparison.json',results)
# Build full-horizon validation observations using exactly the previous causal segment builder.
ns=runpy.run_path(str(ROOT/'independent_20261028/retrain_paired_20261101.py'),run_name='diagnostic_helpers')
long=ns['pose_pairs'](ns['SPLITS']['val'],horizon=50);torch.save(long,O/'validation_50.pt')
horizon={}
for name in ['absolute_existing','persistence','absolute_xy','anchored_mse','anchored_reliable']:
    if name=='persistence':p=long['lx'][:,-1:].expand_as(long['ly'])
    else:
        model=SkeletonLSTM().cuda().eval()
        path=OLD/'lstm/best.pt' if name=='absolute_existing' else O/name/'best.pt'
        model.load_state_dict(torch.load(path,weights_only=True));_,p=evaluate(model,long,'absolute' if name.startswith('absolute') else 'anchored',50)
    blocks=[]
    for k in range(5):
        sl=slice(k*10,(k+1)*10);blocks.append(metrics(p[:,sl].cuda(),long['ly'][:,sl].cuda(),long['lx'].cuda()))
    horizon[name]=blocks
save('validation_horizon.json',dict(samples=len(long['lx']),results=horizon));log(horizon)
assert hashlib.sha256((ROOT/'st_gcn.py').read_bytes()).hexdigest()==json.loads((O/'protocol.json').read_text())['architecture_sha256']
save('complete.json',dict(test_used=False,architecture_unchanged=True,validation_only=True))
