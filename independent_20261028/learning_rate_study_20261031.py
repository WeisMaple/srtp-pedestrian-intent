"""Controlled ST-GCN learning-rate study using the saved internal split.

Each run restarts from the same initialization and uses the same minibatch
order. Original demonstration weights are never modified.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, json, time, csv, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Subset
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root',type=Path,default=Path(r'F:\Artificial_Intelligence\srtp'))
parser.add_argument('--dataset-root',type=Path,default=Path(r'D:\SRTP_modern_internal_20261028'))
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'learning_rate_results')
args=parser.parse_args()
ROOT = args.project_root
sys.path.insert(0, str(ROOT))
from st_gcn import Model
from independent_20261028.evaluate_internal_metrics import summarize

W = args.output
W.mkdir(parents=True,exist_ok=True)
SOURCE = args.dataset_root
DATA = SOURCE / 'datasets/stgcn.pt'
saved_split = json.loads((SOURCE/'models/stgcn/split.json').read_text())
data = torch.load(DATA, map_location='cpu', weights_only=True)
x, y = data['x'], data['y']
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.is_available()
device = torch.device('cuda:0')
config = dict(learning_rates=[0.0005,0.001,0.002], seed=2026, epochs=125,
    batch_size=128, optimizer='SGD', momentum=0., weight_decay=0.,
    loss='CrossEntropyLoss', selection='highest validation accuracy; earliest tie',
    train_samples=796, validation_samples=200,
    scope='internal overlapping-window validation; normalization fitted before split',
    dataset=str(DATA), dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
    model_source_sha256=hashlib.sha256((ROOT/'st_gcn.py').read_bytes()).hexdigest(),
    device=torch.cuda.get_device_name(), torch=torch.__version__,
    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    identical_initialization=True, identical_minibatch_order=True)
(W/'configuration.json').write_text(json.dumps(config,indent=2),'utf8')
(W/'split.json').write_text(json.dumps(saved_split,indent=2),'utf8')
results=[]
for lr in config['learning_rates']:
    run=W/('lr_'+str(lr));run.mkdir(exist_ok=True)
    if (run/'result.json').exists():
        results.append(json.loads((run/'result.json').read_text()));continue
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    ids=torch.randperm(len(x));tr=ids[:796];va=ids[796:]
    assert tr.tolist()==saved_split['train_indices'] and va.tolist()==saved_split['validation_indices']
    full=TensorDataset(x,nn.functional.one_hot(y,3).float())
    train=DataLoader(Subset(full,tr),batch_size=128,shuffle=True,num_workers=0)
    valid=DataLoader(Subset(full,va),batch_size=128,shuffle=False,num_workers=0)
    model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).to(device)
    h=hashlib.sha256()
    for value in model.state_dict().values():h.update(value.detach().cpu().numpy().tobytes())
    initial_hash=h.hexdigest()
    opt=torch.optim.SGD(model.parameters(),lr=lr);criterion=nn.CrossEntropyLoss()
    best=-1.;history=[];start=time.perf_counter()
    for epoch in range(1,126):
        model.train();train_loss=0.;correct=0
        for xb,yb in train:
            xb=xb.to(device);yb=yb.to(device);opt.zero_grad()
            out=model(xb);loss=criterion(out,yb)
            assert torch.isfinite(loss)
            loss.backward();opt.step()
            train_loss+=loss.item()*len(xb);correct+=(out.argmax(1)==yb.argmax(1)).sum().item()
        model.eval();vloss=0.;cm=np.zeros((3,3),dtype=np.int64);preds=[]
        with torch.no_grad():
            for xb,yb in valid:
                xb=xb.to(device);yb=yb.to(device);out=model(xb)
                vloss+=criterion(out,yb).item()*len(xb)
                truth=yb.argmax(1).cpu().numpy();pred=out.argmax(1).cpu().numpy()
                cm+=np.bincount(3*truth+pred,minlength=9).reshape(3,3)
                preds.extend(pred.tolist())
        metrics=summarize(cm)
        row=dict(epoch=epoch,train_loss=train_loss/796,validation_loss=vloss/200,
                 train_accuracy=correct/796,validation_accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
        history.append(row)
        if metrics['accuracy']>best:
            best=metrics['accuracy'];chosen=dict(epoch=epoch,validation_loss=vloss/200,**metrics)
            torch.save(model.state_dict(),run/'best.pt')
            (run/'best_predictions.json').write_text(json.dumps(dict(indices=va.tolist(),truth=y[va].tolist(),prediction=preds)),'utf8')
        with (run/'history.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(row));wr.writeheader();wr.writerows(history)
        if epoch%10==0 or epoch==1:
            print(json.dumps(dict(lr=lr,elapsed_seconds=time.perf_counter()-start,**row)),flush=True)
    # Independently reload the selected checkpoint and check its confusion matrix.
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True));model.eval()
    recheck=np.zeros((3,3),dtype=np.int64)
    with torch.inference_mode():
        for xb,yb in valid:
            truth=yb.argmax(1).numpy();pred=model(xb.to(device)).argmax(1).cpu().numpy()
            recheck+=np.bincount(3*truth+pred,minlength=9).reshape(3,3)
    assert recheck.tolist()==chosen['confusion_matrix']
    result=dict(learning_rate=lr,initial_state_sha256=initial_hash,elapsed_seconds=time.perf_counter()-start,
                checkpoint_verified=True,selected=chosen,final=history[-1])
    (run/'result.json').write_text(json.dumps(result,indent=2),'utf8');results.append(result)
    del model,opt;torch.cuda.empty_cache()
assert len({r['initial_state_sha256'] for r in results})==1
(W/'summary.json').write_text(json.dumps(dict(configuration=config,results=results),indent=2),'utf8')
print('COMPLETE '+json.dumps([dict(lr=r['learning_rate'],epoch=r['selected']['epoch'],accuracy=r['selected']['accuracy'],macro_f1=r['selected']['macro_f1']) for r in results]),flush=True)
