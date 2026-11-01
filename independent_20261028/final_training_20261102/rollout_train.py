"""Fine tune the same LSTM on its five-step recursive use, validation only."""
from pathlib import Path
source=Path(__file__).with_name('investigate.py').read_text('utf8')
exec(compile(source.split('\nold=SkeletonLSTM')[0],str(Path(__file__).with_name('investigate.py')),'exec'))
assert (O/'controls_complete.json').exists()
ns=runpy.run_path(str(ROOT/'independent_20261028/retrain_paired_20261101.py'),run_name='rollout_helpers')
tr50=ns['pose_pairs'](ns['SPLITS']['train'],horizon=50)
va50=torch.load(O/'validation_50.pt',weights_only=False)
dest=O/'anchored_rollout';dest.mkdir(exist_ok=True)
seed();model=SkeletonLSTM().cuda();model.load_state_dict(torch.load(O/'anchored_reliable/best.pt',weights_only=True))
opt=torch.optim.Adam(model.parameters(),lr=.0001)
loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tr50['lx'],tr50['ly']),batch_size=64,shuffle=True)
model.eval();baseline,_=evaluate(model,va50,'anchored',50);best=baseline['xy_mse']
torch.save(model.state_dict(),dest/'best.pt');hist=[dict(epoch=0,**baseline)]
save('anchored_rollout/protocol.json',dict(epochs=30,lr=.0001,batch_size=64,clip=1,train_samples=len(tr50['lx']),val_samples=len(va50['lx']),
    initialization='anchored_reliable best epoch13',selection='full50 validation all-joint XY MSE, including epoch0',
    loss='valid-target XY MSE + 0.1 confidence MSE over all50 future observations; backprop through all five recursive calls',test_used=False))
save('anchored_rollout/best.json',hist[0]);begin=time.perf_counter()
for epoch in range(1,31):
    model.train();total=0
    for x,y in loader:
        x,y=x.cuda(),y.cuda();series=x;pieces=[];opt.zero_grad()
        for k in range(5):
            anchor=series[:,-1:];p=model(((series-anchor)/std).reshape(-1,50,51))*std+anchor
            pieces.append(p);series=torch.cat((series[:,10:],p),1)
        p=torch.cat(pieces,1);err=((p-y)/std).square()
        loss=err[...,:2][valid(y)].mean()+.1*err[...,2].mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(x)
    model.eval();m,_=evaluate(model,va50,'anchored',50)
    row=dict(epoch=epoch,train_loss=total/len(tr50['lx']),**m);hist.append(row)
    if m['xy_mse']<best:
        best=m['xy_mse'];torch.save(model.state_dict(),dest/'best.pt');save('anchored_rollout/best.json',row)
    if epoch%5==0:log(dict(seconds=time.perf_counter()-begin,**row));save('anchored_rollout/history.json',hist)
model.load_state_dict(torch.load(dest/'best.pt',weights_only=True));model.eval()
m10,_=evaluate(model,va,'anchored');m50,p=evaluate(model,va50,'anchored',50)
blocks=[metrics(p[:,k*10:(k+1)*10].cuda(),va50['ly'][:,k*10:(k+1)*10].cuda(),va50['lx'].cuda()) for k in range(5)]
save('anchored_rollout/results.json',dict(before50=baseline,after50=m50,after10=m10,blocks=blocks,seconds=time.perf_counter()-begin))
save('anchored_rollout/history.json',hist)
save('anchored_rollout/complete.json',dict(test_used=False,architecture_unchanged=True))
