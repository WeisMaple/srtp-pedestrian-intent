"""Matched absolute-coordinate control and two independent seeds of anchored MSE."""
from pathlib import Path
# Load only shared definitions, not the experiment execution block.
source=Path(__file__).with_name('investigate.py').read_text('utf8')
exec(compile(source.split('\nold=SkeletonLSTM')[0],str(Path(__file__).with_name('investigate.py')),'exec'))
assert (O/'complete.json').exists()
long=torch.load(O/'validation_50.pt',weights_only=False)
controls={}
save('controls_protocol.json',dict(variants=[['absolute_mse',2026],['anchored_mse',2027],['anchored_mse',2028]],
    epochs=100,lr=.001,batch_size=128,clip=1,selection='validation all-joint XY MSE',test_used=False))
for mode,s in [('absolute_mse',2026),('anchored_mse',2027),('anchored_mse',2028)]:
    name=mode+'_seed'+str(s);dest=O/name;dest.mkdir(exist_ok=True)
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    model=SkeletonLSTM().cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tr['lx'],tr['ly']),batch_size=128,shuffle=True)
    hist=[];best=1e9;begin=time.perf_counter()
    for epoch in range(1,101):
        model.train();total=0
        for x,y in loader:
            x,y=x.cuda(),y.cuda();anchor=mean if mode=='absolute_mse' else x[:,-1:]
            opt.zero_grad();loss=(model(((x-anchor)/std).reshape(-1,50,51))-(y-anchor)/std).square().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(x)
        model.eval();m,_=evaluate(model,va,'absolute' if mode=='absolute_mse' else 'anchored')
        row=dict(epoch=epoch,train_loss=total/len(tr['lx']),**m);hist.append(row)
        if m['xy_mse']<best:
            best=m['xy_mse'];torch.save(model.state_dict(),dest/'best.pt');save(name+'/best.json',row)
        if epoch%20==0:log(dict(name=name,seconds=time.perf_counter()-begin,**row))
    save(name+'/history.json',hist)
    model.load_state_dict(torch.load(dest/'best.pt',weights_only=True));model.eval()
    _,p=evaluate(model,long,'absolute' if mode=='absolute_mse' else 'anchored',50)
    blocks=[metrics(p[:,k*10:(k+1)*10].cuda(),long['ly'][:,k*10:(k+1)*10].cuda(),long['lx'].cuda()) for k in range(5)]
    controls[name]=dict(best=json.loads((dest/'best.json').read_text()),blocks=blocks,seconds=time.perf_counter()-begin)
    save('controls_results.json',controls)
    del model,opt;torch.cuda.empty_cache()
save('controls_complete.json',dict(test_used=False,architecture_unchanged=hashlib.sha256((ROOT/'st_gcn.py').read_bytes()).hexdigest()==json.loads((O/'protocol.json').read_text())['architecture_sha256']))
