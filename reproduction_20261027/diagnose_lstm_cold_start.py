import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,sys,json,torch
ROOT=pathlib.Path(r'F:\Artificial_Intelligence\srtp');sys.path.insert(0,str(ROOT));ART=pathlib.Path(r'D:\SRTP_reproduction_20261027');torch.set_num_threads(2)
from st_gcn import SkeletonLSTM
d=torch.load(ART/'datasets/lstm.pt',map_location='cpu',weights_only=True)
ids=json.loads((ART/'models/lstm/split.json').read_text())['validation_indices'][:16]
x=d['x'][ids];y=d['y'][ids];result={}
for name,path in [('source_decay_from_scratch',ART/'models/lstm/final_state_dict.pt'),('supplied_original',ROOT/'Models/lstm_gpu_pre.pth')]:
    model=SkeletonLSTM();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.train();out=model(x);loss=torch.nn.functional.mse_loss(out,y);loss.backward()
    result[name]={'MSE_on_16_windows':float(loss),'mean_output_std_across_different_inputs':float(out.detach().std(0).mean()),'layer0_input_weight_norm':float(model.lstm.weight_ih_l0.detach().norm()),'layer0_input_gradient_norm':float(model.lstm.weight_ih_l0.grad.norm()),'layer9_input_gradient_norm':float(model.lstm.weight_ih_l9.grad.norm())}
(ART/'LSTM_cold_start_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
