import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,sys,json,torch
ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));ART=pathlib.Path(r'D:\SRTP_reproduction_20261027');torch.set_num_threads(2)
from st_gcn import Model,SkeletonLSTM
c=torch.load(ART/'datasets/stgcn.pt',map_location='cpu',weights_only=True);l=torch.load(ART/'datasets/lstm.pt',map_location='cpu',weights_only=True)
ids=json.loads((ART/'models/stgcn/split.json').read_text())['validation_indices']
g=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).cuda().eval();g.load_state_dict(torch.load(ART/'models/stgcn/final_state_dict.pt',map_location='cuda',weights_only=True))
r=SkeletonLSTM().cuda().eval();r.load_state_dict(torch.load(ART/'models/lstm_wd0/final_state_dict.pt',map_location='cuda',weights_only=True))
cm=torch.zeros(3,3,dtype=torch.long);baseline=torch.zeros(3,3,dtype=torch.long)
mean=c['mean'].cuda();std=c['std'].cuda();lm=l['mean'].cuda();ls=l['std'].cuda()
with torch.inference_mode():
    for start in range(0,len(ids),32):
        ix=ids[start:start+32];truth=c['y'][ix]
        raw=c['x'][ix].cuda()*std+mean
        history=raw.squeeze(-1).permute(0,2,3,1)
        series=(history-lm)/ls
        for _ in range(5):series=torch.cat([series[:,10:],r(series.reshape(-1,50,51))],dim=1)
        future=(series*ls+lm).permute(0,3,1,2).unsqueeze(-1)
        out=g((future-mean)/std).argmax(1).cpu();cm+=torch.bincount(3*truth+out,minlength=9).reshape(3,3)
        hold=raw[:,:,-1:,:,:].expand(-1,-1,50,-1,-1)
        out=g((hold-mean)/std).argmax(1).cpu();baseline+=torch.bincount(3*truth+out,minlength=9).reshape(3,3)
result={'protocol':'Additional diagnostic only: use saved training normalization for both networks; NOT the paper two-sample inference recipe','windows':len(ids),'LSTM_then_STGCN_accuracy':cm.diag().sum().item()/len(ids),'hold_last_then_STGCN_accuracy':baseline.diag().sum().item()/len(ids),'LSTM_confusion_matrix':cm.tolist(),'hold_last_confusion_matrix':baseline.tolist()}
(ART/'training_stats_alignment_ablation.json').write_text(json.dumps(result,indent=2))
results=json.loads((ART/'reproduction_results.json').read_text(encoding='utf-8'));results['nonpaper_training_stats_alignment_ablation']=result;(ART/'reproduction_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
