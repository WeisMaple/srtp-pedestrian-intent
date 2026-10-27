import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,sys,hashlib,torch
ROOT=pathlib.Path(r'F:\Artificial_Intelligence\srtp');sys.path.insert(0,str(ROOT));ART=pathlib.Path(r'D:\SRTP_reproduction_20261027');torch.set_num_threads(2)
from st_gcn import Model
manifest=json.loads((ART/'reproduction_input_manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
m=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).cuda().eval();m.load_state_dict(torch.load(ART/'models/stgcn/final_state_dict.pt',map_location='cuda',weights_only=True))
d=torch.load(ART/'datasets/stgcn.pt',map_location='cpu',weights_only=True);ids=json.loads((ART/'models/stgcn/split.json').read_text())['validation_indices'];cm=torch.zeros(3,3,dtype=torch.long)
with torch.inference_mode():
    for start in range(0,len(ids),128):
        ix=ids[start:start+128];out=m(d['x'][ix].cuda());assert torch.isfinite(out).all();pred=out.argmax(1).cpu();cm+=torch.bincount(3*d['y'][ix]+pred,minlength=9).reshape(3,3)
expected=json.loads((ART/'models/stgcn/final_validation.json').read_text());assert cm.tolist()==expected['confusion_matrix']
result={'original_sources_and_weights_unchanged':True,'saved_STGCN_readback_confusion_matrix_verified':True,'verified_accuracy':float(cm.diag().sum()/cm.sum()),'trained_weight_sha256':{name:hashlib.sha256((ART/'models'/name/'final_state_dict.pt').read_bytes()).hexdigest() for name in ['stgcn','lstm','lstm_wd0']}}
(ART/'final_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
