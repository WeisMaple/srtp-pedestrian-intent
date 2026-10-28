"""Local balanced training with the original LSTM and ST-GCN architectures."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,sys,json
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modern_pipeline.pose_filter import PoseFilter
source=ROOT/'independent_20261028/experiment.py'
text=source.read_text(encoding='utf-8')
text=text.replace("loss_fn=torch.nn.CrossEntropyLoss() if kind=='stgcn' else torch.nn.MSELoss()", "counts=torch.bincount(data['train']['y'],minlength=3).float().clamp_min(1); weights=counts.rsqrt(); weights=weights/weights.mean()\n    loss_fn=torch.nn.CrossEntropyLoss(weight=weights.cuda()) if kind=='stgcn' else torch.nn.MSELoss()")
ns={'__name__':'classroom_training','__file__':str(source)};exec(compile(text,str(source),'exec'),ns)
original_out=ns['OUT'];raw=ns['raw'];out=pathlib.Path(r'D:\SRTP_classroom_20261028');out.mkdir(exist_ok=True)
def filtered_raw(video,variant):
    ns['OUT']=original_out
    try:rows,poses=raw(video,variant)
    finally:ns['OUT']=out
    f=PoseFilter();poses=poses.copy()
    for i,row in enumerate(rows):poses[i]=f.update(int(row[1]),int(row[0]),poses[i])
    return rows,poses
ns['OUT']=out;ns['raw']=filtered_raw
(out/'split.json').write_text(json.dumps(ns['SPLITS'],indent=2))
(out/'configuration.json').write_text(json.dumps(dict(pose_filter='causal EMA alpha=.7, missing joint hold<=2 frames',class_weights='inverse square root of training class counts',selection='validation weighted loss',architecture='unchanged original ST-GCN and LSTM',test_split='unchanged official test videos'),indent=2))
data,stats=ns['build']('modern')
for kind in ['stgcn','lstm']:ns['train']('modern',data,stats,kind)
ns['evaluate']('modern',data)
(out/'complete.json').write_text(json.dumps(dict(status='complete',default_demo_weights_changed=False)))
