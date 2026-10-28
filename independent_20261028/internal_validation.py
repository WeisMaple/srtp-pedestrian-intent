"""Reproduce the old window-validation protocol using YOLO26 pose caches.

This intentionally includes overlapping windows and pre-split normalization.
It is an internal-validation experiment, never an independent test result.
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,sys,json,ast,collections,hashlib,subprocess,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from reproduction_20261027.build_training_datasets import assemble
OUT=pathlib.Path(r'D:\SRTP_modern_internal_20261028')
CACHE=pathlib.Path(r'D:\SRTP_independent_20261028\modern_raw')
OUT.mkdir(exist_ok=True);(OUT/'datasets').mkdir(exist_ok=True)
torch.set_num_threads(2)

if not (OUT/'datasets/stgcn.pt').exists():
    # Import the two unchanged pure matching functions without importing legacy TF.
    source=(ROOT/'DataProcess.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'calculate_iou','match_bboxes'}]
    assert len(functions)==2
    ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(ROOT/'DataProcess.py'),'exec'),ns)
    seqs=[];inventory=[]
    paths=sorted(CACHE.glob('video_*.npz'));assert len(paths)==346
    for path in paths:
        video=path.stem
        labels={p.get('id'):int(p.get('crossing')) for p in ET.parse(ROOT/'data/annotations_attributes'/f'{video}_attributes.xml').getroot().findall('pedestrian')}
        database=collections.OrderedDict()
        for tr in ET.parse(ROOT/'data/annotations'/f'{video}.xml').getroot().findall('track'):
            boxes=tr.findall('box')
            if not boxes:continue
            pid=boxes[0].findtext("attribute[@name='id']")
            database[pid]=dict(bbox=[[float(b.get(k)) for k in ['xtl','ytl','xbr','ybr']] for b in boxes],frames=[int(b.get('frame')) for b in boxes])
        with np.load(path) as z:rows=z['rows'];poses=z['poses']
        first=collections.OrderedDict()
        for row in rows:
            tid=int(row[1]);first.setdefault(tid,[])
            if len(first[tid])<=10:first[tid].append(row[2:].astype(int).tolist())
        pairs,_=ns['match_bboxes'](database,first)
        accepted={int(tid):(pid,float(score)) for pid,tid,score in pairs if score>.5 and pid.endswith('b') and pid in labels}
        for tid,(pid,score) in accepted.items():
            validframes=set(database[pid]['frames'])
            ids=[i for i,r in enumerate(rows) if int(r[1])==tid and int(r[0]) in validframes]
            meta=dict(video=video,track_id=tid,person_id=pid,crossing=labels[pid],match_iou=score,observations=len(ids))
            meta['class']={-1:0,0:1,1:2}[labels[pid]]
            seqs.append(dict(meta=meta,poses=poses[ids].copy(),frames=rows[ids,0].astype(np.int32)))
        inventory.append(dict(video=video,tracks=len(first),accepted_tracks=len(accepted)))
    seqs.sort(key=lambda r:(r['meta']['video'],'data_'+str(r['meta']['track_id'])+'.npy'))
    data=assemble(seqs);torch.save(data['stgcn'],OUT/'datasets/stgcn.pt')
    summary=dict(videos=len(paths),people=data['people'],observations=data['observations'],windows=len(data['stgcn']['x']),classes=torch.bincount(data['stgcn']['y'],minlength=3).tolist())
    (OUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'matching_inventory.json').write_text(json.dumps(inventory));print('DATASET',summary,flush=True)

config=json.loads((ROOT/'reproduction_20261027/reproduction_config.json').read_text(encoding='utf-8-sig'))
config['artifact_root']=str(OUT);config['data'].update(detector='YOLO26s-pose',tracker='ByteTrack',pose='YOLO26s-pose full-frame',detector_conf=.1,pose_batch=8)
config['data']['cache']=str(CACHE);config['purpose']='Paper/code-style ST-GCN internal validation, intentionally not an independent test'
(OUT/'reproduction_config.json').write_text(json.dumps(config,indent=2))
# Execute an archived copy of the exact earlier training script. Only paths change.
trainer=ROOT/'reproduction_20261027/train_reproduction.py';original=trainer.read_text(encoding='utf-8-sig')
old_art="ART=pathlib.Path(r'D:\\SRTP_reproduction_20261027')"
old_root="ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))"
assert original.count(old_art)==1 and original.count(old_root)==1
text=original.replace(old_art,'ART=pathlib.Path('+repr(str(OUT))+')').replace(old_root,'ROOT=pathlib.Path('+repr(str(ROOT))+');sys.path.insert(0,str(ROOT))')
snapshot=OUT/'train_used.py';snapshot.write_text(text,encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps(dict(original_training_script=str(trainer),original_sha256=hashlib.sha256(original.encode()).hexdigest(),only_training_script_changes=['ROOT path','ART path'],normalization='all windows before random split, same as old protocol',warning='Window overlap and normalization leakage are intentionally retained; never compare as an independent generalization score'),indent=2))
if not (OUT/'models/stgcn/final_validation.json').exists():
    command=[sys.executable,'-s',str(snapshot),'--model','stgcn']
    if (OUT/'models/stgcn/checkpoint_latest.pt').exists():command.append('--resume')
    subprocess.run(command,cwd=ROOT,check=True)
