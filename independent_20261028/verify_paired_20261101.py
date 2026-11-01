"""Verify the paired retraining artifact and benchmark identical video inputs."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from modern_pipeline.behavior import BehaviorPredictor
from modern_pipeline.cli import arguments,run

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(r'D:\SRTP_lstm_retrain_20261101'))
parser.add_argument('--skip-video',action='store_true')
args=parser.parse_args();OUT=args.output
torch.set_num_threads(2)
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
load=lambda n:json.loads((OUT/n).read_text('utf8'))
split=load('split.json');data=torch.load(OUT/'train_val.pt',weights_only=False);features=torch.load(OUT/'features.pt',weights_only=False)
assert not any(set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')])
for s in ['train','val']:
    assert set(k[0] for k in data[s]['keys']).issubset(split[s])
    assert set(k[0] for k in data[s]['lkeys']).issubset(split[s])
folds=load('forecast_folds.json');all_held=[]
for i,held in enumerate(folds):
    fit=load(f'fold{i}/complete.json')['fit_videos'];assert not(set(held)&set(fit));all_held+=held
assert len(all_held)==len(set(all_held)) and set(all_held)==set(split['train'])
stats=torch.load(OUT/'lstm/stats.pt',weights_only=True)
assert torch.allclose(stats['lstm_mean'],data['train']['lx'].mean((0,1)),atol=1e-6)
assert torch.allclose(stats['lstm_std'],data['train']['lx'].std((0,1),correction=0).clamp_min(1e-6),atol=1e-6)
checks={}
for branch in ['observed','forecast']:
    st=torch.load(OUT/branch/'normalization.pt',weights_only=True)
    for k in ['stgcn_mean','stgcn_std']:
        assert torch.isfinite(st[k]).all() and torch.equal(st[k],st[k][:,:,:1].expand_as(st[k]))
    if branch=='observed':
        raw=data['train']['x'];mean=raw.mean((0,1));std=raw.std((0,1),correction=0).clamp_min(1e-6)
        assert torch.allclose(st['stgcn_mean'][0,:,0,:,0],mean.T,atol=1e-6)
        assert torch.allclose(st['stgcn_std'][0,:,0,:,0],std.T,atol=1e-6)
    m=BehaviorPredictor(OUT/branch/'best.pt',OUT/'lstm/best.pt' if branch=='forecast' else None,'cpu','training_stats',OUT/branch/'normalization.pt')
    x=data['val']['x'][:32].numpy();batch=m.predict(x)['probabilities'];single=np.stack([m.predict(xx[None])['probabilities'][0] for xx in x])
    err=float(np.abs(batch-single).max());assert err<1e-4,err
    checks[branch]=dict(cpu_batch_single_max_error=err)
    rows=load(branch+'/test_predictions.json');truth=np.array([r['truth'] for r in rows]);pred=np.array([r['prediction'] for r in rows])
    cm=np.bincount(3*truth+pred,minlength=9).reshape(3,3)
    assert cm.tolist()==load('comparison.json')['branches'][branch]['confusion_matrix']
    checks[branch]['prediction_file_matches_metrics']=True
    del m
assert [r['key'] for r in load('observed/test_predictions.json')]==[r['key'] for r in load('forecast/test_predictions.json')]
protected=load('protected_weights.json')
assert all(hashlib.sha256((ROOT/'weights/final'/n).read_bytes()).hexdigest()==h for n,h in protected.items())
assert hashlib.sha256((ROOT/'st_gcn.py').read_bytes()).hexdigest()==load('protocol.json')['architecture_sha256']
(OUT/'verification.json').write_text(json.dumps(dict(split_video_disjoint=True,fold_fit_disjoint=True,train_only_stats=True,
    identical_test_keys=True,unchanged_architecture=True,original_weights_unchanged=True,branches=checks),indent=2),'utf8')
print('Verification passed.',flush=True)
if not args.skip_video:
    bench=OUT/'video_benchmark';bench.mkdir(exist_ok=True);results={b:[] for b in ['observed','forecast']}
    for repeat in range(3):
        order=['observed','forecast'] if repeat%2==0 else ['forecast','observed']
        for branch in order:
            output=bench/f'{branch}_{repeat}.mp4';met=output.with_suffix('.metrics.json')
            if met.exists():result=json.loads(met.read_text('utf8'))
            else:
                cmd=['--input',str(ROOT/'data/JAAD_clips/video_0001.mp4'),'--output',str(output),
                    '--pose-model',str(ROOT/'weights/final/yolo26s-pose.pt'),'--device','cuda:0','--behavior-device','cpu',
                    '--task','classify' if branch=='observed' else 'predict','--stgcn',str(OUT/branch/'best.pt'),
                    '--lstm',str(OUT/'lstm/best.pt'),'--normalization','training_stats','--stats',str(OUT/branch/'normalization.pt'),
                    '--max-frames','150','--presentation']
                if repeat>0:cmd.append('--no-video')
                result=run(arguments(cmd))
            assert result['status']=='complete' and result['frames']==150 and result['classification_updates']>0
            results[branch].append(result)
    # First run includes MP4 encoding; repeats 1/2 share no-video output mode.
    summary={b:dict(encoded_run_fps=rr[0]['processing_fps'],
        repeated_no_video_fps=[r['processing_fps'] for r in rr[1:]],
        median_no_video_fps=float(np.median([r['processing_fps'] for r in rr[1:]])),
        prediction_calls=[r['classification_updates'] for r in rr]) for b,rr in results.items()}
    (bench/'summary.json').write_text(json.dumps(dict(input='video_0001.mp4',frames=150,
        pose_device='GPU',behavior_device='CPU',same_detector_and_tracker=True,results=summary),indent=2),'utf8')
    print('VIDEO_BENCHMARK '+json.dumps(summary),flush=True)
