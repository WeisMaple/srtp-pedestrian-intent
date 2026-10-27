import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg'
import pathlib,sys,json,argparse,collections,shutil
import tempfile
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
ART=pathlib.Path(r'D:\SRTP_reproduction_20261027');torch.set_num_threads(2)

def load_sequences(paths):
    sequences=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['meta']))
            for i,m in enumerate(meta):sequences.append({'meta':m,'poses':z[f'p{i}'].copy(),'frames':z[f'f{i}'].copy()})
    return sorted(sequences,key=lambda r:(r['meta']['video'],'data_'+str(r['meta']['track_id'])+'.npy'))

def assemble(sequences):
    cls=[];y=[];cm=[];lx=[];ly=[];lm=[]
    nonempty=[r['poses'] for r in sequences if len(r['poses'])]
    if not nonempty:raise RuntimeError('No matched skeleton observations')
    all_poses=np.concatenate(nonempty,axis=0)
    lmean=all_poses.mean(0);lstd=all_poses.std(0)+1e-8
    for seq in sequences:
        poses=seq['poses'];frames=seq['frames'];meta=seq['meta']
        for start in range(0,len(poses)-49,10):
            cls.append(poses[start:start+50][None]);y.append(meta['class'])
            cm.append({**meta,'start_pose':start,'first_frame':int(frames[start]),'last_frame':int(frames[start+49])})
        normalized=(poses-lmean)/lstd
        for start in range(0,len(poses)-59,10):
            lx.append(normalized[start:start+50]);ly.append(normalized[start+50:start+60])
            lm.append({**meta,'start_pose':start,'first_frame':int(frames[start]),'last_frame':int(frames[start+59])})
    if len(cls)<2 or not lx:raise RuntimeError('Not enough 50/60-observation windows')
    x=torch.from_numpy(np.array(cls,dtype=np.float32)).permute(0,4,2,3,1)
    cmean=x.mean(0,keepdim=True);cstd=x.std(0,keepdim=True);cstd[cstd==0]=1e-6
    x=(x-cmean)/cstd
    return {'stgcn':{'x':x.contiguous(),'y':torch.tensor(y,dtype=torch.long),'mean':cmean,'std':cstd,'metadata':cm},'lstm':{'x':torch.from_numpy(np.array(lx,dtype=np.float32)).reshape(-1,50,51),'y':torch.from_numpy(np.array(ly,dtype=np.float32)),'mean':torch.from_numpy(lmean),'std':torch.from_numpy(lstd),'metadata':lm},'people':len(sequences),'observations':len(all_poses)}

def validate_adapter():
    from DataSet import SkeletonDataset,LSTM_SkeletonDataset
    paths=sorted((ART/'features').glob('*.npz'))
    available=load_sequences(paths)
    selected=[]
    for cls in range(3):
        matches=[r for r in available if r['meta']['class']==cls and len(r['poses'])>=60]
        if matches:selected.append(matches[0])
    shorts=[r for r in available if 0<len(r['poses'])<50]
    if shorts:selected.append(shorts[0])
    sequences=sorted(selected,key=lambda r:(r['meta']['video'],'data_'+str(r['meta']['track_id'])+'.npy'))
    if not sequences:raise RuntimeError('Need completed feature files for adapter validation')
    fixtures=ART/'adapter_validations';fixtures.mkdir(exist_ok=True)
    reference_root=pathlib.Path(tempfile.mkdtemp(prefix='all_classes_',dir=str(fixtures)))
    for r in sequences:
        m=r['meta'];video=m['video'];name='data_'+str(m['track_id'])
        folder=reference_root/'data/final_data'/video;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/(name+'.npy'),[{'skleton':p} for p in r['poses']])
        intent=reference_root/'data/intention_data'/video;intent.mkdir(parents=True,exist_ok=True)
    for video in {r['meta']['video'] for r in sequences}:
        entries=[{'data_'+str(r['meta']['track_id']):r['meta']['crossing']} for r in sequences if r['meta']['video']==video]
        (reference_root/'data/intention_data'/video/'intention_data.json').write_text(json.dumps(entries))
    previous=os.getcwd();os.chdir(reference_root)
    try:
        original_cls=SkeletonDataset(str(reference_root/'data/final_data'),3,2)
        original_lstm=LSTM_SkeletonDataset(str(reference_root/'data/final_data'))
    finally:os.chdir(previous)
    current=assemble(sequences)
    comparisons={
        'STGCN_features_max_abs':float((original_cls.data-current['stgcn']['x']).abs().max()),
        'STGCN_labels_equal':bool(torch.equal(original_cls.labels.argmax(1),current['stgcn']['y'])),
        'classes_tested':sorted(set(r['meta']['class'] for r in sequences)),
        'short_sequence_tested':any(0<len(r['poses'])<50 for r in sequences),
        'LSTM_inputs_max_abs':float((original_lstm.all_inputs.reshape(-1,50,51)-current['lstm']['x']).abs().max()),
        'LSTM_targets_max_abs':float((original_lstm.all_outputs.reshape(-1,10,17,3)-current['lstm']['y']).abs().max())}
    assert comparisons['STGCN_labels_equal']
    assert max(v for k,v in comparisons.items() if k.endswith('_abs'))<1e-5,comparisons
    (ART/'adapter_validation_results.json').write_text(json.dumps(comparisons,indent=2))
    print(json.dumps(comparisons),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.validate_only:validate_adapter();raise SystemExit(0)
    paths=sorted((ART/'features').glob('video_*.npz'))
    expected=sorted(p.stem for p in (ROOT/'data/JAAD_clips').glob('*.mp4'))
    if [p.stem for p in paths]!=expected:raise RuntimeError('Refusing partial dataset: '+str(len(paths))+' / '+str(len(expected)))
    validate_adapter();sequences=load_sequences(paths);data=assemble(sequences)
    target=ART/'datasets';target.mkdir(exist_ok=True)
    summary={'videos':len(paths),'matched_people':data['people'],'skeleton_observations':data['observations'],'classes':{str(i):int((data['stgcn']['y']==i).sum()) for i in range(3)}}
    for name in ['stgcn','lstm']:
        path=target/(name+'.pt');tmp=target/(name+'.tmp.pt');torch.save(data[name],tmp);tmp.replace(path)
        summary[name]={'samples':len(data[name]['x']),'input_shape':list(data[name]['x'].shape),'target_shape':list(data[name]['y'].shape)}
    (ART/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    print('DATASETS_READY '+json.dumps(summary),flush=True)

