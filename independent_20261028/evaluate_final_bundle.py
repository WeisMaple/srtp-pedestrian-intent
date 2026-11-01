"""Score frozen final bundle; official test videos overlap classifier training."""
import json,time,hashlib
from pathlib import Path
import experiment as e

def main():
    out=Path('D:/SRTP_presentation_20261029/evaluation');out.mkdir(parents=True,exist_ok=True)
    weights=e.ROOT/'weights/final'
    data,refs=e.causal_test('modern')
    model=e.BehaviorPredictor(weights/'stgcn.pt',weights/'lstm.pt','cuda:0','legacy_pair')
    predictions=[None]*len(data['keys']);start=time.perf_counter()
    for video in sorted(refs):
        ids=[i for i,k in enumerate(data['keys']) if k[0]==video]
        for j in range(0,len(ids),32):
            ix=ids[j:j+32];p=model.predict(data['x'][ix].numpy(),refs[video])['class_ids']
            for i,label in zip(ix,p):predictions[i]=int(label)
    result=e.metrics(data['y'].tolist(),predictions)
    result.update(eligible=data['eligible'],coverage=len(predictions)/data['eligible'],videos=len(e.SPLITS['test']),
        scope='Overlapping-data diagnostic: classifier trained on windows from all JAAD videos. NOT an independent generalization test.',
        frontend='Reused YOLO26s-Pose/ByteTrack cached detections, current-frame IoU>=0.5; evaluate every 10 frames after 50 observations',
        target='JAAD pedestrian crossing attribute, NOT a measured future crossing-event lead time',
        behavior_batch_seconds=time.perf_counter()-start,
        sha256={f:hashlib.sha256((weights/f).read_bytes()).hexdigest() for f in ['stgcn.pt','lstm.pt']})
    rows=[dict(video=k[0],person=k[1],frame=k[2],truth=int(y),prediction=p) for k,y,p in zip(data['keys'],data['y'],predictions)]
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
