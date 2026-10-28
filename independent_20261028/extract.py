"""Cache full-frame YOLO26 poses and sequential ByteTrack IDs; no labels enter inference."""
import os, sys, pathlib, json, time
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modern_pipeline.bootstrap import configure, ARTIFACT_ROOT
configure()
import torch, cv2, numpy as np
from ultralytics import YOLO
OUT=pathlib.Path(r'D:\SRTP_independent_20261028'); (OUT/'modern_raw').mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
model=YOLO(str(ARTIFACT_ROOT/'weights/yolo26s-pose.pt'))
videos=sorted((ROOT/'data/JAAD_clips').glob('*.mp4')); started=time.perf_counter()
for index,video in enumerate(videos,1):
    dest=OUT/'modern_raw'/(video.stem+'.npz')
    if dest.exists(): continue
    if model.predictor is not None and hasattr(model.predictor,'trackers'):
        for tracker in model.predictor.trackers: tracker.reset()
    cap=cv2.VideoCapture(str(video)); expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frame=0; rows=[]; poses=[]; begin=time.perf_counter()
    while True:
        images=[]
        for _ in range(8):
            ok,img=cap.read()
            if not ok: break
            images.append(img)
        if not images: break
        results=model.track(images,persist=True,tracker='bytetrack.yaml',imgsz=640,conf=.1,device=0,quantize=32,classes=[0],verbose=False,max_det=100)
        for result in results:
            if result.boxes is not None and result.boxes.is_track and result.keypoints is not None:
                for tid,box,pose in zip(result.boxes.id.cpu().numpy(),result.boxes.xyxy.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                    if np.isfinite(pose).all(): rows.append([frame,int(tid),*box]); poses.append(pose)
            frame+=1
    cap.release(); assert frame==expected,(video,frame,expected)
    np.savez_compressed(dest,rows=np.asarray(rows,dtype=np.float32).reshape(-1,6),poses=np.asarray(poses,dtype=np.float32).reshape(-1,17,3),frames=frame)
    progress=dict(completed=index,total=len(videos),video=video.stem,frames=frame,observations=len(rows),video_seconds=time.perf_counter()-begin,elapsed=time.perf_counter()-started)
    (OUT/'extraction_progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
(OUT/'extraction_complete.json').write_text(json.dumps(dict(videos=len(videos),seconds=time.perf_counter()-started)))
