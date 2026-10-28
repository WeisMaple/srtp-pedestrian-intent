import pathlib,sys,json
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modern_pipeline.bootstrap import configure,ARTIFACT_ROOT
configure()
import torch,numpy as np,cv2
from modern_pipeline.frontend import YOLOPoseTracker
torch.set_num_threads(2);out=pathlib.Path(r'D:\SRTP_independent_20261028')
front=YOLOPoseTracker(ARTIFACT_ROOT/'weights/yolo26s-pose.pt','cuda:0');cap=cv2.VideoCapture(str(ROOT/'data/JAAD_clips/video_0001.mp4'))
single={}
for f in range(60):
    ok,img=cap.read();assert ok
    for p in front(img):single[(f,p.track_id)]=p.pose
cap.release()
with np.load(out/'modern_raw/video_0001.npz') as z:batch={(int(r[0]),int(r[1])):p for r,p in zip(z['rows'],z['poses']) if r[0]<60}
shared=set(single)&set(batch)
delta=np.stack([np.abs(single[k]-batch[k]) for k in sorted(shared)])
report=dict(single_observations=len(single),batch_observations=len(batch),shared_track_frame_keys=len(shared),max_pose_abs_difference=float(delta.max()),mean_xy_pixel_difference=float(delta[:,:,:2].mean()),p95_xy_pixel_difference=float(np.percentile(delta[:,:,:2],95)),max_confidence_difference=float(delta[:,:,2].max()),note='Batch and single-frame CUDA inference are not numerically identical; tracking keys agree on this 60-frame check.')
assert set(single)==set(batch),report
assert np.isfinite(delta).all()
(out/'extraction_batch_check.json').write_text(json.dumps(report,indent=2));print(report)
