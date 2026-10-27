import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2';os.environ['YOLO_AUTOINSTALL']='false'
import pathlib,sys,json,time,argparse,collections,xml.etree.ElementTree as ET
import numpy as np
import torch
import cv2
ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));os.chdir(ROOT)
ART=pathlib.Path(r'D:\SRTP_reproduction_20261027');DATA=ROOT/'data'
for n in ['tracks','features','matches','logs']:(ART/n).mkdir(parents=True,exist_ok=True)
np.random.seed(2026);torch.manual_seed(2026);torch.set_num_threads(2)
def refresh_GPU_budget():
    free,total=torch.cuda.mem_get_info()
    budget=min(total*0.75,free+torch.cuda.memory_reserved()-512*1024**2)
    torch.cuda.set_per_process_memory_fraction(max(0.10,budget/total),0)
refresh_GPU_budget()
from api.yolo import YOLOPersonDetector
from api.deepsort import DeepSORTTracker
from deepsort.generate_detections import create_box_encoder
from deepsort.tracker import Tracker
from deepsort.nn_matching import NearestNeighborDistanceMetric
from yolo.utils.general import non_max_suppression,scale_coords
from DataProcess import match_bboxes
from ultralytics import YOLO
from ultralytics.utils import ops as ultra_ops
original_pose_nms=ultra_ops.non_max_suppression
def complete_pose_nms(*args,**kwargs):
    kwargs['max_time_img']=float('inf')
    return original_pose_nms(*args,**kwargs)
ultra_ops.non_max_suppression=complete_pose_nms
parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
videos=sorted(p.stem for p in (DATA/'JAAD_clips').glob('*.mp4'))
if args.limit:videos=videos[:args.limit]
detector=YOLOPersonDetector();detector.load(str(ROOT/'checkpoints/yolov7x.pt'))
encoder=create_box_encoder(str(ROOT/'checkpoints/ReID.pb'),batch_size=32)
pose=YOLO(str(ROOT/'Models/yolov8n-pose.pt')).to('cuda:0')
parity_checked=False;started=time.perf_counter();inventory=[]

def annotations(video):
    labels={p.get('id'):int(p.get('crossing')) for p in ET.parse(DATA/'annotations_attributes'/f'{video}_attributes.xml').getroot().findall('pedestrian')}
    database=collections.OrderedDict()
    for tr in ET.parse(DATA/'annotations'/f'{video}.xml').getroot().findall('track'):
        rows=tr.findall('box')
        if not rows:continue
        pid=rows[0].findtext("attribute[@name='id']")
        database[pid]={'bbox':[[float(b.get(k)) for k in ['xtl','ytl','xbr','ybr']] for b in rows],'frames':[int(b.get('frame')) for b in rows]}
    return database,labels

def new_tracker():
    t=object.__new__(DeepSORTTracker);t.encoder=encoder
    t.tracker=Tracker(NearestNeighborDistanceMetric('cosine',0.5,None),max_age=100)
    import matplotlib.pyplot as plt
    t.colors=plt.get_cmap('hsv')(np.linspace(0,1,20,False))[:,:3]*255
    return t

@torch.inference_mode()
def detect_batch(frames):
    tensors=[detector.preprocess(img)[0] for img in frames]
    shapes={tuple(x.shape) for x in tensors}
    if len(shapes)!=1:raise RuntimeError('Inconsistent shapes within video detection batch')
    images=torch.stack(tensors)
    pred=detector.model(images,augment=False)[0]
    detections=non_max_suppression(pred,detector.conf_th,detector.iou_th,classes=[0])
    results=[]
    for frame,d in zip(frames,detections):
        d[:,:4]=scale_coords(images.shape[2:],d[:,:4],frame.shape).round()
        results.append(d.detach().cpu().numpy())
    return results

def get_tracks(video):
    global parity_checked
    path=ART/'tracks'/f'{video}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:return z['rows'],int(z['frames'])
    cap=cv2.VideoCapture(str(DATA/'JAAD_clips'/f'{video}.mp4'))
    if not cap.isOpened():raise RuntimeError('Cannot open '+video)
    torch.cuda.empty_cache();refresh_GPU_budget()
    tracker=new_tracker();rows=[];frame_idx=0
    while True:
        frames=[]
        for _ in range(4):
            ok,img=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if not frames:break
        ds=detect_batch(frames)
        if not parity_checked:
            reference=detector.detect(frames[0].copy());actual=ds[0]
            comparison={'video':video,'single_count':len(reference),'batch_count':len(actual),'batch_size':len(frames)}
            if len(reference)==len(actual) and len(reference):comparison.update(max_coordinate_difference=float(np.max(np.abs(reference[:,:4]-actual[:,:4]))),max_score_difference=float(np.max(np.abs(reference[:,4]-actual[:,4]))))
            (ART/'detector_batch_parity.json').write_text(json.dumps(comparison,indent=2))
            print('DETECTOR_PARITY '+json.dumps(comparison),flush=True);parity_checked=True
        for frame,d in zip(frames,ds):
            if len(d):
                bboxes=d[:,:4].copy();bboxes[:,2:]-=bboxes[:,:2]
                people=tracker.track(frame,bboxes,d[:,4].copy())
                for person in people:rows.append([frame_idx,int(person['track_id'])]+list(map(int,person['bbox'])))
            frame_idx+=1
    cap.release();arr=np.asarray(rows,dtype=np.int32).reshape(-1,6)
    tmp=path.with_name(path.stem+'.tmp.npz');np.savez_compressed(tmp,rows=arr,frames=np.array(frame_idx));tmp.replace(path)
    return arr,frame_idx

def extract_poses(video,rows,pairs,database,labels):
    torch.cuda.empty_cache();refresh_GPU_budget()
    accepted={int(tid):(pid,float(score)) for pid,tid,score in pairs if score>0.5 and pid.endswith('b') and pid in labels}
    events=collections.defaultdict(list)
    gt_frames={pid:set(database[pid]['frames']) for pid,_ in accepted.values()}
    for row in rows:
        frame,tid,*box=map(int,row)
        if tid in accepted and frame in gt_frames[accepted[tid][0]]:events[frame].append((tid,box))
    outputs={tid:[] for tid in accepted};indices={tid:[] for tid in accepted}
    cap=cv2.VideoCapture(str(DATA/'JAAD_clips'/f'{video}.mp4'))
    batch=[];info=[]
    def flush():
        if not batch:return
        results=pose(batch,device=0,imgsz=640,verbose=False)
        for result,(tid,frame,x,y) in zip(results,info):
            points=result.keypoints.data.detach().cpu().numpy()
            if points.size and points.shape[1:]==(17,3):
                p=points[0].copy();p[:,0]+=x;p[:,1]+=y
                outputs[tid].append(p);indices[tid].append(frame)
        batch.clear();info.clear()
    for frame in range(max(events,default=-1)+1):
        ok,img=cap.read()
        if not ok:break
        for tid,(x1,y1,x2,y2) in events.get(frame,[]):
            crop=img[y1:y2,x1:x2]
            if crop.shape[0]<=0 or crop.shape[1]<=0:continue
            batch.append(crop.copy());info.append((tid,frame,x1,y1))
            if len(batch)>=8:flush()
    flush();cap.release()
    metadata=[];arrays={}
    for i,tid in enumerate(accepted):
        pid,score=accepted[tid];raw_label=labels[pid]
        metadata.append({'video':video,'track_id':tid,'person_id':pid,'crossing':raw_label,'class':{-1:0,0:1,1:2}[raw_label],'match_iou':score,'observations':len(outputs[tid])})
        arrays[f'p{i}']=np.asarray(outputs[tid],dtype=np.float32).reshape(-1,17,3);arrays[f'f{i}']=np.asarray(indices[tid],dtype=np.int32)
    arrays['meta']=np.array(json.dumps(metadata));path=ART/'features'/f'{video}.npz';tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
    return metadata

for i,video in enumerate(videos,1):
    start=time.perf_counter();path=ART/'features'/f'{video}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:metadata=json.loads(str(z['meta']))
        row={'video':video,'people':len(metadata),'observations':sum(m['observations'] for m in metadata),'cache':True,'seconds':0.}
    else:
        database,labels=annotations(video);tracks,frames=get_tracks(video)
        first_boxes=collections.OrderedDict()
        for _,tid,*bbox in tracks:
            tid=int(tid)
            if tid not in first_boxes:first_boxes[tid]=[]
            if len(first_boxes[tid])<=10:first_boxes[tid].append(list(map(int,bbox)))
        pairs,log=match_bboxes(database,first_boxes)
        (ART/'matches'/f'{video}.json').write_text(json.dumps({'pairs':pairs,'log':log},ensure_ascii=False),encoding='utf-8')
        metadata=extract_poses(video,tracks,pairs,database,labels)
        row={'video':video,'frames':frames,'tracks':len(first_boxes),'people':len(metadata),'observations':sum(m['observations'] for m in metadata),'cache':False,'seconds':time.perf_counter()-start}
    inventory.append(row)
    (ART/'preprocessing_progress.json').write_text(json.dumps({'completed':i,'total':len(videos),'last':row,'elapsed_seconds':time.perf_counter()-started,'people':sum(r['people'] for r in inventory),'observations':sum(r['observations'] for r in inventory)},ensure_ascii=False,indent=2),encoding='utf-8')
    print('VIDEO '+str(i)+'/'+str(len(videos))+' '+json.dumps(row),flush=True)
(ART/'preprocessing_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
print('PREPROCESSING_COMPLETE '+str(len(videos)),flush=True)

