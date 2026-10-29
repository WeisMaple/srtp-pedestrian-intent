"""Video CLI for the TensorFlow-free YOLO pose / LSTM / ST-GCN pipeline."""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from .bootstrap import ARTIFACT_ROOT, PROJECT_ROOT, configure

LABELS = ('irrelevant', 'not crossing', 'crossing')
EDGES = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


class PredictionCancelled(RuntimeError):
    """A caller requested cancellation at a decoded-frame boundary."""

def arguments(argv=None):
    p = argparse.ArgumentParser(description='PyTorch-only YOLO26 Pose pedestrian intent pipeline')
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', type=Path, help='New .mp4 file; default is a timestamped file on D:')
    p.add_argument('--pose-model', type=Path, default=ARTIFACT_ROOT/'weights/yolo26s-pose.pt')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0, or 0')
    p.add_argument('--behavior-device', default='same', help='same, cpu, or cuda:0; device for LSTM and ST-GCN')
    p.add_argument('--presentation', action='store_true', help='Show a compact intent display for video demonstrations')
    p.add_argument('--pose-filter', action='store_true', help='Use causal pose processing with matching trained weights')
    p.add_argument('--task', choices=['predict','classify'], default='predict',
                   help='predict: classify forecast poses; classify: classify observed poses')
    p.add_argument('--weights', choices=['reproduced','original'], default='reproduced')
    p.add_argument('--stgcn', type=Path)
    p.add_argument('--lstm', type=Path)
    p.add_argument('--normalization', choices=['legacy_pair','training_stats'], default='legacy_pair')
    p.add_argument('--stats', type=Path, default=ARTIFACT_ROOT/'weights/normalization.pt')
    p.add_argument('--tracker', choices=['bytetrack.yaml','botsort.yaml'], default='bytetrack.yaml')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.1)
    p.add_argument('--half', action='store_true', help='FP16 YOLO on CUDA; behavior models stay FP32')
    p.add_argument('--classify-every', type=int, default=1, help='Infer every N video frames once history is ready (default: every frame); values >1 enable cached display')
    p.add_argument('--max-gap', type=int, default=2, help='Reset history after a larger observed-frame gap')
    p.add_argument('--track-ttl', type=int, default=30)
    p.add_argument('--min-joints', type=int, default=5)
    p.add_argument('--joint-conf', type=float, default=0.2)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--save-keypoints', action='store_true')
    p.add_argument('--no-video', action='store_true', help='Save JSONL/metrics without encoding video')
    a = p.parse_args(argv)
    if a.imgsz < 32 or a.classify_every < 1 or a.max_gap < 1 or a.track_ttl < a.max_gap:
        p.error('Require imgsz>=32, classify-every>=1, max-gap>=1, track-ttl>=max-gap')
    if not 0 <= a.conf <= 1 or not 0 <= a.joint_conf <= 1 or not 0 <= a.min_joints <= 17:
        p.error('Confidence must be in [0,1], min-joints in [0,17]')
    if a.max_frames is not None and a.max_frames < 1:
        p.error('max-frames must be positive')
    if a.normalization == 'training_stats' and a.weights == 'original':
        p.error('training_stats is fitted for reproduced weights; choose --weights reproduced')
    return a

def draw_people(frame, records, task, joint_conf, presentation=False):
    import cv2
    h,w = frame.shape[:2]
    for r in records:
        color = (60,200,70) if r['class_id'] is None else ((0,200,255),(80,210,70),(40,70,255))[r['class_id']]
        x1,y1,x2,y2 = [int(v) for v in r['bbox']]
        cv2.rectangle(frame,(max(x1,0),max(y1,0)),(min(x2,w-1),min(y2,h-1)),color,2)
        label = ('Collecting motion %d/50' if presentation else 'warming %d/50') % r['history_length']
        if r['class_id'] is not None:
            names=('Other','Not crossing','Crossing') if presentation else LABELS
            label = names[r['class_id']] + ' | score %.2f' % r['class_probability']
        label = ('ID %d | %s' % (r['track_id'],label)) if presentation else ('ID %d %s %s' % (r['track_id'], 'future:' if task=='predict' else 'history:', label))
        cv2.putText(frame,label,(max(0,min(x1,w-250)),max(18,min(y1-6,h-5))),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
        pose = r['_pose']
        for i,j in EDGES:
            if pose[i,2] >= joint_conf and pose[j,2] >= joint_conf:
                p1=tuple(int(v) for v in pose[i,:2]);p2=tuple(int(v) for v in pose[j,:2])
                cv2.line(frame,p1,p2,(255,180,40),1,cv2.LINE_AA)
        for x,y,c in pose:
            if c >= joint_conf:
                cv2.circle(frame,(int(x),int(y)),2,(40,240,240),-1,cv2.LINE_AA)
    if presentation:
        cv2.rectangle(frame,(0,0),(min(w,680),58),(30,30,30),-1)
        cv2.putText(frame,'Pedestrian Crossing Intent',(16,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2,cv2.LINE_AA)
        ready=sum(r['class_id'] is not None for r in records)
        cv2.putText(frame,'People: %d | Predictions: %d' % (len(records),ready),(16,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(210,210,210),1,cv2.LINE_AA)
    return frame

def run(a, on_event=None, cancel=None, on_computation=None):
    configure()
    if on_computation is None and getattr(a, 'live_telemetry', False):
        on_computation = on_event
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from .behavior import BehaviorPredictor
    from .frontend import YOLOPoseTracker
    from .history import HistoryStore
    torch.set_num_threads(2)
    source = a.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError('Input video does not exist: '+str(source))
    output = a.output.expanduser().resolve() if a.output else ARTIFACT_ROOT/'results'/(source.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_intent.mp4')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must have .mp4 suffix')
    json_path=output.with_suffix('.jsonl'); metrics_path=output.with_suffix('.metrics.json')
    for p in ([json_path,metrics_path] if a.no_video else [output,json_path,metrics_path]):
        if p == source or p.exists():
            raise ValueError('Refusing to overwrite: '+str(p))
    requested = a.device
    if requested == 'auto': requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif requested.isdigit(): requested = 'cuda:'+requested
    device=torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu')
    if a.half and device.type != 'cuda':
        raise ValueError('--half requires CUDA')
    default_g = ARTIFACT_ROOT/'weights/stgcn_reproduced.pt' if a.weights=='reproduced' else PROJECT_ROOT/'Models/stgcn_gpu_pre.pth'
    default_l = ARTIFACT_ROOT/'weights/lstm_reproduced_wd0.pt' if a.weights=='reproduced' else PROJECT_ROOT/'Models/lstm_gpu_pre.pth'
    stgcn=(a.stgcn or default_g).resolve(); lstm=(a.lstm or default_l).resolve() if a.task=='predict' else None
    started=time.perf_counter()
    front=YOLOPoseTracker(a.pose_model, str(device), a.imgsz, a.conf, a.tracker, a.half)
    behavior_device=str(device) if a.behavior_device=='same' else a.behavior_device
    behavior=BehaviorPredictor(stgcn,lstm,behavior_device,a.normalization,a.stats)
    store=HistoryStore(max_gap=a.max_gap,ttl=a.track_ttl)
    from .pose_filter import PoseFilter
    pose_filter=PoseFilter() if a.pose_filter else None
    if device.type=='cuda':
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    cap=cv2.VideoCapture(str(source));writer=None;records_file=None
    count=0;annotations=0;updates=0;pose_count=0;seen=set();reference=None
    frontend_seconds=behavior_seconds=0.;frame_times=[];processing_start=None;status='failed'
    meta={'input':str(source),'output':str(output) if not a.no_video else None,
          'pose_model':str(a.pose_model.resolve()),'STGCN':str(stgcn),'LSTM':str(lstm) if lstm else None,
          'task':a.task,'normalization':a.normalization,'weights':a.weights,'tracker':a.tracker,
          'device':str(device),'ultralytics':ultralytics.__version__,'torch':torch.__version__,
          'behavior_device':str(behavior.device),'presentation':a.presentation,
          'pose_filter':a.pose_filter,
          'imgsz':a.imgsz,'conf':a.conf,'half_pose':a.half,'classify_every':a.classify_every,
          'max_gap':a.max_gap,'track_ttl':a.track_ttl,'min_joints':a.min_joints,
          'joint_conf':a.joint_conf,'save_keypoints':a.save_keypoints,
          'label_map':{str(i):v for i,v in enumerate(LABELS)},
          'forecast_observations':50 if lstm else 0,
          'note':'New frontend changes skeleton distribution; previous validation scores do not apply.'}
    live = None
    try:
        if on_computation is not None:
            from .live_trace import LiveTrace
            live = LiveTrace(on_computation)
            live.attach_yolo(front.model)
            live.attach_behavior(behavior)
        if not cap.isOpened():raise RuntimeError('Cannot open input video')
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if width<1 or height<1 or not np.isfinite(fps) or fps<=0:
            raise RuntimeError('Invalid video dimensions or frame rate')
        meta.update(width=width,height=height,input_fps=fps,input_frames_declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if on_event is not None:
            on_event({'kind': 'metadata', 'width': width, 'height': height, 'fps': fps,
                      'frame_count': meta['input_frames_declared']})
        output.parent.mkdir(parents=True,exist_ok=True)
        if not a.no_video:
            writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
            if not writer.isOpened():raise RuntimeError('Cannot create MP4 writer')
        records_file=json_path.open('w',encoding='utf-8')
        processing_start=time.perf_counter()
        while a.max_frames is None or count<a.max_frames:
            if cancel is not None and cancel.is_set():
                status = 'cancelled'
                raise PredictionCancelled('Analysis cancelled')
            if on_event is not None:
                on_event({'kind': 'stage', 'stage': 0, 'frame': count})
            frame_start=time.perf_counter();ok,frame=cap.read()
            if not ok:break
            if live is not None:
                live.begin_frame(count, count / fps)
            if on_event is not None:
                on_event({'kind': 'stage', 'stage': 1, 'frame': count})
            t=time.perf_counter();people=front(frame);frontend_seconds+=time.perf_counter()-t
            if on_event is not None:
                on_event({'kind': 'stage', 'stage': 2, 'frame': count})
            ready=[];eligible=[];eligible_ids=[];current=[]
            for person in people:
                if pose_filter is not None:person.pose=pose_filter.update(person.track_id,count,person.pose)
                seen.add(person.track_id);pose_count+=1
                valid=(person.pose[:,2]>=a.joint_conf)&(person.pose[:,0]>0)&(person.pose[:,1]>0)
                state=store.states.get(person.track_id)
                accepted=int(valid.sum())>=a.min_joints
                if accepted:
                    state=store.update(person.track_id,count,person.pose)
                    if state.ready:
                        if reference is None:
                            reference=state.as_array().copy()
                            # Preserve the legacy reference initialization warm-up.
                            if a.normalization=='legacy_pair':
                                current.append((person,state,accepted));continue
                        if state.prediction is None or count-state.last_prediction>=a.classify_every:
                            ready.append(state.as_array());eligible.append(state);eligible_ids.append(person.track_id)
                if state is not None and (count-state.last_seen>a.max_gap or not accepted):
                    state.prediction=None
                current.append((person,state,accepted))
            if ready:
                if live is not None:
                    live.begin_behavior(eligible_ids, np.stack(ready))
                t=time.perf_counter()
                if on_event is None and live is None:
                    prediction=behavior.predict(np.stack(ready),reference)
                else:
                    def observe_behavior(event):
                        if live is not None:
                            live.forecast_step(event)
                        if on_event is not None:
                            on_event(dict(event, frame=count))
                    prediction=behavior.predict(np.stack(ready),reference,on_event=observe_behavior)
                behavior_seconds+=time.perf_counter()-t
                for i,state in enumerate(eligible):
                    cls=int(prediction['class_ids'][i]);probs=prediction['probabilities'][i]
                    state.prediction={'class_id':cls,'probabilities':probs.tolist(),'class_probability':float(probs[cls])}
                    if on_event is not None:
                        state.prediction['future'] = prediction['future_poses'][i].copy()
                    state.last_prediction=count;updates+=1
            records=[]
            for person,state,accepted in current:
                pred=state.prediction if state is not None else None
                row={'track_id':person.track_id,'bbox':person.bbox.tolist(),
                     'detection_confidence':person.confidence,'pose_accepted':accepted,
                     'history_length':len(state.poses) if state is not None else 0,
                     'class_id':pred['class_id'] if pred else None,
                     'class_probability':pred['class_probability'] if pred else None,
                     'probabilities':pred['probabilities'] if pred else None,
                     'prediction_updated':state is not None and state.last_prediction==count,
                     '_pose':person.pose}
                if pred:annotations+=1
                records.append(row)
            if on_event is not None and not ready:
                on_event({'kind': 'stage', 'stage': 3, 'frame': count})
            if writer is not None:writer.write(draw_people(frame,records,a.task,a.joint_conf,a.presentation))
            serial=[]
            for row in records:
                pose=row.pop('_pose')
                if a.save_keypoints:row['keypoints']=pose.tolist()
                serial.append(row)
            records_file.write(json.dumps({'frame':count,'time_seconds':count/fps,'tracks':serial},allow_nan=False)+'\n')
            if on_event is not None:
                trace = []
                for person, state, accepted in current[:3]:
                    if state is None:
                        continue
                    row = next(r for r in serial if r['track_id'] == person.track_id)
                    future = state.prediction.get('future') if state.prediction else None
                    trace.append(dict(row, history=state.as_array(), future=future))
                on_event({'kind': 'frame', 'frame': count, 'time_seconds': count / fps,
                          'image': frame, 'tracks': serial, 'trace': trace})
            if live is not None:
                live_tracks = []
                selected_ids = eligible_ids[:3] if eligible_ids else [person.track_id for person, _, _ in current[:3]]
                selected = {person.track_id: (person, state, accepted) for person, state, accepted in current}
                for track_id in selected_ids:
                    person, state, accepted = selected[track_id]
                    row = next(r for r in serial if r['track_id'] == person.track_id)
                    live_tracks.append(dict(row, keypoints=person.pose.tolist(),
                                            prediction_frame=state.last_prediction if state is not None else -1))
                live.end_frame(live_tracks)
            store.expire(count);count+=1;frame_times.append(time.perf_counter()-frame_start)
            if count%30==0:
                records_file.flush();print('Processed',count,'frames; intent annotations',annotations,flush=True)
        if count==0:raise RuntimeError('Input video contains no decodable frames')
        if any(x=='tensorflow' or x.startswith('tensorflow.') for x in sys.modules):
            raise RuntimeError('Unexpected TensorFlow import in modern pipeline')
        status='complete'
    finally:
        if live is not None:
            live.close()
        cap.release()
        if writer is not None:writer.release()
        if records_file is not None:records_file.close()
        elapsed=time.perf_counter()-processing_start if processing_start else 0.
        meta.update(status=status,frames=count,pose_observations=pose_count,unique_track_ids=len(seen),
                    classification_updates=updates,classification_annotations=annotations,
                    frontend_seconds=frontend_seconds,behavior_seconds=behavior_seconds,
                    processing_seconds=elapsed,processing_fps=count/elapsed if elapsed else 0.,
                    total_seconds_including_model_load=time.perf_counter()-started,
                    median_frame_ms=float(np.median(frame_times)*1000) if frame_times else None,
                    fps_excluding_first_frame=(count-1)/sum(frame_times[1:]) if count>1 else None,
                    peak_torch_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0,
                    tensorflow_loaded=any(x=='tensorflow' or x.startswith('tensorflow.') for x in sys.modules))
        if records_file is not None:metrics_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    return meta

def main(argv=None):
    a=arguments(argv)
    try:return run(a)
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print('Prediction error:',str(exc),file=sys.stderr);raise SystemExit(1)

if __name__=='__main__':main()


