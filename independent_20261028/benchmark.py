"""Same reencoded held-out clips and MP4V writer; full pipeline wall time, every frame."""
import os,sys,pathlib,json,time,argparse
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2'
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=pathlib.Path(r'D:\SRTP_independent_20261028');dest=OUT/'speed';dest.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['legacy','modern'],required=True);args=p.parse_args()
if args.variant=='modern':
    from modern_pipeline.bootstrap import configure
    configure()
import cv2,torch
torch.set_num_threads(2)
videos=(ROOT/'data/split_ids/all_videos/test.txt').read_text().split()[:3]
records=[]
for video in videos:
    source=dest/(video+'_input.mp4')
    if not source.exists():
        cap=cv2.VideoCapture(str(ROOT/'data/JAAD_clips'/f'{video}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);size=tuple(int(cap.get(k)) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT])
        writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),fps,size);assert writer.isOpened()
        for i in range(150):
            ok,frame=cap.read()
            if not ok:break
            writer.write(frame)
        cap.release();writer.release()
    output=dest/(video+'_'+args.variant+'.mp4')
    if output.with_suffix('.benchmark.json').exists():records.append(json.loads(output.with_suffix('.benchmark.json').read_text()));continue
    if args.variant=='modern':
        from modern_pipeline.cli import arguments,run
        a=arguments(['--input',str(source),'--output',str(output),'--stgcn',str(OUT/'modern/stgcn/best.pt'),'--lstm',str(OUT/'modern/lstm/best.pt'),'--classify-every','1','--normalization','legacy_pair'])
        start=time.perf_counter();run(a);seconds=time.perf_counter()-start
        m=json.loads(output.with_suffix('.metrics.json').read_text());record={'video':video,'seconds_call':seconds,'details':m}
    else:
        import predict
        annotations=[];times=[];base_detector=predict.YOLOPersonDetector;draw=predict.predict_draw
        class Metered(base_detector):
            def detect(self,image):times.append(time.perf_counter());return super().detect(image)
        def draw_record(frame,pred,bbox):
            annotations.append(dict(frame=len(times)-1,class_id=int(pred),bbox=list(map(float,bbox))))
            return draw(frame,pred,bbox)
        def writer_same(video,path):
            w=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),video.get(cv2.CAP_PROP_FPS),(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))));assert w.isOpened();return w
        predict.YOLOPersonDetector=Metered;predict.predict_draw=draw_record;predict.video_writer_same_codec=writer_same
        start=time.perf_counter();predict.predict_video(str(source),str(output),str(OUT/'legacy/stgcn/best.pt'),str(OUT/'legacy/lstm/best.pt'),3);seconds=time.perf_counter()-start
        record=dict(video=video,seconds_call=seconds,frames=len(times),annotations=len(annotations),steady_seconds_after_first_detector=(time.perf_counter()-times[1]) if len(times)>1 else None)
        output.with_suffix('.predictions.json').write_text(json.dumps(annotations))
        predict.YOLOPersonDetector=base_detector;predict.predict_draw=draw
    cap=cv2.VideoCapture(str(output));frames=0
    while cap.read()[0]:frames+=1
    cap.release();record['decoded_frames']=frames;record['fps_including_setup']=frames/seconds
    output.with_suffix('.benchmark.json').write_text(json.dumps(record,indent=2));records.append(record);print('BENCHMARK',args.variant,record,flush=True)
(dest/(args.variant+'_summary.json')).write_text(json.dumps(records,indent=2))
