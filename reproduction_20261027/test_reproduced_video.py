import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2'
import pathlib,sys,json,time,collections,xml.etree.ElementTree as ET
import cv2,torch
ROOT=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));os.chdir(ROOT);torch.set_num_threads(2)
ART=pathlib.Path(r'D:\SRTP_reproduction_20261027')
import predict
source=ROOT/'data/JAAD_clips/video_0001.mp4';sample=ART/'demo_raw_90frames.mp4';output=ART/'reproduced_demo_90frames.mp4'
if output.exists():raise RuntimeError('Demo output already exists')
cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer=cv2.VideoWriter(str(sample),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writer.isOpened()
for _ in range(90):
    ok,frame=cap.read();assert ok;writer.write(frame)
writer.release();cap.release()
state={'frame':-1};rows=[]
OriginalDetector=predict.YOLOPersonDetector
class MeteredDetector(OriginalDetector):
    def detect(self,image):state['frame']+=1;return super().detect(image)
original_draw=predict.predict_draw
def record(frame,pred,bbox):
    rows.append({'frame':state['frame'],'class':int(pred),'bbox':list(map(int,bbox))});return original_draw(frame,pred,bbox)
predict.YOLOPersonDetector=MeteredDetector;predict.predict_draw=record
start=time.perf_counter()
predict.predict_video(str(sample),str(output),str(ART/'models/stgcn/final_state_dict.pt'),str(ART/'models/lstm_wd0/final_state_dict.pt'),3)
seconds=time.perf_counter()-start
cap=cv2.VideoCapture(str(output));decoded=0
while True:
    ok,frame=cap.read()
    if not ok:break
    decoded+=1
    if decoded==85:cv2.imwrite(str(ART/'reproduced_demo_frame85.jpg'),frame)
cap.release();assert decoded==90
summary={'source':str(source),'frames':90,'decoded_output_frames':decoded,'classification_annotations':len(rows),'predicted_class_counts':dict(collections.Counter(r['class'] for r in rows)),'seconds_including_setup':seconds,'STGCN_weights':str(ART/'models/stgcn/final_state_dict.pt'),'LSTM_weights':str(ART/'models/lstm_wd0/final_state_dict.pt'),'video':str(output),'note':'Original predict_video logic, including original CPU assignment and two-sample normalization; LSTM is the declared weight_decay=0 control trained from scratch'}
(ART/'demo_results.json').write_text(json.dumps(summary,indent=2));(ART/'demo_classifications.json').write_text(json.dumps(rows,indent=2))
print('DEMO_COMPLETE '+json.dumps(summary),flush=True)
