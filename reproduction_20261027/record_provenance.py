import pathlib,hashlib,json,sys,importlib.metadata as md,time
ROOT=pathlib.Path(r'F:\Artificial_Intelligence\srtp');ART=pathlib.Path(r'D:\SRTP_reproduction_20261027')
files=['DataSet.py','DataProcess.py','train.py','predict.py','st_gcn.py','api/yolo.py','api/deepsort.py','deepsort/generate_detections.py','Models/stgcn_gpu_pre.pth','Models/lstm_gpu_pre.pth','Models/yolov8n-pose.pt','checkpoints/yolov7x.pt','checkpoints/ReID.pb','data/JAAD_video_inventory.csv','data/annotation_source.json']
sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
paper=pathlib.Path(r'G:\CQU\srtp\080904K_20204244_吕崇发_LW_01.pdf')
packages={name:md.version(name) for name in ['torch','torchvision','numpy','scipy','ultralytics','tensorflow','opencv-python']}
manifest={'date':'2026-10-27','python':sys.executable,'python_version':sys.version,'source_sha256':sources,'paper':{'path':str(paper),'sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),'experiment_PDF_pages':[33,34,35,37,38]},'package_versions':packages}
(ART/'reproduction_input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'packages':packages,'tracked_source_files':len(sources)},ensure_ascii=False))
