import pathlib,json,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent.parent;CODE=ROOT/'reproduction_20261027';ART=pathlib.Path(r'D:\SRTP_reproduction_20261027')
status_path=ART/'pipeline_state.json'
def status(phase,detail):status_path.write_text(json.dumps({'phase':phase,'detail':detail,'time':time.strftime('%Y-%m-%d %H:%M:%S')},indent=2))
def run(name,arguments):
    status(name,'Running '+str(arguments));log=ART/'logs'/(name+'.txt')
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,'-s']+list(map(str,arguments)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate();proc.wait();raise
    if code:raise RuntimeError(name+' failed with exit '+str(code)+'; '+str(log))
try:
    status('waiting_for_preprocessing','Will start only after all 346 detected/IoU-matched video feature files are complete')
    while True:
        try:
            progress=json.loads((ART/'preprocessing_progress.json').read_text())
            ready=progress['completed']==346 and progress['total']==346 and len(list((ART/'features').glob('video_*.npz')))==346
        except (FileNotFoundError,json.JSONDecodeError):ready=False
        if ready:break
        time.sleep(10)
    run('build_datasets',[CODE/'build_training_datasets.py'])
    for model in ['stgcn','lstm']:run('pretrained_'+model,[CODE/'train_reproduction.py','--model',model,'--baseline-only'])
    for model,total in [('stgcn',125),('lstm',100)]:
        final=ART/'models'/model/'final_validation.json'
        if final.exists() and json.loads(final.read_text())['epochs']==total:continue
        args=[CODE/'train_reproduction.py','--model',model]
        if (ART/'models'/model/'checkpoint_latest.pt').exists():args+=['--resume']
        run('train_'+model,args)
    status('training_complete','Both fresh models completed the configured epochs; analysis and report remain to be generated')
    print('TRAINING_PIPELINE_COMPLETE',flush=True)
except BaseException as exc:
    status('failed',str(exc));(ART/'pipeline_error.txt').write_text(traceback.format_exc());raise
