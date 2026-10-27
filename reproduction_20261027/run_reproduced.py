"""Run recovered paper inference with separately stored reproduced weights."""
import os,argparse,pathlib
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg'
os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2'
ROOT=pathlib.Path(__file__).resolve().parent.parent
ART=pathlib.Path(r'D:\SRTP_reproduction_20261027')
def main():
    p=argparse.ArgumentParser(description='Run paper-style inference using separately trained reproduction weights.')
    p.add_argument('--input',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path)
    p.add_argument('--lstm-variant',choices=['paper_default','source_code'],default='paper_default',help='paper_default: fresh weight_decay=0 control; source_code: fresh weight_decay=1e-4 run')
    a=p.parse_args();source=a.input.resolve();out=a.output.resolve()
    if not source.is_file():p.error('Input video does not exist')
    if out.exists():p.error('Output already exists; choose a new path')
    stgcn=ART/'models/stgcn/final_state_dict.pt';lstm=ART/'models'/('lstm_wd0' if a.lstm_variant=='paper_default' else 'lstm')/'final_state_dict.pt'
    if not stgcn.is_file() or not lstm.is_file():p.error('Reproduction training is incomplete')
    out.parent.mkdir(parents=True,exist_ok=True);os.chdir(ROOT)
    import sys;sys.path.insert(0,str(ROOT))
    import torch;torch.set_num_threads(2)
    from predict import predict_video
    print('ST-GCN:',stgcn,flush=True);print('LSTM:',lstm,flush=True)
    predict_video(str(source),str(out),str(stgcn),str(lstm),3)
    print('Output:',out,flush=True)
if __name__=='__main__':main()
