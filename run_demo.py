"""Single presentation entry point for pedestrian crossing-intent prediction."""
import argparse,os,json
from pathlib import Path
from modern_pipeline.bootstrap import configure,PROJECT_ROOT

def main():
    parser=argparse.ArgumentParser(description='Pedestrian crossing-intent video demonstration')
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--model-dir',type=Path,help='Directory containing a matched model_config.json bundle')
    a=parser.parse_args()
    configure()
    from modern_pipeline.cli import main as predict
    weights=a.model_dir or Path(os.environ.get('SRTP_DEMO_MODEL_DIR',str(PROJECT_ROOT/'weights/final')))
    config_path=weights/'model_config.json'
    if not config_path.is_file():
        parser.error('The model directory must contain model_config.json and its matching weights/statistics.')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    args=['--input',str(a.input),'--stgcn',str(weights/config['stgcn']),'--lstm',str(weights/config['lstm']),
          '--task','predict','--normalization',config['normalization'],'--stats',str(weights/config['stats']),'--classify-every','1',
          '--behavior-device','cpu','--presentation']
    if config['pose_filter']:args+=['--pose-filter']
    if config.get('pose_model'):args+=['--pose-model',str(weights/config['pose_model'])]
    if a.output:args+=['--output',str(a.output)]
    if a.max_frames is not None:args+=['--max-frames',str(a.max_frames)]
    return predict(args)

if __name__=='__main__':main()
