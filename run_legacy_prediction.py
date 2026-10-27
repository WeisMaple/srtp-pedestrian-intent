"""Run the recovered SRTP video pipeline using explicit input/output paths."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run pedestrian intent prediction with the recovered SRTP models.')
    parser.add_argument('--input', required=True, type=Path, help='Input video path')
    parser.add_argument('--output', type=Path, help='Output MP4 path; defaults to results/<input>_intent.mp4')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error('Input video does not exist: ' + str(source))
    output = args.output.expanduser().resolve() if args.output else root / 'results' / (source.stem + '_intent.mp4')
    if output == source:
        parser.error('Output path must differ from input path')
    if output.exists():
        parser.error('Output already exists; choose a new output path: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    import torch
    torch.set_num_threads(2)
    import predict
    print('Detector device:', 'cuda:0 (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu', flush=True)
    print('The original pipeline keeps pose/LSTM/ST-GCN on CPU.', flush=True)
    predict.predict_video(str(source), str(output), str(root/'Models/stgcn_gpu_pre.pth'), str(root/'Models/lstm_gpu_pre.pth'), 3)
    print('Output:', output, flush=True)


if __name__ == '__main__':
    main()

