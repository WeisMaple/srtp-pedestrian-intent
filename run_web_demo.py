"""Launch the local React/Python video-analysis workbench."""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='本地行人意图分析工作台')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model-dir', type=Path, default=PROJECT_ROOT / 'weights/final')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'artifacts/web/jobs')
    parser.add_argument('--max-upload-mb', type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.max_upload_mb < 1:
        parser.error('端口或上传大小无效。')
    os.environ.setdefault('SRTP_MODERN_HOME', str(PROJECT_ROOT / 'artifacts'))
    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'artifacts/runtime/matplotlib'))
    from modern_pipeline.bootstrap import configure
    configure()
    import uvicorn
    from modern_pipeline.web.app import create_app
    from modern_pipeline.web.jobs import JobManager
    manager = JobManager(args.data_dir, args.model_dir)
    app = create_app(manager, max_upload_bytes=args.max_upload_mb * 1024 * 1024)
    print('打开 http://127.0.0.1:%d 选择视频并开始分析。' % args.port, flush=True)
    try:
        uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='info')
    finally:
        manager.close()


if __name__ == '__main__':
    main()
