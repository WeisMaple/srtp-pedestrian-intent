"""Same-origin localhost API and built React frontend."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from modern_pipeline.bootstrap import PROJECT_ROOT
from .jobs import JobBusy, JobManager


def create_app(manager=None, max_upload_bytes=500 * 1024 * 1024):
    owns_manager = manager is None
    manager = manager or JobManager(PROJECT_ROOT / 'artifacts/web/jobs')

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owns_manager:
            await run_in_threadpool(manager.close)

    app = FastAPI(title='深度学习大作业 · 行人意图分析', lifespan=lifespan)
    app.state.manager = manager

    @app.get('/api/health')
    def health():
        return manager.health()

    @app.post('/api/jobs', status_code=201)
    async def upload(file: UploadFile = File(...)):
        filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
        if Path(filename).suffix.lower() not in ('.mp4', '.mov', '.avi'):
            raise HTTPException(400, '请选择 MP4、MOV 或 AVI 视频文件。')
        temporary = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), dir=manager.root, delete=False)
        path = Path(temporary.name)
        try:
            size = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_upload_bytes:
                    raise HTTPException(413, '视频超过上传大小限制，请选择较短的片段。')
                temporary.write(chunk)
            temporary.close()
            if not size:
                raise HTTPException(400, '视频文件为空。')
            return await run_in_threadpool(manager.register, path, filename)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            temporary.close()
            path.unlink(missing_ok=True)
            await file.close()

    def find(job_id):
        try:
            return manager.snapshot(job_id)
        except KeyError:
            raise HTTPException(404, '未找到这个分析任务。')

    @app.get('/api/jobs/{job_id}')
    def state(job_id: str):
        return find(job_id)

    @app.post('/api/jobs/{job_id}/start')
    def start(job_id: str):
        find(job_id)
        try:
            return manager.start(job_id)
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        find(job_id)
        return manager.cancel_job(job_id)

    @app.get('/api/jobs/{job_id}/trace')
    def trace(job_id: str):
        find(job_id)
        return manager.trace(job_id)

    @app.get('/api/jobs/{job_id}/network')
    def network(job_id: str, frame: int = Query(..., ge=0), track_id: int = Query(..., ge=0)):
        find(job_id)
        try:
            result = manager.network(job_id, frame, track_id)
            return JSONResponse(result, headers={'Cache-Control': 'no-store'})
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get('/api/jobs/{job_id}/frontend-network')
    def frontend_network(job_id: str, frame: int = Query(..., ge=0)):
        find(job_id)
        try:
            result = manager.frontend_network(job_id, frame)
            return JSONResponse(result, headers={'Cache-Control': 'no-store'})
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get('/api/jobs/{job_id}/preview')
    def preview(job_id: str):
        find(job_id)
        job = manager.get(job_id)
        path = job.directory / 'preview.jpg'
        if not path.is_file():
            path = job.directory / 'poster.jpg'
        return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.get('/api/jobs/{job_id}/files/{name}')
    def download(job_id: str, name: str):
        find(job_id)
        try:
            path = manager.file(job_id, name)
        except KeyError:
            raise HTTPException(404, '结果尚未生成或文件不可访问。')
        media = {'video': 'video/mp4', 'source': 'video/mp4', 'poster': 'image/jpeg',
                 'bundle': 'application/zip', 'data': 'application/x-ndjson', 'metrics': 'application/json'}[name]
        return FileResponse(path, media_type=media, filename=path.name,
                            content_disposition_type='inline' if name in ('video', 'source', 'poster') else 'attachment')

    @app.get('/api/jobs/{job_id}/live')
    async def live(job_id: str, request: Request):
        find(job_id)
        last_id = request.headers.get('last-event-id')
        try:
            after = int(last_id) if last_id is not None else None
            if after is not None and after < 0:
                raise ValueError
        except ValueError:
            raise HTTPException(400, 'Last-Event-ID must be a non-negative integer.')

        async def stream():
            cursor, ticks = after, 0
            while not await request.is_disconnected():
                rows, status = manager.live_events(job_id, cursor)
                for row in rows:
                    data = row['data']
                    cursor = data['seq']
                    yield ('id: ' + str(cursor) + '\nevent: ' + row['event'] + '\ndata: '
                           + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n\n')
                if status in ('completed', 'failed', 'cancelled'):
                    break
                ticks += 1
                if ticks % 200 == 0:
                    yield ': keepalive\n\n'
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/jobs/{job_id}/events')
    async def events(job_id: str, request: Request):
        find(job_id)

        async def stream():
            revision, ticks = -1, 0
            while not await request.is_disconnected():
                snapshot = manager.snapshot(job_id)
                terminal = snapshot['status'] in ('completed', 'failed', 'cancelled')
                # Status can become terminal before the worker increments revision.
                # Always deliver that final snapshot before closing the stream.
                if snapshot['revision'] != revision or terminal:
                    revision = snapshot['revision']
                    yield 'data: ' + json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + '\n\n'
                if terminal:
                    break
                ticks += 1
                if ticks % 40 == 0:
                    yield ': keepalive\n\n'
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    frontend = PROJECT_ROOT / 'web_demo/dist'
    if (frontend / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=str(frontend / 'assets')), name='assets')

    @app.get('/')
    @app.get('/process')
    def index():
        if (frontend / 'index.html').is_file():
            return FileResponse(frontend / 'index.html')
        return HTMLResponse('<h1>请先构建网页</h1><p>在 web_demo 目录运行 npm install 和 npm run build，然后重新启动。</p>', status_code=503)

    return app
