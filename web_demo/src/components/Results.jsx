import { jobUrl } from '../api';
import EmptyState from './EmptyState';

export default function Results({ job, onOpenCorridor }) {
    if (!job) return <section className="results-page">
        <EmptyState title="没有可显示的任务">先在工作台上传并分析一段视频，结果与下载会出现在这里。</EmptyState>
    </section>;
    const metrics = job.metrics || {};
    return <section className="results-page">
        <header><span className="mono">RESULT / 一次真实推理</span><h1>结果与下载</h1>
            <button className="blueprint-button" onClick={onOpenCorridor}>回到长廊</button></header>
        <div className="results-grid">
            <div className="results-video blueprint-panel">
                {job.outputs?.video ? <video src={jobUrl(job.id, '/files/video')} controls playsInline /> : <p>结果视频尚未生成。</p>}
            </div>
            <aside className="results-metrics blueprint-panel">
                <div><span>处理帧</span><b>{metrics.frames ?? job.processed_frames ?? '—'}</b></div>
                <div><span>有效观测</span><b>{metrics.pose_observations ?? '—'}</b></div>
                <div><span>预测更新</span><b>{metrics.classification_updates ?? '—'}</b></div>
                <div><span>处理 FPS</span><b>{Number.isFinite(metrics.processing_fps) ? metrics.processing_fps.toFixed(2) : '—'}</b></div>
                <div className="downloads">
                    {job.outputs?.video ? <a href={jobUrl(job.id, '/files/video')}>标注 MP4</a> : <span>标注 MP4 尚未生成</span>}
                    {job.outputs?.data ? <a href={jobUrl(job.id, '/files/data')}>逐帧 JSONL</a> : <span>逐帧 JSONL 尚未生成</span>}
                    {job.outputs?.bundle ? <a href={jobUrl(job.id, '/files/bundle')}>完整 ZIP</a> : <span>完整 ZIP 尚未生成</span>}
                </div>
            </aside>
        </div>
    </section>;
}
