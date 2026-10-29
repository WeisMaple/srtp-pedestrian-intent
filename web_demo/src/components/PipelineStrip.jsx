import Icon from './Icon';
import Skeleton from './Skeleton';

export const STAGES = [
    { title: '视频解码', short: '视频解码', icon: 'video', detail: '解析视频，按帧提取图像', description: '读取输入视频，逐帧解码为图像。后续步骤使用同一帧画面，保留原始视频的尺寸与时间信息。' },
    { title: '姿态检测 YOLO26', short: '姿态检测', icon: 'person', detail: '检测行人并提取 COCO17 骨架', description: 'YOLO26-Pose 在画面中定位行人，为每个人提取 COCO17 关键点。无效或置信度不足的姿态不会当作有效骨架观测。' },
    { title: '身份跟踪 ByteTrack', short: '身份跟踪', icon: 'target', detail: '跨帧关联，生成稳定的行人 ID', description: 'ByteTrack 将相邻画面中的检测结果关联，为同一行人维持跟踪 ID。检测与跟踪来自同一次前端调用。' },
    { title: '历史缓存 50 帧', short: '历史缓存', icon: 'stack', detail: '为每个行人缓存最近 50 次观测', description: '按行人 ID 收集最近 50 次有效骨架观测。缺失姿态与轨迹间断会影响历史是否可用；观测不足时不会填造分类结果。' },
    { title: '未来骨架 LSTM', short: '未来骨架', icon: 'forecast', detail: '未来 10 帧 × 5 次迭代预测', description: '将最近 50 次骨架观测送入 LSTM，每轮预测 10 帧，共 5 轮得到未来 50 帧骨架，再交由意图分类。绿色虚线表示模型预测。' },
    { title: '意图分类 ST-GCN', short: '意图分类', icon: 'graph', detail: '结合时空骨架特征进行意图分类', description: 'ST-GCN 沿 COCO17 的关节连接学习时空特征，输出其他、不过街、过街三个模型分数。图传播动画为连接结构示意。' },
];

export default function PipelineStrip({ job, onProcess }) {
    const active = job && !['ready','failed','cancelled'].includes(job.status);
    const firstTrack = job?.tracks?.[0];
    return <section className="panel pipeline-strip">
        <div className="pipeline-heading"><div><h2>从骨架到意图</h2><span>六个步骤，完成从视频到行人过街意图的时序分析。</span></div><button className="text-button" onClick={() => onProcess(0)}>展开运算过程<Icon name="arrow" size={17} /></button></div>
        <ol className="pipeline-stages">{STAGES.map((stage, index) => <li key={stage.short} className={active && job.stage === index ? 'current' : ''}>
            <button className="stage-overview" onClick={() => onProcess(index)}><div className="stage-copy"><span className="stage-number">{index + 1}</span><div><h3>{stage.title}</h3><p>{stage.detail}</p></div></div><div className={`stage-mini stage-mini-${index}`}>
                {index === 0 ? <div className="mini-frames"><i /><i /><Icon name="video" size={33} /></div> : index === 1 || index === 2 ? <div className="mini-detection">{firstTrack?.keypoints ? <Skeleton pose={firstTrack.keypoints} color="#259b7a" /> : <Icon name={stage.icon} size={38} />}<i /><i /></div> : index === 3 ? <div className="mini-history"><Icon name="stack" size={33} /><span>最近 50 次有效观测</span></div> : index === 4 ? <div className="mini-forecast">{[1,2,3,4,5].map((n) => <span key={n}>{n * 10}</span>)}</div> : <div className="mini-classifier"><Icon name="graph" size={28} /><i /><span>ST-GCN</span><i /><div><b>其他</b><b>不过街</b><b>过街</b></div></div>}
            </div></button>{index < 5 && <Icon name="arrow" size={18} className="stage-connector" />}
        </li>)}</ol>
    </section>;
}
