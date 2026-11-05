import { Button, Details, Summary } from './ui/Controls';
import { processMetrics } from '../domain/process';
import { STEPS } from '../domain/timeline';
import ArithmeticInspector from './ArithmeticInspector';
import NumericLedger from './NumericLedger';
import '../styles/process.css';

export default function ComputationDesk({ job, live, track, state, calculation, flow, terms, filter, page,
    onFilter, onPage, onSeek, onInspect, stream, onRetry, record }) {
    const metrics = processMetrics(state.step, { job, live, track, calculation, module: state.module, block: state.block });
    const step = STEPS.find(item => item.step === state.step);
    const connection = record?.active ? (record.status === 'ready' ? '历史记录重算 · 可逐项回看' : record.status === 'error' ? '记录读取失败' : record.status === 'empty' ? '没有完整历史' : '正在准备记录重算')
        : state.frozen ? '数值已冻结 · 后台继续分析' : stream?.connection === 'complete' ? '采集已结束'
        : stream?.connection === 'connected' ? '实时连接正常' : stream?.connection === 'reconnecting' ? '连接中断 · 正在重连'
            : stream?.connection === 'error' ? '实时连接失败' : job ? '正在连接数据' : '等待分析任务';
    return <section className="computation-desk" aria-label="计算过程说明">
        <header className="process-heading">
            <div><span className="process-kicker mono">计算过程 / {String(state.step).padStart(2, '0')}</span>
                <h2>{step?.title}：这一步在做什么</h2></div>
            <span className={`process-connection ${stream?.error ? 'has-error' : ''}`} role="status">{connection}</span>
            {stream?.error && <Button type="button" className="process-button" onClick={onRetry}>重新连接</Button>}
            {calculation && <Button type="button" className="process-button" onClick={onInspect}>查看计算明细 ↗</Button>}
        </header>
        <p className="process-description">{metrics.description}</p>
        <div className="process-flow">
            {[['输入', metrics.input], ['处理', metrics.operation], ['输出', metrics.output]].map(([label, value]) =>
                <div key={label}><small>{label}</small><strong>{value}</strong></div>)}
        </div>
        {metrics.notice && !record?.active && <p className="process-notice">{metrics.notice}</p>}
        {calculation && <>
            <ArithmeticInspector calculation={calculation} state={flow} onSeek={onSeek} />
            <Details className="process-terms"><Summary>展开全部 {calculation.terms.length} 项参与明细</Summary>
                <NumericLedger calculation={calculation} state={flow} terms={terms} filter={filter} page={page}
                    onFilter={onFilter} onPage={onPage} onSeek={onSeek} /></Details>
        </>}
    </section>;
}
