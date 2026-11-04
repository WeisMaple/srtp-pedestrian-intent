import { TERM_FILTERS, formatValue, statusLabel, termOrdinal, termStatus } from '../domain/ledger';
import '../styles/ledger.css';

export default function NumericLedger({ calculation, state, terms = [], filter = 'all', page = 0, onPage, onFilter, onSeek }) {
    if (!calculation) return <div className="ledger-empty">等待可检查的真实计算。</div>;
    const pageSize = 8;
    const maxPage = Math.max(0, Math.ceil(terms.length / pageSize) - 1);
    const currentPage = Math.min(Math.max(0, page), maxPage);
    const visible = terms.slice(currentPage * pageSize, currentPage * pageSize + pageSize);
    return <section className="numeric-ledger" aria-label="参与项数值">
        <div className="ledger-head">
            <div><span>PARTICIPATION / 针对当前目标</span><h3>{calculation.title}</h3></div>
            <p className="mono">{calculation.counts.connected} 有连接 · {calculation.counts.disconnected} 无连接 · {calculation.counts.zeroContribution} 贡献为零{calculation.counts.unrecorded ? ` · ${calculation.counts.unrecorded} 未采集` : ''}</p>
        </div>
        <div className="ledger-tools">
            <div>{TERM_FILTERS.map(item => <button key={item.value} aria-pressed={filter === item.value}
                onClick={() => onFilter(item.value)}>{item.label}</button>)}</div>
            <div><button aria-label="上一页" disabled={currentPage === 0} onClick={() => onPage(currentPage - 1)}>←</button>
                <span className="mono">{currentPage + 1} / {maxPage + 1}</span>
                <button aria-label="下一页" disabled={currentPage === maxPage} onClick={() => onPage(currentPage + 1)}>→</button></div>
        </div>
        <div className="ledger-table" role="table" aria-label="完整参与项">
            <div role="row" className="ledger-row head"><span>来源 / 索引</span><span>输入</span><span>权重</span><span>乘积</span><span>作用</span></div>
            {visible.map(term => {
                const ordinal = termOrdinal(term);
                const status = termStatus(term, ordinal, state);
                return <button role="row" key={ordinal} className={`ledger-row ${status}`}
                    aria-label={`查看第 ${ordinal + 1} 项，${statusLabel(status)}，输入 ${formatValue(term.input)}，权重 ${formatValue(term.weight)}，乘积 ${formatValue(term.product)}`}
                    onClick={() => onSeek(ordinal + 0.5)}>
                    <span><b>#{ordinal + 1}</b> {term.label}</span>
                    <span>{formatValue(term.input)}</span>
                    <span>{formatValue(term.weight)}</span>
                    <span>{formatValue(term.product)}</span>
                    <span>{statusLabel(status)}</span>
                </button>;
            })}
        </div>
    </section>;
}
