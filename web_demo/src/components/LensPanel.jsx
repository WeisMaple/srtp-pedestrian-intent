import NumericLedger from './NumericLedger';
import { formatValue } from '../domain/ledger';

export default function LensPanel({ open, calculation, state, terms, filter, page, onFilter, onPage, onSeek, onClose }) {
    if (!open) return null;
    return <aside className="lens-panel blueprint-panel" role="dialog" aria-label="计算放大镜">
        <header><div><span>CALCULATION LENS</span><strong>{calculation?.title || '等待选择对象'}</strong></div>
            <button onClick={onClose} aria-label="关闭放大镜">×</button></header>
        <div className="lens-equation mono">
            {state?.currentTerm ? `${formatValue(state.currentTerm.input)} × ${formatValue(state.currentTerm.weight)} = ${formatValue(state.currentTerm.product)}` : '—'}
        </div>
        <NumericLedger calculation={calculation} state={state} terms={terms} filter={filter}
            page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />
    </aside>;
}
