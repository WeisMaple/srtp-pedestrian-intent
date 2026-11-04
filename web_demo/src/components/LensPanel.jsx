import { useEffect } from 'react';
import NumericLedger from './NumericLedger';
import { formatValue } from '../domain/ledger';

export default function LensPanel({ open, calculation, state, terms, waitingLabel, filter, page, onFilter, onPage, onSeek, onClose }) {
    useEffect(() => {
        if (!open) return undefined;
        const close = event => { if (event.key === 'Escape') onClose?.(); };
        window.addEventListener('keydown', close);
        return () => window.removeEventListener('keydown', close);
    }, [open, onClose]);
    if (!open) return null;
    return <aside className="lens-panel blueprint-panel" role="dialog" aria-label="计算放大镜">
        <header><div><span>CALCULATION LENS</span><strong>{calculation?.title || waitingLabel || '等待选择对象'}</strong></div>
            <button onClick={onClose} aria-label="关闭放大镜">×</button></header>
        <div className="lens-equation mono">
            {state?.currentTerm ? `${formatValue(state.currentTerm.input)} × ${formatValue(state.currentTerm.weight)} = ${formatValue(state.currentTerm.product)}` : '—'}
        </div>
        {calculation && <div className="lens-summary mono">
            <span>实际输出 <b>{formatValue(calculation.actual)}</b></span>
            <span>重建 <b>{formatValue(calculation.reconstructed)}</b></span>
            <span>部分和 <b>{formatValue(state?.partialSum)}</b></span>
            {calculation.absoluteError != null && <span>误差 <b>{formatValue(calculation.absoluteError)}</b></span>}
        </div>}
        <NumericLedger calculation={calculation} state={state} terms={terms} filter={filter}
            page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />
    </aside>;
}
