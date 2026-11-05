import { Button } from './ui/Controls';
import { useEffect, useRef } from 'react';
import NumericLedger from './NumericLedger';
import ArithmeticInspector from './ArithmeticInspector';

export default function LensPanel({ open, calculation, state, terms, waitingLabel, filter, page, onFilter, onPage, onSeek, onClose }) {
    const panel = useRef(null);
    useEffect(() => {
        if (!open) return undefined;
        const trigger = document.activeElement;
        panel.current?.querySelector('button')?.focus();
        return () => { if (trigger?.isConnected) trigger.focus(); };
    }, [open]);
    useEffect(() => {
        if (!open) return undefined;
        const close = event => { if (event.key === 'Escape') onClose?.(); };
        window.addEventListener('keydown', close);
        return () => window.removeEventListener('keydown', close);
    }, [open, onClose]);
    if (!open) return null;
    return <aside ref={panel} className="lens-panel blueprint-panel" role="dialog" aria-label="计算放大镜">
        <header><div><span>CALCULATION LENS</span><strong>{calculation?.title || waitingLabel || '等待选择对象'}</strong></div>
            <Button onClick={onClose} aria-label="关闭放大镜">×</Button></header>
        <ArithmeticInspector calculation={calculation} state={state} onSeek={onSeek} />
        <NumericLedger calculation={calculation} state={state} terms={terms} filter={filter}
            page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />
    </aside>;
}
