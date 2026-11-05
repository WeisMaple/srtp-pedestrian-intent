import { Button, Range } from './ui/Controls';
import { formatValue } from '../domain/ledger';

const ACTIVATIONS = { identity: '恒等映射', relu: 'ReLU', silu: 'SiLU', tanh: 'tanh',
    sigmoid: 'Sigmoid', softmax: 'Softmax', batch_norm: 'BatchNorm', batch_norm_silu: 'BatchNorm → SiLU' };

export default function ArithmeticInspector({ calculation, state, onSeek }) {
    if (!calculation || !state) return null;
    const count = calculation.terms.length;
    const position = Math.min(calculation.maxCursor, Math.max(0, state.cursor));
    const phase = state.done ? '输出结果' : state.biasApplied
        ? (state.phaseName === 'bias' ? '加偏置' : '应用激活') : `第 ${state.termIndex + 1} / ${count} 项`;
    return <section className="arithmetic-inspector" aria-label="逐项计算过程">
        <div className="arithmetic-title"><strong>{calculation.title}</strong><span>{calculation.metadata?.source_kind === 'recomputed_from_saved_history'
            ? `历史重算 · 帧 #${calculation.metadata.source_frame}` : '最近已采集'} · 代表算式{calculation.track_id != null ? ` · ID ${calculation.track_id}` : ''}</span></div>
        <div className="arithmetic-equation mono">
            <span>{formatValue(state.currentTerm.input)}</span><i>×</i>
            <span>{formatValue(state.currentTerm.weight)}</span><i>=</i>
            <b>{formatValue(state.currentTerm.product)}</b>
        </div>
        <dl className="arithmetic-values">
            {[
                ['当前累计', state.partialSum], ['完整乘积和 Σ', calculation.dot], ['偏置 b', calculation.bias],
                ['重建输出', calculation.reconstructed], ['实际采集输出', calculation.actual], ['绝对误差', calculation.absoluteError],
            ].map(([label, value]) => <div key={label}><dt>{label}</dt><dd className="mono">{formatValue(value)}</dd></div>)}
        </dl>
        <div className="arithmetic-controls">
            <Button type="button" aria-label="上一计算项" disabled={position <= 0}
                onClick={() => onSeek(Math.max(0, Math.ceil(position) - 1))}>←</Button>
            <label><span>{phase}</span><Range aria-label="逐项计算进度" type="range" min="0" max={calculation.maxCursor}
                step="0.1" value={position} onChange={event => onSeek(Number(event.target.value))} /></label>
            <Button type="button" aria-label="下一计算项" disabled={state.done}
                onClick={() => onSeek(Math.min(calculation.maxCursor, Math.floor(position) + 1))}>→</Button>
            <Button type="button" onClick={() => onSeek(count)}>加偏置</Button>
            <Button type="button" onClick={() => onSeek(count + 1)}>{ACTIVATIONS[calculation.activation] || calculation.activation}</Button>
            <Button type="button" onClick={() => onSeek(calculation.maxCursor)}>看结果</Button>
        </div>
        <p className="arithmetic-note">Σ(输入 × 权重) + 偏置 → {ACTIVATIONS[calculation.activation] || calculation.activation}。滑动仅回看已采集数值，不代表模型执行耗时。</p>
    </section>;
}
