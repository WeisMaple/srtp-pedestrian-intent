import { memo, useId, useMemo, useState } from 'react';
import { accumulationFrame, activationValue, clamp01, displayNumber as number, weightedTerms } from './arithmeticMath';
import '../arithmetic.css';

const PAGE_SIZE = 64;
const WINDOW_SIZE = 9;
const mix = (start, end, t) => start + (end - start) * t;
const smooth = value => { const p = clamp01(value); return p * p * (3 - 2 * p); };
const tint = (value, peak, weight = false) => {
    const opacity = .13 + Math.min(1, Math.abs(value) / peak) * .7;
    return `rgba(${value < 0 ? '105, 166, 245' : weight ? '183, 156, 244' : '75, 218, 163'}, ${opacity})`;
};

const TensorCells = memo(function TensorCells({ values, peak, start, selected, onSelect, label, weight = false }) {
    const slice = values.slice(start, start + PAGE_SIZE);
    return <div className={`at-tensor ${weight ? 'at-weights' : 'at-inputs'}`}>
        <div className="at-tensor-label"><b>{label}</b><span>[{start} … {start + slice.length - 1}]</span></div>
        <div className="at-tensor-grid" style={{ '--tensor-columns': Math.min(16, slice.length) }} role="group" aria-label={`${label}，点击维度查看实际数值`}>
            {slice.map((value, offset) => <button type="button" key={start + offset} className={selected === start + offset ? 'selected' : ''} style={{ backgroundColor: tint(value, peak, weight) }} aria-pressed={selected === start + offset} aria-label={`${label}，维度 ${start + offset}，数值 ${value}`} title={`[${start + offset}] = ${value}`} onClick={() => onSelect(start + offset)}><span>{Math.abs(value) < 10 ? value.toFixed(1) : number(value, 0)}</span></button>)}
        </div>
        <div className="at-tensor-axis"><span>维度 j →</span><span>负数为蓝色 · 亮度表示绝对值</span></div>
    </div>;
});

function Operand({ x, y, value, name, kind = 'input', opacity = 1, scale = 1 }) {
    return <g className={`at-svg-token at-token-${kind}`} opacity={opacity} transform={`translate(${x} ${y}) scale(${scale})`}><rect x="-53" y="-22" width="106" height="44" rx="5" /><text textAnchor="middle" y="6">{number(value)}</text>{name ? <text className="at-token-name" textAnchor="middle" y="-33">{name}</text> : null}</g>;
}

function ArithmeticStage({ index, phase, before, after, input, weight, product, bias, biasPhase, outputPhase, total, output, compact = false }) {
    const pos = compact ? { x: [70, 67], w: [290, 67], product: [180, 166], sum: [87, 306], out: [285, 306], width: 360, height: 385 } : { x: [90, 85], w: [90, 213], product: [342, 149], sum: [580, 149], out: [822, 149], width: 940, height: 285 };
    const { x, w, product: p, sum: a, out: o } = pos;
    const move = smooth(phase / .4);
    const productMove = smooth((phase - .49) / .29);
    const isAdded = phase >= .78;
    const biasAdded = biasPhase >= 1;
    const sum = (isAdded ? after : before) + (biasAdded ? bias : 0);
    const productVisible = phase >= .4;
    const pathX = `M${x[0]} ${x[1]} C${mix(x[0], p[0], .6)} ${x[1]} ${mix(x[0], p[0], .6)} ${p[1]} ${p[0]} ${p[1]}`;
    const pathW = `M${w[0]} ${w[1]} C${mix(w[0], p[0], .6)} ${w[1]} ${mix(w[0], p[0], .6)} ${p[1]} ${p[0]} ${p[1]}`;
    const stageText = outputPhase > 0 ? '完整加权和输出' : biasPhase > 0 ? '将偏置加入完整内积' : phase < .4 ? '取出输入与同一维权重' : phase < .49 ? '两数相乘，得到贡献值' : phase < .78 ? '贡献值进入累加器' : '完成本项，等待下一项';
    return <svg className={`at-arithmetic-stage ${compact ? 'at-stage-compact' : 'at-stage-wide'}`} viewBox={`0 0 ${pos.width} ${pos.height}`} role="img" aria-label={`第 ${index + 1} 项：${input} 乘 ${weight} 等于 ${product}。此前和 ${before}，加入后 ${after}。${stageText}`}>
        <g className="at-stage-guides"><path d={pathX} /><path d={pathW} /><path d={`M${p[0]} ${p[1]} L${a[0]} ${a[1]}`} /><path d={`M${a[0]} ${a[1]} L${o[0]} ${o[1]}`} /></g>
        <g className="at-stage-traces"><path d={pathX} pathLength="1" strokeDasharray={`${move} 1`} /><path d={pathW} pathLength="1" strokeDasharray={`${move} 1`} className="weight" /><path d={`M${p[0]} ${p[1]} L${a[0]} ${a[1]}`} pathLength="1" strokeDasharray={`${productMove} 1`} /><path d={`M${a[0]} ${a[1]} L${o[0]} ${o[1]}`} pathLength="1" strokeDasharray={`${outputPhase} 1`} /></g>
        <g className="at-source-slot"><rect x={x[0] - 55} y={x[1] - 23} width="110" height="46" rx="5" /><rect x={w[0] - 55} y={w[1] - 23} width="110" height="46" rx="5" /><text x={x[0]} y={x[1] - 34} textAnchor="middle">x[{index}]</text><text x={w[0]} y={w[1] - 34} textAnchor="middle">w[{index}]</text><text x={x[0]} y={x[1] + 6} textAnchor="middle">{number(input)}</text><text x={w[0]} y={w[1] + 6} textAnchor="middle">{number(weight)}</text></g>
        <g className={`at-product-node ${productVisible ? 'active' : ''}`} transform={`translate(${p[0]} ${p[1]})`}><circle r="40" /><text className="at-operation-symbol" textAnchor="middle" y={productVisible ? -5 : 10}>{productVisible ? 'x × w' : '×'}</text>{productVisible ? <text className="at-operation-value" textAnchor="middle" y="18">{number(product)}</text> : null}<text className="at-stage-label" textAnchor="middle" y="-52">逐项乘法</text></g>
        {phase < .4 ? <><Operand x={mix(x[0], p[0] - 24, move)} y={mix(x[1], p[1], move)} value={input} opacity={1 - clamp01((phase - .31) / .09)} scale={1 - move * .2} /><Operand x={mix(w[0], p[0] + 24, move)} y={mix(w[1], p[1], move)} value={weight} kind="weight" opacity={1 - clamp01((phase - .31) / .09)} scale={1 - move * .2} /></> : null}
        {phase >= .49 && phase < .78 ? <Operand x={mix(p[0], a[0], productMove)} y={mix(p[1], a[1], productMove)} value={product} kind={product < 0 ? 'negative' : 'product'} /> : null}
        <g className={`at-accumulator ${isAdded ? 'active' : ''}`} transform={`translate(${a[0]} ${a[1]})`}><rect x="-73" y="-40" width="146" height="80" rx="8" /><text className="at-stage-label" textAnchor="middle" y="-53">累加器 Σ</text><text className="at-accumulator-count" textAnchor="middle" y="-15">前 {index + (isAdded ? 1 : 0)} 项{biasAdded ? ' + b' : ''}</text><text className="at-accumulator-value" textAnchor="middle" y="16">{number(sum)}</text>{isAdded && biasPhase === 0 ? <text className="at-add-label" textAnchor="middle" y="64">已加入 {number(product)}</text> : null}</g>
        <g className={`at-result-node ${outputPhase >= 1 ? 'active' : ''}`} transform={`translate(${o[0]} ${o[1]})`}><rect x="-61" y="-35" width="122" height="70" rx="7" /><text className="at-stage-label" textAnchor="middle" y="-53">输出</text><text className="at-result-value" textAnchor="middle" y="7">{outputPhase >= 1 ? number(Number.isFinite(output) ? output : total) : '···'}</text>{outputPhase >= 1 ? <text className="at-add-label" textAnchor="middle" y="60">全部维度完成</text> : null}</g>
        {biasPhase > 0 ? <g opacity={clamp01(biasPhase * 3)}><text className="at-bias-label" x={a[0]} y={a[1] - (compact ? 82 : 85)} textAnchor="middle">+ 偏置 b = {number(bias)}</text><path className="at-bias-arrow" d={`M${a[0]} ${a[1] - 73} V${a[1] - 43}`} pathLength="1" strokeDasharray={`${biasPhase} 1`} /></g> : null}
        {!compact ? <text className="at-stage-caption" x="470" y="264" textAnchor="middle">{stageText}</text> : null}
    </svg>;
}

/** A fully seekable multiplication / accumulation view, using every supplied dimension. */
export function WeightedSum({ title = '展开一次加权求和', subtitle, inputs, weights, bias = 0, output, progress = 1, labels, onPause, inputLabel = '输入 x', weightLabel = '权重 w', outputLabel = '加权输出', compact = true, className = '' }) {
    const id = useId();
    const terms = useMemo(() => weightedTerms(inputs, weights, bias), [inputs, weights, bias]);
    const [inspection, setInspection] = useState(null);
    const [pageChoice, setPageChoice] = useState(null);
    const count = terms?.products.length || 0;
    const frame = accumulationFrame(progress, count);
    const manual = inspection?.atProgress === progress && inspection.index < count;
    const index = manual ? Math.min(count - 1, inspection.index) : frame.index;
    const phase = manual ? inspection.phase : frame.phase;
    const page = pageChoice?.atProgress === progress ? pageChoice.page : Math.floor(index / PAGE_SIZE);
    const pageStart = Math.max(0, Math.min(Math.ceil(count / PAGE_SIZE) - 1, page)) * PAGE_SIZE;
    const choose = (nextIndex, nextPhase = 1) => {
        onPause?.();
        setInspection({ atProgress: progress, inputs, weights, index: nextIndex, phase: nextPhase });
        setPageChoice({ atProgress: progress, inputs, page: Math.floor(nextIndex / PAGE_SIZE) });
    };
    if (!terms) return <section className={`arithmetic-theater at-empty ${className}`} aria-label={title}><div className="at-eyebrow">NUMERICAL COMPUTATION</div><h3>{title}</h3><p>等待完整的输入向量与对应权重。</p><span>读取真实计算记录后，这里会逐项展示乘法、累加和输出。</span></section>;
    const windowStart = Math.min(Math.max(0, index - 4), Math.max(0, count - WINDOW_SIZE));
    const displayed = terms.products.slice(windowStart, windowStart + WINDOW_SIZE);
    const complete = index + (phase >= .78 ? 1 : 0);
    const partial = terms.prefixes[complete];
    const biasPhase = manual ? 0 : frame.biasPhase;
    const outputPhase = manual ? 0 : frame.outputPhase;
    const outputError = Number.isFinite(output) ? Math.abs(output - terms.total) : null;
    const stageProps = { index, phase, before: terms.prefixes[index], after: terms.prefixes[index + 1], input: inputs[index], weight: weights[index], product: terms.products[index], bias, biasPhase, outputPhase, total: terms.total, output };
    const selectedLabel = Array.isArray(labels) ? labels[index] : null;
    const termCards = <div className="at-term-window" role="group" aria-label={`逐项贡献，展示维度 ${windowStart} 到 ${windowStart + displayed.length - 1}`}>
        {displayed.map((product, offset) => { const j = windowStart + offset; return <button type="button" key={j} onClick={() => choose(j)} aria-pressed={j === index} className={`${j === index ? 'selected' : ''} ${j < complete ? 'completed' : ''}`} title={`x[${j}] × w[${j}] = ${inputs[j]} × ${weights[j]} = ${product}`}><span className="at-term-index">j = {j}{j < complete ? <i>✓</i> : null}</span><span className="at-term-input">{number(inputs[j], 3)}</span><span className="at-term-weight">× {number(weights[j], 3)}</span><b className={product < 0 ? 'negative' : ''}>{number(product, 4)}</b></button>; })}
    </div>;
    return <section className={`arithmetic-theater ${compact ? 'at-compact' : ''} ${className}`} aria-labelledby={`${id}-title`}>
        <header className="at-header"><div><div className="at-eyebrow">MULTIPLY · ACCUMULATE · EMIT</div><h3 id={`${id}-title`}>{title}</h3>{subtitle ? <p>{subtitle}</p> : null}</div><div className="at-step-badge"><b>{String(index + 1).padStart(2, '0')}</b><span>/ {count} 项</span></div></header>
        <div className="at-array-toolbar"><span className="at-array-hint">点击色块，检查该维乘加</span><div className="at-pager"><button type="button" disabled={pageStart === 0} onClick={() => { onPause?.(); setPageChoice({ atProgress: progress, inputs, page: page - 1 }); }} aria-label="查看前 64 个维度">←</button><span>{pageStart}–{Math.min(count, pageStart + PAGE_SIZE) - 1} <small>/ {count} 维</small></span><button type="button" disabled={pageStart + PAGE_SIZE >= count} onClick={() => { onPause?.(); setPageChoice({ atProgress: progress, inputs, page: page + 1 }); }} aria-label="查看后 64 个维度">→</button></div></div>
        <div className="at-tensor-pair"><TensorCells values={inputs} peak={terms.inputPeak} start={pageStart} selected={index} onSelect={choose} label={inputLabel} /><span className="at-pair-operator" aria-hidden="true">⊙</span><TensorCells values={weights} peak={terms.weightPeak} start={pageStart} selected={index} onSelect={choose} label={weightLabel} weight /></div>
        <div className="at-computation-paper">
            <div className="at-paper-topline"><span><i /> {manual ? '单步检查' : '正在计算'} · j = {index}{selectedLabel ? ` · ${selectedLabel}` : ''}</span><code>Σ x[j] × w[j] + b</code></div>
            <ArithmeticStage {...stageProps} /><ArithmeticStage {...stageProps} compact />
            <div className="at-current-equation"><span>本项贡献</span><code><b className="at-input-number">{number(inputs[index], 6)}</b><i>×</i><b className="at-weight-number">{number(weights[index], 6)}</b><i>=</i><b>{number(terms.products[index], 6)}</b></code></div>
            <div className="at-operation-scrub"><label htmlFor={`${id}-phase`}>检查本项动画</label><input id={`${id}-phase`} type="range" min="0" max="1000" value={Math.round(phase * 1000)} onChange={event => choose(index, Number(event.target.value) / 1000)} aria-valuetext={`${Math.round(phase * 100)}%，${phase < .4 ? '输入与权重汇合' : phase < .78 ? '乘积进入累加器' : '完成累加'}`} /><span>{phase < .4 ? '取数 → 相乘' : phase < .78 ? '乘积 → 累加' : '完成本项'}</span></div>
        </div>
        {compact ? <details className="at-terms-details"><summary>逐项数值表 <span>当前窗口 {windowStart}–{windowStart + displayed.length - 1} / {count} 维</span></summary>{termCards}</details> : termCards}
        <div className="at-term-selector"><label htmlFor={`${id}-dimension`}>检查维度</label><input id={`${id}-dimension`} type="range" min="0" max={Math.max(0, count - 1)} value={index} onChange={event => choose(Number(event.target.value))} aria-valuetext={`第 ${index + 1} 项，共 ${count} 项`} /><output htmlFor={`${id}-dimension`}>j = {index}</output>{manual ? <button type="button" onClick={() => { setInspection(null); setPageChoice(null); }}>返回播放位置</button> : null}</div>
        <footer className="at-sum-ledger"><div><span>当前部分和 <small>{complete} / {count} 项</small></span><b>{number(partial, 6)}</b><small>Σ 前 {complete} 项，不含偏置</small></div><div><span>完整内积 <small>全部 {count} 项</small></span><b>{number(terms.dot, 6)}</b><small>偏置 b = {number(bias, 6)}</small></div><div className={outputPhase >= 1 ? 'is-emitted' : ''}><span>{outputLabel}</span><b>{number(Number.isFinite(output) ? output : terms.total, 6)}</b><small>{Number.isFinite(output) ? '记录中的实际输出' : '完整内积 + 偏置'}</small></div></footer>
        <p className="at-data-note">每次展示最多 {WINDOW_SIZE} 项贡献；完整求和使用全部 {count} 维。{count > WINDOW_SIZE ? '前 3 项慢速展开，其余按同一规则累加。' : ''}{outputError !== null ? `逐项重算与记录输出差值 ${outputError.toExponential(2)}。` : '显示值已舍入，计算使用原始精度。'}</p>
    </section>;
}

export function ValueTransform({ value, output, kind = 'sigmoid', progress = 1, title, className = '' }) {
    const p = clamp01(progress);
    const fn = ['sigmoid', 'tanh', 'relu'].includes(kind) ? kind : 'sigmoid';
    const computed = activationValue(value, fn);
    const actual = Number.isFinite(output) ? output : computed;
    const domain = Math.max(4, Math.min(20, Math.abs(value || 0) * 1.25));
    const bounds = fn === 'relu' ? [0, domain] : fn === 'tanh' ? [-1, 1] : [0, 1];
    const xToPlot = x => 48 + (x + domain) / (domain * 2) * 340;
    const yToPlot = y => 182 - (y - bounds[0]) / (bounds[1] - bounds[0]) * 142;
    const points = Array.from({ length: 101 }, (_, i) => { const x = -domain + i / 100 * domain * 2; return `${i ? 'L' : 'M'}${xToPlot(x).toFixed(2)} ${yToPlot(activationValue(x, fn)).toFixed(2)}`; }).join(' ');
    const px = xToPlot(Math.max(-domain, Math.min(domain, value || 0)));
    const py = yToPlot(computed ?? 0);
    const enter = clamp01(p / .45);
    const exit = clamp01((p - .45) / .55);
    const pointX = mix(px, px, enter);
    const pointY = mix(yToPlot(0), py, enter);
    const formula = fn === 'sigmoid' ? 'σ(z) = 1 / (1 + e⁻ᶻ)' : fn === 'tanh' ? 'tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)' : 'ReLU(z) = max(0, z)';
    return <section className={`at-transform ${className}`} aria-label={title || `${fn} 激活函数`}><div className="at-transform-heading"><span className="at-eyebrow">NONLINEAR ACTIVATION</span><h3>{title || `${fn === 'sigmoid' ? 'Sigmoid' : fn === 'tanh' ? 'Tanh' : 'ReLU'}：把加权和变为激活值`}</h3><code>{formula}</code></div><div className="at-transform-body"><div className="at-transform-value"><span>加权和 z</span><b>{number(value)}</b><small>输入沿曲线查找映射</small></div><svg className="at-function-plot" viewBox="0 0 430 220" role="img" aria-label={`${fn} 曲线：输入 ${number(value, 6)}，输出 ${number(actual, 6)}`}><path className="at-function-grid" d={`M48 40 H388 M48 111 H388 M48 182 H388 M48 40 V182 M218 40 V182 M388 40 V182`} /><path className="at-function-axis" d={`M48 ${yToPlot(0)} H403 M218 194 V27`} /><path className="at-function-curve" d={points} />{computed !== null ? <><path className="at-function-projection" d={`M${px} ${yToPlot(0)} V${pointY}`} /><path className="at-function-projection" d={`M${px} ${py} H${mix(px, 409, exit)}`} /><circle className="at-function-dot" cx={pointX} cy={pointY} r="6" /><text className="at-function-annotation" x={px < 280 ? px + 10 : px - 10} y={Math.max(22, py - 12)} textAnchor={px < 280 ? 'start' : 'end'}>{number(actual)}</text></> : null}<text x="48" y="207" textAnchor="middle">−{number(domain, 1)}</text><text x="218" y="207" textAnchor="middle">0</text><text x="388" y="207" textAnchor="middle">{number(domain, 1)}</text><text x="36" y="45" textAnchor="end">{bounds[1]}</text><text x="36" y="185" textAnchor="end">{bounds[0]}</text><text x="410" y="207">z</text></svg><div className={`at-transform-value at-transform-output ${p >= 1 ? 'ready' : ''}`}><span>{fn === 'sigmoid' ? '门开合程度' : '激活输出'}</span><b>{p >= .9 ? number(actual) : '···'}</b><small>{fn === 'sigmoid' ? '范围 0 到 1' : fn === 'tanh' ? '范围 −1 到 1' : '负值归零，保留正值'}</small></div></div>{Math.abs(value || 0) > domain ? <p className="at-data-note">输入超出曲线显示范围，标在边界；输出使用完整输入计算。</p> : null}</section>;
}

export function GateMemory({ step, progress = 1 }) {
    const p = clamp01(progress);
    const f = step?.f, previous = step?.c_previous, i = step?.i, g = step?.g, o = step?.o;
    const valid = [f, previous, i, g, o].every(Number.isFinite);
    const retained = valid ? f * previous : null;
    const written = valid ? i * g : null;
    const cell = Number.isFinite(step?.c) ? step.c : valid ? retained + written : null;
    const hidden = Number.isFinite(step?.h) ? step.h : cell !== null && Number.isFinite(o) ? o * Math.tanh(cell) : null;
    return <section className="at-memory" aria-label="LSTM 记忆与隐藏状态的数值计算"><div className="at-memory-header"><span className="at-eyebrow">CELL STATE → HIDDEN STATE</span><h3>保留旧记忆，写入新信息</h3></div><div className="at-memory-flow"><div className={`at-memory-branch ${p >= .2 ? 'active' : ''}`}><span>保留的旧记忆</span><code>{number(f)} <i>×</i> {number(previous)}</code><b>{p >= .2 ? number(retained) : '···'}</b><small>遗忘门 fₜ × cₜ₋₁</small></div><span className="at-memory-operator" style={{ opacity: .25 + clamp01(p * 3) * .75 }}>+</span><div className={`at-memory-branch ${p >= .4 ? 'active' : ''}`}><span>写入的新记忆</span><code>{number(i)} <i>×</i> {number(g)}</code><b>{p >= .4 ? number(written) : '···'}</b><small>输入门 iₜ × 候选 gₜ</small></div><span className="at-memory-operator" style={{ opacity: .25 + clamp01((p - .4) * 3) * .75 }}>=</span><div className={`at-memory-cell ${p >= .6 ? 'active' : ''}`}><span>新记忆 cₜ</span><b>{p >= .6 ? number(cell) : '···'}</b><div className="at-memory-bar"><i style={{ transform: `scaleX(${clamp01((p - .4) / .2)})` }} /></div><small>{p >= .6 ? '两条支路汇合' : '等待乘积汇入'}</small></div></div><div className="at-memory-hidden"><span>传给下一时刻的隐藏状态</span><code>hₜ = oₜ × tanh(cₜ)</code><div><b>{number(o)}</b><i>×</i><b>tanh({number(cell)})</b><i>=</i><strong className={p >= 1 ? 'active' : ''}>{p >= 1 ? number(hidden) : '···'}</strong></div></div></section>;
}
