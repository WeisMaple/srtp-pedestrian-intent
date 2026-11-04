import { useEffect, useState } from 'react';
import CorridorStage from './CorridorStage';
import ContextRail from './ContextRail';
import StepNav from './StepNav';
import TransportDeck from './TransportDeck';
import NumericLedger from './NumericLedger';
import LensPanel from './LensPanel';
import RecordLens from './RecordLens';
import { beatsForStep, STEPS } from '../domain/timeline';
import { flowState } from '../domain/calculation';

function yoloModuleLabel(module) {
    const type = module.type || '模块';
    return `#${module.index} · ${type}${module.representative_unavailable ? ' · 无代表卷积' : ''}`;
}

function stgcnBlockLabel(block) {
    const channels = block.in_channels != null && block.out_channels != null
        ? ` · ${block.in_channels}→${block.out_channels} 通道` : '';
    return `第 ${block.index} 块${channels}`;
}

function StructureRetry({ error, onRetry }) {
    if (!error) return null;
    return <span className="structure-retry"><small className="mono">{error}</small>
        <button type="button" onClick={onRetry}>重试模型结构</button></span>;
}

function StageSelectors({ state, actions, structure, live, track, job, recordOpen, onToggleRecord,
    structureError, onRetryStructure }) {
    const modules = structure?.yoloModules || [];
    const blocks = structure?.stgcnBlocks || [];
    const collectedModules = Object.keys(live?.details?.yolo || {}).length;
    const collectedBlocks = Object.keys(live?.details?.stgcn || {}).length;
    if (state.step === 2) {
        return <div className="stage-step-tools">
            <label><span className="mono">YOLO 模块</span>
                <select aria-label="选择 YOLO 模块" value={state.module}
                    disabled={!modules.length}
                    onChange={event => actions.selectModule(Number(event.target.value))}>
                    {modules.length
                        ? modules.map(module => <option key={module.index} value={module.index}>{yoloModuleLabel(module)}</option>)
                        : <option value={state.module}>模型结构未就绪</option>}
                </select>
            </label>
            {job?.status === 'completed' && <button type="button" className="record-toggle"
                aria-pressed={recordOpen} onClick={onToggleRecord}>{recordOpen ? '返回实时' : '记录重算'}</button>}
            {modules.length > 0 && <small className="mono">点击特征平面打开放大镜 · 详细采集 {collectedModules} / {modules.length}</small>}
            {!modules.length && <StructureRetry error={structureError} onRetry={onRetryStructure} />}
        </div>;
    }
    if (state.step === 3) {
        return <div className="stage-step-tools">
            <small className="mono">姿态带为实时采样历史（最多 10 个时间步）；有效观测 {track?.history_length ?? '—'} / 50 来自真实观测，不补造完整 50 帧。</small>
        </div>;
    }
    if (state.step === 5) {
        return <div className="stage-step-tools">
            <label><span className="mono">ST-GCN 块</span>
                <select aria-label="选择 ST-GCN 块" value={state.block}
                    disabled={!blocks.length}
                    onChange={event => actions.selectBlock(Number(event.target.value))}>
                    {blocks.length
                        ? blocks.map(block => <option key={block.index} value={block.index}>{stgcnBlockLabel(block)}</option>)
                        : <option value={state.block}>模型结构未就绪</option>}
                </select>
            </label>
            {blocks.length > 0 && <small className="mono">点击骨架关节打开放大镜 · 详细采集 {collectedBlocks} / {blocks.length}</small>}
            {!blocks.length && <StructureRetry error={structureError} onRetry={onRetryStructure} />}
        </div>;
    }
    return null;
}

export default function Corridor({ job, live, structure, structureError, onRetryStructure, state, actions,
    selectedTrack, tracks, onSelectTrack, calculation, terms, filter, page, onFilter, onPage, onSeek,
    lensOpen, onCloseLens }) {
    const beats = beatsForStep(state.step);
    const stepTitle = STEPS.find(item => item.step === state.step)?.title || '视频帧';
    const flow = calculation ? flowState(calculation, state.termIndex) : null;
    const [recordOpen, setRecordOpen] = useState(false);
    const waitingLabel = state.step === 2 ? `等待 YOLO 模块 ${state.module} 的详细采集`
        : state.step === 5 ? `等待 ST-GCN 第 ${state.block} 块的详细采集` : '等待该步骤的真实计算';

    useEffect(() => {
        if (state.step !== 2 || job?.status !== 'completed') setRecordOpen(false);
    }, [state.step, job?.status]);

    useEffect(() => {
        const inField = target => target?.isContentEditable
            || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'].includes(target?.tagName);
        const onKey = event => {
            if (event.metaKey || event.ctrlKey || event.altKey || inField(event.target)) return;
            if (event.code === 'Space') {
                event.preventDefault(); actions.togglePause(); return;
            }
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
            event.preventDefault();
            const direction = event.key === 'ArrowRight' ? 1 : -1;
            if (lensOpen && terms.length) {
                const next = Math.max(0, Math.min(terms.length - 1, state.termIndex + direction));
                actions.seekTerm(next);
            } else actions.stepBeat(direction);
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [actions, lensOpen, terms.length, state.termIndex]);

    return <section className="corridor-page">
        <ContextRail job={job} tracks={tracks} track={selectedTrack} onSelect={onSelectTrack} />
        <div className="corridor-main">
            <div className="corridor-stage-wrap">
                <CorridorStage step={state.step} track={selectedTrack} live={live} trackId={selectedTrack?.track_id}
                    structure={structure} onCameraInteract={actions.cameraInteract} onPick={packet => {
                        if (packet?.kind === 'bay') actions.selectStep(packet.step);
                        else if (packet?.kind === 'module') {
                            actions.selectStep(2); actions.selectModule(packet.module); onSeek(0);
                        } else if (packet?.kind === 'joint') {
                            actions.selectStep(5); onSeek(0);
                        }
                    }} />
                <div className="stage-overlay">
                    <StepNav state={state} actions={actions} />
                    <StageSelectors state={state} actions={actions} structure={structure} live={live} track={selectedTrack}
                        job={job} recordOpen={recordOpen} onToggleRecord={() => setRecordOpen(value => !value)}
                        structureError={structureError} onRetryStructure={onRetryStructure} />
                    <LensPanel open={lensOpen} calculation={calculation} state={flow} terms={terms}
                        waitingLabel={waitingLabel} filter={filter} page={page} onFilter={onFilter} onPage={onPage}
                        onSeek={index => { onSeek(index); actions.seekTerm(index); }} onClose={onCloseLens} />
                </div>
            </div>
            <TransportDeck state={state} actions={actions} currentBeat={state.beat}
                totalBeats={beats.length} stepTitle={stepTitle} />
            {recordOpen ? <RecordLens job={job} structure={structure} onClose={() => setRecordOpen(false)} />
                : !lensOpen && <NumericLedger calculation={calculation} state={flow} terms={terms}
                    filter={filter} page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />}
        </div>
    </section>;
}
