import CorridorStage from './CorridorStage';
import ContextRail from './ContextRail';
import StepNav from './StepNav';
import TransportDeck from './TransportDeck';
import NumericLedger from './NumericLedger';
import LensPanel from './LensPanel';
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

function StageSelectors({ state, actions, structure, live }) {
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
            {modules.length > 0 && <small className="mono">详细采集 {collectedModules} / {modules.length}</small>}
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
            {blocks.length > 0 && <small className="mono">详细采集 {collectedBlocks} / {blocks.length}</small>}
        </div>;
    }
    return null;
}

export default function Corridor({ job, live, structure, state, actions, selectedTrack, tracks, onSelectTrack,
    calculation, terms, filter, page, onFilter, onPage, onSeek, lensOpen, onCloseLens }) {
    const beats = beatsForStep(state.step);
    const stepTitle = STEPS.find(item => item.step === state.step)?.title || '视频帧';
    const flow = calculation ? flowState(calculation, state.termIndex) : null;
    return <section className="corridor-page">
        <ContextRail job={job} tracks={tracks} track={selectedTrack} onSelect={onSelectTrack} />
        <div className="corridor-main">
            <div className="corridor-stage-wrap">
                <CorridorStage step={state.step} track={selectedTrack} live={live} trackId={selectedTrack?.track_id}
                    structure={structure} onPick={packet => { if (packet?.kind === 'bay') actions.selectStep(packet.step); }} />
                <div className="stage-overlay">
                    <StepNav state={state} actions={actions} />
                    <StageSelectors state={state} actions={actions} structure={structure} live={live} />
                    <LensPanel open={lensOpen} calculation={calculation} state={flow} terms={terms}
                        filter={filter} page={page} onFilter={onFilter} onPage={onPage}
                        onSeek={index => { onSeek(index); actions.seekTerm(index); }} onClose={onCloseLens} />
                </div>
            </div>
            <TransportDeck state={state} actions={actions} currentBeat={state.beat}
                totalBeats={beats.length} stepTitle={stepTitle} />
            {!lensOpen && <NumericLedger calculation={calculation} state={flow} terms={terms}
                filter={filter} page={page} onFilter={onFilter} onPage={onPage} onSeek={onSeek} />}
        </div>
    </section>;
}
