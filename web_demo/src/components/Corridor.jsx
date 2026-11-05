import { Button, Select } from './ui/Controls';
import { useEffect, useState } from 'react';
import CorridorStage from './CorridorStage';
import ContextRail from './ContextRail';
import StepNav from './StepNav';
import TransportDeck from './TransportDeck';
import ComputationDesk from './ComputationDesk';
import LensPanel from './LensPanel';
import RecordLens from './RecordLens';
import PoseBindingStage from './PoseBindingStage';
import SpatiotemporalComputation from './SpatiotemporalComputation';
import LSTMComputation from './LSTMComputation';
import ReplayTimeline from './ReplayTimeline';
import IntentComputation from './IntentComputation';
import { beatsForStep, STEPS } from '../domain/timeline';
import { flowState } from '../domain/calculation';
import { currentGraphRows } from '../domain/spatiotemporal';

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
        <Button type="button" onClick={onRetry}>重试模型结构</Button></span>;
}

function StageSelectors({ state, actions, structure, live, track, job, recordOpen, onToggleRecord,
    structureError, onRetryStructure }) {
    const modules = structure?.yoloModules || [];
    const blocks = structure?.stgcnBlocks || [];
    const collectedModules = Object.keys(live?.details?.yolo || {}).length;
    const availableBlocks = [...new Set(currentGraphRows(live).filter(row =>
        String(row.track_id) === String(track?.track_id) && row.stage === 'temporal'
        && Boolean(row.metadata?.boundary) === state.boundary).map(row => String(row.metadata.block)))];
    if (state.step === 2) {
        return <div className="stage-step-tools">
            <label><span className="mono">YOLO 模块</span>
                <Select aria-label="选择 YOLO 模块" value={state.module}
                    disabled={!modules.length}
                    onChange={event => actions.selectModule(Number(event.target.value))}>
                    {modules.length
                        ? modules.map(module => <option key={module.index} value={module.index}>{yoloModuleLabel(module)}{live?.details?.yolo?.[module.index] ? ' · 已采集' : job?.status === 'completed' ? ' · 可重算' : ' · 待采集'}</option>)
                        : <option value={state.module}>模型结构未就绪</option>}
                </Select>
            </label>
            {job?.status === 'completed' && <Button type="button" className="record-toggle"
                aria-pressed={recordOpen} onClick={onToggleRecord}>{recordOpen ? '收起记录重算' : '记录重算'}</Button>}
            {modules.length > 0 && <small className="mono">{job?.status === 'completed' ? '选择回看帧，点击记录重算查看该帧的模块算式。' : `模块结构中可点击平面查看数值 · 详细采集 ${collectedModules} / ${modules.length}`}</small>}
            {!live?.details?.yolo?.[state.module] && collectedModules > 0 && <Button type="button" className="process-button"
                onClick={() => actions.selectModule(Number(Object.keys(live.details.yolo)[0]))}>查看已采集模块</Button>}
            {!modules.length && <StructureRetry error={structureError} onRetry={onRetryStructure} />}
        </div>;
    }
    if (state.step === 3) {
        return <div className="stage-step-tools">
            <small className="mono">检测框 → COCO17 骨架 → ByteTrack ID → 有效历史；当前选中行人已积累 {track?.history_length ?? '—'} / 50 次有效观测。</small>
        </div>;
    }
    if (state.step === 5) {
        return <div className="stage-step-tools">
            <label><span className="mono">ST-GCN 块</span>
                <Select aria-label="选择 ST-GCN 块" value={state.block}
                    disabled={!blocks.length}
                    onChange={event => actions.selectBlock(Number(event.target.value))}>
                    {blocks.length
                        ? blocks.map(block => <option key={block.index} value={block.index}>{stgcnBlockLabel(block)}{availableBlocks.includes(String(block.index - 1)) ? ' · 已采集' : ' · 待采集'}</option>)
                        : <option value={state.block}>模型结构未就绪</option>}
                </Select>
            </label>
            <label>时间窗口 <Select aria-label="时序卷积窗口位置" value={state.boundary ? 'boundary' : 'middle'}
                onChange={event => actions.selectBoundary(event.target.value === 'boundary')}>
                <option value="middle">中间窗口</option><option value="boundary">边界窗口 · 零填充</option></Select></label>
            {blocks.length > 0 && <small className="mono">同帧时间窗口 {availableBlocks.length} / {blocks.length} 块 · 空间与时间同时展示</small>}
            {!availableBlocks.includes(String(state.block - 1)) && availableBlocks.length > 0 && <Button type="button" className="process-button"
                onClick={() => actions.selectBlock(1 + Number(availableBlocks[0]))}>查看已采集块</Button>}
            {!blocks.length && <StructureRetry error={structureError} onRetry={onRetryStructure} />}
        </div>;
    }
    return null;
}

export default function Corridor({ job, live, stream, structure, structureError, onRetryStructure, state, actions,
    selectedTrack, tracks, onSelectTrack, calculation, terms, filter, page, onFilter, onPage, onSeek,
    lensOpen, onCloseLens, record, replay }) {
    const beats = beatsForStep(state.step);
    const stepTitle = STEPS.find(item => item.step === state.step)?.title || '视频帧';
    const flow = calculation ? flowState(calculation, state.termIndex) : null;
    const [recordOpen, setRecordOpen] = useState(false);
    const [showModules, setShowModules] = useState(false);
    const videoReplay = replay?.active && !record?.active;
    const inspect = () => { actions.cameraInteract(); replay?.pause(); };
    const seekFrame = frame => { inspect(); replay.seek(frame); };
    const waitingLabel = state.step === 2 ? `等待 YOLO 模块 ${state.module} 的详细采集`
        : state.step === 5 ? `等待 ST-GCN 第 ${state.block} 块的详细采集` : '等待该步骤的真实计算';

    useEffect(() => {
        if (state.step !== 2 || job?.status !== 'completed') setRecordOpen(false);
    }, [state.step, job?.status]);

    useEffect(() => {
        const inField = target => target?.isContentEditable
            || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY', 'A'].includes(target?.tagName);
        const onKey = event => {
            if (event.metaKey || event.ctrlKey || event.altKey || inField(event.target)) return;
            if (event.code === 'Space') {
                event.preventDefault();
                if (videoReplay) { actions.cameraInteract(); if (replay.playing) replay.pause(); else replay.play(); }
                else if (record?.active) actions.setGraphAuto(!state.autoGraph || state.frozen);
                else actions.togglePause();
                return;
            }
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
            event.preventDefault();
            const direction = event.key === 'ArrowRight' ? 1 : -1;
            if (lensOpen && terms.length) {
                const next = Math.max(0, Math.min(terms.length - 1, state.termIndex + direction));
                actions.seekTerm(next);
            } else if (videoReplay) { actions.cameraInteract(); replay.step(direction); }
            else actions.stepBeat(direction);
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [actions, lensOpen, terms.length, state.termIndex, state.autoGraph, state.frozen, record?.active, videoReplay, replay?.playing, replay?.pause, replay?.play, replay?.step]);

    return <section className={`corridor-page ${state.step === 5 ? 'is-spatiotemporal' : ''}`}>
        <ContextRail job={job} live={live} tracks={tracks} track={selectedTrack} onSelect={id => { inspect(); onSelectTrack(id); }} recorded={replay?.active}
            onFrameReady={state.step > 3 || (state.step === 2 && showModules) ? replay?.onFrameReady : undefined} />
        <div className="corridor-main">
            <div className="corridor-stage-header">
                <StepNav state={state} actions={actions} />
                {state.step===6 && <div className="stage-view-switch" role="group" aria-label="意图展示方式">
                    <Button aria-pressed={state.intentDetail} onClick={() => actions.setIntentDetail(true)}>计算拆解 · 3D</Button>
                    <Button aria-pressed={!state.intentDetail} onClick={() => actions.setIntentDetail(false)}>逐帧结果</Button>
                </div>}
                {videoReplay && <ReplayTimeline replay={replay} onInteract={actions.cameraInteract} />}
                {([4, 5].includes(state.step) || (state.step===6 && state.intentDetail)) && <div className="graph-session">
                    <span className="graph-source">{record?.active ? '记录重算' : '实时采集'} · {record?.active
                        ? record.selection ? `源帧 #${record.selection.frame} · ID ${record.selection.track.track_id}` : '正在读取历史'
                        : `采样帧 #${live?.frame ?? '—'}`}</span>
                    {record?.active && record.choices.length > 0 && <label>记录对象 <Select aria-label="选择计算记录"
                        value={record.selection?.key || ''} onChange={event => { actions.seekTerm(0); record.select(event.target.value); }}>
                        {record.choices.map(item => <option key={item.key} value={item.key}>帧 #{item.frame} · ID {item.track.track_id}</option>)}
                    </Select></label>}
                    <Button type="button" className="process-button" aria-pressed={state.autoGraph}
                        onClick={() => actions.setGraphAuto(!state.autoGraph)}>{state.autoGraph ? '暂停自动演示' : state.step === 5 ? '逐项联动演示' : '自动演示与切换'}</Button>
                    <small>{record?.active ? '就近选择有完整 50 次骨架历史的记录；可切换帧回看。使用原权重重算，数值可能有舍入差异。' : state.step === 4 ? '收到新采样值后更新；完成后自动接入逐门重算。' : '自动选择有数据的块；收到新采样值后更新。'}</small>
                </div>}
                <StageSelectors state={state} actions={actions} structure={structure} live={live} track={selectedTrack}
                    job={job} recordOpen={recordOpen} onToggleRecord={() => { inspect(); setRecordOpen(value => !value); }}
                    structureError={structureError} onRetryStructure={onRetryStructure} />
                {state.step === 2 && <div className="stage-view-switch" role="group" aria-label="YOLO 展示方式">
                    <Button type="button" aria-pressed={!showModules} onClick={() => { actions.cameraInteract(); setShowModules(false); }}>骨架与 ID 绑定</Button>
                    <Button type="button" aria-pressed={showModules} onClick={() => { actions.cameraInteract(); setShowModules(true); }}>YOLO 模块结构</Button>
                </div>}
            </div>
            <div className={`corridor-stage-wrap ${state.step === 5 ? 'is-spacetime' : state.step === 4 ? 'is-memory' : state.step===6 ? 'is-intent' : ''}`}>
                {(state.step <= 3 && !(state.step === 2 && showModules)) ? <PoseBindingStage key={job?.id || 'empty'}
                    job={job} live={live} tracks={tracks} track={selectedTrack} step={state.step} beat={state.beat}
                    onSelect={onSelectTrack} onInteract={inspect} onFrameReady={replay?.onFrameReady} />
                    : state.step === 4 ? <LSTMComputation key={`${job?.id}-${selectedTrack?.track_id}-${record?.selection?.key || 'live'}`}
                        live={live} record={record} track={selectedTrack} state={state} actions={actions} />
                    : state.step === 5 ? <SpatiotemporalComputation job={job} live={live} record={record} track={selectedTrack}
                        structure={structure} state={state} actions={actions} />
                    : state.step===6 ? <IntentComputation live={live} record={record} track={selectedTrack} state={state} actions={actions} />
                    : <CorridorStage step={state.step} track={selectedTrack} live={live} trackId={selectedTrack?.track_id}
                    structure={structure} onCameraInteract={actions.cameraInteract} onPick={packet => {
                        if (packet?.kind === 'bay') actions.selectStep(packet.step);
                        else if (packet?.kind === 'module') {
                            actions.selectStep(2); actions.selectModule(packet.module); onSeek(0);
                        } else if (packet?.kind === 'joint') {
                            actions.selectStep(5); onSeek(0);
                        }
                    }} />}
                {state.step !== 5 && <div className="stage-overlay">
                    <LensPanel open={lensOpen} calculation={calculation} state={flow} terms={terms}
                        waitingLabel={waitingLabel} filter={filter} page={page} onFilter={onFilter} onPage={onPage}
                        onSeek={index => { onSeek(index); actions.seekTerm(index); }} onClose={onCloseLens} />
                </div>}
            </div>
            {state.step !== 5 && <TransportDeck state={state} actions={actions} currentBeat={state.beat}
                totalBeats={beats.length} stepTitle={stepTitle} recorded={replay?.active} />}
            {state.step === 5 ? null : recordOpen ? <RecordLens job={job} structure={structure} frame={replay.frame} module={state.module}
                onFrame={seekFrame} onModule={actions.selectModule} paused={!replay.playing && replay.status === 'ready'} onClose={() => setRecordOpen(false)} />
                : <ComputationDesk job={job} live={live} track={selectedTrack} state={state} calculation={calculation}
                    flow={flow} terms={terms} filter={filter} page={page} onFilter={onFilter} onPage={onPage}
                    onSeek={actions.seekTerm} onInspect={() => onSeek(state.termIndex)} stream={stream} onRetry={stream?.retry} record={record} />}
        </div>
    </section>;
}
