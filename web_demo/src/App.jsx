import { useCallback, useEffect, useMemo, useState } from 'react';
import AppShell from './components/AppShell';
import Workbench from './components/Workbench';
import Corridor from './components/Corridor';
import Results from './components/Results';
import LiveCamera from './components/LiveCamera';
import { hashForPage, pageFromHash } from './app/page';
import { useJob } from './hooks/useJob';
import { useLiveComputation } from './hooks/useLiveComputation';
import { useModelStructure } from './hooks/useModelStructure';
import { useCorridor } from './app/useCorridor';
import { normalizeCalculation } from './domain/calculation';
import { filterTerms } from './domain/ledger';
import { latestCalculation } from './domain/workbench';
import { visibleTracks } from './domain/process';
import { useRecordedGraph } from './hooks/useRecordedGraph';
import { useAutomaticGraphSelection } from './hooks/useAutomaticGraphSelection';
import { useFrameReplay } from './hooks/useFrameReplay';
import { stepReady } from './domain/timeline';
import { currentGraphRows } from './domain/spatiotemporal';
import './styles/workbench.css';
import './styles/ledger.css';

export default function App() {
    const [page, setPage] = useState(() => pageFromHash(window.location.hash));
    const jobState = useJob();
    const live = useLiveComputation(jobState.job);
    const model = useModelStructure();
    const [selectedTrackId, setSelectedTrackId] = useState(null);
    const [filter, setFilter] = useState('all');
    const [pageIndex, setPageIndex] = useState(0);
    const [lensOpen, setLensOpen] = useState(false);
    const tracks = visibleTracks(live.snapshot, jobState.job);
    const selectedTrack = tracks.find(track => String(track.track_id) === String(selectedTrackId)) || tracks[0] || null;
    const externalReadiness = page === 'corridor' && jobState.job?.status === 'completed';
    const { state, actions } = useCorridor({ job: jobState.job, live: live.snapshot, trackId: selectedTrack?.track_id, externalReadiness });
    const replay = useFrameReplay(jobState.job, page === 'corridor', state.speed);
    const replayTrack = replay.snapshot?.tracks.find(track => String(track.track_id) === String(selectedTrackId))
        || replay.snapshot?.tracks[0];
    const selectRecord = useCallback((frame, trackId) => {
        replay.seek(frame); setSelectedTrackId(trackId);
    }, [replay.seek]);
    const record = useRecordedGraph(jobState.job, page === 'corridor' && ([4, 5].includes(state.step) || (state.step === 6 && state.intentDetail)),
        replay.active ? { frame: replay.frame, trackId: replayTrack?.track_id, onSelect: selectRecord } : null);
    const viewLive = record.active ? record.snapshot : replay.active ? replay.snapshot : live.snapshot;
    const viewTracks = replay.active ? viewLive?.tracks || [] : tracks;
    const initialTrack = record.active ? record.selection?.track || null : replay.active
        ? viewTracks.find(track => String(track.track_id) === String(selectedTrackId)) || viewTracks[0] || null : selectedTrack;
    // Once a saved network window is selected, every view returns to that same
    // source frame. The video cursor survives switching between stages.
    useEffect(() => {
        if (record.active && record.selection && replay.status === 'ready'
            && (replay.frame !== record.selection.frame || String(selectedTrackId) !== String(record.selection.track.track_id)))
            selectRecord(record.selection.frame, record.selection.track.track_id);
    }, [record.active, record.selection?.frame, record.selection?.track.track_id, replay.status, replay.frame, selectedTrackId, selectRecord]);
    useEffect(() => { if (replay.active) actions.pauseNarration(); }, [replay.active, actions]);
    useEffect(() => {
        if (externalReadiness) actions.setReady(stepReady(state.step, viewLive, jobState.job));
    }, [externalReadiness, state.step, viewLive, jobState.job, actions]);
    useEffect(() => { replay.pause(); }, [state.step, state.intentDetail, replay.pause]);
    const graphSnapshot = useMemo(() => {
        if (!viewLive || state.step !== 5) return viewLive;
        const stgcn = {};
        for (const row of currentGraphRows(viewLive)) (stgcn[row.metadata.block] ||= []).push(row);
        return { ...viewLive, details: { ...viewLive.details, stgcn } };
    }, [viewLive, state.step]);
    const graphPick = useAutomaticGraphSelection(graphSnapshot, state.step === 5 || replay.active ? [] : live.calculations,
        state, initialTrack?.track_id, { onWindow: actions.nextGraph, onTrack: record.active ? null : setSelectedTrackId });
    const viewState = graphPick ? { ...state, block: graphPick.block, boundary: graphPick.boundary } : state;
    const viewTrack = graphPick ? viewTracks.find(track => String(track.track_id) === String(graphPick.trackId)) || initialTrack : initialTrack;
    useEffect(() => {
        if (state.frozen) live.freeze(true);
        else if (live.frozen) live.resume();
        // Freeze only on the transport pause control; step navigation still follows live data.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [state.frozen]);
    useEffect(() => { if (selectedTrackId == null && tracks.length) setSelectedTrackId(tracks[0].track_id); }, [tracks, selectedTrackId]);
    useEffect(() => {
        const onChange = () => setPage(pageFromHash(window.location.hash));
        window.addEventListener('hashchange', onChange);
        return () => window.removeEventListener('hashchange', onChange);
    }, []);
    const navigate = next => { window.location.hash = hashForPage(next); setPage(pageFromHash(hashForPage(next))); };
    const raw = useMemo(() => latestCalculation(viewLive, state.step, model.structure,
        state.module, viewState.block, viewTrack?.track_id ?? null, replay.active ? [] : live.calculations,
        { operation: state.operation, boundary: viewState.boundary, classIndex: state.intentClass }),
        [viewLive, live.calculations, state.step, state.module, viewState.block, state.operation, viewState.boundary, state.intentClass, model.structure, viewTrack?.track_id, replay.active]);
    const calculation = useMemo(() => normalizeCalculation(raw), [raw]);
    const terms = useMemo(() => calculation ? filterTerms(calculation.terms, filter) : [], [calculation, filter]);
    const status = { ready: Boolean(jobState.health?.models_ready), label: jobState.health?.model_name || '本地模型' };
    return <AppShell page={page} onNavigate={navigate} status={status}>
        {page === 'workbench' && <Workbench job={jobState.job} health={jobState.health} error={jobState.error}
            live={live.snapshot}
            pending={jobState.pending} onUpload={jobState.upload} onStart={jobState.start} onCancel={jobState.cancel}
            onOpenCorridor={step => { if (Number.isInteger(step)) actions.selectStep(step); navigate('corridor'); }} onOpenResults={() => navigate('results')} />}
        {page === 'corridor' && <Corridor key={jobState.job?.id || 'empty'} job={jobState.job} live={viewLive} structure={model.structure} record={record}
            stream={live} replay={replay}
            structureError={model.error} onRetryStructure={model.retry}
            state={viewState} actions={actions} selectedTrack={viewTrack} tracks={viewTracks}
            onSelectTrack={id => { if (record.active) record.selectTrack(id); else setSelectedTrackId(id); actions.selectTrack(id); actions.setGraphAuto(false); }}
            calculation={calculation} terms={terms} filter={filter} page={pageIndex}
            onFilter={value => { setFilter(value); setPageIndex(0); }} onPage={setPageIndex}
            onSeek={index => { actions.seekTerm(index); setLensOpen(true); }} lensOpen={lensOpen}
            onCloseLens={() => setLensOpen(false)} />}
        {page === 'camera' && <LiveCamera health={jobState.health} />}
        {page === 'results' && <Results job={jobState.job} onOpenCorridor={() => navigate('corridor')} />}
    </AppShell>;
}
