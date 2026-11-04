import { useEffect, useMemo, useState } from 'react';
import AppShell from './components/AppShell';
import Workbench from './components/Workbench';
import Corridor from './components/Corridor';
import Results from './components/Results';
import { hashForPage, pageFromHash } from './app/page';
import { useJob } from './hooks/useJob';
import { useLiveComputation } from './hooks/useLiveComputation';
import { useModelStructure } from './hooks/useModelStructure';
import { useCorridor } from './app/useCorridor';
import { normalizeCalculation } from './domain/calculation';
import { filterTerms } from './domain/ledger';
import { latestCalculation } from './domain/workbench';
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
    const tracks = live.snapshot?.tracks?.length ? live.snapshot.tracks : (jobState.job?.tracks || []);
    const selectedTrack = tracks.find(track => String(track.track_id) === String(selectedTrackId)) || tracks[0] || null;
    const { state, actions } = useCorridor({ job: jobState.job, live: live.snapshot, trackId: selectedTrack?.track_id });
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
    const raw = useMemo(() => latestCalculation(live.snapshot, state.step, model.structure,
        state.module, state.block, selectedTrack?.track_id, live.calculations),
        [live.snapshot, live.calculations, state.step, state.module, state.block, model.structure, selectedTrack?.track_id]);
    const calculation = useMemo(() => normalizeCalculation(raw), [raw]);
    const terms = useMemo(() => calculation ? filterTerms(calculation.terms, filter) : [], [calculation, filter]);
    const status = { ready: Boolean(jobState.health?.models_ready), label: jobState.health?.model_name || '本地模型' };
    return <AppShell page={page} onNavigate={navigate} status={status}>
        {page === 'workbench' && <Workbench job={jobState.job} health={jobState.health} error={jobState.error}
            pending={jobState.pending} onUpload={jobState.upload} onStart={jobState.start} onCancel={jobState.cancel}
            onOpenCorridor={() => navigate('corridor')} onOpenResults={() => navigate('results')} />}
        {page === 'corridor' && <Corridor job={jobState.job} live={live.snapshot} structure={model.structure}
            state={state} actions={actions} selectedTrack={selectedTrack} tracks={tracks}
            onSelectTrack={id => { setSelectedTrackId(id); actions.selectTrack(id); }}
            calculation={calculation} terms={terms} filter={filter} page={pageIndex}
            onFilter={value => { setFilter(value); setPageIndex(0); }} onPage={setPageIndex}
            onSeek={index => { actions.seekTerm(index); setLensOpen(true); }} lensOpen={lensOpen}
            onCloseLens={() => setLensOpen(false)} />}
        {page === 'results' && <Results job={jobState.job} onOpenCorridor={() => navigate('corridor')} />}
    </AppShell>;
}
