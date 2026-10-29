import { useEffect, useState } from 'react';
import { useJob } from './hooks/useJob';
import Header from './components/Header';
import Icon from './components/Icon';
import ProcessPage from './components/ProcessPage';

export default function App() {
    const state = useJob();
    const [selectedId, setSelectedId] = useState(null);
    useEffect(() => { setSelectedId(null); }, [state.job?.id]);
    useEffect(() => {
        if (selectedId == null && state.job?.tracks?.length) setSelectedId(state.job.tracks[0].track_id);
    }, [state.job?.tracks, selectedId]);
    useEffect(() => {
        const onVisibility = () => document.documentElement.classList.toggle('page-hidden', document.hidden);
        document.addEventListener('visibilitychange', onVisibility);
        onVisibility();
        return () => document.removeEventListener('visibilitychange', onVisibility);
    }, []);

    return <><Header health={state.health} />
        {state.error && <div className="error-toast" role="alert"><Icon name="info" size={20} /><span>{state.error}</span><button className="icon-button" aria-label="关闭提示" onClick={state.dismissError}><Icon name="close" size={16} /></button></div>}
        <ProcessPage state={state} job={state.job} selectedId={selectedId} onSelect={setSelectedId} />
        <footer className="app-footer"><span>深度学习课程大作业 · 基于骨架的行人意图分析</span><span>结果仅在本地处理</span></footer>
    </>;
}
