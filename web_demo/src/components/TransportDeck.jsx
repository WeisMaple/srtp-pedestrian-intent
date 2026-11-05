import { Button, Range } from './ui/Controls';
import { SPEEDS, usesComputationClock } from '../app/useCorridor';

export default function TransportDeck({ state, actions, currentBeat, totalBeats, stepTitle, recorded = false }) {
    const graph = usesComputationClock(state);
    const playing = graph ? state.autoGraph && !state.frozen : !state.paused;
    const togglePlayback = () => {
        if (graph) actions.setGraphAuto(!playing);
        else if (recorded) { if (playing) actions.pauseNarration(); else actions.followLatest(); }
        else actions.togglePause();
    };
    return <div className="transport-deck">
        <Button className="transport-play" aria-label={graph ? playing ? '暂停计算演示' : '播放计算演示' : recorded ? playing ? '暂停讲解' : '播放讲解' : state.paused ? '播放并跟随最新数值' : '暂停并冻结数值'}
            aria-pressed={playing} onClick={togglePlayback}>{playing ? 'Ⅱ' : '▶'}</Button>
        <Button className="transport-step" aria-label="从头开始讲解" onClick={actions.restart}>从头讲解</Button>
        <Button className="transport-step" aria-label="上一讲解节拍"
            onClick={actions.previousBeat}>←</Button>
        <div className="transport-progress">
            <div><strong>{stepTitle} · 讲解节拍</strong><span className="mono">{currentBeat + 1} / {totalBeats}</span></div>
            <Range aria-label="讲解进度" type="range" min="0" max={Math.max(0, totalBeats - 1)}
                value={Math.min(currentBeat, Math.max(0, totalBeats - 1))}
                onChange={event => actions.seekBeat?.(Number(event.target.value))} />
        </div>
        <div className="transport-speed" role="group" aria-label="讲解速度">
            {SPEEDS.map(speed => <Button key={speed} aria-pressed={state.speed === speed}
                onClick={() => actions.setSpeed(speed)}>{speed}×</Button>)}
        </div>
        <Button className="transport-next" disabled={state.step <= 1} onClick={() => actions.selectStep(state.step - 1)}>← 上一步</Button>
        <Button className="transport-next" disabled={state.step >= 6} onClick={actions.next}>下一步 →</Button>
        {!recorded && <Button className="transport-follow" aria-pressed={state.following} onClick={actions.followLatest}>跟随最新</Button>}
    </div>;
}
