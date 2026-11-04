import { SPEEDS } from '../app/useCorridor';

export default function TransportDeck({ state, actions, currentBeat, totalBeats, stepTitle }) {
    return <div className="transport-deck">
        <button className="transport-play" aria-label={state.paused ? '播放并跟随最新数值' : '暂停并冻结数值'}
            aria-pressed={state.frozen} onClick={actions.togglePause}>{state.paused ? '▶' : 'Ⅱ'}</button>
        <button className="transport-step" aria-label="从头开始讲解" onClick={actions.restart}>从头</button>
        <button className="transport-step" aria-label="上一讲解节拍"
            onClick={actions.previousBeat}>←</button>
        <div className="transport-progress">
            <div><strong>{stepTitle}</strong><span className="mono">{currentBeat + 1} / {totalBeats}</span></div>
            <input aria-label="讲解进度" type="range" min="0" max={Math.max(0, totalBeats - 1)}
                value={Math.min(currentBeat, Math.max(0, totalBeats - 1))}
                onChange={event => actions.seekBeat?.(Number(event.target.value))} />
        </div>
        <div className="transport-speed" role="group" aria-label="讲解速度">
            {SPEEDS.map(speed => <button key={speed} aria-pressed={state.speed === speed}
                onClick={() => actions.setSpeed(speed)}>{speed}×</button>)}
        </div>
        <button className="transport-next" onClick={actions.next}>下一步 →</button>
        <button className="transport-follow" aria-pressed={state.following} onClick={actions.followLatest}>跟随最新</button>
    </div>;
}
