export default function TransportDeck({ state, actions, currentBeat, totalBeats, stepTitle }) {
    return <div className="transport-deck">
        <button className="transport-play" aria-label={state.paused ? '播放讲解' : '暂停讲解'}
            onClick={actions.togglePause}>{state.paused ? '▶' : 'Ⅱ'}</button>
        <div className="transport-progress">
            <div><strong>{stepTitle}</strong><span className="mono">{currentBeat + 1} / {totalBeats}</span></div>
            <input aria-label="讲解进度" type="range" min="0" max={Math.max(0, totalBeats - 1)}
                value={Math.min(currentBeat, Math.max(0, totalBeats - 1))}
                onChange={event => actions.seekBeat?.(Number(event.target.value))} />
        </div>
        <button className="transport-next" onClick={actions.next}>下一步 →</button>
        <button className="transport-follow" aria-pressed={state.following} onClick={actions.followLatest}>跟随最新</button>
    </div>;
}
