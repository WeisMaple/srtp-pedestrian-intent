import { Button, NumberField, Range } from './ui/Controls';
export default function ReplayTimeline({ replay, onInteract }) {
    if (!replay?.active) return null;
    const run = action => { onInteract(); action(); };
    if (replay.status !== 'ready') return <div className="frame-replay" role="status">
        {replay.status === 'error' ? <>{replay.error} <Button onClick={replay.retry}>重试回看</Button></>
            : replay.status === 'empty' ? '没有可回看的逐帧记录。' : '正在读取逐帧回看记录…'}
    </div>;
    return <section className="frame-replay" aria-label="视频逐帧回看">
        <div className="replay-heading"><strong>视频回看</strong>
            <span className="mono">帧 #{replay.frame} / {replay.maxFrame} · {(replay.frame / replay.fps).toFixed(2)} s</span>
            <small>画面、骨架与 ID 同步 · ← / → 逐帧，空格播放</small></div>
        <div className="replay-controls">
            <Button aria-label={replay.playing ? '暂停视频回看' : '播放视频回看'} aria-pressed={replay.playing}
                onClick={() => run(replay.playing ? replay.pause : replay.play)}>{replay.playing ? 'Ⅱ 暂停' : '▶ 播放'}</Button>
            <Button onClick={() => run(() => { replay.seek(replay.minFrame); replay.play(); })}>从头播放</Button>
            <Button aria-label="回看上一帧" disabled={replay.frame <= replay.minFrame} onClick={() => run(() => replay.step(-1))}>←</Button>
            <Range aria-label="视频回看进度" type="range" min={replay.minFrame} max={replay.maxFrame} step="1" value={replay.frame}
                onChange={event => run(() => replay.seek(event.target.value))} />
            <Button aria-label="回看下一帧" disabled={replay.frame >= replay.maxFrame} onClick={() => run(() => replay.step(1))}>→</Button>
            <label>帧 <NumberField aria-label="回看源帧" type="number" min={replay.minFrame} max={replay.maxFrame} value={replay.frame}
                onChange={event => run(() => replay.seek(event.target.value))} /></label>
            {replay.firstPersonFrame != null && <Button onClick={() => run(() => replay.seek(replay.firstPersonFrame))}>首个行人</Button>}
        </div>
        {replay.snapshot?.replayMissing && <small role="status">此帧未保存推理记录，仅显示源画面。</small>}
    </section>;
}
