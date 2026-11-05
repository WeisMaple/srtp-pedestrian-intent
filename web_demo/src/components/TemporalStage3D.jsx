import { Button } from './ui/Controls';
import { useEffect, useRef, useState } from 'react';
import { createTemporalScene } from '../scene/temporalScene';

export default function TemporalStage3D({ calculation, channel, flow, onSeek, onInteract, auto }) {
    const host = useRef(null), engine = useRef(null), callbacks = useRef({});
    const [error, setError] = useState('');
    callbacks.current = { onSeek, onInteract };
    useEffect(() => {
        try {
            engine.current = createTemporalScene(host.current, {
                onSelect: index => callbacks.current.onSeek(index), onInteract: () => callbacks.current.onInteract(),
                onError: message => { setError(message); engine.current?.dispose(); engine.current = null; },
            });
        } catch (cause) { setError(cause.message || 'WebGL 不可用'); }
        return () => { engine.current?.dispose(); engine.current = null; };
    }, []);
    useEffect(() => { engine.current?.setData(calculation, channel); }, [calculation, channel]);
    useEffect(() => { engine.current?.setCursor(flow); }, [flow]);
    useEffect(() => { engine.current?.setPlayback(auto); }, [auto]);
    return <div className="temporal-3d">
        {error ? <div role="alert" className="temporal-3d-error">{error} 请使用下方数值明细。</div> : <div className="temporal-3d-canvas" ref={host} />}
        <div className="temporal-camera" role="group" aria-label="卷积视角">
            {[['overview', '立体透视'], ['top', '俯视通道'], ['focus', '聚焦乘加']].map(([mode, label]) =>
                <Button type="button" key={mode} onClick={() => { onInteract(); engine.current?.view(mode); }}>{label}</Button>)}
        </div>
        <p className="temporal-3d-key">横向：9 个时间位置 · 立体分层：最多 4 个真实输入通道 · 蓝 / 红：正 / 负值 · 金色：当前乘加项 · 线框：补零。拖动旋转、滚轮缩放、点击方块查看。</p>
    </div>;
}
