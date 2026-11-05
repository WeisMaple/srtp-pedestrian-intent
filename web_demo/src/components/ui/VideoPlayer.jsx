import { useEffect, useRef, useState } from 'react';
import { Button, Range, Select } from './Controls';
import { formatTime } from '../../api';
import { useVideoPlayback } from './useVideoPlayback';

export default function VideoPlayer({src,poster,muted=false,preload='metadata',label='视频',className=''}) {
    const video=useRef(null),container=useRef(null);
    const [playing,setPlaying]=useState(false),[current,setCurrent]=useState(0),[duration,setDuration]=useState(0);
    const [silent,setSilent]=useState(muted),[speed,setSpeed]=useState(1),[fullscreen,setFullscreen]=useState(false),[error,setError]=useState('');
    const {toggle,cancel}=useVideoPlayback(video,src,setError);
    useEffect(()=>{setPlaying(false);setCurrent(0);setDuration(0);setError('');setSpeed(1);setSilent(muted);
        const element=video.current;return ()=>element?.pause();},[src,muted]);
    useEffect(()=>{const change=()=>setFullscreen(document.fullscreenElement===container.current);
        document.addEventListener('fullscreenchange',change);return ()=>document.removeEventListener('fullscreenchange',change);},[]);
    const seek=value=>{if(duration>0){const next=Math.max(0,Math.min(duration,Number(value)||0));video.current.currentTime=next;setCurrent(next);}};
    const full=async()=>{try {if(document.fullscreenElement===container.current)await document.exitFullscreen();else await container.current.requestFullscreen();}
        catch {setError('当前浏览器暂时无法进入全屏。');}};
    return <div ref={container} className={`ui-video ${className}`} role="region" aria-label={`${label}播放器`} tabIndex="0"
        onKeyDown={event=>{if(event.target!==event.currentTarget)return;
            if(event.code==='Space'){event.preventDefault();event.stopPropagation();toggle();}
            if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();event.stopPropagation();seek(current+(event.key==='ArrowRight'?5:-5));}}}>
        <div className="ui-video-picture"><video ref={video} src={src} poster={poster} muted={silent} playsInline preload={preload}
            aria-label={label} onClick={toggle} onPlay={()=>setPlaying(true)} onPause={()=>{cancel();setPlaying(false);}}
            onEnded={()=>setPlaying(false)} onTimeUpdate={event=>setCurrent(event.currentTarget.currentTime)}
            onDurationChange={event=>setDuration(Number.isFinite(event.currentTarget.duration)?event.currentTarget.duration:0)}
            onRateChange={event=>setSpeed(event.currentTarget.playbackRate)} onVolumeChange={event=>setSilent(event.currentTarget.muted)}
            onError={()=>{setPlaying(false);setError('无法加载视频，请检查文件或连接。');}} />
            {!playing&&!error&&<Button className="ui-video-overlay" aria-label={`播放${label}`} onClick={toggle}>▶</Button>}
        </div>
        <div className="ui-video-controls">
            <Range aria-label={`${label}播放进度`} min="0" max={duration||1} step="0.1" value={current} disabled={!duration} onChange={event=>seek(event.target.value)} />
            <div className="ui-video-actions"><Button className="ui-video-icon" aria-label={playing?`暂停${label}`:`播放${label}`} onClick={toggle}>{playing?'Ⅱ':'▶'}</Button>
                <span className="ui-video-time mono">{formatTime(current)} / {duration?formatTime(duration):'--:--'}</span>
                <Button className="ui-video-icon" aria-label={silent?'开启视频声音':'静音视频'} aria-pressed={silent} onClick={()=>setSilent(value=>!value)}>{silent?'静音':'声音'}</Button>
                <Select aria-label={`${label}播放速度`} value={speed} onChange={event=>{const next=Number(event.target.value);video.current.playbackRate=next;setSpeed(next);}}>
                    {[0.5,1,1.5,2].map(value=><option key={value} value={value}>{value}×</option>)}</Select>
                <Button className="ui-video-icon" aria-label={fullscreen?'退出全屏':'全屏播放'} disabled={!document.fullscreenEnabled} onClick={full}>{fullscreen?'收起':'全屏'}</Button>
            </div>
        </div>
        {error&&<p className="ui-video-error" role="alert">{error}</p>}
    </div>;
}
