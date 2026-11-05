import { useEffect, useRef } from 'react';

/** Ignore play() completions belonging to a replaced or interrupted video. */
export function useVideoPlayback(video,src,onError) {
    const operation=useRef(0);
    const cancel=()=>{operation.current+=1;};
    useEffect(()=>{cancel();return cancel;},[src]);
    const toggle=async()=>{
        const element=video.current;
        if(!element)return;
        const request=++operation.current;
        onError('');
        try {
            if(element.paused){if(element.ended)element.currentTime=0;await element.play();}
            else element.pause();
        } catch(error) {
            if(request===operation.current&&error?.name!=='AbortError')onError('暂时无法播放，请重试或检查视频文件。');
        }
    };
    return {toggle,cancel};
}
