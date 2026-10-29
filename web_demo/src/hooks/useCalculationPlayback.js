import { useEffect, useMemo, useRef, useState } from 'react';

/** Spatial reading dwells on actual messages; zero/nonexistent edges stay distinct.
 * Every candidate retains a time interval and remains in the exact prefix sum. */
export function graphTimeline(calculation){
    if(calculation?.stage!=='spatial')return null;
    const stops=[0];
    for(const term of calculation.terms)stops.push(stops.at(-1)+(term.connected?1.6:.045));
    stops.push(stops.at(-1)+.65);stops.push(stops.at(-1)+.95);return stops;
}
export function timelineTime(cursor,stops){
    const c=Math.max(0,Math.min(stops.length-1,cursor)),i=Math.floor(c);
    return stops[i]+(c-i)*((stops[i+1]??stops[i])-stops[i]);
}
export function timelineCursor(time,stops){
    if(time<=0)return 0;
    const i=stops.findIndex(value=>value>time);
    return i<0?stops.length-1:i-1+(time-stops[i-1])/(stops[i]-stops[i-1]);
}

export function cursorTime(cursor,count) {
    const first=Math.min(3,count), end=count+2;
    const c=Math.max(0,Math.min(end,cursor));
    const tail=count-first;
    if(c<first)return c*1.15;
    if(c<count)return first*1.15+(c-first)*Math.min(1.15,5/Math.max(1,tail));
    return first*1.15+tail*Math.min(1.15,5/Math.max(1,tail))+(c-count)*.95;
}
export function timeCursor(time,count) {
    const first=Math.min(3,count),tail=count-first,step=Math.min(1.15,5/Math.max(1,tail));
    if(time<first*1.15)return Math.max(0,time/1.15);
    if(time<first*1.15+tail*step)return first+(time-first*1.15)/step;
    return Math.min(count+2,count+(time-first*1.15-tail*step)/.95);
}
export function useCalculationPlayback(calculation) {
    const [cursor,setCursor]=useState(0),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1);
    const value=useRef(0);const count=calculation?.terms?.length||0;
    const timeline=useMemo(()=>graphTimeline(calculation),[calculation]);
    const key=calculation?`${calculation.source}:${calculation.frame}:${calculation.track_id}:${calculation.id}`:'';
    useEffect(()=>{value.current=0;setCursor(0);},[key]);
    useEffect(()=>{
        if(!playing||!calculation)return;
        if(window.matchMedia('(prefers-reduced-motion: reduce)').matches){value.current=count+2;setCursor(count+2);setPlaying(false);return;}
        let raf,last=performance.now(),lastPaint=last;
        const tick=now=>{const dt=Math.min(.15,(now-last)/1000);last=now;value.current=timeline?timelineCursor(timelineTime(value.current,timeline)+dt*speed,timeline):timeCursor(cursorTime(value.current,count)+dt*speed,count);if(now-lastPaint>=32||value.current>=count+2){setCursor(value.current);lastPaint=now;}if(value.current>=count+2){setPlaying(false);return;}raf=requestAnimationFrame(tick);};
        raf=requestAnimationFrame(tick);return()=>cancelAnimationFrame(raf);
    },[playing,key,count,speed,timeline]);
    return {cursor,playing,speed,setSpeed,start:()=>{value.current=0;setCursor(0);setPlaying(true);},pause:()=>setPlaying(false),toggle:()=>{if(value.current>=count+2){value.current=0;setCursor(0);}setPlaying(v=>!v);},seek:n=>{value.current=Math.max(0,Math.min(count+2,n));setCursor(value.current);setPlaying(false);}};
}
