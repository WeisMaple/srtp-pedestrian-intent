import { useEffect, useRef, useState } from 'react';

export function useOperator(progress, boundaries, onPause, defaultIndex = 0) {
    const [choice,setChoice]=useState(null);
    const manual=choice?.at===progress;
    const index=manual?choice.index:progress>=1?defaultIndex:Math.max(0,boundaries.findIndex((end)=>progress<end));
    const start=index?boundaries[index-1]:0;
    const local=manual||progress>=1?1:Math.max(0,Math.min(1,(progress-start)/(boundaries[index]-start)/.97));
    return {index,local,choose:value=>{onPause?.();setChoice({at:progress,index:value});}};
}
export default function OperatorTabs({labels,index,onChange}) {
    const tabs=useRef(null);
    useEffect(()=>{ const scroll=tabs.current?.closest('.lab-canvas-scroll'); if(scroll) scroll.scrollTop=0; },[index]);
    return <div ref={tabs} className="operator-tabs" role="group" aria-label="展开计算算子">{labels.map((label,i)=><button key={label} aria-pressed={i===index} onClick={()=>onChange(i)}><span>{i+1}</span>{label}</button>)}</div>;
}
