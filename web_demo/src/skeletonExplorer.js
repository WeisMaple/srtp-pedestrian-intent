import {asPose,normalizePose,poseCoordinateFrame,evenlySpacedIndices} from './three/sceneData.js';

/** Preserve native time indices. Missing samples are never interpolated. */
export function sequenceFrames(node) {
    return (node?.values||[]).map((value,i)=>({pose:asPose(value),time:node.indices?.[0]?.[i]??i,sourceSlot:i})).filter(f=>f.pose&&Number.isInteger(f.time)).map((f,slot)=>({...f,slot}));
}
export function frameStrip(frames,selected,count=5) {
    if(!frames.length)return [];
    const indices=new Set(evenlySpacedIndices(frames.length,count));
    indices.add(Math.max(0,Math.min(frames.length-1,selected)));
    return [...indices].sort((a,b)=>a-b).map(i=>frames[i]);
}
export function projectedSequence(frames,size=220) {
    const bounds=poseCoordinateFrame(frames.map(f=>f.pose));
    return frames.map(f=>({...f,points:normalizePose(f.pose,size,bounds).map(([x,y,c])=>[x,-y,c])}));
}
/** One input channel across the true temporal kernel, including padding. */
export function temporalTerms(calculation) {
    if(calculation?.stage!=='temporal'||calculation.operation==='residual_add')return [];
    return calculation.terms.flatMap(t=>{
        const ix=calculation.metadata?.term_indices?.[t.index]??t.inputIndex;
        if(!Array.isArray(ix)||!Number.isInteger(ix[0]))return [];
        const nativeTime=ix.length===3&&calculation.metadata?.input_axes?.includes('time');
        const slot=nativeTime?t.index%(calculation.metadata?.kernel_size||9):ix[1];
        return [{...t,channel:ix[0],slot,time:nativeTime?ix[1]:calculation.metadata?.input_timesteps?.[slot]??null}];
    });
}
export function temporalKernel(calculation,channel=0) {
    return temporalTerms(calculation).filter(t=>t.channel===channel).sort((a,b)=>a.slot-b.slot);
}
