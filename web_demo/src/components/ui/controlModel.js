export function rangePercent(value,min=0,max=100) {
    const [v,lo,hi]=[value,min,max].map(Number);
    return [v,lo,hi].every(Number.isFinite)&&hi>lo?Math.max(0,Math.min(100,(v-lo)/(hi-lo)*100)):0;
}
export function stepNumber(value,direction,min=-Infinity,max=Infinity,step=1) {
    const base=value!==''&&Number.isFinite(Number(value))?Number(value):Number.isFinite(Number(min))?Number(min):0;
    const increment=Number.isFinite(Number(step))&&Number(step)>0?Number(step):1;
    return Math.max(Number(min),Math.min(Number(max),Number((base+direction*increment).toFixed(10))));
}
export function selectionChange(value,next,name) {
    if(String(value)===String(next)) return null;
    const target={value:String(next),name};
    return {target,currentTarget:target};
}
export function selectKey(state,key,options) {
    const enabled=options.map((option,index)=>option.disabled?-1:index).filter(index=>index>=0);
    if(!enabled.length) return {open:false,active:-1};
    const active=enabled.includes(state.active)?state.active:enabled[0];
    let next={open:state.open,active};
    if(key==='Escape') return {...next,open:false,handled:true};
    if(key==='Tab') return {...next,open:false,...(state.open?{commit:options[active].value}:{} )};
    if(key==='Enter'||key===' ') return {...next,open:!state.open,handled:true,...(state.open?{commit:options[active].value}:{})};
    if(key==='Home'||key==='End') return {open:true,active:key==='Home'?enabled[0]:enabled.at(-1),handled:true};
    if(key==='ArrowDown'||key==='ArrowUp') {
        const position=enabled.indexOf(active)+(key==='ArrowDown'?1:-1);
        return {open:true,active:state.open?enabled[Math.max(0,Math.min(enabled.length-1,position))]:active,handled:true};
    }
    return next;
}
export function menuPosition(rect,viewportWidth,viewportHeight,wantedHeight=280) {
    const width=Math.min(Math.max(rect.width,200),Math.max(0,viewportWidth-16));
    const below=viewportHeight-rect.bottom-14,above=rect.top-14;
    const upward=below<Math.min(wantedHeight,180)&&above>below;
    const maxHeight=Math.max(0,Math.min(wantedHeight,upward?above:below));
    return {position:'fixed',left:Math.max(8,Math.min(rect.left,viewportWidth-width-8)),width,maxHeight,
        top:upward?rect.top-maxHeight-6:rect.bottom+6};
}
