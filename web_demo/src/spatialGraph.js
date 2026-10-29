/** Exact graph slice adapters. Direction is A[partition, source, receiver]. */
export const JOINT_NAMES=['鼻尖','左眼','右眼','左耳','右耳','左肩','右肩','左肘','右肘','左腕','右腕','左髋','右髋','左膝','右膝','左踝','右踝'];
export const PARTITION_NAMES=['自身','自身 + 向心','离心'];
const finite=Number.isFinite;
export const spatialTarget=c=>c?.metadata?.target_indices?.[2]??c?.target?.joint??null;

export function validGraphSlice(c){
    const s=c?.metadata?.graph_slice;
    if(c?.stage!=='spatial'||!s||s.joint_count!==17||c.inputs?.length!==51||!Array.isArray(s.outputs)||s.outputs.length!==17||!s.outputs.every(finite)||!Array.isArray(s.edges))return null;
    const time=c.metadata?.target_indices?.[1]??c.target?.timestep;
    const channel=c.metadata?.target_indices?.[0]??c.target?.channel;
    if(s.timestep!==time||s.channel!==channel)return null;
    const seen=new Set();
    for(const e of s.edges){
        if(!Array.isArray(e)||e.length!==4||!e.slice(0,3).every(Number.isInteger)||e[0]<0||e[0]>2||e[1]<0||e[1]>16||e[2]<0||e[2]>16||!finite(e[3]))return null;
        const key=e.slice(0,3).join(':');if(seen.has(key))return null;seen.add(key);
    }
    return s;
}

export function recordedGraphSlice(graph,block,spatial){
    const outputs=block.spatial?.values?.[spatial.output_channel];
    if(block.spatial?.timestep!==spatial.timestep||!Array.isArray(outputs)||outputs.length!==17||!outputs.every(finite))return null;
    const edges=[];
    for(let k=0;k<3;k++)for(let v=0;v<17;v++)for(let w=0;w<17;w++){
        const base=graph.adjacency?.[k]?.[v]?.[w],weight=block.effective_adjacency?.[k]?.[v]?.[w];
        if(!finite(base)||!finite(weight))return null;
        if(base!==0)edges.push([k,v,w,weight]);
    }
    return {channel:spatial.output_channel,timestep:spatial.timestep,joint_count:17,edges,outputs:[...outputs],source:'recorded_gcn_forward_same_channel_and_timestep'};
}

/** Return raw operands for createCalculation; never edit a pinned record. */
export function retargetSpatial(c,joint){
    const s=validGraphSlice(c);
    if(!s||!Number.isInteger(joint)||joint<0||joint>=17)return null;
    const weights=Array(51).fill(0),connected=Array(51).fill(false);
    for(const [k,v,w,weight] of s.edges)if(w===joint){weights[k*17+v]=weight;connected[k*17+v]=true;}
    const baseId=c.base_id||c.id;
    return {...c,id:`${baseId}.joint${joint}`,base_id:baseId,weights,connected,actual:s.outputs[joint],
        target:{block:c.metadata.block,channel:s.channel,timestep:s.timestep,joint},
        title:`骨架邻域聚合 · ${JOINT_NAMES[joint]} J${joint}`,
        metadata:{...c.metadata,target_indices:[s.channel,s.timestep,joint]}};
}

export function spatialNeighborhood(c,state){
    if(c?.stage!=='spatial'||!state)return null;
    const target=spatialTarget(c),messages=c.terms.filter(t=>t.connected);
    const partitions=PARTITION_NAMES.map((name,k)=>{
        const terms=c.terms.slice(k*17,(k+1)*17),complete=terms.every(t=>t.recorded);
        return {name,index:k,count:terms.filter(t=>t.connected).length,
            sum:complete?terms.reduce((sum,t)=>sum+t.product,0):null,
            partial:terms.filter(t=>t.index<state.completedCount).reduce((sum,t)=>sum===null||!t.recorded?null:sum+t.product,0)};
    });
    return {target,messages,partitions,neighbors:[...new Set(messages.map(t=>t.index%17))],
        completed:messages.filter(t=>t.recorded&&t.index<state.completedCount).length,
        first:messages.find(t=>t.recorded)?.index??0,
        peak:Math.max(1e-12,...messages.filter(t=>t.recorded).map(t=>Math.abs(t.product))),
        unknown:c.terms.filter(t=>!t.recorded).length};
}
