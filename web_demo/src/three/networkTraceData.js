/** Bind scalar examples to model objects, never to an arbitrary sampled voxel. */
import {validGraphSlice} from '../spatialGraph.js';
const array=value=>Array.isArray(value)?value:[];
const layer=(key,index=null,extra={})=>({kind:'layer',key,index,...extra});
const pose=(key,time,joint,extra={})=>({kind:'joint',key,time,joint,...extra});
export function calculationBinding(c,termIndex=0,lastLayer=9) {
    if(!c)return null;
    const term=c.terms[termIndex],m=c.metadata||{},i=term?.inputIndex;
    const block=Math.max(0,(m.block??c.target?.block??(c.source==='live'?0:1))-(c.source==='live'?0:1));
    const target=m.target_indices||[c.target?.channel??0,c.target?.timestep??0,c.target?.joint??0];
    const result={chapter:c.chapter,source:null,target:null,weight:null,title:c.title};
    if(c.stage==='yolo') {
        const index=[Math.floor(termIndex/9),Math.floor(termIndex/3)%3,termIndex%3];
        result.source=layer('yolo.patch',Array.isArray(i)?i:index,{fallback:'yolo.0',fallbackPort:'input'});
        result.weight=layer('yolo.weights',index,{fallback:'yolo.0',fallbackPort:'weight'});
        result.target=layer('yolo.conv_output',[0],{fallback:'yolo.0',fallbackPort:'output'});
    }else if(c.stage==='normalization') {
        result.source=termIndex===0?pose('history',0,0):layer('history',null,{port:'statistics'});
        result.target=layer('history',null,{port:'normalized'});
    }else if(c.stage==='denormalization') {
        const time=m.target_indices?.[0]??((m.round??1)-1)*10;
        result.source=pose('future',time,0,{port:'normalized'});result.target=pose('future',time,0);
    }else if(c.stage==='decoder') {
        if(c.operation==='lstm_gate') {
            result.source=layer('lstm.layer.0',null,{port:termIndex<51?'input_t':'hidden_previous'});
            result.target=layer('lstm.gates',[['i','f','g','o'].indexOf(c.target?.gate)],{fallback:'lstm.layer.0'});
        }else if(c.operation==='lstm_cell'||c.operation==='lstm_hidden') {
            result.source=layer('lstm.gates',c.operation==='lstm_hidden'?[4]:termIndex?[2]:null,{fallback:'lstm.layer.0',port:termIndex?'gate':'previous_state'});
            result.target=layer('lstm.gates',[c.operation==='lstm_cell'?4:5],{fallback:'lstm.layer.0'});
        }else {
            result.source=layer(`lstm.layer.${lastLayer}`,[termIndex]);
            result.target=layer(`lstm.layer.${lastLayer}`,null,{port:'decoder_output'});
        }
    }else if(c.stage==='spatial') {
        const partition=Math.floor(termIndex/17),joint=termIndex%17;
        result.source=pose('future',target[1],joint,{partition,projected:true});
        result.target=layer(`stgcn.${block}.spatial`,[target[0],target[2]],{time:target[1]});
    }else if(c.stage==='temporal') {
        if(c.operation==='residual_add'){
            result.source=layer(`stgcn.${block}.${termIndex?'output':'temporal'}`,null,{port:termIndex?'residual_input':'temporal_output'});
            result.target=layer(`stgcn.${block}.output`,null,{fallback:`stgcn.${block}.temporal`,port:'residual_output'});
        }else {
            const kernel=m.kernel_size||9,k=termIndex%kernel,channel=Math.floor(termIndex/kernel);
            const time=Array.isArray(i)?i[1]:m.input_timesteps?.[k];
            result.source=layer('stgcn.window',[channel,time],{slot:k,fallback:`stgcn.${block}.temporal`,fallbackPort:'input'});
            result.weight=layer('stgcn.kernel',[channel,k],{slot:k,fallback:`stgcn.${block}.temporal`,fallbackPort:'weight'});
            result.target=layer(`stgcn.${block}.temporal`,[target[0],target[2]],{time:target[1]});
        }
    }else if(c.stage==='pool') {
        result.source=layer('stgcn.pool',null,{port:'pool_input'});result.target=layer('stgcn.pool',[target[0]]);
    }else if(c.stage==='classifier') {
        const cls=m.selected_class??c.target?.class??target[0]??0;
        result.source=layer('stgcn.pool',[termIndex]);result.weight=layer('stgcn.weights',[cls,termIndex],{port:'weight'});
        result.target=layer('stgcn.logits',[cls],{fallback:'stgcn.probabilities',classIndex:cls,fallbackPort:'logit_output'});
    }else if(c.stage==='softmax') {
        const cls=m.selected_class??c.target?.class??0;
        result.source=layer('stgcn.logits',[termIndex],{fallback:'stgcn.probabilities',classIndex:termIndex,port:'exp_logit'});
        result.target=layer('stgcn.probabilities',[cls],{classIndex:cls});
    }
    return result.source&&result.target?result:null;
}

export function exactCellIndex(cells,binding,node) {
    if(!binding?.index||binding.port||binding.time!=null&&node?.timestep!==binding.time)return -1;
    return cells.findIndex(cell=>array(cell.index).length===binding.index.length&&cell.index.every((value,i)=>value===binding.index[i]));
}

export function exactPoseAnchor(binding,skeleton,node) {
    // Projected graph features belong to a joint in the topology; they are
    // explicitly labelled as features, not as that skeleton's pixel coordinates.
    if(binding.projected)return true;
    if(!skeleton.pose||skeleton.timestep!==binding.time)return false;
    const time=node?.indices?.[0]?.[skeleton.actualTimestep]??skeleton.actualTimestep;
    const offset=binding.key==='future'&&node?.indices?.[0]?.[0]===1?1:0;
    return time===binding.time+offset;
}

/** Fill only structures already present in the network scene with the actual example. */
export function traceSceneNodes(base,c,lastLayer=9) {
    if(!c)return base;
    const nodes={...base},source='当前算式 · 实际参与值';
    const node=(values,shape,indices)=>({values,shape,indices,source});
    if(c.stage==='spatial'){
        const binding=calculationBinding(c),key=binding.target.key,time=binding.target.time,channel=binding.target.index[0],joint=binding.target.index[1];
        const outputs=validGraphSlice(c)?.outputs;
        const values=Array.isArray(outputs)?outputs:Array.from({length:17},(_,i)=>i===joint?c.actual:null);
        nodes[key]={...node([values],[1,17],[[channel],Array.from({length:17},(_,i)=>i)]),timestep:time};
    }
    if(c.stage==='yolo'&&c.terms.length===27){
        const cube=items=>Array.from({length:3},(_,ch)=>Array.from({length:3},(_,y)=>items.slice(ch*9+y*3,ch*9+y*3+3)));
        const native=Array.isArray(c.input_indices[0])?[0,1,2].map(axis=>[...new Set(c.input_indices.map(i=>i[axis]))]):[[0,1,2],[0,1,2],[0,1,2]];
        nodes['yolo.patch']=node(cube(c.inputs),[3,3,3],native);
        nodes['yolo.weights']=node(cube(c.weights),[3,3,3],[[0,1,2],[0,1,2],[0,1,2]]);
        nodes['yolo.conv_output']=node([c.actual],[1],[[0]]);
    }
    if(c.stage==='temporal'&&c.operation!=='residual_add'){
        const kernel=c.metadata.kernel_size||9,count=c.terms.length/kernel;
        if(Number.isInteger(count)){
            const rows=values=>Array.from({length:count},(_,channel)=>values.slice(channel*kernel,(channel+1)*kernel));
            const times=c.metadata.input_timesteps||c.input_indices.slice(0,kernel).map(index=>Array.isArray(index)?index[1]:null);
            nodes['stgcn.window']=node(rows(c.inputs),[count,kernel],[Array.from({length:count},(_,i)=>i),times]);
            nodes['stgcn.kernel']=node(rows(c.weights),[count,kernel],[Array.from({length:count},(_,i)=>i),Array.from({length:kernel},(_,i)=>i)]);
        }
    }
    if(c.stage==='decoder'&&c.operation!=='lstm_gate'&&c.operation!=='lstm_cell'&&c.operation!=='lstm_hidden'){
        const key=`lstm.layer.${lastLayer}`;
        nodes[key]={...base[key],...node(c.inputs,[c.inputs.length],[c.inputs.map((_,i)=>i)])};
    }
    if(c.stage==='classifier'){
        const cls=c.metadata.selected_class??c.target?.class??c.metadata.target_indices?.[0]??0;
        const values=Array.from({length:3},(_,i)=>i===cls?c.weights:base['stgcn.weights']?.values?.[i]||Array(c.weights.length).fill(null));
        nodes['stgcn.weights']=node(values,[3,c.weights.length],[[0,1,2],c.weights.map((_,i)=>i)]);
    }
    return nodes;
}
