import test from 'node:test';
import assert from 'node:assert/strict';
import {createCalculation,flowState} from '../src/calculationFlow.js';
import {validGraphSlice,retargetSpatial,spatialNeighborhood,recordedGraphSlice} from '../src/spatialGraph.js';
const make=()=>createCalculation({id:'graph',stage:'spatial',inputs:Array.from({length:51},(_,i)=>i+1),weights:Array(51).fill(0),connected:Array(51).fill(false),bias:0,actual:0,
    metadata:{block:0,target_indices:[0,25,0],graph_slice:{channel:0,timestep:25,joint_count:17,
        edges:[[0,1,4,2],[1,7,4,-.25],[2,5,4,0],[0,4,1,3]],outputs:Array.from({length:17},(_,i)=>i===4?4-25*.25:i===1?15:0)}}},
    {source:'live',frame:92,track_id:7});

test('receiver selection preserves direction, zero learned edges, all terms and provenance',()=>{
    const base=make(),c=createCalculation(retargetSpatial(base,4),base.provenance);
    assert.equal(c.actual,-2.25);assert.equal(c.reconstructed,c.actual);assert.equal(c.terms.length,51);
    assert.deepEqual(c.terms.filter(t=>t.connected).map(t=>t.index),[1,24,39]);
    assert.equal(c.terms[39].membership,'zeroContribution');assert.equal(c.terms[4].membership,'disconnected');
    assert.equal(c.source,'live');assert.equal(c.frame,92);assert.equal(c.track_id,7);
    assert.equal(base.actual,0);assert.equal(base.metadata.target_indices[2],0);
    assert.equal(retargetSpatial(c,1).id,'graph.joint1');
});
test('every available receiver matches its independent output',()=>{
    const c=make();for(let j=0;j<17;j++){
        const selected=createCalculation(retargetSpatial(c,j),c.provenance);
        assert.equal(selected.absoluteError,0);
    }
});
test('missing, mismatched, duplicate and malformed graph slices cannot enable arbitrary targets',()=>{
    const c=make();for(const patch of [{timestep:26},{channel:1},{outputs:[0]},{edges:[[0,1,4,2],[0,1,4,2]]},{edges:[[0,17,4,2]]}]){
        const bad={...c,metadata:{...c.metadata,graph_slice:{...c.metadata.graph_slice,...patch}}};
        assert.equal(validGraphSlice(bad),null);assert.equal(retargetSpatial(bad,4),null);
    }
    assert.equal(retargetSpatial({...c,metadata:{}},4),null);assert.equal(retargetSpatial(c,17),null);
});
test('partition subtotals follow the same deterministic playback, including connected zero',()=>{
    const c=createCalculation(retargetSpatial(make(),4));
    const before=spatialNeighborhood(c,flowState(c,24.5)),after=spatialNeighborhood(c,flowState(c,24.9));
    assert.deepEqual(before.partitions.map(p=>p.sum),[4,-6.25,0]);
    assert.deepEqual(before.partitions.map(p=>p.partial),[4,0,0]);
    assert.deepEqual(after.partitions.map(p=>p.partial),[4,-6.25,0]);
    assert.deepEqual(after.neighbors,[1,7,5]);assert.equal(after.completed,2);assert.equal(after.messages.length,3);
    assert.deepEqual(spatialNeighborhood(c,flowState(c,24.5)),before);
});
test('missing operands never become zero contribution or a complete partition total',()=>{
    const raw=retargetSpatial(make(),4);raw.inputs=[...raw.inputs];raw.inputs[24]=null;
    const c=createCalculation(raw),n=spatialNeighborhood(c,flowState(c,53));
    assert.equal(n.unknown,1);assert.equal(n.completed,2);assert.equal(n.partitions[1].sum,null);assert.equal(n.partitions[1].partial,null);assert.equal(c.reconstructed,null);
});
test('recorded slices require matching output time and original topology, including zero effective edges',()=>{
    const base=Array.from({length:3},()=>Array.from({length:17},()=>Array(17).fill(0)));
    base[1][7][4]=.5;
    const weights=base.map(p=>p.map(row=>row.map(()=>0)));
    const graph={adjacency:base},block={spatial:{values:[Array(17).fill(0)],timestep:25},effective_adjacency:weights};
    const spatial={timestep:25,output_channel:0};
    assert.deepEqual(recordedGraphSlice(graph,block,spatial).edges,[[1,7,4,0]]);
    assert.equal(recordedGraphSlice(graph,block,{...spatial,timestep:24}),null);
    assert.equal(recordedGraphSlice({},block,spatial),null);
});
