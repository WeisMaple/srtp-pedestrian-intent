import test from 'node:test';
import assert from 'node:assert/strict';
import { asPose, normalizePose, poseCoordinateFrame, poseSampleIndex, sceneNodes, tensorCells, latticeSlots, evenlySpacedIndices, eventStage } from '../src/three/sceneData.js';

test('voxel sampling preserves values and original native tensor coordinates',()=>{
    const node={values:[[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]],indices:[[2,7],[10,30],[4,12,20]],shape:[1,64,80,80]};
    const cells=tensorCells(node,[2,2,2]);
    assert.deepEqual(cells.map(c=>c.value),[0,2,3,5,6,8,9,11]);
    assert.deepEqual(cells.at(-1).index,[7,30,20]);
    assert.deepEqual(cells.at(-1).layoutIndex,[1,1,1]);
    assert.equal(cells[0].value,0);
});

test('missing or nonfinite activations never become invented numeric values',()=>{
    assert.deepEqual(tensorCells(null),[]);
    const cells=tensorCells({values:[[null,NaN,-2]]},[1,3]);
    assert.deepEqual(cells.map(c=>c.value),[null,null,-2]);
    const nodes=sceneNodes({});assert.deepEqual(nodes,{});
});

test('a partial live frame never mixes with saved snapshot tensors',()=>{
    const liveNodes={'yolo.0':{values:[[[42]]],shape:[1,1,1]}};
    const nodes=sceneNodes({liveNodes,data:{stgcn:{pool:[99]}},frontend:{layers:[{values:[[[1]]]}]}});
    assert.equal(nodes,liveNodes);assert.equal(nodes['stgcn.pool'],undefined);
    assert.deepEqual(sceneNodes({liveNodes:{},track:{history:[Array.from({length:17},()=>[1,2,3])]}}),{});
});

test('snapshot LSTM states match selected round and gates are not reused across rounds',()=>{
    const data={lstm:{hidden_size:256,rounds:[{layers:[{h:[1,2],c:[3,4]}]},{layers:[{h:[5,6],c:[7,8]}]}],gates:{steps:[{i:1,f:2,g:3,o:4,c:5,h:6}]}}};
    assert.deepEqual(sceneNodes({data,round:1})['lstm.layer.0'].values,[5,6]);
    assert.deepEqual(sceneNodes({data,round:1})['lstm.layer.0'].cell,[7,8]);
    assert.equal(sceneNodes({data,round:1})['lstm.gates'],undefined);
    assert.deepEqual(sceneNodes({data,round:0})['lstm.gates'].values,[1,2,3,4,5,6]);
});

test('pose conversion validates coordinates and shares a fixed frame across time',()=>{
    const first=Array.from({length:17},(_,i)=>[i,i*2,.7]);const second=first.map(([x,y,c])=>[x+10,y,c]);
    assert.deepEqual(asPose(first.flat()),first);
    const invalid=first.flat();invalid[0]=NaN;assert.equal(asPose(invalid),null);
    const bounds=poseCoordinateFrame([first,second]);
    const a=normalizePose(first,4,bounds),b=normalizePose(second,4,bounds);
    assert.ok(b[0][0]>a[0][0]);assert.equal(a[0][2],.7);
    assert.equal(normalizePose(null)[0][2],null);
});

test('event stage routing follows actual operator identity',()=>{
    assert.equal(eventStage({node:'yolo.12'}),1);
    assert.equal(eventStage({node:'lstm.3'}),3);
    assert.equal(eventStage({node:'stgcn.2.spatial'}),5);
    assert.equal(eventStage({node:'stgcn.4.temporal'}),6);
    assert.equal(eventStage({node:'stgcn.logits'}),7);
    assert.equal(eventStage({node:'pipeline.output'}),7);
    assert.equal(eventStage({node:'unknown'}),null);
});

test('uniform sampling has no duplicates and includes both edges',()=>{
    assert.deepEqual(evenlySpacedIndices(3,8),[0,1,2]);
    assert.deepEqual(evenlySpacedIndices(16,4),[0,5,10,15]);
    assert.deepEqual(evenlySpacedIndices(0,4),[]);
});


test('sparse live history samples use native time coordinates, not array offsets',()=>{
    const node={values:Array.from({length:10},()=>[]),indices:[[0,5,11,16,22,27,33,38,44,49]]};
    assert.equal(poseSampleIndex(node,25),5);
    assert.equal(poseSampleIndex(node,45),8);
    assert.equal(poseSampleIndex(node,49),9);
    assert.equal(poseSampleIndex({values:[[],[],[]],indices:[[1,2,3]]},0,true),0);
    assert.equal(poseSampleIndex({values:[[],[],[]],indices:[[1,2,3]]},1,true),1);
});


test('partial transposed LSTM samples occupy unique native-time slots without gray overlap',()=>{
    const times=[0,16,33,49],units=[0,2,4,6,8,10,12,14];
    const node={values:times.map((time,t)=>units.map((unit,u)=>t*10+u)),indices:[times,units]};
    const samples=tensorCells(node,[50,8]).map(cell=>({...cell,layoutIndex:[cell.layoutIndex[1],cell.index[0]]}));
    const slots=latticeSlots(samples,{cols:50,rows:8,depth:1,capacity:400});
    assert.equal(slots.length,400);
    assert.equal(new Set(slots.map(c=>c.layoutIndex.join(','))).size,400);
    assert.equal(slots.filter(c=>c.value!==null).length,32);
    assert.equal(slots.filter(c=>c.value===null).length,368);
    times.forEach((time,t)=>units.forEach((unit,u)=>{
        const cell=slots[u*50+time];
        assert.equal(cell.value,node.values[t][u]);
        assert.deepEqual(cell.index,[time,unit]);
        assert.deepEqual(cell.layoutIndex,[0,u,time]);
    }));
    assert.equal(slots[50].value,1); // Previously covered by gray placeholder instance 50.
    assert.equal(slots[1].value,null); // Unsampled native time stays empty.
});

test('pose slots beyond the known time range remain structural, not repeated final observations',()=>{
    assert.equal(poseSampleIndex({values:[[],[],[]]},3),-1);
    assert.equal(poseSampleIndex({values:[[],[],[]],indices:[[1,2,3]]},3,true),-1);
    assert.equal(poseSampleIndex({values:[[],[],[]],indices:[[0,5,10]]},12),-1);
});
