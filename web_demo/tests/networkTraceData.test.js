import test from 'node:test';
import assert from 'node:assert/strict';
import {createCalculation} from '../src/calculationFlow.js';
import {calculationBinding,exactCellIndex,exactPoseAnchor,traceSceneNodes} from '../src/three/networkTraceData.js';
const calculation=(stage,count,extra={})=>createCalculation({id:stage,stage,inputs:Array(count).fill(2),weights:Array(count).fill(.5),bias:0,actual:count,...extra});

test('a native cell is matched exactly; absent indices, ports and other time slices never borrow a voxel',()=>{
    const cells=[{index:[0,1]},{index:[0,9]}];
    assert.equal(exactCellIndex(cells,{index:[0,9]}),1);
    assert.equal(exactCellIndex(cells,{index:[0,2]}),-1);
    assert.equal(exactCellIndex(cells,{index:[0,1],port:'input'}),-1);
    assert.equal(exactCellIndex(cells,{index:[0,1],time:25},{timestep:49}),-1);
    assert.equal(exactCellIndex(cells,{index:[0,1],time:25},{timestep:25}),0);
});

test('a nearby skeleton time is a layer annotation, never an exact coordinate sample',()=>{
    const node={indices:[[1,13,25,37]]},skeleton={pose:[[1,2]],timestep:36,actualTimestep:3};
    assert.equal(exactPoseAnchor({key:'future',time:40},skeleton,node),false);
    assert.equal(exactPoseAnchor({key:'future',time:36},skeleton,node),true);
    assert.equal(exactPoseAnchor({key:'future',time:25,projected:true},skeleton,node),true);
    assert.equal(exactPoseAnchor({key:'history',time:0},{pose:null,timestep:0,actualTimestep:-1},{}),false);
});
test('YOLO operands populate the existing RGB and weight blocks with exact native coordinates',()=>{
    const indices=Array.from({length:27},(_,i)=>[Math.floor(i/9),191+Math.floor(i/3)%3,319+i%3]);
    const c=calculation('yolo',27,{input_indices:indices}),nodes=traceSceneNodes({},c);
    assert.deepEqual(nodes['yolo.patch'].indices,[[0,1,2],[191,192,193],[319,320,321]]);
    assert.deepEqual(calculationBinding(c,26).source.index,[2,193,321]);
    assert.deepEqual(calculationBinding(c,26).weight.index,[2,2,2]);
    assert.equal(nodes['yolo.conv_output'].values[0],27);
    assert.deepEqual(Object.keys(nodes),['yolo.patch','yolo.weights','yolo.conv_output']);
});
test('spatial connections refer to original partition joints and exact output block without changing membership',()=>{
    const live=calculation('spatial',51,{source:'live',metadata:{block:0,target_indices:[0,25,6]}});
    const saved=calculation('spatial',51,{metadata:{block:1},target:{channel:0,timestep:25,joint:6}});
    for(const c of [live,saved]){
        const b=calculationBinding(c,35);assert.equal(b.source.partition,2);assert.equal(b.source.joint,1);
        assert.equal(b.target.key,'stgcn.0.spatial');assert.deepEqual(b.target.index,[0,6]);assert.equal(b.target.time,25);
    }
});
test('LSTM decoder uses last hidden layer, while a gate previous state stays a layer annotation',()=>{
    assert.deepEqual(calculationBinding(calculation('decoder',256),19,9).source,{kind:'layer',key:'lstm.layer.9',index:[19]});
    const gate=calculation('decoder',307,{operation:'lstm_gate',target:{gate:'f'}}),binding=calculationBinding(gate,52);
    assert.equal(binding.source.port,'hidden_previous');assert.equal(binding.source.index,null);
    assert.equal(binding.target.key,'lstm.gates');assert.deepEqual(binding.target.index,[1]);
});
test('temporal input uses true padded timesteps and current weights in the original window',()=>{
    const indices=Array.from({length:18},(_,i)=>[Math.floor(i/9),i%9-4,0]);
    const c=calculation('temporal',18,{source:'live',metadata:{kernel_size:9,block:0,target_indices:[0,0,0]},input_indices:indices});
    const b=calculationBinding(c,10),nodes=traceSceneNodes({},c);
    assert.equal(b.source.key,'stgcn.window');assert.deepEqual(b.source.index,[1,-3]);assert.deepEqual(b.weight.index,[1,1]);
    assert.deepEqual(nodes['stgcn.window'].indices[1],[-4,-3,-2,-1,0,1,2,3,4]);
});
test('classifier and softmax attach to original pooled cells, class weights, logits and score bars',()=>{
    const c=calculation('classifier',256,{metadata:{selected_class:2}}),b=calculationBinding(c,7);
    assert.equal(b.source.key,'stgcn.pool');assert.deepEqual(b.source.index,[7]);assert.deepEqual(b.weight.index,[2,7]);
    assert.equal(b.target.key,'stgcn.logits');assert.deepEqual(b.target.index,[2]);
    const softmax=calculation('softmax',3,{metadata:{selected_class:2}});
    assert.equal(calculationBinding(softmax,1).target.key,'stgcn.probabilities');
    assert.equal(calculationBinding(softmax,1).source.port,'exp_logit');
});
