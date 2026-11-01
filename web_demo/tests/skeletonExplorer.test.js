import test from 'node:test';
import assert from 'node:assert/strict';
import {sequenceFrames,projectedSequence,frameStrip,temporalKernel} from '../src/skeletonExplorer.js';
const pose=(x=0)=>Array.from({length:17},(_,j)=>[x+j,j*3,.9]);
test('native sparse indices survive missing poses without inventing frames',()=>{
 const f=sequenceFrames({values:[pose(),null,pose(20)],indices:[[0,12,49]]});
 assert.deepEqual(f.map(v=>[v.time,v.slot,v.sourceSlot]),[[0,0,0],[49,1,2]]);
 assert.deepEqual(frameStrip(f,1).map(v=>v.time),[0,49]);
});
test('sequence uses one shared coordinate scale and preserves image displacement',()=>{
 const f=projectedSequence(sequenceFrames({values:[pose(),pose(20)]}));
 assert.ok(f[1].points[9][0]>f[0].points[9][0]);
 assert.equal(f[0].points[9][1],f[1].points[9][1]);
 assert.equal(f[0].pose[9][0],9);
});
test('selected sparse frame is retained in the overview',()=>{
 const f=sequenceFrames({values:Array.from({length:50},(_,i)=>pose(i))});
 const strip=frameStrip(f,17,5);
 assert.ok(strip.some(v=>v.time===17));assert.equal(strip[0].time,0);assert.equal(strip.at(-1).time,49);
});
test('temporal window respects channel, native time, unknown operands and padding',()=>{
 const c={stage:'temporal',operation:'temporal_convolution',metadata:{input_timesteps:[-1,0,1]},terms:[
 {inputIndex:[0,0],input:0,product:0,padding:true,recorded:true},
 {inputIndex:[0,1],input:null,product:null,recorded:false},
 {inputIndex:[1,1],input:2,product:3,recorded:true}]};
 const t=temporalKernel(c,0);assert.equal(t.length,2);assert.equal(t[0].time,-1);assert.equal(t[0].padding,true);assert.equal(t[1].input,null);
 assert.deepEqual(temporalKernel({...c,operation:'residual_add'},0),[]);
});
test('record labels retain numeric kernel coordinates and live time indices keep negative padding',()=>{
 const saved={stage:'temporal',metadata:{term_indices:[[3,0],[3,1]],input_timesteps:[24,25]},terms:[{index:0,inputIndex:'channel 3 · t=24 · kernel 0'},{index:1,inputIndex:'channel 3 · t=25 · kernel 1'}]};
 assert.deepEqual(temporalKernel(saved,3).map(t=>[t.slot,t.time]),[[0,24],[1,25]]);
 const live={stage:'temporal',metadata:{input_axes:['channel','time','joint'],kernel_size:9},terms:[{index:0,inputIndex:[0,-4,0]},{index:1,inputIndex:[0,-3,0]},{index:9,inputIndex:[1,-4,0]}]};
 assert.deepEqual(temporalKernel(live,0).map(t=>[t.slot,t.time]),[[0,-4],[1,-3]]);
 assert.deepEqual(temporalKernel(live,1).map(t=>[t.slot,t.time]),[[0,-4]]);
});
