import test from 'node:test';
import assert from 'node:assert/strict';
import {cursorTime,timeCursor,graphTimeline,timelineTime,timelineCursor} from '../src/hooks/useCalculationPlayback.js';
test('graph message timing dwells on connected zero too, without dropping disconnected candidates',()=>{
    const stops=graphTimeline({stage:'spatial',terms:[{connected:true,product:0},{connected:false},{connected:true}]});
    assert.equal(stops[1],1.6);assert.ok(stops[2]-stops[1]<.05);assert.equal(stops.length,6);
    for(const c of [0,.2,.5,.85,1.2,2.5,3,4.5,5])assert.ok(Math.abs(timelineCursor(timelineTime(c,stops),stops)-c)<1e-9);
    assert.equal(timelineCursor(100,stops),5);assert.equal(graphTimeline({stage:'decoder'}),null);
});
test('dataflow timing preserves every cursor when slowing early terms and accelerating long sums',()=>{
    for(const count of [1,3,27,307,576,2304])for(const fraction of [0,.1,.5,.85,.999,1]){
        const cursor=fraction*(count+2);
        assert.ok(Math.abs(timeCursor(cursorTime(cursor,count),count)-cursor)<1e-9);
    }
});
test('terminal bias and activation each remain visible and seeking backwards is deterministic',()=>{
    assert.ok(cursorTime(309,307)-cursorTime(307,307)>1.8);
    assert.equal(timeCursor(100,307),309);
    assert.equal(timeCursor(-1,307),0);
    assert.ok(cursorTime(1,307)>1);
    assert.ok(cursorTime(307,307)<9);
});
