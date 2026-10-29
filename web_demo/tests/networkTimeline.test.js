import test from 'node:test';
import assert from 'node:assert/strict';
import { networkPosition, networkStart, NETWORK_CHAPTERS, NETWORK_DURATION } from '../src/networkTimeline.js';

test('each chapter begins exactly at its boundary, scrubbing back preserves stage position',()=>{
    for(let i=0;i<NETWORK_CHAPTERS.length;i++) {
        assert.deepEqual(networkPosition(networkStart(i)),{chapter:i,progress:0});
        assert.deepEqual(networkPosition(networkStart(i)+NETWORK_CHAPTERS[i].duration/2),{chapter:i,progress:.5});
    }
    assert.equal(networkPosition(networkStart(5)+1).chapter,5);
    assert.equal(networkPosition(networkStart(2)+1).chapter,2);
    assert.equal(networkPosition(4).chapter,0);
});
test('end and out-of-bounds playback stay within a valid diagram',()=>{
    const end={chapter:NETWORK_CHAPTERS.length-1,progress:1};
    assert.deepEqual(networkPosition(NETWORK_DURATION),end);
    assert.deepEqual(networkPosition(NETWORK_DURATION+100),end);
    assert.deepEqual(networkPosition(-3),{chapter:0,progress:0});
});
