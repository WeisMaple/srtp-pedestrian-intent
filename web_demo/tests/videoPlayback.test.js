import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import TestRenderer, { act } from 'react-test-renderer';
import { useVideoPlayback } from '../src/components/ui/useVideoPlayback.js';

test('replaced, paused and unmounted videos ignore pending play failures; current errors remain visible',async()=>{
    globalThis.IS_REACT_ACT_ENVIRONMENT=true;
    let renderer,playback,rejectPlay;
    const errors=[];
    const media={current:{paused:true,ended:false,pause(){},play(){return new Promise((resolve,reject)=>{rejectPlay=reject;});}}};
    function Probe({src}) {playback=useVideoPlayback(media,src,error=>errors.push(error));return null;}
    try {
        await act(async()=>{renderer=TestRenderer.create(React.createElement(Probe,{src:'old.mp4'}));});
        let pending;
        await act(async()=>{pending=playback.toggle();});
        await act(async()=>{renderer.update(React.createElement(Probe,{src:'new.mp4'}));});
        await act(async()=>{rejectPlay(new Error('stale source failure'));await pending;});
        assert.equal(errors.filter(Boolean).length,0);
        await act(async()=>{pending=playback.toggle();playback.cancel();rejectPlay(new Error('paused'));await pending;});
        assert.equal(errors.filter(Boolean).length,0);
        await act(async()=>{pending=playback.toggle();rejectPlay(Object.assign(new Error('interrupted'),{name:'AbortError'}));await pending;});
        assert.equal(errors.filter(Boolean).length,0);
        await act(async()=>{pending=playback.toggle();rejectPlay(new Error('unsupported'));await pending;});
        assert.equal(errors.filter(Boolean).length,1);
        await act(async()=>{pending=playback.toggle();renderer.unmount();});
        await act(async()=>{rejectPlay(new Error('removed source'));await pending;});
        assert.equal(errors.filter(Boolean).length,1);
    } finally {
        await act(async()=>{renderer?.unmount();});
        delete globalThis.IS_REACT_ACT_ENVIRONMENT;
    }
});
