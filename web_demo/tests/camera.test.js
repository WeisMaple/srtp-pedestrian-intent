import test from 'node:test';
import assert from 'node:assert/strict';
import { createCameraController } from '../src/camera/controller.js';

const flush = async () => { for (let i=0;i<8;i++) await Promise.resolve(); };
function fixture(overrides={}) {
    const tracks = [{ stopCount:0, stop(){this.stopCount++;}, onended:null }];
    const stream = { getTracks:()=>tracks, getVideoTracks:()=>tracks };
    const states=[], sockets=[], rendered=[], timers=new Map(); let id=0, time=100;
    const video={srcObject:null, play:async()=>{}};
    const controller=createCameraController({ video, socketUrl:'ws://localhost/api/camera/live',
        mediaDevices:{getUserMedia:async()=>stream,enumerateDevices:async()=>[]},
        makeSocket:()=>{const socket={readyState:1,sent:[],send(blob){this.sent.push(blob);},close(){this.readyState=3;}};sockets.push(socket);return socket;},
        capture:async()=>({blob:'jpeg',width:160,height:120}), render:(frame,result)=>rendered.push(result),
        onUpdate:state=>states.push(state), now:()=>time+=100,
        schedule:(fn,delay)=>{timers.set(++id,{fn,delay});return id;},cancel:token=>timers.delete(token), ...overrides });
    const tick=async()=>{const next=[...timers].find(([,row])=>row.delay<=250);assert.ok(next);timers.delete(next[0]);await next[1].fn();await flush();};
    const receive=data=>sockets.at(-1).onmessage({data:JSON.stringify(data)});
    return {controller,tracks,stream,states,sockets,rendered,timers,video,tick,receive};
}

test('camera sends one frame at a time and renders each result against its captured dimensions', async()=>{
    const f=fixture(); await f.controller.start();
    f.receive({type:'ready'}); await f.tick();
    assert.equal(f.sockets[0].sent.length,1);
    assert.equal([...f.timers.values()].filter(row=>row.delay<=100).length,0);
    f.receive({type:'result',frame:0,width:160,height:120,tracks:[{track_id:7}],processing_ms:12});
    assert.equal(f.rendered[0].frame,0);
    await f.tick(); assert.equal(f.sockets[0].sent.length,2);
    f.receive({type:'result',frame:1,width:160,height:120,tracks:[],processing_ms:10});
    assert.deepEqual(f.states.at(-1).result.tracks,[]);
    f.controller.stop();
    assert.equal(f.tracks[0].stopCount,1);
    assert.equal(f.video.srcObject,null);
    assert.equal(f.timers.size,0);
    assert.equal(f.states.at(-1).result,null);
});

test('stopping while permission is pending releases a late media stream without opening a connection', async()=>{
    let grant;
    const f=fixture({mediaDevices:{getUserMedia:()=>new Promise(resolve=>{grant=resolve;})}});
    const starting=f.controller.start(); f.controller.stop(); grant(f.stream); await starting;
    assert.equal(f.tracks[0].stopCount,1); assert.equal(f.sockets.length,0);
    assert.equal(f.states.at(-1).status,'stopped');
});

test('backend errors and unexpected closure release the camera and clear displayed results', async()=>{
    const f=fixture();await f.controller.start();
    f.receive({type:'error',message:'模型忙碌'});
    assert.equal(f.states.at(-1).error,'模型忙碌');assert.equal(f.tracks[0].stopCount,1);
    assert.equal(f.states.at(-1).result,null);
    const g=fixture();await g.controller.start();g.sockets[0].onclose();
    assert.equal(g.states.at(-1).status,'error');assert.equal(g.tracks[0].stopCount,1);
});

test('permission denial is recoverable and frame mismatch is never painted', async()=>{
    const f=fixture({mediaDevices:{getUserMedia:async()=>{throw Object.assign(new Error('denied'),{name:'NotAllowedError'});}}});
    await f.controller.start();assert.equal(f.states.at(-1).status,'error');
    assert.match(f.states.at(-1).error,/权限/);
    const g=fixture();await g.controller.start();g.receive({type:'ready'});await g.tick();
    g.receive({type:'result',frame:0,width:320,height:120,tracks:[]});
    assert.equal(g.rendered.length,0);assert.equal(g.states.at(-1).status,'error');
    assert.equal(g.tracks[0].stopCount,1);
});

test('a capture completing after stop cannot send into the old or new session', async()=>{
    let captured;
    const f=fixture({capture:()=>new Promise(resolve=>{captured=resolve;})});
    await f.controller.start();f.receive({type:'ready'});
    const capturing=f.tick();await flush();f.controller.stop();
    captured({blob:'late',width:160,height:120});await capturing;
    assert.equal(f.sockets[0].sent.length,0);assert.equal(f.rendered.length,0);
});

test('quick camera restart waits for prior inference cleanup and stop cancels the retry', async()=>{
    const f=fixture();await f.controller.start('camera-a');f.receive({type:'ready'});await f.tick();
    await f.controller.start('camera-b');
    f.receive({type:'error',code:'camera_busy',message:'prior camera is closing'});
    assert.equal(f.states.at(-1).status,'connecting');
    assert.equal(f.states.at(-1).streaming,true);
    await f.tick();assert.equal(f.sockets.length,3);
    f.receive({type:'ready'});await f.tick();
    f.receive({type:'result',frame:0,width:160,height:120,tracks:[],processing_ms:10});
    assert.equal(f.states.at(-1).status,'running');
    assert.equal(f.rendered.length,1);
    await f.controller.start();f.receive({type:'error',code:'camera_busy',message:'closing'});
    f.controller.stop();assert.equal(f.timers.size,0);assert.equal(f.states.at(-1).streaming,false);
});
