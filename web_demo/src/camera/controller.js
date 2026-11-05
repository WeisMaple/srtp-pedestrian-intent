const ERRORS = {
    NotAllowedError: '摄像头权限未开启。请在浏览器地址栏允许摄像头后重试。',
    NotFoundError: '没有找到摄像头，请连接设备后重试。',
    NotReadableError: '无法读取摄像头，可能正被其他应用占用。',
    OverconstrainedError: '所选摄像头已不可用，请重新选择设备。',
};

/** One outstanding captured frame; all asynchronous work belongs to one generation. */
export function createCameraController({ video, socketUrl, capture, render, onUpdate,
    mediaDevices = navigator.mediaDevices, makeSocket = url => new WebSocket(url),
    schedule = setTimeout, cancel = clearTimeout, now = () => performance.now() }) {
    let generation=0, stream=null, socket=null, timer=null, deadline=null, pending=null;
    let lastFrame=null, lastResult=null, skeleton=true, expectedFrame=0, lastReceived=null;
    let state={status:'idle',error:'',message:'',result:null,devices:[],streaming:false,fps:null,latency:null};
    const publish=patch=>{state={...state,...patch};onUpdate(state);};
    const clearTimers=()=>{cancel(timer);cancel(deadline);timer=null;deadline=null;};
    function stop(message='摄像头已停止', error='') {
        generation++;
        clearTimers();
        if(socket) { socket.onmessage=null;socket.onerror=null;socket.onclose=null;socket.close();socket=null; }
        stream?.getTracks().forEach(track=>{track.onended=null;track.stop();});
        stream=null;video.srcObject=null;pending=null;lastFrame=null;lastResult=null;
        publish({status:error?'error':'stopped',error,message,result:null,streaming:false,fps:null,latency:null});
    }
    const fail=error=>stop('',ERRORS[error?.name] || error?.message || '摄像头连接失败，请重试。');
    async function refreshDevices() {
        try {
            const devices=await mediaDevices?.enumerateDevices?.();
            if(devices) publish({devices:devices.filter(device=>device.kind==='videoinput')});
        } catch { /* A device list is optional; default capture still works. */ }
    }
    async function start(deviceId='') {
        stop('');
        const token=generation, current=()=>token===generation;
        expectedFrame=0;lastReceived=null;
        publish({status:'requesting',message:'请允许浏览器使用摄像头',error:''});
        if(!mediaDevices?.getUserMedia) { fail(new Error('当前环境无法访问摄像头，请使用 localhost 或 HTTPS 打开页面。'));return; }
        try {
            const acquired=await mediaDevices.getUserMedia({audio:false,video:{
                width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:30},
                ...(deviceId?{deviceId:{exact:deviceId}}:{facingMode:'environment'}),
            }});
            if(!current()) { acquired.getTracks().forEach(track=>track.stop());return; }
            stream=acquired;video.srcObject=stream;
            stream.getVideoTracks().forEach(track=>{track.onended=()=>{if(current())fail(new Error('摄像头已断开，请检查设备后重新开始。'));};});
            publish({streaming:true,status:'connecting',message:'正在连接本地识别模型…'});
            refreshDevices();
            await video.play();
            if(!current()) return;
            let busyRetries=0;
            function connect() {
                if(!current()) return;
                const connection=makeSocket(socketUrl);socket=connection;
                const armTimeout=()=>{cancel(deadline);deadline=schedule(()=>{if(current())fail(new Error('识别服务响应超时，请检查服务后重试。'));},30000);};
                async function nextFrame() {
                    if(!current() || pending || connection.readyState!==1) return;
                    try {
                        // Reserve before asynchronous JPEG encoding to prevent overlapping capture.
                        pending={capturing:true};
                        const frame=await capture();
                        if(!current()) return;
                        if(!frame) {pending=null;timer=schedule(nextFrame,100);return;}
                        pending={...frame,sentAt:now()};
                        connection.send(frame.blob);armTimeout();
                    } catch(error) {if(current())fail(error);}
                }
                connection.onmessage=event=>{
                    if(!current()) return;
                    try {
                        const data=JSON.parse(event.data);
                        if(data.type==='error') {
                            if(data.code==='camera_busy' && busyRetries++<20) {
                                cancel(deadline);
                                connection.onmessage=null;connection.onerror=null;connection.onclose=null;
                                connection.close();socket=null;
                                publish({status:'connecting',message:'等待上一段现场识别结束…'});
                                timer=schedule(connect,250);return;
                            }
                            throw new Error(data.message);
                        }
                        if(data.type==='ready') {
                            if(state.status!=='connecting') return;
                            cancel(deadline);
                            publish({status:'running',message:'正在读取现场画面'});
                            timer=schedule(nextFrame,0);
                        } else if(data.type==='result') {
                            if(!pending || pending.capturing || data.frame!==expectedFrame
                                || data.width!==pending.width || data.height!==pending.height || !Array.isArray(data.tracks))
                                throw new Error('识别画面与返回结果不同步，请重新开始。');
                            cancel(deadline);
                            const received=now();
                            lastFrame=pending;lastResult=data;render(lastFrame,data,skeleton);
                            publish({result:data,message:'正在识别',latency:received-pending.sentAt,
                                fps:lastReceived==null?null:1000/(received-lastReceived)});
                            lastReceived=received;pending=null;expectedFrame++;
                            timer=schedule(nextFrame,Math.max(0,100-(received-lastFrame.sentAt)));
                        }
                    } catch(error) {fail(error);}
                };
                connection.onerror=()=>{if(current())fail(new Error('无法连接本地识别服务，请确认服务已启动。'));};
                connection.onclose=()=>{if(current())fail(new Error('识别连接已断开，摄像头已关闭，可重新开始。'));};
                armTimeout();
            }
            connect();
        } catch(error) {if(current())fail(error);}
    }
    return { start, stop, refreshDevices, setSkeleton(value) {
        skeleton=value;if(lastFrame&&lastResult)render(lastFrame,lastResult,skeleton);
    } };
}
