import { Button, Checkbox, Progress, Select } from './ui/Controls';
import { useEffect, useRef, useState } from 'react';
import { CLASS_LABELS, trackLabel } from '../api';
import { createCameraController } from '../camera/controller';
import { captureCameraFrame, drawCameraResult } from '../camera/render';
import '../styles/camera.css';

const BUSY=new Set(['requesting','connecting','running']);
const LABELS={idle:'待开启',requesting:'等待权限',connecting:'连接中',running:'现场识别',stopped:'已停止',error:'未连接'};

export default function LiveCamera({ health }) {
    const video=useRef(null),canvas=useRef(null),controller=useRef(null);
    const [state,setState]=useState({status:'idle',devices:[],result:null,error:'',streaming:false});
    const [device,setDevice]=useState(''),[skeleton,setSkeleton]=useState(true);
    useEffect(()=>{
        let mounted=true;
        const client=createCameraController({video:video.current,
            socketUrl:`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/camera/live`,
            capture:()=>captureCameraFrame(video.current),
            render:(frame,result,showSkeleton)=>drawCameraResult(canvas.current,frame,result,showSkeleton),
            onUpdate:value=>{if(mounted)setState(value);},
        });
        controller.current=client;client.refreshDevices();
        const hide=()=>{if(document.hidden)client.stop('页面已进入后台，摄像头已停止');};
        const leave=()=>client.stop();
        document.addEventListener('visibilitychange',hide);window.addEventListener('pagehide',leave);
        navigator.mediaDevices?.addEventListener?.('devicechange',client.refreshDevices);
        return ()=>{mounted=false;client.stop();controller.current=null;
            document.removeEventListener('visibilitychange',hide);window.removeEventListener('pagehide',leave);
            navigator.mediaDevices?.removeEventListener?.('devicechange',client.refreshDevices);};
    },[]);
    const active=BUSY.has(state.status),tracks=state.result?.tracks || [];
    const classified=tracks.filter(track=>track.class_id!=null).length;
    return <div className="camera-page">
        <header className="camera-heading"><div><span className="mono">LIVE / CAMERA</span><h1>现场视频</h1>
            <p>打开摄像头，直接查看行人骨架、跟踪 ID 与意图结果。</p></div>
            <span className={`camera-status ${state.status==='running'?'is-live':''}`} role="status"><i />{LABELS[state.status]}</span></header>
        <div className="camera-toolbar">
            <label>摄像头<Select aria-label="选择摄像头" value={device} onChange={event=>{
                setDevice(event.target.value);if(active)controller.current.start(event.target.value);
            }}><option value="">默认摄像头</option>{state.devices.map((item,index)=><option key={item.deviceId||index} value={item.deviceId}>
                {item.label||`摄像头 ${index+1}`}</option>)}</Select></label>
            <Button className="blueprint-button camera-start" disabled={!active&&health?.models_ready===false}
                onClick={()=>active?controller.current.stop():controller.current.start(device)}>{active?'停止摄像头':'开启摄像头'}</Button>
            <label className="camera-skeleton"><Checkbox type="checkbox" checked={skeleton} onChange={event=>{
                setSkeleton(event.target.checked);controller.current.setSkeleton(event.target.checked);
            }} />显示骨架</label>
            <span className="camera-policy">仅视频 · 本机识别 · 不保存画面</span>
        </div>
        {state.error&&<div className="camera-error" role="alert">{state.error}</div>}
        {health?.models_ready===false&&<div className="camera-error" role="alert">模型尚未就绪：{health.detail || '请检查本地模型文件。'}</div>}
        <div className="camera-layout">
            <section className="camera-view" aria-label="现场识别画面">
                <div className="camera-screen">
                    <canvas ref={canvas} className={state.result?'':'is-hidden'} aria-label="与骨架和意图同步的识别画面" />
                    <div className={`camera-preview ${state.result?'is-inset':''} ${state.streaming?'':'is-hidden'}`}>
                        <video ref={video} muted playsInline autoPlay aria-label="摄像头实时预览" />
                        <span>实时预览</span>
                    </div>
                    {!state.streaming&&<div className="camera-placeholder"><svg viewBox="0 0 64 64" aria-hidden="true"><rect x="8" y="18" width="34" height="30" rx="5"/><path d="M42 28 56 20v28L42 40Z"/><circle cx="25" cy="33" r="7"/></svg>
                        <h2>{state.status==='requesting'?'等待摄像头授权':state.status==='error'?'摄像头未连接':'让现场进入画面'}</h2>
                        <p>{state.status==='requesting'?'请在浏览器提示中允许摄像头访问。':'点击“开启摄像头”开始，支持内置和 USB 摄像头。'}</p></div>}
                    {state.streaming&&!state.result&&<div className="camera-loading" role="status">{state.message}</div>}
                    {state.result&&<div className="camera-frame-label">识别画面 · #{state.result.frame+1}</div>}
                </div>
                <div className="camera-metrics"><span>当前行人 <strong>{tracks.length}</strong></span><span>已输出意图 <strong>{classified}</strong></span>
                    <span>识别帧率 <strong>{state.fps==null?'—':`${state.fps.toFixed(1)} fps`}</strong></span>
                    <span>往返耗时 <strong>{state.latency==null?'—':`${Math.round(state.latency)} ms`}</strong></span></div>
                <p className="camera-footnote">识别画面与骨架、ID、分数对应同一帧；右上角为连续实时预览。完整看到行人身体，并积累 50 次有效骨架后显示意图。</p>
                {state.status==='stopped'&&<p className="camera-footnote" role="status">{state.message}</p>}
            </section>
            <aside className="camera-people" aria-label="现场行人结果"><div className="camera-people-heading"><h2>现场行人</h2><span>{tracks.length} 人</span></div>
                {!tracks.length&&<div className="camera-empty"><strong>{state.result?'当前未检测到行人':'等待现场画面'}</strong>
                    <p>{state.result?'请让行人全身进入镜头，并保持光线充足。':'开启后将在此显示每位行人的 ID 和意图结果。'}</p></div>}
                {tracks.map(track=><article className="camera-person" key={track.track_id}>
                    <header><span className="mono">ID {track.track_id}</span><strong className={track.class_id===2?'is-crossing':''}>{trackLabel(track)}</strong></header>
                    {track.class_id==null?<div className="camera-warmup"><span>{track.pose_accepted===false?'姿态不完整，请露出身体':`骨架积累 ${track.history_length} / 50`}</span>
                        <Progress max="50" value={track.history_length||0} aria-label={`ID ${track.track_id} 骨架积累`} /></div>
                    :<div className="camera-class-scores">{CLASS_LABELS.map((name,index)=><div key={name}>
                        <span>{name}</span><i><em style={{width:`${(track.probabilities?.[index]||0)*100}%`}} /></i>
                        <b>{Number.isFinite(track.probabilities?.[index])?`${(track.probabilities[index]*100).toFixed(1)}%`:'—'}</b></div>)}</div>}
                </article>)}
                <p className="camera-footnote">显示模型分数，不代表识别正确率。停止或离开本页后清空现场结果。</p>
            </aside>
        </div>
    </div>;
}
