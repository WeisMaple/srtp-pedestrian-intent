import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { ACTIVE_STATUSES, trackLabel } from '../api';
import { useTrace } from '../hooks/useTrace';
import { useNetwork } from '../hooks/useNetwork';
import { useFrontendNetwork } from '../hooks/useFrontendNetwork';
import { useNetworkPlayback } from '../hooks/useNetworkPlayback';
import { useLiveComputation } from '../hooks/useLiveComputation';
import { normalizeLiveNodes } from '../liveComputation';
import { NETWORK_CHAPTERS, NETWORK_DURATION } from '../networkTimeline';
import { LstmView } from './NetworkViews';
import { PipelineOverview, YoloView, TrackingView, ForecastView, SpatialView, TemporalView, ClassifierView } from './PipelineViews';
import { LiveControls, LiveTensorView, LiveActivity } from './LiveComputationPanel';
import InputPanel from './InputPanel';
import VideoPanel from './VideoPanel';
import Skeleton from './Skeleton';
import { ScoreBars } from './TrackPanel';
import Icon from './Icon';
import '../live-workbench.css';

const NeuralScene3D=lazy(()=>import('./NeuralScene3D'));
const CalculationWorkbench=lazy(()=>import('./CalculationWorkbench'));
const EMPTY=[];
export default function ProcessPage({ state, job, selectedId, onSelect }) {
    const busy=ACTIVE_STATUSES.has(job?.status);
    const live=useLiveComputation(job);
    const [sourceChoice,setSourceChoice]=useState(null);
    const hasLive=Object.keys(live.snapshot?.nodes||{}).length>0;
    const hasCalculations=Object.values(live.snapshot?.nodes||{}).some(event=>event.calculations?.length);
    const source=sourceChoice||(busy||hasCalculations?'live':'inspection');
    const isLive=source==='live';
    const { samples, error:traceError, retry:retryTrace }=useTrace(job,!isLive);
    const [selectedSample,setSelectedSample]=useState(null);
    const [presentation,setPresentation]=useState(false);
    const [dimension,setDimension]=useState('3d');
    const [sceneError,setSceneError]=useState('');
    const [zoom,setZoom]=useState(1);
    const canvasScroll=useRef(null);
    const sampleIndex=selectedSample===null?samples.length-1:Math.min(selectedSample,samples.length-1);
    const sample=samples[sampleIndex];
    const tracks=isLive?(live.snapshot?.tracks||EMPTY):(sample?.tracks||EMPTY);
    const track=tracks.find(item=>String(item.track_id)===String(selectedId));
    const recordedTrack=sample?.tracks?.find(item=>String(item.track_id)===String(selectedId));
    const network=useNetwork(job,isLive?null:sample,isLive?null:recordedTrack);
    const frontend=useFrontendNetwork(job,isLive?null:sample);
    const playback=useNetworkPlayback();
    const [preview,setPreview]=useState(true);
    const nodes=useMemo(()=>normalizeLiveNodes(live.snapshot,selectedId),[live.snapshot,selectedId]);
    useEffect(()=>{setSelectedSample(null);setSourceChoice(null);setSceneError('');playback.chapterSeek(0);setPreview(true);},[job?.id]);
    useEffect(()=>{if(busy){setSourceChoice('live');setDimension('3d');playback.pause();}},[busy]);
    useEffect(()=>{playback.pause();},[selectedId,selectedSample]);
    useEffect(()=>{
        if(tracks.length&&!tracks.some(item=>String(item.track_id)===String(selectedId)))onSelect(tracks[0].track_id);
    },[tracks,selectedId]);
    useEffect(()=>{const escape=event=>{if(event.key==='Escape')setPresentation(false);};window.addEventListener('keydown',escape);return()=>window.removeEventListener('keydown',escape);},[]);
    const navigate=chapter=>{playback.chapterSeek(chapter);setPreview(true);setZoom(1);if(canvasScroll.current)canvasScroll.current.scrollTop=0;};
    const changeSource=value=>{setSourceChoice(value);playback.pause();setPreview(true);};
    const inspectTime=time=>{if(!samples.length||busy||isLive)return;const nearest=samples.reduce((best,value,index)=>Math.abs(value.time_seconds-time)<Math.abs(samples[best].time_seconds-time)?index:best,0);setSelectedSample(nearest);};
    const progress=!preview&&playback.time>=NETWORK_DURATION?.999999:preview||playback.reducedMotion?1:playback.progress;
    const viewProps={data:network.data,frontend:frontend.data,progress,playing:playback.playing&&!playback.reducedMotion,onPause:playback.pause,job,track:recordedTrack,sample,samples,onChapter:navigate};
    const status=isLive?live.frozen?'已冻结当前计算':busy?'来自当前推理的前向输出':hasLive?'本次推理采集记录':'开始分析后接收实时计算':network.loading||frontend.loading?'正在读取记录计算…':network.data?'记录重算 / 真实张量':'选择记录以查看数值';
    const errors=isLive?live.error:traceError||network.error||frontend.error;
    const views=[PipelineOverview,YoloView,TrackingView,LstmView,ForecastView,SpatialView,TemporalView,ClassifierView];
    const View=views[playback.chapter];
    const displayFrame=isLive?live.snapshot?.frame:sample?.frame;
    useEffect(()=>{if(canvasScroll.current)canvasScroll.current.scrollTop=0;},[playback.chapter,dimension,isLive]);
    const historyCount=track?.history?.length??track?.history_length??0;
    const layerKeys=Object.keys(nodes);
    const received=[hasLive,layerKeys.some(k=>k.startsWith('yolo.')),Boolean(nodes.history)||historyCount>0,layerKeys.some(k=>k.startsWith('lstm.')),Boolean(nodes.future),layerKeys.some(k=>k.endsWith('.spatial')),layerKeys.some(k=>k.endsWith('.temporal')),Boolean(nodes['stgcn.logits'])];
    const stageStatus=index=>{
        if(!job)return '等待视频输入';
        if(!isLive)return (index===1?frontend.data:network.data)?'记录数值已就绪':'等待读取数值';
        if(index===0)return '最新采集帧 · 阶段状态';
        if(received[index])return index===2?`有效观测 ${historyCount} / 50`:'本帧已有实测值';
        if(index>=3&&historyCount>0&&historyCount<50)return '等待 50 次有效观测';
        return '本帧尚未收到数值';
    };

    return <main id="workbench" className={`unified-workbench ${presentation?'lab-presentation':''} ${dimension==='3d'?'three-dimensional':''}`}>
        <div className="lab-heading"><div><span className="network-kicker">LIVE NEURAL COMPUTATION / 3D EXPLORER</span><h1>看见，正在发生的推理<span>深度学习计算实验台</span></h1></div><button className="network-focus-button" aria-pressed={presentation} onClick={()=>setPresentation(!presentation)}><Icon name={presentation?'close':'expand'} size={16}/>{presentation?'退出演示':'专注演示'}</button></div>
        <div className="lab-input"><InputPanel {...state}/></div>
        <div className="lab-layout">
            <aside className="lab-context"><div className="lab-context-heading"><span>01 / VIDEO CONTEXT</span><b>视频与行人</b></div>
                <VideoPanel job={job} seekTime={!busy&&!isLive?sample?.time_seconds:undefined} onInspectTime={inspectTime}/>
                <div className="sample-inspector panel"><div className="sample-heading"><h2>{isLive?'当前采集的行人':'沿着同一个行人，追踪计算'}</h2><span>{displayFrame>=0?`帧 ${displayFrame+1}`:'等待输入'}</span></div>
                    {isLive?<div className="live-sample-caption"><i/>{live.frozen?'固定查看这份采集记录':busy?'网络层输出正在流入三维场景':'保留本次推理的最后采集结果'}</div>:<label className="record-select">计算记录<select aria-label="选择真实记录帧" value={sampleIndex} disabled={!samples.length} onChange={event=>setSelectedSample(Number(event.target.value))}>{!samples.length&&<option value="-1">暂无记录</option>}{samples.map((s,i)=><option key={s.frame} value={i}>第 {s.frame+1} 帧 · {s.time_seconds.toFixed(2)} s</option>)}</select></label>}
                    <div className="sample-people" aria-label="选择行人">{tracks.map(person=><button key={person.track_id} aria-pressed={String(person.track_id)===String(selectedId)} onClick={()=>onSelect(person.track_id)}>ID {String(person.track_id).padStart(2,'0')}</button>)}{!tracks.length&&<span>{busy?'当前采集帧正在检测行人':'分析后显示可追踪的行人 ID'}</span>}</div>
                    <div className="sample-person"><Skeleton pose={track?.keypoints} label="所选记录的行人骨架"/><div><span className="sample-prediction">{track?trackLabel(track):'等待有效观测'}</span><p>历史 {historyCount} / 50 · 每帧 17 关节</p><small className="sample-score-origin">{isLive?'本次实际推理分数':'视频原分析分数'}</small><ScoreBars probabilities={track?.probabilities}/></div></div>
                    <p className="sample-link-note">{isLive?'图像特征属于整帧；行为特征对应所选行人。预览显示最近完成的画面，计算区标明正在采集的帧。':'选择记录帧会定位视频；视频拖动后取最近保存记录。最多 60 个记录、每帧 3 人。'}</p>
                </div>
                {isLive?<LiveActivity live={live}/>:<div className="lab-source-note"><i/>记录讲解可慢放与逐项检查。<br/>切换“实时推理”可查看本次分析时采集的输出。</div>}
            </aside>
            <section className="lab-computation" aria-label="全流程神经网络计算">
                <div className="lab-canvas-heading"><div><span>02 / SPATIAL COMPUTATION</span><b>走进神经网络的计算空间</b></div><div className={`network-data-status ${isLive?hasLive?'ready':'':network.data?'ready':''}`}><i/>{status}</div></div>
                <div className="computation-mode-bar"><div className="dimension-switch" aria-label="计算展示维度"><button aria-pressed={dimension==='3d'} onClick={()=>setDimension('3d')}><span>◈</span>三维网络</button><button aria-pressed={dimension==='2d'} onClick={()=>setDimension('2d')}>数值推导</button></div><div className="source-switch" aria-label="计算数据来源"><button aria-pressed={isLive} onClick={()=>changeSource('live')}><i/>实时推理</button><button aria-pressed={!isLive} disabled={busy} onClick={()=>changeSource('inspection')}>记录讲解</button></div></div>
                <nav className="lab-stage-nav" aria-label="神经网络讲解章节">{NETWORK_CHAPTERS.map((chapter,i)=><button key={chapter.title} className={playback.chapter===i?'active':''} aria-current={playback.chapter===i?'step':undefined} onClick={()=>navigate(i)}><span>{String(i).padStart(2,'0')}</span><strong>{chapter.title}</strong><small className={received[i]?'stage-received':''} title={chapter.subtitle}>{stageStatus(i)}</small></button>)}</nav>
                {errors&&<div className="network-error" role="alert"><Icon name="info" size={17}/><span>{errors}</span><button onClick={()=>{live.retry();retryTrace();network.retry();frontend.retry();}}>重新连接</button></div>}
                {sceneError&&dimension==='3d'&&<div className="network-error" role="alert"><span>{sceneError}</span><button onClick={()=>setDimension('2d')}>查看数值视图</button></div>}
                {dimension==='3d'?<div className="three-canvas-frame" data-source={source} data-frame={displayFrame??-1} data-seq={live.snapshot?.seq??0} data-frozen={live.frozen}>
                    <Suspense fallback={<div className="three-loading"><span>◈</span><strong>正在构建三维计算空间</strong><p>准备张量体素、网络层与相机</p></div>}><CalculationWorkbench job={job} stage={playback.chapter} snapshot={isLive?live.snapshot:undefined} data={network.data} frontend={frontend.data} trackId={selectedId} frame={displayFrame} isLive={isLive}
                        sceneContext={{liveSnapshot:isLive?live.snapshot:null,selectedId,data:isLive?null:network.data,frontend:isLive?null:frontend.data,track,liveNodes:isLive?nodes:undefined,liveEvent:isLive?live.latestEvent:null}}
                        renderScene={({calculation,state:calculationState,context,onTermSelect,onCursorChange,onGraphTarget,playing,onTogglePlay})=><NeuralScene3D stage={playback.chapter} {...context} calculation={calculation} calculationState={calculationState} onTermSelect={onTermSelect} onCursorChange={onCursorChange} onGraphTarget={onGraphTarget} calculationPlaying={playing} onToggleCalculation={onTogglePlay} running={false} frozen={Boolean(calculation)} progress={isLive?1:progress} onStageSelect={navigate} onError={setSceneError}/>}/></Suspense>
                    {isLive&&!hasLive&&<div className="three-waiting-label">{busy?'正在等待本帧的第一个网络层输出':'选择视频并开始分析，即可实时观察网络计算'}</div>}
                </div>:<><div className="lab-canvas-tools"><span>激活 <i className="activation-dot"/> · 权重 <i className="weight-dot"/> · <b>{displayFrame>=0?`ID ${selectedId??'—'} / 帧 ${displayFrame+1}`:'结构浏览'}</b></span>{!isLive&&<div><button aria-label="缩小计算画布" onClick={()=>setZoom(v=>Math.max(.75,v-.25))}>−</button><button onClick={()=>setZoom(1)} aria-label="重置计算画布缩放">{Math.round(zoom*100)}%</button><button aria-label="放大计算画布" onClick={()=>setZoom(v=>Math.min(1.75,v+.25))}>＋</button></div>}</div><div className="lab-canvas-scroll" ref={canvasScroll}><section className="network-canvas" style={{width:`${zoom*100}%`}} data-chapter={playback.chapter} data-progress={progress.toFixed(4)} data-playing={playback.playing}>{isLive?<LiveTensorView nodes={nodes} stage={playback.chapter}/>:<View {...viewProps}/>}</section></div></>}
                {dimension==='3d'?null:isLive?<LiveControls live={live} job={job}/>:<div className="network-player panel"><div className="network-player-buttons"><button className="replay-play" aria-label={playback.playing?'暂停网络讲解':'播放网络讲解'} onClick={()=>{setPreview(false);playback.toggle();}}><Icon name={playback.playing?'pause':'play'} size={21}/></button><button className="icon-button" aria-label="从头播放网络讲解" onClick={()=>{setPreview(false);playback.restart();}}><Icon name="refresh" size={17}/></button><div className="speed-control" aria-label="讲解速度">{[.5,1,2].map(speed=><button key={speed} className={playback.speed===speed?'active':''} aria-pressed={playback.speed===speed} onClick={()=>playback.setSpeed(speed)}>{speed}×</button>)}</div></div><div className="network-player-timeline"><div><strong>{NETWORK_CHAPTERS[playback.chapter].title}</strong><span>{playback.time.toFixed(1)} / {NETWORK_DURATION} s</span></div><input aria-label="网络讲解进度" type="range" min="0" max={NETWORK_DURATION} step=".05" value={playback.time} onChange={e=>{setPreview(false);playback.seek(Number(e.target.value));}} style={{'--fill':`${playback.time/NETWORK_DURATION*100}%`}}/></div><button className="lab-step" aria-label="下一计算章节" disabled={playback.chapter===7} onClick={()=>navigate(playback.chapter+1)}><Icon name="arrow" size={18}/></button></div>}
                <p className="lab-provenance">{isLive?'实时输出直接采自视频分析中的前向计算，每秒最多采集 4 帧；光迹适当延长以便观察，实际层耗时单独显示。冻结画面不会暂停推理。':'记录讲解：YOLO 使用源视频帧重算，LSTM / ST-GCN 使用保存的骨架重算。播放速度是讲解速度。三维中的骨架深度轴表示时间或特征布局。'}</p>
            </section>
        </div>
    </main>;
}
