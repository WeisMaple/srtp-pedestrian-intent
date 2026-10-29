import { useMemo, useState } from 'react';
import { TensorMap, fmt, shape } from './NetworkPrimitives';
import { WeightedSum, ValueTransform } from './ArithmeticTheater';
import Skeleton, { poseBounds } from './Skeleton';
import OperatorTabs, { useOperator } from './OperatorTabs';
import { NETWORK_CHAPTERS } from '../networkTimeline';

const clamp = v=>Math.max(0,Math.min(1,v));
const phase = (p,start,end)=>clamp((p-start)/(end-start));
const flat = v=>v?.flat(Infinity);
const titles = ['其他','不过街','过街'];
const jointPositions = [[100,22],[87,15],[113,15],[74,23],[126,23],[65,65],[135,65],[45,109],[155,109],[30,155],[170,155],[78,146],[122,146],[73,202],[127,202],[65,254],[135,254]];
function Title({number,title,children}) { return <div className="network-section-title"><span>{number} / INSIDE THE COMPUTATION</span><h2>{title}</h2><p>{children}</p></div>; }
function DepthLink({onClick,children}) {return <button className="depth-link" onClick={onClick}>{children}<span>↗</span></button>;}
function Missing({children}) {return <div className="trace-missing">{children || '等待真实计算数据。上传视频并完成分析后，这里会出现可逐项检查的数值。'}</div>;}

export function PipelineOverview({data,frontend,track,progress,onChapter}) {
    const groups=[
        {name:'视觉编码',shape:shape(frontend?.input?.tensor_shape),chapter:1,items:(frontend?.layers||[]).filter((_,i)=>[0,2,4,6,8,10,16,19,22].includes(i)).map(v=>({name:`${v.index} · ${v.type}`,values:v.values?.[0]}))},
        {name:'时序记忆',shape:'50 × 51 → 10 × 51',chapter:3,items:Array.from({length:10},(_,i)=>({name:`LSTM ${i+1}`,values:data?.lstm?.rounds?.[0]?.layers?.[i]?.h?[data.lstm.rounds[0].layers[i].h]:undefined}))},
        {name:'时空图网络',shape:'3 → 64 → 128 → 256',chapter:5,items:Array.from({length:10},(_,i)=>({name:`ST-GCN ${i+1}`,values:data?.stgcn?.blocks?.[i]?.output?.values}))},
    ];
    return <div><Title number="00" title="把整个模型铺开，再走进每一次计算">沿网络从左向右阅读。每个小矩阵是实际中间数值的采样；点击任意一列深入计算。</Title>
        <div className="architecture-loom"><div className="loom-input"><div className="loom-image">{frontend?.input?.image_data_url?<img src={frontend.input.image_data_url} alt="所选源视频帧"/>:<Skeleton pose={track?.keypoints}/>}</div><b>一帧图像</b><small>像素 → 特征</small></div>{groups.map((group,index)=><button className="loom-column" key={group.name} onClick={()=>onChapter(group.chapter)} style={{'--loom-reveal':clamp(progress*3-index)}}><header><b>{group.name}</b><small>{group.shape}</small></header><div className="loom-tensors">{(group.items.length?group.items:Array.from({length:9},(_,i)=>({name:`阶段 ${i+1}`}))).map((item,i)=><div className="loom-tensor" key={i} style={{opacity:progress===1?1:.3+.7*clamp(progress*30-index*10-i)}}><span>{item.name}</span><TensorMap values={item.values} label={item.values?'实际值采样':'等待数值'} rows={4} cols={8}/></div>)}</div><footer>展开网络 ↗</footer></button>)}<div className="loom-output"><span>意图分数</span>{titles.map((s,i)=><div key={s}><b>{s}</b><i><em style={{transform:`scaleY(${(data?.stgcn?.probabilities?.[i]||0)*phase(progress,.7,1)})`}}/></i><small>{fmt(data?.stgcn?.probabilities?.[i])}</small></div>)}</div></div>
        <div className="pipeline-bridges"><DepthLink onClick={()=>onChapter(2)}>身份关联 → 50 次观测</DepthLink><DepthLink onClick={()=>onChapter(4)}>五轮预测 → 50 帧未来</DepthLink><DepthLink onClick={()=>onChapter(6)}>9 步时间窗口 → 残差</DepthLink><DepthLink onClick={()=>onChapter(7)}>256 维特征 → 3 类输出</DepthLink></div>
        <div className="overview-instruction"><span>探索方式</span><b>选一个行人 → 进入网络 → 展开算子 → 检查一个输出值</b><p>可直接跳转，也可按播放键完成 {NETWORK_CHAPTERS.length} 个阶段的连续讲解。输入、权重、乘积与累加值共用同一条时间轴。</p></div>
    </div>;
}

export function YoloView({frontend,progress,onPause}) {
    const [selected,setSelected]=useState(0);
    const [stepIndex,setStep]=useState(0);
    const op=useOperator(progress,[.55,.7,.9,1],onPause);
    const layers=frontend?.layers||[];
    const layer=layers[selected];
    const steps=frontend?.first_conv?.steps||[];
    const example=steps[stepIndex]||frontend?.first_conv?.example;
    const operationProgress=op.local;
    const conv=frontend?.first_conv;
    const convInputs=useMemo(()=>flat(example?.patch),[example]);
    const convWeights=useMemo(()=>flat(example?.weights),[example]);
    return <div className="yolo-view" data-operator={op.index}><Title number="01" title="像素进入卷积核，逐层形成可识别的姿态特征">查看卷积如何生成一个数，再沿 23 个模块检查 Backbone、Neck 和姿态分支的真实特征。</Title>
        <OperatorTabs labels={['卷积乘加','BN 与激活','逐层特征','检测与姿态']} index={op.index} onChange={op.choose}/>
        <div className="vision-inputs"><div><figure>{frontend?.input?.image_data_url&&<img src={frontend.input.image_data_url} alt="所选原始帧"/>}<figcaption>源帧 · {shape(frontend?.input?.original_shape)}</figcaption></figure></div><span>→<small>Letterbox / RGB / ÷255</small></span><div><figure>{frontend?.input?.tensor_image_data_url&&<img src={frontend.input.tensor_image_data_url} alt="YOLO 实际预处理输入"/>}<figcaption>模型输入 · {shape(frontend?.input?.tensor_shape)}</figcaption></figure></div><div className="pixel-kernel"><b>第一层卷积</b><code>3 × 3 × RGB</code><p>27 个输入 × 27 个权重<br/>→ 1 个特征值</p></div></div>
        <div className="operator-heading conv-heading"><h3>卷积核滑动到一个位置</h3><div className="mini-tabs">{steps.map((s,i)=><button key={i} aria-pressed={stepIndex===i} onClick={()=>{onPause();setStep(i);}}>位置 {i+1}</button>)}</div></div>
        {example&&<details className="kernel-peek calculation-detail"><summary>展开 RGB 输入窗口与对应的 3×3 核</summary><div className="kernel-matrices">{['R','G','B'].map((name,c)=><div key={name}><span>{name} 输入窗口</span><TensorMap values={example.patch?.[c]} label={`通道 ${c} · 输入像素/255`} rows={3} cols={3}/><b>×</b><TensorMap values={example.weights?.[c]} label="对应 3×3 核权重" rows={3} cols={3}/></div>)}</div></details>}
        <WeightedSum title="卷积：27 个乘积，汇聚成一个特征" subtitle={`输出通道 ${example?.output_channel??'—'}，位置 [${example?.position?.join(', ')||'—'}] · 步长 ${shape(conv?.stride)}`} inputs={convInputs} weights={convWeights} bias={example?.bias||0} output={example?.conv_output} progress={operationProgress} onPause={onPause}/>
        <div className="numeric-bridge"><div><span>BatchNorm</span><code>γ · (z − μ) / √(σ² + ε) + β</code><b>{fmt(example?.conv_output)} → {fmt(example?.bn?.output)}</b><small>γ {fmt(example?.bn?.gamma)} · β {fmt(example?.bn?.beta)} · μ {fmt(example?.bn?.running_mean)} · σ² {fmt(example?.bn?.running_var)}</small></div><div><span>SiLU 非线性激活</span><code>y = z · sigmoid(z)</code><b>{fmt(example?.bn?.output)} × {fmt(example?1/(1+Math.exp(-example.bn.output)):undefined)} = {fmt(example?.actual_output)}</b><small>写入输出特征图的一个单元</small></div></div>
        <div className="operator-heading features-heading"><h3>逐层检查网络 · 点击层号选择</h3><span>实际尺寸保留 / 图像采样显示</span></div>
        <div className="yolo-layer-rail">{(layers.length?layers:Array.from({length:23},(_,index)=>({index,type:'模块'}))).map((v,i)=><button key={v.index} aria-pressed={selected===i} onClick={()=>{onPause();setSelected(i);}}><b>{v.index}</b><span>{v.type}</span></button>)}</div>
        <div className="feature-gallery-heading"><b>{layer?.type||'等待网络特征'} <span>来自 {JSON.stringify(layer?.from??'—')}</span></b><code>{shape(layer?.shape)}</code></div>
        <div className="feature-gallery">{layer?.values?.map((v,i)=><TensorMap key={i} values={v} label={`通道 ${layer?.channels?.[i]??i} · 原位采样`}/>)}</div>
        {!layer&&<Missing/>}
        <div className="pose-head-inspection"><h3>三个尺度的检测 / 关键点分支 · 实际输出</h3><div className="head-gallery">{frontend?.heads?.map((head,i)=><section key={i}><h3>{head.name} · stride {head.stride}</h3><code>{shape(head.shape)}</code><TensorMap values={head.values?.[0]} label="输出通道采样"/></section>)}</div><p className="network-note">{frontend?.postprocess?.end2end?'端到端检测输出':'使用与视频分析一致的 NMS 分支'} · 置信度阈值 {frontend?.postprocess?.threshold??'—'} · 当前重算检测 {frontend?.detections?.length??0} 人</p></div>
        <p className="network-note">特征来自选定源帧的单帧重算。姿态检测结果不包含跟踪 ID；下一阶段使用视频运行时记录的 ByteTrack 身份。</p>
    </div>;
}

export function TrackingView({data,track,sample,progress}) {
    const [feature,setFeature]=useState(0);
    const history=track?.history||[];
    const matrix=useMemo(()=>history.map(p=>p.flat()),[track?.history]);
    const t=Math.min(Math.max(0,history.length-1),Math.floor(progress*history.length));
    const mean=data?.normalization?.lstm?.mean?.flat()?.[feature];
    const std=data?.normalization?.lstm?.std?.flat()?.[feature];
    const input=matrix[t]?.[feature];
    return <div><Title number="02" title="同一个 ID，把离散姿态变成一段运动">ByteTrack 在视频帧之间关联检测。以下展示已保存的真实 ID 与历史观测；匹配代价矩阵没有在原分析中记录。</Title>
        <div className="identity-flow"><div><Skeleton pose={track?.keypoints}/><span>当前记录 · 帧 {sample?sample.frame+1:'—'}</span></div><strong>ID {track?.track_id??'—'}<small>身份持续关联</small></strong><div className="pose-film">{[0,7,14,21,28,35,42,49].map((n,i)=><div key={n} className={n<=t?'arrived':''} style={{transform:`translateX(${(1-clamp(progress*8-i))*14}px)`}}><Skeleton pose={history[n]}/><small>t {n+1}</small></div>)}</div></div>
        <div className="tensor-fold"><div><h3>50 × 17 × 3</h3><TensorMap values={matrix.length?matrix:undefined} label="按时间堆叠；每个关节包含 x、y、分数" rows={50} cols={51}/></div><div className="tensor-fold-arrow">→<small>逐关节标准化<br/>展平关节与通道</small></div><div><h3>50 × 51</h3><TensorMap values={data?.lstm?.rounds?.[0]?.input?.values} label="送入 LSTM 的实际标准化张量" rows={50} cols={51}/></div></div>
        <div className="normalization-lens"><label>检查一个特征 <select aria-label="选择标准化特征" value={feature} onChange={e=>setFeature(Number(e.target.value))}>{Array.from({length:51},(_,i)=><option key={i} value={i}>关节 {Math.floor(i/3)} · {['x','y','confidence'][i%3]}</option>)}</select></label><div className="normalization-equation"><span>t = {t+1}</span><code>(<b>{fmt(input)}</b> − {fmt(mean)}) / {fmt(std)} = <b>{fmt(data?.lstm?.rounds?.[0]?.input?.values?.[t]?.[feature])}</b></code></div><p>{data?.normalization?.mode==='training_stats'?'μ 与 σ 来自训练集保存的统计量。':'μ 与 σ 由该关节该通道的 50 次观测计算，使用总体标准差并加 ε。'}历史未满时不运行行为模型。</p></div>
    </div>;
}

export function ForecastView({data,progress,onPause}) {
    const [r,setChoice]=useState(0);
    const op=useOperator(progress,[.6,.8,1],onPause);
    const round=data?.lstm?.rounds?.[r];
    const local=op.local;
    const example=round?.decoder_example;
    const t=Math.min(9,Math.floor(local*10));
    const poses=round?.forecast_poses||[];
    const bounds=useMemo(()=>poses.length?poseBounds(poses):undefined,[poses]);
    const bn=data?.stgcn?.data_bn_example;
    return <div className="forecast-view" data-operator={op.index}><Title number="04" title="从隐藏状态解码，再把未来接回输入窗口">每轮预测 10 帧，保留窗口后 40 帧并接入新预测；重复 5 次。下面每次回填都对应模型真实输出。</Title>
        <OperatorTabs labels={['解码器乘加','滚动窗口','归一化桥接']} index={op.index} onChange={op.choose}/>
        <div className="round-tabs">{Array.from({length:5},(_,i)=><button key={i} className={r===i?'active':''} onClick={()=>{onPause();setChoice(i);}}>第 {i+1} 轮<span>未来 {i*10+1}–{(i+1)*10}</span></button>)}</div>
        <div className="rolling-window"><div className="window-cells">{Array.from({length:60},(_,i)=><i key={i} className={i<10?'dropped':i>=50?'predicted':''} style={{transform:`translateX(${-local*1000}%)`,opacity:i<10?1-local:1}}><span>{i>=50?`P${i-49}`:i+1}</span></i>)}</div><div><span>移除最早 10 帧</span><b>保留 40 帧 + 预测 10 帧 → 新的 50 帧输入</b></div></div>
        <div className="forecast-pose-grid"><div className="forecast-ghosts">{poses.filter((_,i)=>i%3===0).map((p,i)=><div key={i} style={{opacity:.15+i*.15}}><Skeleton pose={p} bounds={bounds}/></div>)}<Skeleton pose={poses[t]} bounds={bounds}/><span>第 {r*10+t+1} 个预测姿态 · 还原坐标后</span></div><TensorMap values={round?.forecast?.values} label="解码器输出：10 × 51 · 标准化域" rows={10} cols={51}/></div>
        <WeightedSum title="解码器：第 10 层最终隐藏状态 → 第一个预测坐标" subtitle="256 维完整内积；同一 Linear 产生 510 个输出，重排为 10 × 17 × 3。" inputs={example?.input_values} weights={example?.weights} bias={example?.bias||0} output={example?.actual} progress={local} onPause={onPause}/>
        <div className="numeric-bridge"><div><span>五轮结束：反标准化</span><code>P = Ŷ · σ + μ</code><b>标准化域 → 像素坐标与分数</b></div><div><span>{data?.normalization?.mode==='training_stats'?'训练统计标准化':'与固定参考序列 R 配对'}</span><code>{data?.normalization?.mode==='training_stats'?'X = (P − μtrain) / σtrain':'X = (P − mean(R,P)) / std(R,P)'}</code><small>{data?.normalization?.mode==='training_stats'?'使用训练集保存的标准化参数。':'参考来自视频中首个满足 50 次有效观测的行人序列，使用样本标准差。'}</small></div></div>
        <div className="tensor-fold"><TensorMap values={data?.stgcn?.input?.values} label="ST-GCN 标准化输入 · 50×51"/><div className="tensor-fold-arrow">BN →</div><TensorMap values={data?.stgcn?.data_bn?.values} label="Data BatchNorm 输出 · 50×51"/></div>
        <div className="normalization-lens"><h3>Data BatchNorm：关节 0 的 x 通道，t = {bn?.timestep??'—'}</h3><code>({fmt(bn?.input_value)} − {fmt(bn?.running_mean)}) / √({fmt(bn?.running_variance)} + ε) × {fmt(bn?.weight)} + {fmt(bn?.bias)} = {fmt(bn?.actual)}</code><p>使用训练保存的运行均值、方差与仿射参数。重排为 [1, 3, 50, 17] 后进入图网络。</p></div>
    </div>;
}

function BlockSelect({index,onChange}) {return <div className="gcn-layer-strip">{[64,64,64,64,128,128,128,256,256,256].map((c,i)=><button key={i} className={index===i?'active':''} onClick={()=>onChange(i)}><b>块 {i+1}</b><span>{c} ch</span><small>T {i<4?50:i<7?25:13}</small></button>)}</div>;}
function useBlock(data,progress,onPause) {
    const [index,setIndex]=useState(0);
    return {index,block:data?.stgcn?.blocks?.[index],choose:i=>{onPause();setIndex(i);},local:progress};
}

export function SpatialView({data,progress,onPause}) {
    const {index,block,choose}=useBlock(data,progress,onPause);
    const op=useOperator(progress,[.25,.4,1],onPause,2);
    const local=op.local;
    const spatial=block?.arithmetic?.spatial;
    const target=spatial?.output_joint??0;
    const projection=spatial?.projection;
    const allProducts=flat(spatial?.products)||[];
    const graphInputs=useMemo(()=>flat(spatial?.projected_values),[spatial]);
    const graphWeights=useMemo(()=>flat(spatial?.adjacency),[spatial]);
    return <div className="spatial-view" data-operator={op.index}><Title number="05" title="不是相邻像素，而是相连的身体关节">先用 1×1 卷积映射通道，再按三个空间分区，用邻接矩阵 A 和可学习边权 E 聚合。展开一个关节的完整加权和。</Title>
        <BlockSelect index={index} onChange={choose}/><OperatorTabs labels={['1×1 通道投影','邻接结构','空间聚合']} index={op.index} onChange={op.choose}/>
        <div className="graph-aggregation"><svg viewBox="0 0 330 280" role="img" aria-label="真实 ST-GCN 邻接关系，信号汇聚到选定关节"><g transform="translate(35,0)">{(data?.stgcn?.edges||[]).filter(([a,b])=>a!==b).map(([a,b],i)=><line key={i} x1={jointPositions[a][0]} y1={jointPositions[a][1]} x2={jointPositions[b][0]} y2={jointPositions[b][1]} stroke="#39555f"/>)}{jointPositions.map(([x,y],j)=>{const linked=spatial?.adjacency?.some(a=>a[j]!==0);const end=jointPositions[target];return <g key={j}>{linked&&j!==target&&<><line x1={x} y1={y} x2={end[0]} y2={end[1]} stroke="#7be7bc" opacity=".55"/><circle cx={x+(end[0]-x)*local} cy={y+(end[1]-y)*local} r="4" fill="#bdffe2"/></>}<circle cx={x} cy={y} r={j===target?12:6} fill={j===target?'#b8f0d3':linked?'#54b894':'#24434d'}/><text x={x+9} y={y+4} fill="#9cbbb9" fontSize="9">{j}</text></g>;})}</g></svg><div><span className="network-label">当前汇聚目标</span><h3>关节 {target} · 通道 {spatial?.output_channel??0} · t {spatial?.timestep??'—'}</h3><div className="adjacency-maps">{['自连接','向心分区','离心分区'].map((label,i)=><TensorMap key={label} values={block?.effective_adjacency?.[i]} label={`A${i} ⊙ E${i} · ${label}`} rows={17} cols={17}/>)}</div><p className="network-note">关节布局是结构示意，边连接和系数来自实际模型；高亮连线对应当前输出的非零邻接项。</p></div></div>
        <WeightedSum title="图聚合：跨分区、跨关节的 51 项求和" subtitle={`Z[k,v] × (A[k,v,${target}] · E[k,v,${target}])，全部 3 × 17 项参与；非零项 ${allProducts.filter(v=>v!==0).length}。`} inputs={graphInputs} weights={graphWeights} output={spatial?.actual} progress={local} onPause={onPause} inputLabel="投影特征 Z" weightLabel="有效邻接 A⊙E"/>
        <div className="projection-inspection"><WeightedSum title="空间聚合前的 1×1 Conv" subtitle={`分区 ${projection?.partition??'—'} · 源关节 ${projection?.source_joint??'—'} · 所有输入通道`} inputs={projection?.input_values} weights={projection?.weights} bias={projection?.bias||0} output={projection?.actual} progress={local} onPause={onPause}/></div>
        <div className="tensor-fold"><TensorMap values={block?.spatial?.values} label="空间聚合后的真实特征采样"/><div className="tensor-fold-arrow">→<small>送入时间卷积</small></div><div className="block-shape"><b>{shape(block?.output_shape)}</b><span>当前块完整输出尺寸</span></div></div>
    </div>;
}

export function TemporalView({data,progress,onPause}) {
    const {index,block,choose}=useBlock(data,progress,onPause);
    const op=useOperator(progress,[.72,.86,1],onPause);
    const local=op.local;
    const temporal=block?.arithmetic?.temporal;
    const residual=block?.arithmetic?.residual;
    const temporalInputs=useMemo(()=>flat(temporal?.input_values),[temporal]);
    const temporalWeights=useMemo(()=>flat(temporal?.weights),[temporal]);
    const cursor=Math.min(8,Math.floor(local*9));
    return <div className="temporal-view" data-operator={op.index}><Title number="06" title="沿时间取一个窗口，识别运动怎样发生">BN → ReLU 后，9×1 卷积在同一关节的 9 个时间位置上跨通道乘加。最后与残差分支相加，再经过 ReLU。</Title>
        <BlockSelect index={index} onChange={choose}/><OperatorTabs labels={['时间卷积','残差相加','ReLU 输出']} index={op.index} onChange={op.choose}/>
        <div className="temporal-kernel"><div className="temporal-window">{Array.from({length:9},(_,i)=><div key={i} className={i===cursor?'selected':''}><span>t = {temporal?.input_timesteps?.[i]??'—'}</span><div>{Array.from({length:12},(_,c)=>{const v=temporal?.input_values?.[c]?.[i];return <i key={c} style={{background:v==null?'#24414b':v<0?'#6495dc':'#5bc29c',opacity:v==null?.3:.3+.7*Math.min(1,Math.abs(v))}}/>;})}</div><small>k {i-4}</small></div>)}</div><p>窗口展示前 12 通道；下方完整计算包含全部 {temporal?.input_channels??'—'} 通道 × 9 时间点。步长 {temporal?.stride??'—'}，填充 {temporal?.padding??'—'}。</p></div>
        <WeightedSum title="时间卷积：逐通道、逐时间点累加" subtitle={`输出通道 ${temporal?.output_channel??0} · 关节 ${temporal?.output_joint??'—'} · t ${temporal?.timestep??'—'}`} inputs={temporalInputs} weights={temporalWeights} bias={temporal?.bias||0} output={temporal?.actual} progress={local} onPause={onPause} inputLabel="BN + ReLU 后的窗口" weightLabel="时间卷积核"/>
        <div className="residual-computation"><div><span>卷积后 BN / Dropout（推理关闭）</span><b>{fmt(residual?.temporal)}</b></div><strong>＋</strong><div><span>残差 · {residual?.mode||'—'}</span><b>{fmt(residual?.residual)}</b></div><strong>＝</strong><div><span>激活前</span><b>{fmt(residual?.sum)}</b></div><strong>→</strong><div className="residual-result"><span>ReLU 输出</span><b>{fmt(residual?.actual)}</b></div></div>
        <ValueTransform title="ReLU：保留正响应，负值归零" value={residual?.sum} output={residual?.actual} kind="relu" progress={local}/>
        <div className="tensor-fold"><TensorMap values={block?.temporal?.values} label="时间分支的输出采样"/><div className="tensor-fold-arrow">+ 残差 → ReLU</div><TensorMap values={block?.output?.values} label={`块 ${index+1} 输出采样 · ${shape(block?.output_shape)}`}/></div>
    </div>;
}

export function ClassifierView({data,progress,onPause}) {
    const [selected,setSelected]=useState(2);
    const op=useOperator(progress,[.25,.75,1],onPause,1);
    const net=data?.stgcn;
    const pool=net?.pool_arithmetic;
    const poolInputs=useMemo(()=>flat(pool?.values),[pool]);
    const poolWeights=useMemo(()=>poolInputs?.map(()=>1/poolInputs.length),[poolInputs]);
    const max=net?.logits?Math.max(...net.logits):0;
    const exponentials=net?.logits?.map(v=>Math.exp(v-max));
    const denominator=exponentials?.reduce((a,b)=>a+b,0);
    return <div className="classifier-view" data-operator={op.index}><Title number="07" title="把整个运动片段，归结为三个类别分数">平均池化压缩时间和关节维度，分类头执行三组 256 维内积，Softmax 将 Logits 转成总和为 1 的分数。</Title><OperatorTabs labels={['全局平均池化','分类头乘加','Softmax 决策']} index={op.index} onChange={op.choose}/>
        <div className="pool-inspection"><WeightedSum title="Global Average Pooling · 通道 0" subtitle={`所有 ${pool?.count??221} 个位置均参与：sum / count`} inputs={poolInputs} weights={poolWeights} output={pool?.actual} progress={op.local} onPause={onPause}/></div>
        <div className="classifier-pool"><TensorMap values={net?.pool?Array.from({length:16},(_,i)=>net.pool.slice(i*16,i*16+16)):undefined} label="实际 256 维池化向量 · 按 16×16 排列" rows={16} cols={16}/><div><h3>分类头 · 选择一个输出类别</h3><div className="class-selector">{titles.map((label,i)=><button key={label} aria-pressed={selected===i} onClick={()=>{onPause();setSelected(i);}}>{label}<b>{fmt(net?.logits?.[i])}</b></button>)}</div><p>每类拥有自己的 256 个权重和 1 个偏置。</p></div></div>
        <WeightedSum title={`${titles[selected]}：256 个特征与权重逐项相乘`} subtitle={`logit[${selected}] = Σ W[${selected},j] · z[j] + b[${selected}]`} inputs={net?.pool} weights={net?.head_weights?.[selected]} bias={net?.head_bias?.[selected]||0} output={net?.logits?.[selected]} progress={op.local} onPause={onPause}/>
        <div className="softmax-theater"><div className="softmax-header"><h3>Softmax · 指数 → 归一化 → Argmax</h3><code>pᵢ = exp(sᵢ − max(s)) / Σ exp(sⱼ − max(s))</code></div>{titles.map((label,i)=><div className={`softmax-row ${net?.class_id===i?'winner':''}`} key={label}><b>{label}</b><span>{fmt(net?.logits?.[i])}<small>Logit</small></span><i>→</i><span>{fmt(exponentials?.[i])}<small>exp(s − max)</small></span><i>÷</i><span>{fmt(denominator)}<small>指数总和</small></span><i>＝</i><div className="probability-bar"><em style={{transform:`scaleX(${(net?.probabilities?.[i]||0)*phase(op.local,.5,.95)})`}}/><strong>{fmt(net?.probabilities?.[i])}</strong></div></div>)}</div>
        <div className="decision-output"><span>最终预测</span><b>{net?titles[net.class_id]:'等待完整观测'}</b><p>分数解释当前模型的输出，不等同于真实道路场景的准确率。</p></div>
    </div>;
}
