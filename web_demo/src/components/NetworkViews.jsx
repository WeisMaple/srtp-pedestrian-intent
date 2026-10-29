import { useMemo, useState } from 'react';
import { WeightedSum, ValueTransform, GateMemory } from './ArithmeticTheater';
import OperatorTabs, { useOperator } from './OperatorTabs';
import { jobUrl } from '../api';
import Skeleton from './Skeleton';
import { Box, Wire, Operator, TensorMap, ValueStrip, fmt, shape } from './NetworkPrimitives';
const clamp = value => Math.max(0, Math.min(1, value));

export function OverviewView({ progress, onChapter, data }) {
    const active = index => progress * 9 >= index;
    return <div className="network-overview">
        <div className="network-section-title"><span>END-TO-END INFERENCE</span><h2>一段视频，如何变成三个类别分数</h2><p>每条连线都对应数据的传递；点击网络模块，展开内部计算。</p></div>
        <svg className="network-svg overview-svg" viewBox="0 0 1140 410" role="img" aria-label="完整算法数据流，包含姿态检测、跟踪、归一化、LSTM、ST-GCN 和分类输出">
            <Wire d="M225 88 H300" active={active(1)} progress={clamp(progress * 9 - 1)} /><Wire d="M525 88 H600" active={active(2)} progress={clamp(progress * 9 - 2)} /><Wire d="M825 88 H900" active={active(3)} progress={clamp(progress * 9 - 3)} />
            <Box x={20} y={45} width={205} title="视频逐帧解码" subtitle="RGB 图像 → 检测输入" detail="保留时间戳与画面尺寸" active={active(0)} onClick={() => onChapter(1)} />
            <Box x={300} y={45} width={225} title="YOLO26s-Pose" subtitle="Backbone → Neck → Pose26" detail="行人框 + 17 个关键点" active={active(1)} onClick={() => onChapter(1)} />
            <Box x={600} y={45} width={225} title="ByteTrack 身份关联" subtitle="相邻帧检测 → 稳定的行人 ID" detail="每个 ID 独立维护历史" active={active(2)} onClick={() => onChapter(1)} />
            <Box x={900} y={45} width={220} title="有效观测缓存" subtitle="50 × 17 × 3" detail="时间 × 关节 × (x, y, 分数)" active={active(3)} onClick={() => onChapter(1)} />
            <Wire d="M1010 123 V168 H120 V240" active={active(4)} progress={clamp(progress * 9 - 4)} label="按每个关节、每个通道沿时间维归一化，再展平为 50 × 51" x={335} y={193} />
            <Box x={20} y={240} width={200} title="LSTM 时序预测" subtitle="10 层 · hidden size 256" detail="最后时刻 h → Linear 256→510" active={active(4)} onClick={() => onChapter(2)} />
            <Wire d="M220 280 H300" active={active(5)} progress={clamp(progress * 9 - 5)} />
            <Box x={300} y={240} width={200} title="五轮自回归" subtitle="10 帧 × 5 → 50 帧未来" detail="移除最早 10 帧，接入预测 10 帧" active={active(5)} onClick={() => onChapter(2)} />
            <Wire d="M500 280 H580" active={active(6)} progress={clamp(progress * 9 - 6)} />
            <Box x={580} y={240} width={250} title="ST-GCN 时空特征" subtitle="10 个块 · 3→64→128→256 通道" detail="配对归一化 → 空间聚合 + 时间卷积" active={active(6)} onClick={() => onChapter(3)} />
            <Wire d="M830 280 H900" active={active(7)} progress={clamp(progress * 9 - 7)} />
            <Box x={900} y={240} width={220} title="三类意图输出" subtitle="全局平均池化 → 1×1 Conv" detail="Logits → Softmax → Argmax" active={active(7)} onClick={() => onChapter(4)} />
            <Wire d="M400 318 V367 H120 V318" active={active(5)} progress={clamp(progress * 9 - 5)} color="#a1b5ff" label="预测结果回填输入窗口，重复 5 次" x={140} y={390} />
        </svg>
        <div className="network-facts"><span><b>17</b> 人体关节</span><span><b>50 → 50</b> 观测到预测</span><span><b>10 + 10</b> LSTM 层 / ST-GCN 块</span><span><b>{data ? '已读取' : '结构可浏览'}</b> {data ? '真实网络计算' : '选择完成分析的记录后查看数值'}</span></div>
    </div>;
}

export function InputView({ job, track, sample, data, progress }) {
    const matrix = useMemo(() => track?.history?.map(pose => pose.flat()), [track?.history]);
    return <div>
        <div className="network-section-title"><span>01 / PIXELS TO TENSOR</span><h2>先找到人，再组织成网络能计算的张量</h2><p>YOLO26 检测身体结构；ByteTrack 关联身份；有效观测按时间堆叠。</p></div>
        <div className="input-network-grid"><div className="network-input-frame">{job && <img src={jobUrl(job.id, '/files/poster')} alt="输入视频首帧，用于展示输入形式" />}<span>视频首帧 · 输入形式</span></div><div className="yolo-architecture"><span className="network-label">YOLO26s-Pose · 已核对本地权重结构</span><div className="yolo-row"><b className={progress > .1 ? 'active' : ''}>Conv / C3k2<small>下采样 · 局部特征</small></b><i>→</i><b className={progress > .25 ? 'active' : ''}>SPPF / C2PSA<small>感受野 · 特征交互</small></b></div><div className="yolo-row"><b className={progress > .4 ? 'active' : ''}>Upsample + Concat<small>多尺度特征融合</small></b><i>→</i><b className={progress > .55 ? 'active' : ''}>Pose26 Head<small>框 + 17×3 关键点</small></b></div><p>此处展示网络结构，未记录 YOLO 内部特征图。</p></div><div className="network-pose"><Skeleton pose={track?.keypoints} color="#72e4b6" /><span>ID {track?.track_id ?? '—'} · 当前记录骨架</span></div></div>
        <div className="tensor-preparation"><div><h3>① 按 ID 收集有效观测</h3><TensorMap values={matrix} label="原始骨架：50 时间步 × 51 个特征" rows={50} cols={51} /></div><div className="tensor-equations"><h3>② 沿时间维标准化</h3><code>μ[j,c] = meanₜ X[t,j,c]</code><code>σ[j,c] = stdₜ X[t,j,c] + ε</code><code>X̂ = (X − μ) / σ</code><p>每个关节独立处理 x、y、关键点分数。LSTM 的 51 维输入保留这三个通道。</p></div><div><h3>③ 展平，送入 LSTM</h3><TensorMap values={data?.lstm?.rounds?.[0]?.input?.values} label="标准化输入：50 × 51" rows={50} cols={51} /></div></div>
        <p className="network-note">记录帧 {sample ? sample.frame + 1 : '—'} · 有效历史 {track?.history?.length || 0} / 50。图中色块有数值时来自真实张量，可悬停读取。</p>
    </div>;
}

function Gate({ x, title, formula, value, active, tone }) {
    return <g className={`lstm-gate ${active ? 'active' : ''}`} style={{ '--gate-color': tone }}><rect x={x} y="167" width="100" height="64" rx="9" /><text x={x+50} y="190" textAnchor="middle">{title}</text><text x={x+50} y="214" textAnchor="middle" className="gate-value">{formula} · {fmt(value)}</text></g>;
}

export function LstmView({ data, progress, onPause }) {
    const [t,setTime]=useState(24);
    const [layer,setLayer]=useState(0);
    const [r,setRound]=useState(0);
    const [gateIndex,setGateIndex]=useState(1);
    const op=useOperator(progress,[.5,.63,.84,1],onPause);
    const gate=data?.lstm?.gates?.steps?.[t];
    const currentRound=data?.lstm?.rounds?.[r];
    const currentLayer=currentRound?.layers?.[layer];
    const gateNames=['输入门 i','遗忘门 f','候选记忆 g','输出门 o'];
    const inputs=useMemo(()=>gate?.input_values&&gate?.previous_hidden?[...gate.input_values,...gate.previous_hidden]:undefined,[gate]);
    const weights=useMemo(()=>data?.lstm?.gates?.input_weights?[...data.lstm.gates.input_weights[gateIndex],...data.lstm.gates.recurrent_weights[gateIndex]]:undefined,[data,gateIndex]);
    const phase=op.local;
    return <div><div className="network-section-title"><span>03 / INSIDE THE LSTM</span><h2>一个门的 307 次乘加，决定如何更新记忆</h2><p>先计算四组加权和，经激活函数得到门值，再更新细胞与隐藏状态。选定样本：第 1 轮 / 第 1 层 / 单元 0。</p></div>
        <div className="gate-toolbar"><strong>时间维检查</strong><label>时间步 <input aria-label="LSTM 单元时间步" type="range" min="0" max="49" value={t} onChange={e=>{onPause();setTime(Number(e.target.value));}}/><b>t = {t+1}</b></label></div>
        <OperatorTabs labels={['门的乘加','非线性激活','记忆更新','层与序列']} index={op.index} onChange={op.choose}/>
        {op.index<2&&<div className="gate-choice">{gateNames.map((name,i)=><button key={name} aria-pressed={gateIndex===i} onClick={()=>{onPause();setGateIndex(i);}}>{name}<b>{fmt(gate?.[['i','f','g','o'][i]])}</b></button>)}</div>}
        {op.index===0&&<WeightedSum title={`${gateNames[gateIndex]}：输入与循环状态逐项乘加`} subtitle="前 51 项是当前输入 x，后 256 项是上一时刻隐藏状态 h。两组偏置已合并。" inputs={inputs} weights={weights} bias={gate?.bias?.[gateIndex]||0} output={gate?.preactivation?.[gateIndex]} progress={op.local} onPause={onPause} inputLabel="输入 x ⊕ 上一步 h" weightLabel="Wᵢₕ ⊕ Wₕₕ"/>}
        {op.index===1&&<><ValueTransform title={`${gateNames[gateIndex]}的非线性激活`} value={gate?.preactivation?.[gateIndex]} output={gate?.[['i','f','g','o'][gateIndex]]} kind={gateIndex===2?'tanh':'sigmoid'} progress={op.local}/><div className="numeric-bridge"><div><span>四门来自四套权重</span><code>[i,f,g,o] = [σ,σ,tanh,σ](Wᵢₕx + Wₕₕh + b)</code></div><div><span>输入 · 循环 · 偏置</span><code>{fmt(gate?.input_dot?.[gateIndex])} + {fmt(gate?.recurrent_dot?.[gateIndex])} + {fmt(gate?.bias?.[gateIndex])} = {fmt(gate?.preactivation?.[gateIndex])}</code></div></div></>}
        {op.index===2&&<>
        <svg className="network-svg lstm-cell-svg" viewBox="0 0 1100 340" role="img" aria-label="LSTM 单元内部：遗忘门、输入门、候选记忆、输出门、细胞状态和隐藏状态计算">
            <text x="20" y="34" className="tensor-label">上一时刻记忆 cₜ₋₁</text><text x="20" y="79" className="value-label">{fmt(gate?.c_previous)}</text>
            <Wire d="M20 53 H395" active progress={phase} /><Wire d="M425 53 H675" active={phase > .2} progress={clamp((phase-.2)*2)} /><Wire d="M705 53 H1060" active={phase > .45} progress={clamp((phase-.45)*2)} />
            <Operator x={410} y={53} text="×" active={phase>.2} /><Operator x={690} y={53} text="+" active={phase>.45} />
            <text x="960" y="34" className="tensor-label">新记忆 cₜ</text><text x="970" y="79" className="value-label">{fmt(gate?.c)}</text>
            <Wire d="M400 167 V86 H410 V68" active={phase>.1} progress={clamp(phase*2)} /><Wire d="M530 167 V113 H565" active={phase>.2} progress={clamp(phase*2)} /><Wire d="M650 167 V113 H595" active={phase>.2} progress={clamp(phase*2)} /><Operator x={580} y={113} text="×" active={phase>.3} /><Wire d="M580 98 V87 H690 V68" active={phase>.35} progress={clamp(phase*2)} />
            <Gate x={350} title="遗忘门 fₜ" formula="σ" value={gate?.f} active={phase>.1} tone="#d4b47b" /><Gate x={480} title="输入门 iₜ" formula="σ" value={gate?.i} active={phase>.2} tone="#7faafb" /><Gate x={600} title="候选记忆 gₜ" formula="tanh" value={gate?.g} active={phase>.3} tone="#b69af6" /><Gate x={760} title="输出门 oₜ" formula="σ" value={gate?.o} active={phase>.6} tone="#69d4ae" />
            <Wire d="M900 53 V100" active={phase>.5} /><g className="tanh-operator"><rect x="874" y="100" width="52" height="31" rx="6" /><text x="900" y="121" textAnchor="middle">tanh</text></g><Wire d="M900 131 V240" active={phase>.6} progress={clamp(phase*1.5)} /><Wire d="M860 199 H872 V255 H885" active={phase>.65} progress={clamp(phase*1.4)} /><Operator x={900} y={255} text="×" active={phase>.7} /><Wire d="M915 255 H1060" active={phase>.75} progress={clamp((phase-.75)*4)} />
            <text x="970" y="233" className="tensor-label">隐藏状态 hₜ</text><text x="970" y="282" className="value-label">{fmt(gate?.h)}</text>
            <Box x={20} y={163} width={200} height={71} title="xₜ 与 hₜ₋₁" subtitle="Wᵢₕ xₜ + Wₕₕ hₜ₋₁ + b" active />
            <Wire d="M220 199 H270 V304 H810" active progress={phase} color="#80aaf0" />{[400,530,650,810].map(x=><Wire key={x} d={`M${x} 304 V231`} active progress={phase} color="#80aaf0" />)}
            <text x="290" y="328" className="tensor-label">同一输入进入四组独立权重；σ 限制门值到 0–1，tanh 产生候选记忆。</text>
        </svg>
<GateMemory step={gate} progress={op.local}/></>}
        {op.index===3&&<>        <div className="lstm-details"><section><h3>10 层堆叠 · 每层 256 个隐藏单元</h3><div className="lstm-layer-stack">{Array.from({length:10},(_,i)=><button key={i} className={i===layer?'active':''} onClick={()=>{onPause();setLayer(i);}}>L{i+1}<small>{i===0?'51→256':'256→256'}</small></button>)}</div><ValueStrip values={currentLayer?.h} label={`第 ${layer+1} 层最终 h`} /><ValueStrip values={currentLayer?.c} label={`第 ${layer+1} 层最终 c`} /><p>各层处理完整 50 步序列。解码器只取第 10 层最后时刻的 256 维输出。</p></section><section><h3>最后状态 → 全连接 → 未来 10 帧</h3><div className="network-formula">h₅₀ [256] <i>→</i> Linear [510] <i>→</i> [10, 17, 3]</div><div className="round-tabs">{Array.from({length:5},(_,i)=><button className={i===r?'active':''} key={i} onClick={()=>{onPause();setRound(i);}}>第 {i+1} 轮<span>未来 {i*10+1}–{(i+1)*10}</span></button>)}</div><TensorMap values={currentRound?.forecast?.values} label={`第 ${r+1} 轮标准化域输出 · 10 × 51`} rows={10} cols={51} /><p>窗口更新：X ← concat(X[10:50], 预测 10 帧)。循环在标准化域内进行；每轮 h₀、c₀ 重置为零。五轮结束后再还原像素坐标。</p></section></div>
</>}
    </div>;
}

export function GcnView({ data, progress, playing, onPause }) {
    const [selected, setSelected] = useState(null);
    const [weighted, setWeighted] = useState(false);
    const manual = selected?.position === progress;
    const index = manual ? selected.value : Math.min(9, Math.floor(progress * 10));
    const block = data?.stgcn?.blocks?.[index];
    const channels=[64,64,64,64,128,128,128,256,256,256];
    const frames=[50,50,50,50,25,25,25,13,13,13];
    const phase=progress >= 1 || manual ? 1 : (progress * 10) % 1;
    const a=weighted?block?.effective_adjacency:data?.stgcn?.adjacency;
    const pixelForecast = useMemo(() => data?.future?.values?.map(pose => pose.flat()), [data?.future?.values]);
    return <div>
        <div className="network-section-title"><span>03 / SPATIAL–TEMPORAL GRAPH CONVOLUTION</span><h2>沿身体结构聚合，再沿时间轴提取运动特征</h2><p>预测序列先做配对标准化与 Data BatchNorm，再进入 10 个时空卷积块。点击一层，查看真实中间特征。</p></div>
        <details className="normalization-bridge"><summary>进入 ST-GCN 之前：反标准化 → 固定参考配对标准化 → Data BatchNorm <span>展开真实张量</span></summary><div className="normalization-steps"><section><h3>1. 从标准化域还原坐标</h3><code>P = Ŷ ⊙ σ + μ</code><TensorMap values={pixelForecast} label="未来 50×51 · 像素坐标与分数" rows={50} cols={51}/></section><section><h3>2. 与固定参考 R 配对</h3><code>X = (P − mean(R,P)) / std(R,P)</code><TensorMap values={data?.stgcn?.input?.values} label="配对标准化后 · 50×51" rows={50} cols={51}/><p>参考来自视频中首个满足 50 次有效观测的行人序列。使用样本标准差。</p></section><section><h3>3. Data BatchNorm</h3><code>51 通道 BN → [1, 3, 50, 17]</code><TensorMap values={data?.stgcn?.data_bn?.values} label="实际 BN 输出 · 50×51" rows={50} cols={51}/><p>推理使用训练时保存的均值、方差和仿射参数。</p></section></div></details>
        <div className="gcn-layer-strip">{channels.map((c,i)=><button key={i} className={i===index?'active':''} onClick={()=>{onPause();setSelected({position:progress,value:i});}}><b>块 {String(i+1).padStart(2,'0')}</b><span>{c} 通道</span><small>T = {frames[i]}{i===4||i===7?' · ↓2':''}</small></button>)}</div>
        <svg className="network-svg gcn-block-svg" viewBox="0 0 1120 230" role="img" aria-label="ST-GCN 块内部计算：1×1 卷积、邻接矩阵聚合、9×1 时间卷积、残差相加、ReLU">
            <Box x={10} y={90} width={155} title="输入特征 X" subtitle={`${index===0?3:channels[index-1]} × ${index===0?50:frames[index-1]} × 17`} active />
            <Wire d="M165 130 H205" active progress={phase} /><Box x={205} y={90} width={175} title="1×1 Conv" subtitle={`映射到 3 × ${channels[index]} 通道`} active={phase>.1} />
            <Wire d="M380 130 H420" active={phase>.2} progress={clamp(phase*2)} /><Box x={420} y={90} width={180} title="空间图聚合" subtitle="Σₖ Zₖ (Aₖ ⊙ Eₖ)" active={phase>.25} />
            <Wire d="M600 130 H640" active={phase>.4} progress={clamp(phase*2)} /><Box x={640} y={90} width={200} title="时间卷积 9×1" subtitle={`BN → ReLU → Conv · stride ${index===4||index===7?2:1}`} detail="→ BN → Dropout（推理关闭）" active={phase>.45} />
            <Wire d="M840 130 H895" active={phase>.6} progress={clamp(phase*2)} /><Operator x={910} y={130} text="+" active={phase>.7} /><Wire d="M925 130 H960" active={phase>.75} /><Box x={960} y={90} width={150} title="ReLU" subtitle={`${channels[index]} × ${frames[index]} × 17`} active={phase>.85} />
            <Wire d="M90 90 V36 H910 V115" active={phase>.1} progress={phase} color="#b1a1ec" label={index===0?'首块残差 = 0':index===4||index===7?'残差分支：1×1 Conv + BN，匹配通道与时间长度':'残差分支：Identity，直接传递输入'} x={290} y={25} />
            <text className="tensor-label" x="430" y="209">空间连接处理“哪个关节”；9 步时间窗口处理“怎样变化”。</text>
        </svg>
        <div className="gcn-inspection"><section className="adjacency-section"><div className="network-subheading"><h3>三个空间分区 · Z = Conv(X)</h3><button onClick={()=>setWeighted(!weighted)}>{weighted?'A ⊙ E：学习后的边权':'A：归一化邻接矩阵'}</button></div><div className="adjacency-maps">{['自连接','根节点 + 向心分区','离心分区'].map((label,i)=><TensorMap key={label} values={a?.[i]} label={label} rows={17} cols={17} />)}</div><p>17×17 矩阵，来自模型的实际图定义。E 是该块可学习的边重要性。</p></section><section><h3>块 {index+1} · 实际中间特征</h3><div className="activation-maps"><TensorMap values={block?.spatial?.values} label="空间聚合后" /><span>→</span><TensorMap values={block?.temporal?.values} label="时间卷积后" /><span>→</span><TensorMap values={block?.output?.values} label="残差 + ReLU 后" /></div><p>采样：第一个样本、前 8 通道、中间时间步、全部 17 关节。完整输出：{shape(block?.output_shape)}。</p><div className="activation-stat"><span>均值 <b>{fmt(block?.output?.mean)}</b></span><span>标准差 <b>{fmt(block?.output?.std)}</b></span><span>范围 <b>{fmt(block?.output?.min)} … {fmt(block?.output?.max)}</b></span></div></section></div>
    </div>;
}

export function OutputView({ data, progress }) {
    const net=data?.stgcn;
    const pool=useMemo(()=>net?.pool ? Array.from({length:16},(_,i)=>net.pool.slice(i*16,i*16+16)) : null,[net?.pool]);
    return <div><div className="network-section-title"><span>04 / FEATURES TO PROBABILITIES</span><h2>256 维时空特征，怎样变成最终类别</h2><p>全局平均池化压缩时间与关节维度；三个线性映射得到 Logits，Softmax 将它们归一化。</p></div>
        <div className="output-computation"><section><h3>① Global Average Pooling</h3><div className="network-formula">[256, 13, 17] → [256]</div><TensorMap values={pool} rows={16} cols={16} label="实际池化向量 · 按 16×16 排列" /><code>z[c] = meanₜ,ᵥ X[c,t,v]</code></section><section><h3>② 三组权重分别做内积</h3>{['其他','不过街','过街'].map((label,i)=><div className="logit-equation" key={label}><span>{label}</span><code>Σⱼ W[{i},j]·z[j] + b[{i}]</code><strong>{fmt(net?.logits?.[i])}</strong></div>)}<TensorMap values={net?.head_weights?.map(row=>row.slice(0,32))} rows={3} cols={32} label="分类头真实权重 · 3 类 × 前 32 维" /><p>分类头为 1×1 Conv，池化后等价于 256 → 3 的线性映射。</p></section><section><h3>③ Softmax → Argmax</h3><code>pᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)</code><div className="network-output-scores">{['其他','不过街','过街'].map((label,i)=><div key={label} className={net?.class_id===i?'winner':''}><span>{label}</span><b>{fmt(net?.probabilities?.[i])}</b><i><em style={{transform:`scaleX(${(net?.probabilities?.[i]||0)*clamp(progress*2)})`}} /></i></div>)}</div><div className="network-final-label">预测类别 <strong>{net ? ['其他','不过街','过街'][net.class_id] : '等待网络计算'}</strong></div><p>数值是模型分数，不是道路场景的已验证准确率。</p></section></div>
    </div>;
}
