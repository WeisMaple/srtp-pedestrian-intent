import pathlib,json,sys,hashlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from independent_20261028.experiment import metrics
OUT=pathlib.Path(r'D:\SRTP_independent_20261028')
result={'split':{k:len(v) for k,v in json.loads((OUT/'split.json').read_text()).items()},'models':{},'paired':{}}
predictions={v:json.loads((OUT/v/'test_predictions.json').read_text()) for v in ['legacy','modern']}
for v in predictions:
    result['models'][v]=json.loads((OUT/v/'test_metrics.json').read_text())
    result['models'][v]['dataset']=json.loads((OUT/v/'dataset.json').read_text())
    result['models'][v]['selected_epochs']={k:json.loads((OUT/v/k/'best.json').read_text())['epoch'] for k in ['stgcn','lstm']}
for norm in ['legacy_pair','training_stats']:
    maps={v:{tuple(r['key']):r for r in rows[norm]} for v,rows in predictions.items()};keys=sorted(set(maps['legacy'])&set(maps['modern']))
    assert keys
    assert all(maps['legacy'][k]['truth']==maps['modern'][k]['truth'] for k in keys)
    y=[maps['legacy'][k]['truth'] for k in keys];rr={v:metrics(y,[maps[v][k]['prediction'] for k in keys]) for v in maps}
    rr['always_crossing']=metrics(y,[2]*len(y));rr['videos']=len(set(k[0] for k in keys))
    # Paired cluster bootstrap respects within-video correlation.
    clusters={}
    for k in keys:
        correct=[int(maps[v][k]['prediction']==maps[v][k]['truth']) for v in ['legacy','modern']]
        a=clusters.setdefault(k[0],[0,0]);a[0]+=correct[1]-correct[0];a[1]+=1
    a=np.array(list(clusters.values()));rng=np.random.default_rng(2026);deltas=[]
    for _ in range(2000):
        sample=a[rng.integers(0,len(a),len(a))].sum(0);deltas.append(sample[0]/sample[1])
    rr['accuracy_delta_modern_minus_legacy']=rr['modern']['accuracy']-rr['legacy']['accuracy'];rr['video_bootstrap_95ci']=np.percentile(deltas,[2.5,97.5]).tolist()
    result['paired'][norm]=rr
    (OUT/(norm+'_paired_keys.json')).write_text(json.dumps(keys))
result['speed']={}
for v in ['legacy','modern']:
    records=json.loads((OUT/'speed'/(v+'_summary.json')).read_text());seconds=sum(r['seconds_call'] for r in records);frames=sum(r['decoded_frames'] for r in records)
    count=sum(r['annotations'] if v=='legacy' else r['details']['classification_updates'] for r in records)
    result['speed'][v]=dict(frames=frames,seconds=seconds,fps_including_setup=frames/seconds,behavior_predictions=count,records=records)
result['speed']['engineering_speed_ratio']=result['speed']['modern']['fps_including_setup']/result['speed']['legacy']['fps_including_setup']
result['speed']['equal_behavior_work']=json.loads((OUT/'speed/behavior_equal_work.json').read_text())
result['extraction_batch_check']=json.loads((OUT/'extraction_batch_check.json').read_text())
manifest={}
for path in list((ROOT/'independent_20261028').glob('*.py'))+[ROOT/'st_gcn.py']+list(OUT.glob('*/normalization.pt'))+list(OUT.glob('*/*/best.pt')):
    manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'comparison.json').write_text(json.dumps(result,indent=2))
lines=['# YOLO26 升级版：独立视频训练与对比','',f"官方视频划分：{result['split']}。网络结构未改，训练与选模均不使用测试集。",'',
'## 独立测试结果','', '| 方法 | 有输出样本 | 预测覆盖率 | 仅 ST-GCN 准确率 | LSTM→ST-GCN 准确率 | 联合 Macro-F1 |','|---|---:|---:|---:|---:|---:|']
for v,name in [('legacy','旧前端重训'),('modern','YOLO26 前端重训')]:
    r=result['models'][v];p=r['legacy_pair'];lines.append(f"| {name} | {p['n']} | {r['coverage']:.2%} | {r['stgcn_observed']['accuracy']:.2%} | {p['accuracy']:.2%} | {p['macro_f1']:.4f} |")
p=result['paired']['legacy_pair'];lines+=['',f"双方共同输出的 {p['legacy']['n']} 个测试点、{p['videos']} 个视频：旧版 {p['legacy']['accuracy']:.2%}，新版 {p['modern']['accuracy']:.2%}；差值 {p['accuracy_delta_modern_minus_legacy']*100:.2f} 个百分点。按视频配对 bootstrap 95% 区间：{[round(x*100,2) for x in p['video_bootstrap_95ci']]} 个百分点。始终猜 crossing 的准确率为 {p['always_crossing']['accuracy']:.2%}。",'',
'准确率仅在有可匹配输出的样本上计算；覆盖率分母为全部具备历史标注的候选点。旧缓存继承旧前处理筛选范围，因此共同样本对比和覆盖率必须一起阅读。类别 -1 为 JAAD irrelevant；采用行人属性标签，不是人工逐帧意图真值。', '',
'## 速度','', '| 实现 | 视频帧数 | 调用总耗时（含初始化） | FPS | 行为预测次数 |','|---|---:|---:|---:|---:|']
for v in ['legacy','modern']:
    r=result['speed'][v];lines.append(f"| {v} | {r['frames']} | {r['seconds']:.2f} 秒 | {r['fps_including_setup']:.2f} | {r['behavior_predictions']} |")
lines+=['',f"本次三段同源视频的工程速度比为 {result['speed']['engineering_speed_ratio']:.2f} 倍。同一输入、MP4V 编码、逐帧预测；但原实现 CPU 姿态/行为计算与新版 CUDA 不同，且实际预测次数不同，不能将全部收益归因于 YOLO。具体覆盖和次数见 comparison.json。",'',
'## 方法与诊断','', '主联合结果保留 legacy_pair 推理归一化。仅 ST-GCN 使用训练集统计，二者不能直接混作同一任务。另测 training_stats 对齐作为诊断，均在测试前确定；不根据测试分数替换默认方法。',f"training_stats 共同样本：旧版 {result['paired']['training_stats']['legacy']['accuracy']:.2%}，新版 {result['paired']['training_stats']['modern']['accuracy']:.2%}。",'',
'完整配置、缓存来源限制及执行命令见项目 independent_20261028/README.md。所有权重另存，本次没有替换原权重。只有一个随机种子，短视频测速不代表所有场景。']
b=result['speed']['equal_behavior_work']
lines+=['','## 相同工作量与批量提取检查','',f"同样 20 个历史骨架、同一权重、batch=1，预热 3 次：CPU {b['cpu']['seconds']:.3f} 秒，CUDA {b['cuda:0']['seconds']:.3f} 秒；分类结果一致率 {b['class_agreement']:.0%}。此条件下 GPU 更慢，说明小批量深层 LSTM 的设备调用开销不能忽略。这个检查隔离行为推理，不包含检测、跟踪或视频编码。",'', '三段视频中 video_0016：原实现输出 73 次行为预测，新实现为 0 次。新实现的连续历史策略与轨迹中断导致该片段无可用行为输出；22.05 FPS 不能解释为每帧都完成了同等数量的行为预测。','',f"批量提取与逐帧抽查：{result['extraction_batch_check']}。",'', '结论：新版扩大了这次独立数据评估的输出覆盖范围；共同样本精度小幅上升但置信区间跨零，且不及多数类基线，不能宣称意图区分能力显著改善。视频处理吞吐提高，但包含预测次数减少和实现设备差异。']
(OUT/'对比报告.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='speed'},indent=2))
