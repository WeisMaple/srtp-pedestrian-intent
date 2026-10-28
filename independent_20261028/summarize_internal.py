import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['MPLBACKEND']='Agg'
import pathlib,sys,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from st_gcn import Model
OUT=pathlib.Path(r'D:\SRTP_modern_internal_20261028');RUN=OUT/'models/stgcn'
torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
split=json.loads((RUN/'split.json').read_text());data=torch.load(OUT/'datasets/stgcn.pt',weights_only=True)
ids=split['validation_indices'];x=data['x'][ids];y=data['y'][ids];model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).cuda().eval()
results={}
for tag,filename,recordfile in [('epoch100','epoch_100_state_dict.pt','epoch_100_validation.json'),('final125','final_state_dict.pt','final_validation.json'),('best','best_state_dict.pt','best_validation.json')]:
    model.load_state_dict(torch.load(RUN/filename,map_location='cuda:0',weights_only=True));pred=[]
    with torch.inference_mode():
        for xx in x.split(128):pred.extend(model(xx.cuda()).argmax(1).cpu().tolist())
    cm=torch.bincount(3*y+torch.tensor(pred),minlength=9).reshape(3,3);tp=cm.diagonal();f1=2*tp/(cm.sum(0)+cm.sum(1)).clamp_min(1)
    result=dict(accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),confusion_matrix=cm.tolist(),samples=len(y))
    recorded=json.loads((RUN/recordfile).read_text());assert abs(result['accuracy']-recorded['accuracy'])<1e-7
    if tag=='best':result['epoch']=recorded['epoch']
    results[tag]=result
counts=torch.bincount(y,minlength=3).tolist();results['validation_classes']=counts;results['majority_baseline']=max(counts)/len(y)
results['overlap_fraction']=split['validation_history_overlap_fraction'];results['dataset']=json.loads((OUT/'dataset_summary.json').read_text())
results['normalization']='Before split over all windows, intentionally the original protocol'
(OUT/'verified_results.json').write_text(json.dumps(results,indent=2))
# Remove inherited legacy-frontend fields from the descriptive configuration.
config=json.loads((OUT/'reproduction_config.json').read_text());config['data']={k:config['data'][k] for k in ['videos','detector','detector_imgsz','detector_conf','tracker','pose','pose_imgsz','pose_batch','match_policy','match_threshold','keep','window','stride','cache']}
(OUT/'reproduction_config.json').write_text(json.dumps(config,indent=2));runconfig=json.loads((RUN/'run_configuration.json').read_text());runconfig['config']=config;(RUN/'run_configuration.json').write_text(json.dumps(runconfig,indent=2))
import matplotlib.pyplot as plt,csv
with (RUN/'history.csv').open() as f:history=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(8,4.5));epochs=[int(r['epoch']) for r in history]
ax.plot(epochs,[float(r['train_accuracy'])*100 for r in history],label='Training');ax.plot(epochs,[float(r['validation_accuracy'])*100 for r in history],label='Window validation')
ax.axhline(97.5,ls='--',color='gray',label='Paper reported ~97.5%');ax.set(xlabel='Epoch',ylabel='Accuracy (%)',title='YOLO26 + original ST-GCN: internal window validation');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'accuracy_curve.png',dpi=160);plt.close(fig)
final=results['final125'];best=results['best']
text=f'''# 新版 YOLO 骨架：论文式 ST-GCN 内部验证

最终第125轮：**{final['accuracy']:.2%}**；最佳第{best['epoch']}轮：**{best['accuracy']:.2%}**。第100轮：{results['epoch100']['accuracy']:.2%}。

| 指标 | 结果 |
|---|---:|
| 最终 Macro-F1 | {final['macro_f1']:.4f} |
| 验证样本 | {len(y)} |
| 验证类别0/1/2数量 | {counts} |
| 多数类准确率基线 | {results['majority_baseline']:.2%} |
| 验证窗口与训练历史重叠比例 | {results['overlap_fraction']:.2%} |

论文报告约97.5%；旧前端的上轮复现最终96.38%、最佳96.69%。本轮复用同一训练脚本，仅替换数据产物路径，ST-GCN 结构、优化器、125轮、batch128、seed2026与旧轮一致。

从346个视频的新版 YOLO26/ByteTrack 缓存重新按原代码的前11框匹配规则筛选；共{results['dataset']['people']}条匹配轨迹、{results['dataset']['observations']}次姿态、{results['dataset']['windows']}个窗口。随机80%/20%窗口划分，归一化先于划分，保留原代码的泄漏口径。两种前端的有效样本数和类别比例不同，此对照不是逐样本的升级效果检验。

这是一组重新训练的内部验证模型，不是把之前独立视频训练模型改个测试集后直接打分。本次使用全部视频构建窗口，因此这些新权重不再满足此前独立测试集的未见条件，不能拿回该测试集宣称泛化成绩。

该分数仅对应已提取骨架的ST-GCN分类，不包含LSTM预测误差，也不代表原始视频到意图输出的端到端准确率。独立视频实验的模型、结果和默认运行权重均未替换。

最终混淆矩阵（真实类别为行、预测为列）：
```
{np.array(final['confusion_matrix'])}
```

最终、最佳和第100轮权重已分别重新加载，并复核准确率。训练脚本快照、配置、划分索引及完整曲线保存在本目录。
'''
(OUT/'结果说明.md').write_text(text,encoding='utf-8');print(json.dumps(results,indent=2))
