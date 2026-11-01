# -*- coding: utf-8 -*-
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,csv,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;BASE=Path(r'D:\SRTP_relative_pipeline_20261101');ROOT=Path(r'F:\Artificial_Intelligence\srtp');OLD=Path(r'D:\SRTP_lstm_retrain_20261101')
sys.path[:0]=[str(ROOT),str(BASE)]
from relative_runtime import RelativeBehaviorPredictor
from independent_20261028.evaluate_internal_metrics import summarize
load=lambda n:json.loads((O/n).read_text('utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (O/'complete.json').exists()
r=load('comparison.json');p=load('protocol.json');sel=load('selection_before_test.json');chosen=r['selected_by_validation']
names=['inverse','sqrt_inverse','effective_0999'];labels=['类别频率倒数（对照）','平方根倒数','有效样本数（β=0.999）'];classes=['其他','不过街','过街']
assert chosen==max(names,key=lambda n:(sel['validation'][n]['macro_f1'],-sel['validation'][n]['val_loss']))
assert all(sha(path)==h for path,h in p['frozen_sha256'].items())
assert sha(O/'inverse/best.pt')==sha(BASE/'forecast/best.pt')
rows=[]
for n,label in zip(names,labels):
    m=r['results'][n];best=sel['validation'][n];cm=np.array(m['confusion_matrix'])
    row=dict(method=n,label=label,selected_by_validation=n==chosen,epoch=best['epoch'],val_macro_f1=best['macro_f1'],test_accuracy=m['accuracy'],test_macro_f1=m['macro_f1'],
        not_crossing_precision=m['per_class'][1]['precision'],not_crossing_recall=m['per_class'][1]['recall'],not_crossing_false_positives=int(cm[:,1].sum()-cm[1,1]),not_crossing_true_positives=int(cm[1,1]))
    rows.append(row)
    pred=load(n+'/test_predictions.json');y=np.array([q['truth'] for q in pred]);yy=np.array([q['prediction'] for q in pred]);cm2=np.bincount(y*3+yy,minlength=9).reshape(3,3)
    assert cm2.tolist()==m['confusion_matrix']
with (O/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
for ax,n,label in zip(axes,names,labels):
    m=r['results'][n];cm=np.array(m['confusion_matrix']);q=cm/cm.sum(1,keepdims=True);ax.imshow(q,cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{q[i,j]:.1%}',ha='center',va='center',color='white' if q[i,j]>.6 else '#163044')
    ax.set_xticks(range(3),classes);ax.set_yticks(range(3),classes)
    ax.set(xlabel='预测类别',ylabel='真实类别',title=f'{label}\nAccuracy {m["accuracy"]:.2%} · Macro F1 {m["macro_f1"]:.4f}')
fig.savefig(O/'test_confusion.png',dpi=180,bbox_inches='tight',pad_inches=.18);fig.savefig(O/'test_confusion.svg',bbox_inches='tight',pad_inches=.18);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for n,label,col in zip(names,labels,['#87919b','#238875','#427dad']):
    hist=load(n+'/history.json')
    axes[0].plot([q['epoch'] for q in hist],[q['macro_f1'] for q in hist],color=col,label=label)
    axes[1].plot([q['epoch'] for q in hist],[q['val_loss'] for q in hist],color=col,label=label)
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('训练轮次')
axes[0].set(title='验证集宏平均 F1',ylabel='Macro F1');axes[1].set(title='验证集未加权交叉熵',ylabel='Loss')
fig.savefig(O/'validation_curves.png',dpi=180,bbox_inches='tight',pad_inches=.18);fig.savefig(O/'validation_curves.svg',bbox_inches='tight',pad_inches=.18);plt.close(fig)
torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
x=torch.load(OLD/'train_val.pt',weights_only=False)['val']['x'][:16].numpy()
cpu=RelativeBehaviorPredictor(O/chosen/'best.pt',BASE/'lstm/best.pt','cpu','training_stats',BASE/'forecast/normalization.pt')
gpu=RelativeBehaviorPredictor(O/chosen/'best.pt',BASE/'lstm/best.pt','cuda:0','training_stats',BASE/'forecast/normalization.pt')
a=cpu.predict(x)['probabilities'];b=np.concatenate([cpu.predict(xx[None])['probabilities'] for xx in x]);c=gpu.predict(x)['probabilities']
errors=dict(cpu_batch_single=float(abs(a-b).max()),cpu_gpu=float(abs(a-c).max()));assert max(errors.values())<1e-4
checks=dict(method_selected_by_validation=True,frozen_inputs_unchanged=True,baseline_exactly_reproduced=True,prediction_files_match_metrics=True,
    candidate_runtime_errors=errors,production_weights_unchanged=all(sha(path)==h for path,h in load('protected_weights.json').items()))
assert checks['production_weights_unchanged'];(O/'verification.json').write_text(json.dumps(checks,indent=2),'utf8')
bundle=O/'selected_bundle';bundle.mkdir(exist_ok=True)
for src,name in [(O/chosen/'best.pt','stgcn.pt'),(BASE/'lstm/best.pt','lstm.pt'),(BASE/'forecast/normalization.pt','normalization.pt'),(BASE/'relative_runtime.py','relative_runtime.py'),(BASE/'forecast_relative.py','forecast_relative.py')]:shutil.copy2(src,bundle/name)
config=dict(method=chosen,selected_epoch=sel['validation'][chosen]['epoch'],runtime='relative_runtime.RelativeBehaviorPredictor',normalization='training_stats',representation='last_observation_relative_17x3',production_deployed=False)
(bundle/'config.json').write_text(json.dumps(config,indent=2),'utf8')
lines=['固定改进LSTM后的类别权重对照实验','',
    'LSTM、预测骨架缓存、网络结构、视频划分和标准化保持固定。ST-GCN初始化与批次顺序相同，种子2026，SGD学习率0.001，batch_size128，训练125轮，无动量和权重衰减。',
    '三种类别权重：频率倒数；频率平方根倒数；有效样本数倒数，β固定为0.999。加权交叉熵按每批次样本权重之和归一化。',
    '每组检查点按验证集宏平均F1选择，同分时选择未加权交叉熵更低者。组间也使用同一规则，方案在读取本轮测试结果前确定。',
    f'验证集选择的方案：{labels[names.index(chosen)]}，第{sel["validation"][chosen]["epoch"]}轮。',
    f'测试范围：{r["test_videos"]}段视频，{r["samples"]}个共同有效样本，预定评价点{r["eligible"]}个，覆盖率{r["coverage"]:.2%}。测试划分沿用此前实验，所有本轮模型选择均使用验证集。','']
for row in rows:
    lines.append(f'{row["label"]}：验证MacroF1={row["val_macro_f1"]:.4f}；测试Accuracy={row["test_accuracy"]:.2%}，MacroF1={row["test_macro_f1"]:.4f}；不过街Precision={row["not_crossing_precision"]:.2%}，Recall={row["not_crossing_recall"]:.2%}；不过街正确{row["not_crossing_true_positives"]}，误判{row["not_crossing_false_positives"]}。')
lines+=['','相对当前倒数权重对照的配对视频重采样区间（2000次）：',json.dumps(r['uncertainty_vs_inverse'],ensure_ascii=False,indent=2),'',
    '权重方案影响训练损失，不改变模型结构与推理计算流程。本轮未重复视频测速。',
    '验证指标经重新加载检查点核对；训练特征与实际推理概率一致；候选权重通过CPU单样本/批量、CPU/GPU一致性检查。原项目默认权重保留。',
    'selected_bundle包含验证集选定的分类权重、固定LSTM、标准化统计量和相对变化量推理代码。使用relative_runtime.RelativeBehaviorPredictor，依次传入stgcn.pt、lstm.pt、设备、training_stats、normalization.pt。输入形状[B,50,17,3]，坐标为原图像素坐标。']
(O/'实验结果说明.txt').write_text('\n'.join(lines),'utf8')
source=O/'source';source.mkdir(exist_ok=True);shutil.copy2(ROOT/'st_gcn.py',source/'st_gcn.py')
shutil.copy2(ROOT/'modern_pipeline/behavior.py',source/'behavior.py')
with zipfile.ZipFile(O/'类别权重对照实验包.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in O.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and (path.suffix in ['.py','.json','.csv','.txt','.png','.svg'] or (path.parent==bundle and path.suffix=='.pt')):z.write(path,path.relative_to(O))
print(json.dumps(dict(selected=chosen,comparison=rows,checks=checks),ensure_ascii=False))
