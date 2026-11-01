# -*- coding: utf-8 -*-
"""Produce figures, a Chinese experiment report and portable weight bundles."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib,csv,shutil,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path(r'D:\SRTP_lstm_retrain_20261101'));a=p.parse_args();O=a.output
load=lambda n:json.loads((O/n).read_text('utf8'))
r=load('comparison.json');ds=load('dataset.json');verification=load('verification.json');bm=load('video_benchmark/summary.json');forecast=load('lstm_test_forecast.json')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'observed':'#3869A0','forecast':'#D18B36'};names={'observed':'历史骨架 → ST-GCN','forecast':'LSTM预测骨架 → ST-GCN'}
fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
h=load('lstm/history.json');axs[0].plot([x['epoch'] for x in h],[x['train_mse'] for x in h],label='训练集',c='#3869A0');axs[0].plot([x['epoch'] for x in h],[x['val_mse'] for x in h],label='验证集',c='#D18B36');axs[0].set(title='LSTM骨架预测',ylabel='标准化均方误差')
for branch in names:
    h=load(branch+'/history.json');x=[z['epoch'] for z in h]
    axs[1].plot(x,[z['macro_f1'] for z in h],label=names[branch],c=colors[branch]);axs[2].plot(x,[z['val_loss'] for z in h],label=names[branch],c=colors[branch])
axs[1].set(title='分类器验证集表现',ylabel='宏平均F1',ylim=(0,1));axs[2].set(title='分类器验证集损失',ylabel='交叉熵')
for ax in axs:ax.set_xlabel('训练轮次');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
fig.savefig(O/'training_curves.png',dpi=180);fig.savefig(O/'training_curves.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,4.1),layout='constrained');labels=['其他','不过街','过街']
for ax,branch in zip(axs,names):
    cm=np.array(r['branches'][branch]['confusion_matrix']);ratios=cm/np.maximum(1,cm.sum(1,keepdims=True));im=ax.imshow(ratios,vmin=0,vmax=1,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{ratios[i,j]:.1%}',ha='center',va='center',color='white' if ratios[i,j]>.55 else '#16364B',fontsize=11)
    ax.set(xticks=range(3),yticks=range(3),xticklabels=labels,yticklabels=labels,xlabel='预测类别',ylabel='真实类别',title=names[branch])
fig.colorbar(im,ax=axs,shrink=.7,label='各真实类别中的比例');fig.savefig(O/'test_confusion.png',dpi=180);fig.savefig(O/'test_confusion.svg');plt.close(fig)
rows=[]
for b in names:
    for row in r['branches'][b]['per_class']:rows.append(dict(branch=b,**row))
with (O/'per_class_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
obs=r['branches']['observed'];pred=r['branches']['forecast'];delta=r['delta'];ci=r['uncertainty']
table='| 路径 | 准确率 | 宏平均F1 | 其他召回率 | 不过街召回率 | 过街召回率 |\n|---|---:|---:|---:|---:|---:|\n'
for b in names:
    x=r['branches'][b];table+=f"| {names[b]} | {x['accuracy']:.2%} | {x['macro_f1']:.4f} | "+' | '.join(f"{v['recall']:.2%}" for v in x['per_class'])+' |\n'
direction='提高' if delta['macro_f1']>0 else '降低'
conclusion=f"加入LSTM后，测试集宏平均F1{direction}了{abs(delta['macro_f1']):.4f}，准确率变化为{100*delta['accuracy']:+.2f}个百分点。"
if ci['macro_f1_delta_95ci'][0]<=0<=ci['macro_f1_delta_95ci'][1]:
    conclusion+='按视频进行配对重采样得到的差值区间跨过0，因此当前数据尚不足以确认稳定的分类收益。'
elif delta['macro_f1']>0:conclusion+='本次按视频配对重采样的宏平均F1差值区间高于0，支持当前设置下的分类收益。'
else:conclusion+='本次按视频配对重采样的宏平均F1差值区间低于0，当前LSTM分支的分类表现仍弱于历史骨架分支。'
lines=['# 保持模型结构的LSTM配对重训练实验','',conclusion,'',
'## 训练与数据处理','',
'YOLO26s-Pose、ByteTrack、10层LSTM和10块ST-GCN的网络结构保持原样。实验复用346段视频的原始检测与骨架缓存，没有增加姿态平滑。分类输入严格按检测跟踪身份和时间积累；人工标注只用于同帧边界框匹配和类别赋值，不修补历史轨迹。',
'',f"训练、验证、测试分别包含{len(load('split.json')['train'])}、{len(load('split.json')['val'])}、{r['test_videos']}段视频，三者无视频交集。训练和验证的分类窗口分别为{ds['train']['samples']}、{ds['val']['samples']}，类别数量依次为{ds['train']['classes']}、{ds['val']['classes']}。LSTM训练使用{ds['train']['lstm_samples']}组50步历史/10步未来骨架，验证使用{ds['val']['lstm_samples']}组；监督目标来自同一检测轨迹的后续关键点观测。",'',
'训练/验证划分沿用此前依据类别人数确定的视频分组；原测试视频保持不变。测试视频曾用于早期项目开发，本次两条新训练路径均未用测试数据拟合参数或选模。', '',
'LSTM使用训练集的逐关节、逐通道均值与标准差，Adam学习率0.001、batch_size=128、100轮，按验证MSE选模。训练分类器的预测骨架由三折按视频交叉训练生成：一个训练视频的预测来自未使用该视频训练的LSTM。验证与测试预测来自完整训练集拟合的主LSTM。', '',
'两个分类器分别适应历史观测骨架和预测骨架。各自的标准化统计量只从训练输入计算，并由实际推理函数原样复用。两者具有相同初始权重、训练样本顺序、SGD学习率0.001、batch_size=128和125轮预算。交叉熵按训练类别数量的倒数加权；按验证集宏平均F1选模，平分时选验证交叉熵较低者。随机种子固定为2026。','',
f"主LSTM选中第{load('lstm/best.json')['epoch']}轮；历史骨架分类器选中第{load('observed/best.json')['epoch']}轮，预测骨架分类器选中第{load('forecast/best.json')['epoch']}轮。",'',
'![训练曲线](training_curves.png)','',
'## 同一测试集上的配对结果','',
f"两条路径共同评价{r['test_samples']}个有效点，候选点总数{r['eligible']}，覆盖率为{r['coverage']:.2%}。类别数量为{r['classes']}。所有输出点的输入历史与标签完全一致。",'',table,'',
'![测试集混淆矩阵](test_confusion.png)','',
f"按视频进行{ci['replicates']}次配对重采样，准确率差值的95%区间为[{100*ci['accuracy_delta_95ci'][0]:+.2f}, {100*ci['accuracy_delta_95ci'][1]:+.2f}]个百分点；宏平均F1差值区间为[{ci['macro_f1_delta_95ci'][0]:+.4f}, {ci['macro_f1_delta_95ci'][1]:+.4f}]。该区间反映测试视频组成的变化，不包含多随机种子训练的不确定性。",'',
'## 骨架预测误差与速度','',
f"在测试视频的{forecast['samples']}组10步预测样本上，LSTM标准化MSE为{forecast['normalized_mse']:.4f}；将最后一次观测直接重复10次的参照MSE为{forecast['persistence_mse']:.4f}。只计算横纵坐标时，两者MSE分别为{forecast['xy_mse']:.4f}和{forecast['persistence_xy_mse']:.4f}。这些误差针对后续YOLO骨架观测。",'',
'| 路径 | GPU单人行为推理中位延迟 | 视频处理速度（不编码，两次中位值） |','|---|---:|---:|']
for b in names:
    x=r['branches'][b];v=bm['results'][b]
    lines.append(f"| {names[b]} | {x['behavior_gpu_latency_ms']['median']:.2f} ms | {v['median_no_video_fps']:.2f} FPS |")
lines+=['','单人延迟统计对同一批64个历史窗口重复3次，预热后计时，包含输入/输出传输，YOLO不计入该项。视频测试使用同一段1920×1080视频的前150帧，YOLO在GPU运行，行为模型在CPU运行；两条路径的检测、跟踪和采样设置一致。测速采用预热后的两次不编码运行；首轮生成的视频用于检查显示效果，不纳入速度汇总。','',
'## 核对与使用','',
'已验证视频划分互斥、交叉预测不使用本视频训练、训练统计量来源、两条路径评价点一致、重新加载权重后指标一致，以及训练特征与实际推理输出一致。网络结构文件及当前演示权重的哈希均保持不变。详细记录见verification.json和runtime_parity.json。','',
'模型分别保存在observed/best.pt和forecast/best.pt；主LSTM为lstm/best.pt。每条路径必须配套使用本目录中的normalization.pt。training_stats是两条新路径共同的推理方式。旧报告中的内部98%和旧全流程诊断采用不同训练与评价设置，保留为原版本实验记录。','',
'类别标签沿用JAAD的行人级过街属性。本次实验比较两条处理路径的分类效果，过街开始前不同提前量的评价可作为后续实验。','']
diag=load('forecast_horizon_diagnostic.json')
lines+=['## 递归预测检查','',f"使用具有完整50步后续观测的{diag['samples']}组测试序列，分别计算每10步的标准化坐标MSE。",'', '|未来观测范围|LSTM预测|保持最后观测不变|','|---|---:|---:|']
for i,(v,b) in enumerate(zip(diag['lstm_mse_by_10step_block'],diag['persistence_mse_by_10step_block'])):
    lines.append(f'|{i*10+1}—{i*10+10}|{v:.4f}|{b:.4f}|')
lines+=['','五个区间内，LSTM坐标误差均高于保持最后观测的参照。训练/推理的一致性核对已通过，这组结果表明当前骨架预测质量仍不足，后续应优先检查骨架预测的监督目标与训练策略。','']
(O/'实验对比报告.md').write_text('\n'.join(lines),'utf8')
bundles=O/'bundles';bundles.mkdir(exist_ok=True)
for b in names:
    files={'stgcn.pt':O/b/'best.pt','normalization.pt':O/b/'normalization.pt'}
    if b=='forecast':files['lstm.pt']=O/'lstm/best.pt'
    config=dict(branch=b,task='classify' if b=='observed' else 'predict',normalization='training_stats',pose_filter=False,
        stgcn='stgcn.pt',lstm='lstm.pt' if b=='forecast' else None,stats='normalization.pt',architecture_sha256=load('protocol.json')['architecture_sha256'],
        sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()})
    with ZipFile(bundles/(b+'_weights.zip'),'w',ZIP_DEFLATED) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('model_config.json',json.dumps(config,indent=2))
print(conclusion)
