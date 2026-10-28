"""Reload ST-GCN checkpoints and report complete original-window validation metrics."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import pathlib,sys,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from st_gcn import Model

def summarize(cm):
    cm=np.asarray(cm,dtype=np.int64);support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
    precision=np.divide(tp,predicted,out=np.zeros(3,dtype=float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(3,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(3,dtype=float),where=(support+predicted)>0)
    accuracy=float(tp.sum()/cm.sum());names=['irrelevant','not_crossing','crossing']
    return dict(samples=int(cm.sum()),correct=int(tp.sum()),accuracy=accuracy,
        macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),macro_f1=float(f1.mean()),
        weighted_precision=float(np.average(precision,weights=support)),weighted_recall=float(np.average(recall,weights=support)),weighted_f1=float(np.average(f1,weights=support)),
        micro_precision=accuracy,micro_recall=accuracy,micro_f1=accuracy,
        balanced_accuracy=float(recall.mean()),confusion_matrix=cm.tolist(),
        per_class=[dict(label=i,name=names[i],support=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i in range(3)])

def main():
    out=pathlib.Path(r'D:\SRTP_modern_internal_20261028');run=out/'models/stgcn'
    split=json.loads((run/'split.json').read_text());data=torch.load(out/'datasets/stgcn.pt',weights_only=True)
    ids=split['validation_indices'];assert len(ids)==len(set(ids))==200
    assert not(set(ids)&set(split['train_indices']))
    x=data['x'][ids];truth=data['y'][ids].numpy();torch.set_num_threads(2)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=Model(3,3,{'layout':'yolov8','strategy':'spatial'},True).to(device).eval()
    results={};predictions={}
    for tag,filename,record in [('epoch100','epoch_100_state_dict.pt','epoch_100_validation.json'),('final125','final_state_dict.pt','final_validation.json'),('best121','best_state_dict.pt','best_validation.json')]:
        model.load_state_dict(torch.load(run/filename,map_location=device,weights_only=True));probs=[]
        with torch.inference_mode():
            for xx in x.split(128):probs.extend(model(xx.to(device)).softmax(1).cpu().tolist())
        probs=np.array(probs);assert np.isfinite(probs).all();pred=probs.argmax(1)
        cm=np.bincount(3*truth+pred,minlength=9).reshape(3,3);r=summarize(cm)
        saved=json.loads((run/record).read_text());assert cm.tolist()==saved['confusion_matrix']
        assert abs(r['accuracy']-saved['accuracy'])<1e-7
        r.update(weights=str(run/filename),sha256=hashlib.sha256((run/filename).read_bytes()).hexdigest(),checkpoint_reloaded=True)
        results[tag]=r
        predictions[tag]=[dict(sample_index=i,truth=int(y),prediction=int(p),probabilities=pp.tolist()) for i,y,p,pp in zip(ids,truth,pred,probs)]
    result=dict(scope='ST-GCN only; original overlapping-window validation, not end-to-end or independent-video testing',device=device,validation_windows=200,train_windows=split['train_samples'],overlapping_history_windows=split['validation_windows_overlapping_training_history'],normalization='fitted before split on all windows',zero_division=0,results=results)
    (out/'detailed_metrics_20261029.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'validation_predictions_20261029.json').write_text(json.dumps(predictions),encoding='utf-8')
    lines=['ST-GCN内部验证指标（2026-10-29重新加载权重复核）','验证样本200个；训练窗口796个。195/200验证窗口与训练历史重叠，归一化在划分前计算。','类别0=irrelevant，类别1=不横穿，类别2=横穿。该结果不代表完整视频意图预测或独立测试性能。','']
    for tag,r in results.items():
        lines.append(f"{tag}: Accuracy={r['accuracy']:.4%}; Macro Precision={r['macro_precision']:.4%}; Macro Recall={r['macro_recall']:.4%}; Macro F1={r['macro_f1']:.6f}; Weighted F1={r['weighted_f1']:.6f}")
        for row in r['per_class']:lines.append(str(row))
        lines.append('Confusion matrix (true rows, predicted columns): '+str(r['confusion_matrix']));lines.append('')
    (out/'指标明细_20261029.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
