# ST-GCN 学习率对照实验

三组实验分别采用 0.0005、0.001、0.002 学习率，固定随机种子 2026、796/200 窗口划分、网络初始参数、批次顺序、SGD、batch_size=128、125 轮、momentum=0、weight_decay=0。每组按最高内部验证准确率选模，相同最高值取最早轮次。选中权重均重新加载并核对混淆矩阵。

| 学习率 | 轮次 | 内部验证准确率 | 宏平均 F1 |
|---|---:|---:|---:|
| 0.0005 | 104 | 95.00% | 0.9238 |
| 0.001 | 121 | 98.00% | 0.9614 |
| 0.002 | 91 | 99.00% | 0.9789 |

数据采用现有内部验证设置：先标准化再按窗口划分，200个验证窗口中195个与训练窗口共享历史观测。本次实验使用一个固定种子。99%对应学习率对照实验中的分类器内部验证；全流程评价和演示系统使用原有0.001权重。

## 复现

在项目根目录运行：

```powershell
python independent_20261028/learning_rate_study_20261031.py --project-root . --dataset-root D:\SRTP_modern_internal_20261028 --output D:\SRTP_lr_repeat
```

dataset-root需要包含 datasets/stgcn.pt 和 models/stgcn/split.json。该缓存由 independent_20261028/internal_validation.py 根据YOLO骨架缓存和JAAD标注构建；数据获取及目录说明见 data/PROJECT_DATA.md。训练使用 d2l 环境中的 PyTorch 2.2.2+cu121、NVIDIA RTX 4060 Laptop GPU。

configuration.json记录配置、输入数据和模型源文件哈希；split.json记录窗口索引；各lr目录包含每轮history.csv、选中权重的逐样本预测和result.json；summary.json汇总结果。history.csv中的loss是按样本数加权的交叉熵均值。图表使用完整125轮数据。best_predictions.json可据此重新计算分类指标。

训练权重保存在实验输出目录的best.pt，不覆盖演示系统权重；提交包保留脚本、配置、曲线、逐样本预测和指标。
