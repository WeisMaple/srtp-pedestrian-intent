# 最终模型训练记录

这里保存本机实际执行的训练脚本，网络定义始终来自根目录 `st_gcn.py`。脚本保留实验时使用的 D 盘缓存路径，迁移机器时需调整各脚本开头的 ROOT/O/BASE 等目录；运行前先按 `data/PROJECT_DATA.md` 准备视频、标注及检测缓存。

1. `../retrain_paired_20261101.py` 构建按视频划分的因果历史、分类标签和未来骨架样本；划分文件见 `docs/final/v2_20261102/split.json`。
2. `investigate.py` 与 `controls.py` 训练并分析骨架表示，`rollout_train.py` 在所选相对骨架 LSTM 上进行50步递归训练。
3. `train_evaluate.py` 依照固定三折视频划分生成训练预测骨架，训练分类器；`forecast_relative.py` 实现相对坐标递归预测。
4. `study.py` 比较三种类别权重，`finalize_study.py` 汇总验证选模与测试结果。
5. `matched_ablation.py` 用相同平方根倒数权重重新训练历史骨架分类对照，比较最终 LSTM 路径。

这些脚本是顺序实验记录，前一步输出是后一步输入。演示不执行训练脚本；下载已校验的最终模型包后直接使用 `run_demo.py` 或 `run_web_demo.py`。
