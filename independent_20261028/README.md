# 独立视频测试：旧前端与 YOLO26 前端

已归档报告、训练曲线和指标见 [实验结果](../docs/experiments/README.md)。新版内部验证由 `internal_validation.py` 执行，完成后运行 `summarize_internal.py`；它使用全视频随机窗口，因此与本目录的独立视频实验分开保存和解释。

完整执行顺序为：`extract.py` → 两次 `experiment.py --variant legacy/modern` → 两次 `benchmark.py --variant legacy/modern` → `benchmark_behavior.py` 与 `check_extraction.py` → `summarize.py`。每一步确认成功后再继续。运行路径仍按本机配置，迁移时需调整脚本中的 ROOT/OUT/缓存路径。

产物目录：`D:\SRTP_independent_20261028`。使用 d2l 的 Python 加 `-s` 运行。

1. `extract.py`：346 个视频，YOLO26s-Pose，640，FP32，检测阈值 0.1，ByteTrack。批量 8 加速离线姿态推理，跟踪仍按帧顺序更新、按视频重置。
2. `experiment.py --variant legacy` / `--variant modern`：JAAD 官方 `all_videos` train/val/test；禁止跨视频集合，训练统计不含验证和测试数据。
3. `benchmark.py --variant legacy` / `--variant modern`：官方测试列表前三个视频，各前 150 帧，同一输入文件、MP4V 输出、每帧预测；记录初始化在内的调用时长与标注次数。

网络直接导入原 `st_gcn.py`：三类 ST-GCN；10 层、hidden 256 的 LSTM，50→10，递归五次。ST-GCN SGD 0.001，125 轮；LSTM Adam 0.001，100 轮、weight_decay=0（沿用已验证的对照设置，原源码的 1e-4 曾出现塌缩）；batch 128，seed 2026。两组均按验证损失最低选择模型，测试结果不参与选择。

训练监督：同一帧框 IoU≥0.5 的一对一匹配；至少 5 个有效关节点；缺口超过 2 帧则切段。测试历史只由预测轨迹形成，不用真实 ID 修复跟踪；真实框仅用于给已输出结果评分。每 10 帧抽样一次以控制离线评估成本，速度测试仍逐帧运行。任务标签维持 JAAD pedestrian crossing 属性，-1/0/1 映射到三类；-1 是 irrelevant，并非概率不确定性。此评估不是逐帧 crossing 事件预测基准。

旧前端复用上轮已提取的 YOLOv7/DeepSORT/裁剪 YOLOv8 骨架，故继承旧预处理的轨迹筛选覆盖范围；本轮按同步真实框重新关联标签。新版重新提取全部轨迹。分别报告各自覆盖样本与双方同一 `(video, person, frame)` 交集，覆盖率分母为具有至少 50 帧标注历史的测试行人采样点。骨架来自自动提取，并非人工骨架真值。

主结果采用原运行逻辑 `legacy_pair`，训练归一化只用训练数据。另列 `training_stats` 推理对齐作为预先指定的诊断；它不是原论文运行归一化。另列观察骨架的 ST-GCN 分数，不能与 LSTM→ST-GCN 联合预测混用。

速度是两种实际实现的工程比较：原路径中姿态及行为模型在 CPU，新路径为 CUDA；不能把全部差值归因于 YOLO。模型初始化、有效轨迹数量、预测次数都会影响速度，必须伴随覆盖次数解释。单次三视频测速与单种子训练不用于声称统计显著提升。
