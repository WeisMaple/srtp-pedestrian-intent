SRTP 论文复现产物（2026-10-27）

已完成全量视频预处理、从头训练、受控对照、验证及演示。
详细结论：D:\SRTP_reproduction_20261027\复现报告.txt
结果/新权重/中间数据：D:\SRTP_reproduction_20261027
原项目与原权重保持不变。

在 PowerShell 中运行新权重视频推理（请选用尚不存在的输出文件名）：
& 'F:\Anaconda\envs\d2l\python.exe' -s 'F:\Artificial_Intelligence\srtp\reproduction_20261027\run_reproduced.py' --input 'F:\Artificial_Intelligence\srtp\data\JAAD_clips\video_0001.mp4' --output 'D:\SRTP_reproduction_20261027\my_result.mp4'
默认使用第 125 轮 ST-GCN 和 weight_decay=0 对照组 LSTM。此 LSTM 设置是论文未指定衰减时的默认参数比较，运行器仍使用原推理归一化方式。改用原代码衰减组可加 --lstm-variant source_code；该组本次训练有退化，详见报告。

脚本均自动设置 MKL_THREADING_LAYER=SEQUENTIAL；使用 python -s 避免用户站点包冲突。新权重的固定存放位置为 D 盘。

现有产物的分析重跑顺序（会更新分析文件，不会修改原始模型）：
& 'F:\Anaconda\envs\d2l\python.exe' -s 'F:\Artificial_Intelligence\srtp\reproduction_20261027\analyze_reproduction.py'
& 'F:\Anaconda\envs\d2l\python.exe' -s 'F:\Artificial_Intelligence\srtp\reproduction_20261027\ablate_training_stats_alignment.py'
& 'F:\Anaconda\envs\d2l\python.exe' -s 'F:\Artificial_Intelligence\srtp\reproduction_20261027\verify_reproduction.py'

训练入口 train_reproduction.py：
--model stgcn：SGD 第 125 轮，含第 100 轮产物。
--model lstm：原代码 Adam weight_decay=1e-4。
--model lstm --weight-decay 0：仅改变衰减的受控对照。
已有断点会拒绝覆盖；--resume 用于从断点续跑。本轮全部训练已完成，无需再训练。
预处理 prepare_detected_skeletons.py、数据集 build_training_datasets.py。run_training_when_ready.py 负责最初两组训练，不含后来追加的衰减 0 对照。

解释分数时使用报告中的窗口重叠说明；96.38% 是分类模块内部验证结果。独立视频测试需按视频重新划分并重新训练。
