SRTP 新版 YOLO / PyTorch 管线
2026-10-27

新入口已经可以运行：F:\Artificial_Intelligence\srtp\run_prediction.py
整体结构：YOLO26s-pose（行人框 + 17 个骨架点一次输出）→ ByteTrack（ID）→ PyTorch LSTM（未来骨架）→ PyTorch ST-GCN（意图分类）。
YOLO、LSTM、ST-GCN 默认全部使用 CUDA；没有 GPU 时自动用 CPU。ByteTrack 的匹配与 Kalman 滤波在 CPU 上。
新管线不导入 TensorFlow。旧 TensorFlow ReID.pb 被新的跟踪方案替代，没有把旧外观模型的权重硬转换成另一种网络。

直接运行
在 PowerShell 执行以下一整行，输出文件应使用新名字：
& 'F:\Anaconda\envs\d2l\python.exe' -s 'F:\Artificial_Intelligence\srtp\run_prediction.py' --input 'F:\Artificial_Intelligence\srtp\data\JAAD_clips\video_0001.mp4' --output 'D:\SRTP_modern_20261027\my_result.mp4'

也可继续使用 run_prediction.ps1：
& 'F:\Artificial_Intelligence\srtp\run_prediction.ps1' -InputVideo 'F:\Artificial_Intelligence\srtp\data\JAAD_clips\video_0001.mp4' -OutputVideo 'D:\SRTP_modern_20261027\my_result2.mp4'

省略 --output 时自动在 D:\SRTP_modern_20261027\results 生成带时间戳的文件。
输出包括：MP4、同名 .jsonl 逐帧结果、同名 .metrics.json 指标。--save-keypoints 会把每个行人的全图像素坐标骨架也写入 JSONL。

常用选项
--task predict：默认。用 50 次骨架观测递归预测未来 50 次观测的姿态，再分类。
--task classify：只对已观察的 50 次骨架观测做 ST-GCN 分类，不调用 LSTM。
--classify-every 1：默认每帧更新一次行为结果，保持原预测循环的频率。课堂展示需要加速时可显式设为 5，中间帧缓存标签，检测和跟踪仍逐帧运行。
--weights reproduced：默认，使用上轮重训的 ST-GCN 第 125 轮与 LSTM weight_decay=0 对照组权重。
--weights original：使用项目 Models 中最初提供的 ST-GCN/LSTM 权重。
--stgcn / --lstm：指定兼容的 state_dict 文件，覆盖预设行为权重路径。
--normalization legacy_pair：默认，保留论文原来的固定参考序列 + 目标序列二样本归一化。
--normalization training_stats：显式选择上轮探索的训练统计归一化，使用小文件 weights\normalization.pt；统计来自旧前端重建数据，只支持 reproduced 预设。若指定其他行为权重，须用 --stats 传入与其训练匹配的统计文件。
--tracker botsort.yaml：切换为 BoT-SORT（库默认未启用外观 ReID），支持相机运动补偿；默认是较简单的 bytetrack.yaml。
--half：只让 YOLO 在 CUDA 上使用 FP16，LSTM/ST-GCN 保持 FP32。
--device cpu：全部神经网络改为 CPU；--device cuda:0 可明确指定 GPU。
--max-frames 90：只测试前 90 帧；--no-video：仅输出 JSONL 和指标，不编码 MP4。

默认 max-gap=2，允许中间缺失一帧骨架；更大间隔会清空该 ID 的历史与缓存分类。track-ttl=30 负责回收旧 ID。min-joints=5、joint-conf=0.2 用于屏蔽过差姿态。
因此这里的 50 是骨架观测次数，可能跨越略多于 50 个视频帧。要求严格连续可设 --max-gap 1。
显示的分类分数是 softmax 值，未做概率校准，不是准确率。

模块与优化
modern_pipeline\frontend.py：YOLO 单次前向输出框和骨架；直接使用全图 COCO 17 点坐标，避免逐人裁剪与重复推理。
modern_pipeline\behavior.py：多人统一 batch；五次 LSTM 递归全过程留在 torch/GPU；inference_mode、eval；保留原模型结构。
modern_pipeline\history.py：有界轨迹、间隔重置与过期回收。
modern_pipeline\cli.py：视频资源释放、输出检查、帧率保留、逐帧记录和耗时统计。
原 train.py / st_gcn.py / DataSet.py / 原始权重均保持不变。原 predict.py、DataProcess.py、DeepSORT 作为旧版参考保留，新入口不导入它们。
旧入口已备份为 run_legacy_prediction.py；可用它运行旧 TensorFlow 管线。

安装与路径
Python 固定为 F:\Anaconda\envs\d2l\python.exe，继续使用 d2l 已有 torch 2.2.2+cu121 / torchvision 0.17.2+cu121。
新版 Ultralytics 8.4.190 和少量附加库独立放在 D:\SRTP_modern_20261027\runtime\site-packages，不覆盖 d2l 原 Ultralytics 8.2.103。
bootstrap.py 自动为新入口选择该目录，避免影响旧复现实验。用 python -s 排除用户目录包；自动设置 MKL_THREADING_LAYER=SEQUENTIAL。
附加库锁定清单：requirements-modern.txt。需要重建时运行：
& 'F:\Anaconda\envs\d2l\python.exe' -s -m pip install --target 'D:\SRTP_modern_20261027\runtime\site-packages' --no-deps -r 'F:\Artificial_Intelligence\srtp\modern_pipeline\requirements-modern.txt' --index-url https://pypi.org/simple
SRTP_MODERN_HOME 环境变量可整体修改新版产物根目录；移动时请连同 runtime 与 weights 一起移动。
权重来源及哈希：D:\SRTP_modern_20261027\weights\source.json、artifact_manifest.json。
行为权重来自上轮复现，没有对新版骨架重新训练。

验证结果
26 项 unittest 全通过：原行为公式数值对齐、batch/单人一致、空检测、框-ID-骨架对应、非有限输入、断续轨迹和过期回收。
执行：
& 'F:\Anaconda\envs\d2l\python.exe' -s -m unittest discover -s 'F:\Artificial_Intelligence\srtp\modern_pipeline\tests' -p 'test_*.py' -v
7 组新管线实际运行共检查 1,140 帧；5 段输出视频共 1,020 帧全部读回；各次运行均未加载 TensorFlow。
历史测试中，显式每 5 帧更新分类的 600 帧完整视频在本机平均处理 21.19 FPS，含读帧、绘制、编码和逐帧记录；含模型准备总耗时 28.66 秒，PyTorch 峰值已分配显存约 258 MiB（不含桌面其他进程与CUDA驱动开销）。
CPU 分类、训练统计归一化、CUDA FP16/BoT-SORT 分支也已运行测试。

精度范围
这次验证完成了前端和框架替换、接口兼容与速度测试。新骨架、置信度和跟踪分布会改变，旧的 96.38% 内部验证分数不适用于新版。
要判断新版意图精度，需要用新版骨架重新提取 JAAD 数据，按视频划分后重训并评测。此前原方法的窗口重叠问题仍需在该实验中处理。
详细速度比较与现有局限：D:\SRTP_modern_20261027\升级与测试报告.txt
演示：D:\SRTP_modern_20261027\results\yolo26_video0001_full.mp4

官方资料
https://docs.ultralytics.com/models/yolo26
https://docs.ultralytics.com/tasks/pose
https://docs.ultralytics.com/modes/track
https://pypi.org/project/ultralytics/8.4.190/
