# 摄像头模拟过街演示（试验版）

固定摄像头拍摄同学，在地面划定模拟道路。这个入口用于现场试跑：展示 YOLO26 骨架、ByteTrack ID、轨迹外推，以及可选的原 LSTM → ST-GCN 模型对照。原 `run_prediction.py` 和离线实验默认参数保持不变。

## 启动

使用 README 中的 Python 3.9、PyTorch 和隔离 Ultralytics 环境，需要带 GUI 的 `opencv-python`。基础演示仅需本地 `yolo26s-pose.pt`；程序不会自动下载权重。

在已准备好权重的 Windows 机器上，从仓库根目录运行：

```powershell
$env:SRTP_MODERN_HOME = 'D:\SRTP_modern_20261027'
python -s run_camera_demo.py --camera 0 --backend dshow
```

如果摄像头打不开，先关闭占用摄像头的软件，检查系统权限，再试 `--camera 1` 或 `--backend msmf`。其他平台可省略 `--backend`；macOS 可用 `--backend avfoundation --device cpu`，现场性能需另测。模型路径不同可指定 `--pose-model 'path/to/yolo26s-pose.pt'`。

## 操作

1. 固定机位，保证同学全身和脚部入镜；用胶带在地面标出道路。
2. 在预览图像中，沿顺时针或逆时针依次点击道路的四个角。区域必须是凸四边形。
3. 按 **Enter** 确认，开始检测。不要将人物框或整幅画面当作道路区域。
4. **R** 重新划区；**C** 清空轨迹与模型历史；**Q / Esc** 退出。

画面右侧列出最多三个 ID 的详细结果，画面中仍显示其他被跟踪的人。当前骨架为实测姿态，箭头为轨迹外推，二者均不代表模型预测的未来骨架。

| 标识 | 含义 |
| --- | --- |
| `warming` | 尚未获得足够的连续观测 |
| `outside` | 按当前运动估计，在设定时段内不会进入区域 |
| `entering` | 外推路径预计与道路相交，旁边显示估计进入时间 |
| `inside` | 当前脚踝中点已在道路内，不属于提前预测 |
| `feet_missing` / `pose_missing` | 脚部或姿态不足，停止相应输出 |
| `expired` | 结果超过时效，不再展示原预测值 |

## 原模型对照

准备好 `stgcn_reproduced.pt` 和 `lstm_reproduced_wd0.pt` 后：

```powershell
python -s run_camera_demo.py --camera 0 --backend dshow --behavior
```

模型仍按 50 姿态 → LSTM 五轮预测 → ST-GCN 分类运行，默认 `legacy_pair` 归一化。用 `--stgcn`、`--lstm` 指定其他成对权重；使用 `--normalization training_stats` 时须通过 `--stats` 提供与权重配套的统计文件。

模型结果单独标记 `Model`。类 0 显示为 JAAD 的 `irrelevant`；缺失、过期属于额外运行状态。`score` 是未校准的模型分数，不是已验证的正确概率。轨迹外推不会修改或替换模型分类。

## 时间、间断与性能

- 应用读取线程仅保留最新一帧，避免 Python 推理队列积压；不保证消除摄像头驱动内部缓存。
- 轨迹使用读取完成时的单调时间，基于最近 0.5 秒脚踝中点的图像运动估计。区域外至少需要三个点、覆盖 0.2 秒；默认外推 1 秒，可用 `--horizon` 调整。
- 模型历史按 `--model-fps 30` 重采样为 50 点，覆盖约 1.63 秒。该采样率是假定配置，须匹配训练协议；重采样是本演示新增处理，尚未验证其模型精度。
- 默认观测间断超过 `--max-gap 0.2` 秒即重置历史；仅在更短间断内插值，关键点置信度保守取相邻观测较小值。
- 行为模型默认每 `--model-interval 0.2` 秒至多更新一次；超过 `--prediction-ttl 0.6` 秒的结果隐藏。计算本身耗时也计入时效。
- 等待下一帧或推理时，主线程继续刷新结果年龄、撤下过期标签并响应退出；不会将上一帧的有效状态冻结在窗口中。
- `Frame age` 为 OpenCV 交付帧至结果就绪的耗时，不包含曝光、驱动缓存和屏幕呈现，不是完整端到端延迟。`Processing FPS` 包含等待新帧的时间。

## 记录与复测

日志为显式启用，文件必须不存在：

```powershell
$log = 'artifacts/results/camera_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.jsonl'
python -s run_camera_demo.py --camera 0 --backend dshow --behavior --log $log --save-keypoints
```

JSONL 包含设置、道路角点、重置事件、逐帧时间/结果和结束状态；`--save-keypoints` 加入观测骨架。日志不包含视频，也不包含人工动作真值。模型文件、现场素材和运行日志应保留在 Git 外。

重复机位可显式指定归一化道路坐标；以下坐标仅作语法示例，须换成自己的区域：

```powershell
python -s run_camera_demo.py --camera 0 --road 0.4,0.2 0.6,0.2 0.6,0.9 0.4,0.9 --no-display --max-frames 300
```

依次试走入道路、沿边界行走、走近后停下、转身离开、短暂遮挡。检查误报、提示提前量、撤回时间与恢复情况。此 baseline 是图像坐标中的恒速外推，未做地面透视标定；停步、转向、脚部抖动和遮挡都会影响结果。课堂效果不能当作道路数据集准确率。

## 验证范围

自动测试覆盖区域相交、并行/远离运动、时间重采样、缺失与间断、缓存过期、最新帧缓冲、断流及日志退出流程：

```powershell
python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v
```

本次开发环境为 macOS / Python 3.9.25 / NumPy 1.23.5 / OpenCV 4.8.1 / PyTorch 2.2.2 CPU。摄像头和检测器替身仅用于流程测试，合成图像用于绘制检查；本机缺少真实权重和现场画面，尚未验证物理摄像头、YOLO 推理及同学动作效果。原有五项真实权重测试会明确跳过。

2026-10-28 本次检查：51 项测试中 46 项通过、5 项真实权重测试跳过；90 帧合成画面写出 MP4 并全部读回，90 行 JSONL 可解析。合成片仅验证几何、绘制与序列化，不代表检测精度或摄像头吞吐。
