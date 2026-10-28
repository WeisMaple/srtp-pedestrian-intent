# SRTP 行人意图预测

本项目以原论文的行人意图预测为目标，保留“历史骨架 → LSTM 预测未来骨架 → ST-GCN 分类”的方法。代码升级集中于新版 YOLO、PyTorch 推理、模块接口和运行稳定性。课堂展示围绕道路视频中的行人检测、骨架和意图输出展开。

原方法基线与工程升级版的边界、参数和结果口径见 [复现范围](docs/REPRODUCTION_SCOPE.md)。

```text
视频 → YOLO26s-Pose（行人框 + COCO17 骨架）→ ByteTrack（行人 ID）
     → 50 次骨架观测 → LSTM（5 × 10 次未来姿态）→ ST-GCN（三类输出）
```

YOLO、LSTM 和 ST-GCN 默认在 CUDA 运行。新入口不导入 TensorFlow；原 YOLOv7 / DeepSORT / TensorFlow ReID 流程保留供对照。

## 已验证环境

- Windows，Python 3.9.23，RTX 4060 Laptop 8 GiB。
- PyTorch 2.2.2+cu121，torchvision 0.17.2+cu121。
- Ultralytics 8.4.190，YOLO26s-Pose。
- 原 d2l 环境保留 Ultralytics 8.2.103；新版附加依赖放在独立目录，由 `bootstrap.py` 选择。

## 安装与运行

仓库只包含代码、配置与说明。模型权重和 JAAD 视频未上传，克隆后须先准备下述外置文件。

使用 Python 3.9 环境。以下为 PowerShell 示例；已存在的本机 d2l 环境无需重复安装。

```powershell
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-base.txt
$env:SRTP_MODERN_HOME = Join-Path $PWD 'artifacts'
python -m pip install --target "$env:SRTP_MODERN_HOME\runtime\site-packages" --no-deps -r modern_pipeline/requirements-modern.txt
```

准备 `$env:SRTP_MODERN_HOME\weights` 下的四个文件：

| 文件 | 用途 |
| --- | --- |
| `yolo26s-pose.pt` | 官方 YOLO 行人框及姿态模型 |
| `stgcn_reproduced.pt` | 上轮复现 ST-GCN 第 125 轮权重 |
| `lstm_reproduced_wd0.pt` | 上轮复现、weight_decay=0 的 LSTM 权重 |
| `normalization.pt` | 显式启用 training_stats 模式时的归一化统计 |

权重来源和校验值见 [外置文件说明](docs/ARTIFACTS.md)。行为权重需要从本机备份复制，或按复现脚本重新训练。

```powershell
python -s run_prediction.py --input 'path/to/video.mp4' --output 'artifacts/results/demo.mp4'
```

不设置 `SRTP_MODERN_HOME` 时，现代管线继续使用原本已验证的本机目录 `D:\SRTP_modern_20261027`。具体本机路径与命令见 [原运行说明](modern_pipeline/README.txt)。

## 运行选项

- `--task predict`：默认，预测未来骨架后分类。
- `--task classify`：仅用于分类模块的诊断与对照；默认课题流程仍为 `predict`。
- `--classify-every 1`：默认逐帧更新行为结果，与原预测循环的更新频率保持一致。展示加速可显式设为 `5`，中间帧缓存标签。
- `--normalization legacy_pair`：默认，保留恢复代码的固定参考二样本归一化。
- `--normalization training_stats`：显式使用上轮训练统计，属于另列的探索分支。
- `--tracker botsort.yaml`：可选 BoT-SORT；默认 ByteTrack。
- `--device cpu`、`--half`：CPU 推理，或仅让 YOLO 使用 CUDA FP16。
- `--max-frames 90 --save-keypoints`：短片测试并把骨架写入逐帧 JSONL。

输出包含 MP4、JSONL 和指标 JSON；拒绝覆盖已有文件。骨架窗口为 50 次观测，默认允许中间缺失一帧；`--max-gap 1` 可要求严格连续。

## 代码结构

- `modern_pipeline/`：YOLO Pose、跟踪接口、行为模型、轨迹历史及视频 CLI。
- `st_gcn.py`：原 ST-GCN 和 SkeletonLSTM 网络结构。
- `run_prediction.py`：默认现代入口；`run_legacy_prediction.py`：旧入口。
- `predict.py`、`DataProcess.py`、`DataSet.py`、`train.py`：恢复的原项目代码。
- `api/`、`yolo/`、`deepsort/`：旧版检测与跟踪依赖的本地源码。
- `reproduction_20261027/`：全量 JAAD 预处理、训练、对照实验与分析入口。
- `independent_20261028/`：YOLO26 骨架提取、独立视频重训/对比、内部验证和测速脚本。
- `docs/experiments/`：报告、指标、曲线及样本划分。

复现脚本保留本机 `D:\SRTP_reproduction_20261027` 等路径，并非通用训练 CLI。迁移到其他电脑时需调整路径，准备官方 JAAD 数据、标注和旧检测权重。详见该目录说明。

## 验证与分数范围

```powershell
python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v
```

本机已有真实权重时 26 项全部通过；缺少复现权重时，5 项真实权重行为测试自动跳过，其余 21 项无需模型文件。

已完成 7 组实际推理、1,140 帧记录校验及 1,020 帧输出视频读回。此前 600 帧完整示例在显式每 5 帧更新行为结果的设置下平均处理 21.19 FPS；该历史速度不代表当前逐帧默认配置。范围及限制见 [验证记录](docs/VALIDATION.md)。

2026-10-28 已完成新版骨架提取和两套前端的独立视频重训；共同598个测试点上，旧版准确率74.75%、新版76.92%，但Macro-F1下降，尚未证实泛化改善。另按原代码式内部验证，新版ST-GCN最终97.00%、最佳98.00%；此结果存在窗口重叠及归一化泄漏，不是独立测试精度。完整结果见 [实验索引](docs/experiments/README.md)。旧代码将类0显示为unclear intent，JAAD原属性-1实际为irrelevant。

## 第三方来源

本项目包含保留的第三方源码及依赖，不为整个混合项目新增统一开源许可证。各部分来源与原许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `third_party_licenses/`。
