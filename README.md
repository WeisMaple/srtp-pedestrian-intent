# SRTP 行人意图预测

保存已在本机跑通的车载视频行人意图预测代码，包括现代 PyTorch 管线、原始基线与论文复现实验。当前默认仍为“历史骨架 → LSTM 预测未来骨架 → ST-GCN 分类”；讨论中的新版当前行为识别任务尚未实施。

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
- `--task classify`：直接分类已观察骨架；沿用当前权重标签，尚未实现新的逐帧动作任务。
- `--classify-every 5`：默认每 5 个视频帧更新行为结果，检测/跟踪逐帧执行；设为 `1` 可逐帧更新。
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

复现脚本保留本机 `D:\SRTP_reproduction_20261027` 等路径，并非通用训练 CLI。迁移到其他电脑时需调整路径，准备官方 JAAD 数据、标注和旧检测权重。详见该目录说明。

## 验证与分数范围

```powershell
python -s -m unittest discover -s modern_pipeline/tests -p 'test_*.py' -v
```

本机已有真实权重时 26 项全部通过；缺少复现权重时，5 项真实权重行为测试自动跳过，其余 21 项无需模型文件。

已完成 7 组实际推理、1,140 帧记录校验及 1,020 帧输出视频读回。600 帧完整示例平均处理 21.19 FPS。范围及限制见 [验证记录](docs/VALIDATION.md)。

先前 96.38% 是旧特征重建下的 ST-GCN 内部验证结果，存在窗口重叠和归一化数据泄漏；不能作为新前端的独立测试精度。新版骨架尚未用于重新训练行为模型。旧代码将类 0 显示为 unclear intent，其 JAAD 属性映射含义仍需按任务重新审定。

## 第三方来源

本项目包含保留的第三方源码及依赖，不为整个混合项目新增统一开源许可证。各部分来源与原许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `third_party_licenses/`。
