# 行人过街意图预测

输入车载视频，显示行人框、人体骨架、行人ID及过街意图。系统根据50次历史骨架观测，由LSTM预测后续骨架，再由ST-GCN输出分类结果。

在项目目录运行：

```powershell
.\run_demo.ps1 -InputVideo "F:\Artificial_Intelligence\srtp\data\JAAD_clips\video_0001.mp4" -OutputVideo "D:\pedestrian_demo.mp4"
```

画面中的 `Collecting motion` 表示正在积累历史；满足条件后显示意图和模型输出分数。`score` 是模型类别分数，不是实测准确率。输出文件需使用尚未存在的路径。除MP4外，还保存逐帧JSONL和运行指标，便于定位问题。

方法保留论文的17点骨架、50次历史观测、原LSTM和ST-GCN结构及递归骨架预测。前端使用YOLO26 Pose和ByteTrack。课堂展示入口固定为意图预测，每帧更新。骨架处理和归一化由模型目录中的配置控制，必须与权重匹配。

本机采用GPU检测/骨架提取与CPU行为推理。最终模型通过 `python download_final_models.py` 安装到项目 `weights/final/`，也可设置 `SRTP_DEMO_MODEL_DIR` 或传入 `--model-dir`。模型包内的配置、权重和统计文件应一起使用。画面中的 `Other` 对应数据集的irrelevant类别，不表示模型置信度低。最终权重来源与评测范围见 [最终模型说明](FINAL_MODEL.md)。

训练衔接修正后的候选模型位于 `D:\SRTP_final_20261028`，可使用 `python -s run_demo.py --model-dir D:\SRTP_final_20261028 --input 视频路径 --output 新输出路径` 检查。候选模型未提升独立测试分类表现，因此没有替换课堂默认模型；详见 [训练衔接记录](TRAINING_ALIGNMENT.md)。

系统界面只展示运行结果。论文复现记录和各项评测数据保存在实验目录，课堂报告中应按相应评测口径引用。

本机已完成150帧视频检查，生成119次行为预测，输出视频150帧全部读回；26项接口与模型测试通过。演示文件：`D:\SRTP_independent_20261028\speed\classroom_demo_150.mp4`。该短片用于验证运行和画面，不是准确率评测。

