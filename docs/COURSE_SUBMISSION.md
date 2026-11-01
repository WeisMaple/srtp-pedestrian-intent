# 课程项目运行与提交说明

项目题目为“基于车载视频的行人过街意图预测”。演示使用 YOLO26s-Pose 与 ByteTrack 提取连续骨架，由 LSTM 预测后续姿态，再由 ST-GCN 输出其他、不过街和过街三类结果。网页将视频结果与真实模型中间特征放在同一流程中展示。

## 成员与分工

| 成员 | 学号 | 承担工作 | 占比 |
|---|---|---|---|
| 魏廷浩（队长） | 20231971 | 总体方案、模型训练、项目统筹与系统集成 | 40% |
| 张启睿 | 20233466 | 姿态跟踪、推理流程、网页演示与功能验证 | 30% |
| 张倩颖 | 20231860 | 数据整理、实验分析、报告撰写与答辩材料 | 30% |

## 本机演示

本机项目目录为 `F:\Artificial_Intelligence\srtp`，Python 环境为 `F:\Anaconda\envs\d2l`。网页已经有构建产物时，可直接执行：

```powershell
Set-Location 'F:\Artificial_Intelligence\srtp'
.\run_final_web.ps1
```

打开 `http://127.0.0.1:8765`，上传道路视频并开始分析。形成 50 次有效观测后，选择行人，依次展示历史骨架、未来骨架、空间与时间卷积以及分类输出。教室现场以道路视频为主；摄像头适合展示姿态感知和历史积累过程。

## 在其他电脑上运行

按根目录 README 安装 Python 3.9、PyTorch 与项目依赖，运行 `python download_final_models.py` 获取固定模型包。脚本会校验 SHA256，安装到 `weights/final/`。数据准备见 `data/PROJECT_DATA.md`。

网页还需安装 `requirements-web.txt`，在 `web_demo` 下运行 `npm install` 和 `npm run build`，并使 `ffmpeg` 可从命令行调用，然后运行 `python run_web_demo.py`。操作细节见 `docs/WEB_DEMO.md`。

只检查视频推理时，可运行：

```powershell
python -s run_demo.py --input "输入视频.mp4" --output "output/课堂测试.mp4"
```

输出路径应使用尚未存在的文件名。模型权重、数据视频和 Python 环境均独立于源码包；模型下载地址与校验值保存在 `docs/final/release_asset.json` 和 `docs/final/manifest.json`。

演示可选择本机 `data/JAAD_clips/video_0001.mp4`。完整训练按数据说明准备346段视频。

## 实验结果与材料

最终实验结果集中于 `docs/final/v2_20261102/`：`validation.json` 记录验证指标，`classification_test.json` 记录测试与类别权重实验，`matched_ablation.json` 记录相同配置下的LSTM消融，`runtime_checks.json` 记录视频速度，`runtime_parity.json` 核对部署结果。

Word报告和PPT见v2.0.0 Release。PPT保留原学校模板，共18页，包含5秒道路推理视频、约35秒操作录屏及骨架GIF。建议重点讲述任务、总体流程、数据构建、关键模块和实验结果；流程与公式页简述，保留时间展示实际系统。

报告包含背景、数据、模型结构与执行流程、少量核心代码、训练参数及曲线、定量结果、定性展示、消融与类别权重实验、成员分工和课程建议。代码、权重、课程材料分别分发，历史实验用于追溯，课堂入口以最终模型说明为准。
