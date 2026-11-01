# 课程项目数据说明

本项目使用 JAAD 道路视频与行人标注训练骨架预测和行为分类模型。代码提交包保留数据来源、目录约定及实验记录；完整视频通过数据集发布方下载，模型权重由项目固定发布包提供。

## 数据来源与放置位置

- 视频来自 [JAAD 数据集页面](https://data.nvision.eecs.yorku.ca/JAAD_dataset/index.html)，共 346 段。下载 `JAAD_clips.zip` 后，将视频放入本目录的 `JAAD_clips/`，例如 `data/JAAD_clips/video_0001.mp4`。
- 标注采用 [JAAD 标注仓库的 JAAD_2.0 分支](https://github.com/ykotseruba/JAAD/tree/JAAD_2.0)。将其中的 `annotations/`、`annotations_attributes/` 和 `split_ids/` 放入本目录，保留原有文件名和许可证。其他标注目录可一并保留。
- 当前本机完整数据位于 `F:\Artificial_Intelligence\srtp\data`。原视频压缩包来源为 `D:\下载\JAAD_clips.zip`。标注归档的 SHA256 为 `47296e9f40cab76268f6eef691a25cb932f5df6a87f6152fb4d3f23658648243`。

```text
data/
  PROJECT_DATA.md
  JAAD_clips/video_0001.mp4 ... video_0346.mp4
  annotations/video_0001.xml ...
  annotations_attributes/video_0001_attributes.xml ...
  split_ids/all_videos/train.txt
  split_ids/all_videos/val.txt
  split_ids/all_videos/test.txt
```

行人标签读取 `crossing` 属性，原值 `-1/0/1` 分别映射为“其他 / 不过街 / 过街”。“其他”对应数据集的 irrelevant。每次有效观测包含 17 个关节的横坐标、纵坐标和置信度；50 次观测构成一个模型输入。

## 样本与实验范围

最终划分为176/44/126段训练/验证/测试视频，同一视频只属于一个集合。分类样本数为1648/368/1506，三类计数依次为训练241/91/1316、验证26/16/326、测试230/43/1233。分类取样每隔10个原始帧，按检测跟踪编号取最近50次有效骨架，同帧IoU≥0.5的标注匹配仅用于赋标签。

LSTM短期样本为50×17×3历史与10×17×3未来，样本数1835/396/1523；递归训练的目标为50×17×3未来，样本数1074/209/831。窗口起点每次移动10次有效观测。置信度≥0.2且坐标>0的关节有效，每次观测至少5个有效关节，相邻帧号差>2时分段。

LSTM按最后观测进行相对表示，标准差由训练数据确定；ST-GCN训练输入由按视频三折交叉预测生成，标准化统计量只由训练预测骨架计算。

## 数据处理与训练入口

先运行 `independent_20261028/extract.py` 提取YOLO26s-Pose/ByteTrack缓存，再按 `independent_20261028/final_training_20261102/README.md` 的顺序构建样本、训练相对骨架LSTM、生成三折预测骨架和训练分类器。实际划分及实验配置见 `docs/final/v2_20261102/`。

训练脚本保留本机D盘实验路径，迁移电脑时先调整脚本开头的路径配置。演示只需下载最终模型包并准备一段视频，无需重跑训练。视频与标注使用条件以JAAD发布方说明为准。
