# 最终模型清单

固定发布标签：`v1.0.0`。GitHub Release提供 `srtp-final-models-v1.0.0.zip`；下载地址和SHA256在 `final/release_asset.json`，包内各文件校验值在 `final/manifest.json`。

| 组件 | 固定来源 |
|---|---|
| YOLO26s-Pose | 官方v8.4.0资产 |
| ST-GCN | 新版骨架、原代码式内部验证的第121轮；200个验证窗口，196个正确 |
| LSTM | 已完成的YOLO26骨架训练，第68轮验证MSE选模权重 |
| 骨架处理 | COCO17，50次观测，ByteTrack，未启用实验性平滑 |
| 视频归一化 | `legacy_pair`，沿用原推理逻辑 |
| 设备调度 | 自动选择检测设备，行为推理为CPU |

分类器是按用户选择固定的内部验证最佳检查点，不代表根据完整视频意图评测选出的全流程最优组合。LSTM与分类器分别训练，使用同一种YOLO26骨架格式；该最终组合已单独做视频运行检查。98%和Macro-F1=0.9614仅对应已保存内部验证窗口上的ST-GCN，不对应视频流程，也不代表独立泛化能力。

包中的 `normalization.pt` 保留该ST-GCN训练统计和配套LSTM的训练统计，便于复核。默认视频流程使用固定参考归一化，不能通过切换归一化选项直接沿用内部验证分数。

最终分类器SHA256：`956b4f50912fe0f716eba54cc7f0e19f939fb49980879aaf40d99c50227d918a`。

重载权重后的完整指标见 `final/detailed_metrics_20261029.json`；最终视频运行记录见 `final/video_check.json`。原始视频和数据集不上传。
