# 最终模型 v2.0.0

| 组件 | 配置 |
|---|---|
| 人体姿态估计 | YOLO26s-Pose，输入长边640，检测阈值0.1 |
| 跟踪 | ByteTrack，17关键点，50次历史观测 |
| LSTM | 10层、隐藏256、每次输出10步，递归5轮；短期13轮＋递归13轮 |
| ST-GCN | 原10个时空卷积块；平方根倒数类别权重，第28轮 |
| 数值变换 | `training_stats`；`last_observation_relative`，每轮更新最后观测参照 |
| 设备 | YOLO自动选择GPU；网页与视频演示的行为模型使用CPU |

模型包同时包含 `model_config.json`、`normalization.pt`、三组权重和来源说明。`normalization.pt` 中的 `lstm_representation` 明确指定相对表示，避免使用绝对坐标解码。逐轮参照骨架也传入可视化记录，因此记录讲解和实时采样显示同一套计算。

校验清单见 `final/manifest.json`；模型包网址与SHA256见 `final/release_asset.json`。部署后与所选实验实现比较8个真实输入，类别、概率、未来骨架完全一致，见 `final/v2_20261102/runtime_parity.json`。

验证准确率88.86%，测试准确率82.40%、Macro-F1 0.5368、覆盖率35.95%；样本和评价范围见 [最终实验记录](final/v2_20261102/README.md)。
