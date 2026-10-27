# 外置模型与数据

代码仓库不上传模型、视频、训练张量、压缩包或环境目录。本机可运行资源保存在以下位置。

现代资源根目录：`D:\SRTP_modern_20261027`；迁移时可通过 `SRTP_MODERN_HOME` 指定新的根目录。将 `weights/` 四个文件复制过去；`runtime/site-packages` 可以复制或根据锁定清单重装。

| 文件 | 字节数 | SHA256 |
| --- | --- | --- |
| `lstm_reproduced_wd0.pt` | 20743635 | `de4e55496a3fc9e36b764b814be584f34cfb5bdd893f03c94f842cdfa3ffa6b1` |
| `normalization.pt` | 22784 | `90ec4c7a6315d713e0decd60d49f782313d5351f075424300ef38ea452d41d7c` |
| `stgcn_reproduced.pt` | 12387146 | `cb65b1148fc64f5034452f74b890d301c99a581dca335efdbceb5fd1566a656a` |
| `yolo26s-pose.pt` | 24151790 | `a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646` |

YOLO 官方下载地址：
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt

行为权重对应：

- `stgcn_reproduced.pt` ← `D:\SRTP_reproduction_20261027\models\stgcn\final_state_dict.pt`
- `lstm_reproduced_wd0.pt` ← `D:\SRTP_reproduction_20261027\models\lstm_wd0\final_state_dict.pt`
- `normalization.pt` 的键为 `stgcn_mean`、`stgcn_std`（形状 1×3×50×17×1），以及 `lstm_mean`、`lstm_std`（17×3）。统计来自上轮旧前端特征重建，并非新版 YOLO 骨架重新拟合。

旧流程另需项目 `Models/` 与 `checkpoints/` 内的原始文件，包括 YOLOv7x、ReID.pb、YOLOv8n-pose 和原行为模型。这些文件均保留在本机，未上传。

JAAD 视频在本机 `F:\Artificial_Intelligence\srtp\data\JAAD_clips`；压缩包在 `D:\下载\JAAD_clips.zip`。官方标注来源：https://github.com/ykotseruba/JAAD 。本仓库不重新分发视频、标注或论文附件。
