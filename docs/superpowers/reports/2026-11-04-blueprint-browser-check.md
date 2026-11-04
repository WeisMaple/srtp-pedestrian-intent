# 蓝图长廊浏览器验收报告（2026-11-04）

本报告记录 Task U10 的真实浏览器验收。所有检查在本地真实权重、真实视频和实际前向事件上执行；没有使用录制回放、补造数值或模拟预测。

## 结论

工作台、六步长廊、数值台/放大镜与结果页的功能检查全部通过。暂停冻结、跟随最新、24 个 YOLO 模块与 10 个 ST-GCN 块的选择器、场景内点击模块/关节、1440×900、平板竖屏、减少动画和 WebGL 上下文丢失均实际验证。没有页面脚本错误；唯一控制台错误是浏览器请求缺失的 `favicon.ico`（404）。LSTM 与意图代表算式来自当前浏览器会话内缓存的真实事件；75 帧视频的详细采集只覆盖 18/24 个 YOLO 模块与 4/10 个 ST-GCN 块，未覆盖项按设计显示等待。

## 环境与输入

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-04 |
| 系统 | macOS 26.6.2 arm64 |
| 服务端 | `.venv/bin/python -s run_web_demo.py --port 8765 --model-dir weights/final` |
| 模型 | `weights/final/`（YOLO26s-Pose、ST-GCN epoch 121、LSTM epoch 68、`legacy_pair`） |
| 视频 | `artifacts/demo/demo_input.mp4`，75 帧，1280×720，30 fps |
| 浏览器 | Playwright 1.64.1 驱动 Google Chrome for Testing 156.0.8078.4（Chromium），headless |
| 主视口 | 1920×1080（另测 1440×900、834×1194 竖屏、1440×900 + `prefers-reduced-motion: reduce`） |
| 任务 ID | `98ea742842374969a511ac981a5d1092` |
| 截图目录 | `artifacts/browser-check/`（在 `.gitignore` 的 `artifacts/` 下，不提交） |

服务端使用最终构建产物 `web_demo/dist`。本轮处理指标：`frames=75`、`pose_observations=300`、`classification_updates=103`、`processing_seconds=11.70`、`processing_fps=6.41`（CPU、本机单次，不是准确率或通用速度承诺）。

## 检查结果

| # | 检查 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | 工作台上传真实视频、读取元数据、开始分析、运行进度与预览 | PASS | 任务创建成功；75 帧 1280×720 @30 fps；状态 running → completed |
| 2 | 长廊显示六个步骤；骨架与 ID 匹配实时快照 | PASS | 六步为视频帧 / YOLO 骨架 / ID 绑定 / LSTM 预测 / ST-GCN / 意图；UI ID 01 存在于真实 job 快照；17 个关节；历史 `ui=6 api=6` |
| 3 | 50 次观测预热；LSTM / ST-GCN / 意图填充 | PASS | 历史 50/50；三类分数 0.009 / 0.005 / 0.985；LSTM 标题“LSTM 隐状态解码”；ST-GCN 块详情与意图分类详情均有 8 行可见参与项 |
| 4 | 步骤 ② 点击模块、步骤 ⑤ 点击关节；放大镜显示完整参与项和实际输出 | PASS | 场景点击命中 YOLO 模块 16；实际输出 −0.25584，重建 −0.255838，误差 1.702e−6；关节点击命中第 3 块空间邻域，实际输出 0.49803，重建 0.498024，误差 6.120e−6；计数显示 288 项 / 51 项 |
| 5 | 暂停冻结数值且分析继续；「跟随最新」恢复 | PASS | 冻结时历史保持 `16/50` 2.5 秒，后台处理帧从 24 增至 50；恢复后历史到 50/50 |
| 6 | 1440×900、平板竖屏、减少动画、WebGL 丢失 | PASS | 两种视口水平溢出 0 px；平板六步按钮各 90 px 且有画布；减少动画下无脚本错误；WebGL 丢失显示“三维渲染上下文已中断，数值台仍可使用”，数值台保留 |
| 7 | 24 个 YOLO 模块与 10 个 ST-GCN 块可达 | PASS | 选择器分别有 24 / 10 个 option，逐项选择均生效；本轮详细覆盖 18 / 24 与 4 / 10，未覆盖项显示等待 |
| 8 | 结果页视频与下载 | PASS | 标注 MP4、逐帧 JSONL、完整 ZIP 三个下载；结果视频可加载 |
| 9 | 控制台与脚本错误 | PASS（1 个已知 404） | 无 `pageerror`；控制台仅有 `GET /favicon.ico` 404；HTTP 响应无其他 4xx/5xx |

### 细节补充

- 暂停后冻结的是当前显示快照：上下文条、三维层和数值台读取同一 held snapshot；后台 SSE 仍继续接收，处理帧继续增加。跟随最新清除 held snapshot。
- 3D 命中使用真实场景射线检测：步骤 ② 的特征平面映射到模块选择器，步骤 ⑤ 的骨架点映射到当前 ST-GCN 块。未采集详细算式的模块/块会打开“等待详细采集”的放大镜，不借用其他块数值。
- 本轮 YOLO 详细覆盖：0–10、13、16、17、19、20、22、23（18/24）。ST-GCN 详细覆盖：2、3、4、5（4/10）。75 帧在 1 fps 详细节奏下的覆盖上限如此，完整轮需要 12 / 20 个详细帧。
- 浏览器会话内的 LSTM 解码与意图分类代表算式按真实事件 id 缓存最新一条。刷新后这些未持久化的事件会回到等待状态；YOLO/ST-GCN 详细行由 `live.json` 的 `details` 持久化。

## 截图

所有 PNG 位于 `artifacts/browser-check/`：

- `00-workbench-empty-1920.png`：工作台空态
- `01-workbench-uploaded-1920.png`：上传后灯箱与开始按钮
- `02-workbench-running-1920.png`：运行进度
- `03-corridor-step1-1920.png`：长廊步骤 ①
- `04-corridor-step2-1920.png`：步骤 ②，模块选择器与详细采集进度
- `05-corridor-frozen-1920.png`：暂停冻结
- `06-corridor-follow-latest-1920.png`：跟随最新恢复
- `07-corridor-step3-warm-1920.png`：50/50 历史、ID、分数与当前骨架
- `08-corridor-step4-lstm-1920.png`：LSTM 步骤与真实算式
- `09-corridor-lens-module-1920.png`：点击模块后的放大镜
- `10-corridor-step5-stgcn-1920.png`：ST-GCN 块选择与骨架
- `11-corridor-lens-joint-1920.png`：点击关节后的放大镜
- `12-corridor-step6-intent-1920.png`：意图步骤与分类输出
- `13-results-1920.png`：结果页与三个下载
- `14-corridor-1440x900.png`：1440×900 长廊
- `15-corridor-tablet-portrait.png`：834×1194 平板竖屏长廊
- `16-workbench-tablet-portrait.png`：834×1194 工作台
- `17-corridor-reduced-motion-1440.png`：减少动画
- `18-webgl-context-lost-1920.png`：WebGL 上下文丢失回退

## 控制台与网络

```
console: Failed to load resource: the server responded with a status of 404 (Not Found)
```

服务端日志对应 `GET /favicon.ico HTTP/1.1 404 Not Found`。页面没有其他 console error、没有 `pageerror`，Playwright 响应监听没有记录其他 4xx/5xx。该 404 不影响功能。

## 未验证 / 限制

- 没有在真实教室投影仪、其他 GPU 或显卡驱动上测试；本轮是 macOS arm64 上的 headless Chromium，Three.js 实际使用软件/系统 WebGL。
- 未测试 Safari、Firefox、移动端真实设备或触控手势。
- 分类分数仅为模型输出，本报告不评估道路准确率，也没有重跑精度指标。
- 75 帧视频没有完成 24 模块与 10 块的全部详细覆盖；未覆盖项只显示等待，未用其他模块/块数值替代。
- 深块边界的完整 20 帧周期（约 60 秒）和所有行人的边界轮转未在本轮 75 帧短片中走完。
- 关节点击打开的是该块已采集代表算式（固定代表关节/通道/时刻）；未实现前端对任意关节的完整算式重定向。
- 实时快照不包含完整 50 帧关键点数组，因此 ID 绑定展位显示当前真实骨架与历史计数，不伪造 50 帧历史骨架带。
- 刷新后未持久化的 LSTM/意图代表算式不会从旧会话恢复；页面只显示新收到的事件或已保存的 YOLO/ST-GCN 详细行。
- 1440×900 与平板竖屏只做了布局与可操作性检查，没有做像素级视觉回归。
