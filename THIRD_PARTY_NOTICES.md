# Third party notices

This repository preserves recovered project code and adapted third-party components. Existing copyright and source comments remain intact. No blanket license is assigned to the entire mixed codebase by this upload.

| Component | Upstream | Upstream license | Local scope |
| --- | --- | --- | --- |
| YOLOv7 | https://github.com/WongKinYiu/yolov7 | GPL-3.0 | `yolo/`, adapted legacy detection wrapper |
| DeepSORT | https://github.com/nwojke/deep_sort | GPL-3.0 | `deepsort/`, legacy tracking wrapper |
| ST-GCN | https://github.com/yysijie/st-gcn | BSD-2-Clause | Architecture provenance for the recovered `st_gcn.py`; project adaptation and LSTM additions retained |
| Ultralytics | https://github.com/ultralytics/ultralytics | AGPL-3.0 or upstream commercial license | External installed dependency, not vendored here |

Upstream license texts for the retained source components are in `third_party_licenses/`. `sources.json` records the official download URLs and SHA256 values. The exact commits used by the recovered legacy adaptation are unavailable; the original files identify their upstream projects.

YOLO weights, JAAD data, runtime packages and pretrained behavior weights are excluded from this repository. Their own source and usage terms apply separately.
