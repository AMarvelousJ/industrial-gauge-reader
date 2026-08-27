# Ubuntu 边缘设备部署检查清单 (deployment_checklist.md)

> 来源工程:`D:\tiaozhanbei\yolo`(Windows 开发环境)
> 目标:Ubuntu 24.04 边缘设备(摄像头已到位;GPU 情况待确认——按 CPU-only 与 GPU 两种档位准备)
> 上下文:M01 139/164、20闭集逐位一致、pytest 129 全绿(当前算法冻结)

## 1. Python 版本要求

| 项 | 开发环境(Windows) | Ubuntu 建议 |
|---|---|---|
| Python | 3.12.13 | **3.10 ~ 3.12 均可**(推荐 3.12;ultralytics/rapidocr/opencv 均支持) |
| 虚拟环境 | `.venv\Scripts\python.exe` | `.venv/bin/python`(venv 布局不同,脚本需区分) |
| venv 创建 | `py -3.12 -m venv .venv` | `python3.12 -m venv .venv` |

**风险**:代码中 PowerShell 脚本(`scripts/*.ps1`)硬编码 `.venv/Scripts/python.exe` → Ubuntu 必须用等价 bash/sh 封装(第二阶段新增 `setup_ubuntu.sh`)。

## 2. pip 依赖(开发机实际冻结)

| 包 | 版本(Windows) | Ubuntu 注意 |
|---|---|---|
| torch | 2.9.1+**cu126** | **必须替换**:cu126 轮子仅 Linux/CUDA 12.6;CPU 设备用 `torch==2.9.1`(CPU版,`pip install torch --index-url https://download.pytorch.org/whl/cpu`) |
| torchvision | 0.24.1+cu126 | 同上,与 torch 配对 |
| ultralytics | 8.4.106 | 通用;首次运行会自动拉取 `yolov8n-pose.pt`(若有网络) |
| rapidocr | 3.8.1 | **onnxruntime CPU 推理**,无需 CUDA;包内自带 7 个 ONNX 模型(见 §5) |
| opencv-python | 5.0.0.93 | **风险点**:5.0 极新,ARM64/aarch64 wheel 可能缺;候选 `opencv-python-headless` 或 4.10.x(边缘无 GUI) |
| numpy | 2.5.2 | 与 opencv/ultralytics 兼容即可;若 opencv 降级需复测 |
| onnxruntime | 1.22.1 | RapidOCR 用;CPU 版即可 |
| PyYAML / matplotlib / pillow / tqdm / requests | — | 通用 |
| **openpyxl** | 3.1.5 | `source_links.xlsx` 读取(仅数据处理;**Demo 不需要**) |
| pytest | 8.4.2 | 仅测试 |

**缺口记录**:`pandas` 未安装(读 xlsx 用 openpyxl,保持一致)。

## 3. CUDA / CPU 依赖

| 场景 | 依赖 | 说明 |
|---|---|---|
| 开发机 | CUDA 12.6, RTX 3070 Ti | 已有 |
| **边缘(推荐 CPU 档)** | torch CPU、onnxruntime CPU | **YOLO 推理在 CPU 上慢**:8.3 GFLOPs 的 yolov8n 在 RK 级 CPU 上单帧可能 >300ms——**性能目标是第四阶段核心指标**,第一版 Demo 以"能跑通"为准 |
| 边缘(若有 NPU/GPU) | torch 对应轮子 | 需确认设备型号(RK3568/3576 有 NPU,但 ultralytics 不走 NPU;需 RKNN 导出=后续工作,不在本阶段) |

**检查方式**:`python -c "import torch; print(torch.cuda.is_available())"`。

## 4. YOLO 模型文件位置(推理必需)

| 文件 | 大小 | 用途 |
|---|---|---|
| `runs/detect/meter_yolov8n_final/weights/best.pt` | 6.0 MB | **冻结单类仪表检测器**(唯一必需;SHA256 固定 `A0447F65...`) |
| `runs/pose/runs/pointer_keypoints/industrial_single_pointer_v1/weights/best.pt` | 6.2 MB | 关键点模型(**仅诊断**,`--pointer-keypoints` 可选参数;Demo 可省以提速) |
| `third_party/Gauge-Pointer-Reading/scale_segment.pt` | **238 MB** | 指针分割掩膜(**管线必用**,对差压/方形表关键;**边缘传输最大项**) |
| `outputs/style_classifier/best.pt` | 43 MB | 样式分类器(**当前管线未使用**,可不同步) |
| `yolov8n-pose.pt` | — | 训练基础模型(**推理不需要**;勿拷贝) |

**传输体积**:核心推理三件套 ≈ 250 MB。

## 5. OCR 模型文件位置

RapidOCR 模型**随 pip 包安装**在:

```
.venv/lib/python3.12/site-packages/rapidocr/models/
  ch_PP-OCRv4_det_infer.onnx / det_mobile.onnx
  ch_PP-OCRv4_rec_infer.onnx / rec_mobile.onnx
  ch_ppocr_mobile_v2.0_cls_infer.onnx / cls_mobile.onnx
```

- 无需额外拷贝;**若 `--no-cache` 安装失败**,首跑时 RapidOCR 会尝试联网下载。
- 运行时默认 CPU(onnxruntime),**零 CUDA 依赖**——边缘友好。

## 6. 权重文件清单(汇总)

| 用途 | 路径 | 必须 |
|---|---|---|
| 仪表检测 | runs/detect/meter_yolov8n_final/weights/best.pt | ✅ |
| 指针掩膜 | third_party/Gauge-Pointer-Reading/scale_segment.pt | ✅(238MB) |
| 关键点(诊断) | runs/pose/runs/pointer_keypoints/industrial_single_pointer_v1/weights/best.pt | 可选 |
| 关键点阈值 | outputs/pointer_keypoints/val_calibration.json | 可选(伴随关键点) |

## 7. 配置文件

| 文件 | 用途 | Ubuntu 需要 |
|---|---|---|
| `docs/reading_images.json` | 评测图片清单 | 可选(回放测试) |
| `all_set/` | 原始数据(947张) | Demo 回放需要;边缘可仅带 M01 子集 |
| `outputs/pointer_keypoints/val_calibration.json` | 关键点阈值(0.75) | 关键点启用时 |
| `requirements.txt` | 需**重写为 Ubuntu 版**(见 §2) | ✅ 新建 ubuntu 分支 |

**硬编码检查**:代码中无 `D:\`/`C:\` 绝对路径(已扫描通过);图片路径用 `/` 且代码内 `replace(chr(92),'/')` 已兼容。

## 8. Windows 特有路径 / 依赖

| 项 | 处理 |
|---|---|
| `.venv\Scripts\python.exe` | → `.venv/bin/python`(所有 PS1 需 bash 等价物) |
| PowerShell 脚本(5个 ps1) | Ubuntu 用 `setup_ubuntu.sh` + 等价 sh 命令 |
| 路径分隔符 `\` | 代码已 `replace(chr(92),'/')`;`Path` 库跨平台 |
| OpenCV 摄像头(V4L2) | 需要 `sudo usermod -aG video $USER`;确认 `/dev/video0` 权限 |
| 中文字符集 | 控制台需 UTF-8;文件均 utf-8 编码 ✓ |
| 磁盘 | 工程 7.2GB(含 runs 历史)→ 边缘只拷代码+3模型+必要数据 ≈ **300MB** |

## 9. 风险清单(按优先级)

| # | 风险 | 证据/影响 | 缓解 |
|---|---|---|---|
| 1 | **CPU 上 YOLO+OCR 帧率** | yolov8n 8.3 GFLOPs;RK级CPU单帧难<300ms | 第四阶段实测;Demo第一版"可用"优先;回放模式兜底 |
| 2 | **torch cu126 轮子不可用** | Windows 定制索引 | Ubuntu 用 CPU 版 torch(边远无 GPU) |
| 3 | **opencv-python 5.0.0.93 无 ARM wheel** | 牛牛牛 | 备选 headless/4.10.x,复测全部功能 |
| 4 | **scale_segment 238MB 传输** | 网络 | 单独 scp;或首次启动提示 |
| 5 | 摄像头权限 / V4L2 | `/dev/video0` | video 组权限;回放模式降级 |
| 6 | RapidOCR 首跑联网 | 离线环境 | 预拷 onnx/离线 wheel |
| 7 | ultralytics 版本行为差异(Windows vs Linux) | 微 | 冻结 8.4.106 |

## 10. 建议的 Ubuntu 部署目录

```
/home/<user>/gauge_demo/
  style_reader/ pointer_keypoints/ third_party/  (仅所需py)
  runs/detect/.../best.pt
  third_party/Gauge-Pointer-Reading/scale_segment.pt
  docs/reading_images.json (可选)
  assets/demo_images/ (回放图集)
  .venv/
```

---

*检查完毕。下一阶段:`scripts/setup_ubuntu.sh`(第二阶段)+ `style_reader/demo_camera.py`(第三阶段)+ `docs/edge_benchmark.md`(第四阶段)——均遵循"不改核心算法"红线。*
