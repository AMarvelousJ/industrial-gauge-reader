# RK3576 Ubuntu 离线部署包设计 (rk3576_deployment_package.md)

> 目标:Windows 开发机整理 `gauge_demo_rk3576_ubuntu.zip`,U盘拷贝至 RK3576(Ubuntu 24.04, arm64)。
> 原则:**不改任何算法逻辑**;仅 环境适配+输入输出封装+Demo 展示。
> 本文件 = 结构设计 + 文件清单 + 体积估计 + 风险(打包执行另起任务)。

## 1. 包目录结构

```
gauge_demo_rk3576_ubuntu/
├── README.md                     # U盘复制三步说明(解压→install→run_demo)
├── install.sh                    # 创建venv+离线装依赖+环境自检
├── run_demo.sh                   # 启动摄像头Demo(无摄像头自动图片回放)
├── requirements_rk3576.txt       # ARM64 CPU 专用(无CUDA)
├── wheels/                       # 离线ARM64 wheel(见§4,关键!离线必需)
│   ├── torch-*.whl  torchvision-*.whl
│   ├── opencv_python-*.whl  onnxruntime-*.whl
│   └── (其余纯python包:pip install时自动装)
├── src/                          # 完整Python源码(只含运行必需)
│   ├── style_reader/             # 读水管线(核心,599KB)
│   ├── pointer_keypoints/        # 关键点(诊断,可选运行,182KB)
│   ├── data_premark/             # 标注/预标记(打包备查,258KB)
│   ├── third_party/              # 掩膜代码+LICENSE(模型单独放models/)
│   ├── scripts/                  # (sh版入口;原ps1另存,不用)
│   └── docs/reading_images.json  # 评测/回放清单(可选)
└── models/                       # 推理模型(250MB)
    ├── meter_detector.pt         # 冻结YOLO检测器(6.0MB,SHA256校验值写入README)
    ├── scale_segment.pt          # 指针分割掩膜(238MB,最大项)
    ├── pointer_keypoints.pt      # 关键点(6.2MB,可选诊断)
    ├── keypoint_threshold.json   # 关键点阈值0.75(可选)
    └── ocr_models/               # RapidOCR onnx(31MB,替代pip内置,离线兜底)
        ├── ch_PP-OCRv4_det_mobile.onnx
        ├── ch_PP-OCRv4_rec_mobile.onnx
        └── ch_ppocr_mobile_v2.0_cls_mobile.onnx
```

## 2. 需要复制的文件列表(含大小)

| 类别 | 路径(源) | 大小 | 说明 |
|---|---|---|---|
| 源码 | style_reader/、pointer_keypoints/、data_premark/ | ~1.1MB | 完整模块;`__pycache__` 剔除 |
| 掩膜代码+许可 | third_party/Gauge-Pointer-Reading/{LICENSE,README*.md,predict.py} | ~100KB | `data_valid/`、`scale_segment.pt` 不入src(模型见models) |
| 检测器 | runs/detect/meter_yolov8n_final/weights/best.pt | 6.0MB | → models/meter_detector.pt |
| 掩膜模型 | third_party/Gauge-Pointer-Reading/scale_segment.pt | 238MB | → models/scale_segment.pt |
| 关键点(可选) | runs/pose/runs/pointer_keypoints/industrial_single_pointer_v1/weights/best.pt | 6.2MB | → models/pointer_keypoints.pt |
| 关键点阈值 | outputs/pointer_keypoints/val_calibration.json | 1KB | → models/keypoint_threshold.json |
| OCR模型 | .venv/.../rapidocr/models/(mobile三件套) | ~17MB | → models/ocr_models/(防离线首跑联网) |
| 脚本(新) | install.sh、run_demo.sh、requirements_rk3576.txt、README.md | ~15KB | 本阶段新增 |
| 回放图 | all_set/M01/images 或精选16张 | 可选 | run_demo --image-dir 需要 |
| **ARM64 wheels**(新增) | wheels/ 见§4 | ~300MB | **离线安装前提** |

## 3. 压缩包大小估计

| 构成 | 大小 |
|---|---|
| 源码+脚本+配置 | ≈ 2 MB |
| 模型(含OCR) | ≈ 267 MB |
| **ARM64 wheels(torch CPU ~190MB + torchvision ~18MB + opencv ~60MB + onnxruntime ~15MB)** | ≈ 285 MB |
| 压缩后(zip,模型/wheel已高压缩) | **≈ 420~480 MB**(U盘传输无压力) |
| 若**RK3576可联网**(仅首次pip装纯python包):可只带 models+源码+wheels减至torch/opencv/onnx ≈ 260MB | ≈ **280 MB** |

## 4. 依赖矩阵(requirements_rk3576.txt 草案,arm64 CPU)

```
--index-url https://download.pytorch.org/whl/cpu
torch==2.4.1            # arm64 CPU wheel(官方PyTorch提供;2.9.1+cu126 不可用,无需CUDA)
torchvision==0.19.1
opencv-python==4.10.0.84 # arm64 wheel;5.0.0.93 无ARM轮子风险
onnxruntime==1.19.2      # arm64 官方wheel
rapidocr==3.8.1          # pure-python+onnxruntime,模型由包/离线目录提供
ultralytics==8.4.106     # pure-python;依赖numpy等自动
numpy==1.26.4            # arm64 wheel稳定;2.5.2兼容性风险
PyYAML, pillow, tqdm, requests, matplotlib(可选)
openpyxl==3.1.5          # 仅数据处理;Demo可省
```

**额外要放入包的 wheel**(pip 无法源码构建的二进制):
`numpy`、`opencv_python`、`onnxruntime`、`torch`、`torchvision`(从各自 arm64 源下载,Windows 阶段执行)。

## 5. deploy 包配套脚本(设计)

**install.sh**:检查 python3.12/uname -m=arm64→`python3 -m venv .venv`→`pip install --no-index --find-links wheels/ -r requirements_rk3576.txt`(或联网fallback)→自检:torch/opencv/onnxruntime版本+`import ultralytics, rapidocr`→校验模型文件存在+SHA256(meter_detector)。

**run_demo.sh**:`source .venv/bin/activate`→`python demo_camera.py`(默认摄像头;检测失败自动 `--image-dir assets/demo_images` 回放)。

**demo_camera.py**(第三阶段实现,本设计引用):摄像头→YOLO→几何→指针→OCR→mapping→unit→Overlay(框+指针+值+单位+置信+FPS);`--image-dir` 回放模式。

## 6. RK3576 部署风险(重点)

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| 1 | **arm64 二进制 wheel 获取/正确版本** | 🔴高 | 打包前必须验证 torch/opencv/onnxruntime arm64 wheel 可下载;torch 2.4.1 是 arm64 官方线,避免追新 |
| 2 | **CPU 单帧时延**(YOLO 8.3GFLOPs+RapidOCR,无GPU/NPU直通) | 🔴高 | 实测为第四阶段核心;Demo"跑通"优先,`--image-dir` 兜底;RKNN 量化=后续 |
| 3 | **238MB 掩膜加载内存/耗时** | 🟡中 | 启动一次加载;内存建议 ≥4GB |
| 4 | **opencv 摄像头 V4L2 权限** | 🟡中 | `sudo usermod -aG video $USER`;回放模式兜底 |
| 5 | **GLIBC/系统库**(onnxruntime/opencv 需 glibc≥2.28) | 🟡中 | Ubuntu 24.04(glibc 2.39)满足 |
| 6 | 磁盘空间(Python+依赖 ≈1.5GB,含wheel) | 🟢低 | 建议预留 3GB |
| 7 | 中文字符集/路径 | 🟢低 | UTF-8;代码已 `replace(chr(92),'/')` |
| 8 | ultralytics 首次运行联网(下载yolov8n-pose) | 🟢低 | Demo不用pose;`YOLO_FORCE_OFFLINE`?模型本地白名单 |

**结论**:包体 ~420-480MB(含wheel)或 ~280MB(设备可联网);**两大硬风险=arm64 wheel 供给 + CPU 帧率**;其余均属常规。设计不变更任何算法逻辑。

## 7. 下一任务(确认后执行)

1. **裁剪打包**:生成 `gauge_demo_rk3576_ubuntu/` 目录树并填充(源码+模型+OCR onnx+脚本+wheel下载)
2. 写 `install.sh`/`run_demo.sh`/`README.md`/`requirements_rk3576.txt`
3. 在 Windows 上**模拟验证**(python venv 装 arm64 不可行——以"文件完整性+语法+脚本逻辑"验证)
4. 输出 zip + SHA256 校验文件
