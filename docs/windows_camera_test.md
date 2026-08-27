# Windows 摄像头 Demo 验证报告 (windows_camera_test.md)

> 日期:2026-08-24 | 主机:Windows + NVIDIA RTX 3070 Ti(验证用;目标设备 RK3576 arm64)
> Demo:`style_reader/demo_camera.py`(复用 `process_image`,零算法改动)

## 1. 摄像头信息

| 项 | 值 |
|---|---|
| 索引 | 0(1/2 不可用) |
| 分辨率(读取) | 640 × 480 |
| 后端 | OpenCV VideoCapture(4.10+/5.0) |
| 打开 | ✅ `cap.isOpened()=True`,`read()` 成功 |

## 2. 实时帧性能(对空场景 120 帧)

| 指标 | 值 |
|---|---|
| 整体耗时 | 中位 10ms / 均值 25ms / P95 15ms |
| FPS | ≈40(空场景;YOLO无目标快速路径) |

**注意**:空场景/YOLO未检出时走最短路径;画面**有仪表**时额外交付 OCR(见§4)。

## 3. 成功识别样例(图片回放,12张 M01 代表)

| 图 | 读数 | 单位 | 总耗时 | YOLO | OCR | 掩膜 |
|---|---|---|---|---|---|---|
| M01-001 | 7.906 | bar | 8.3s | 1.9s | 4.6s | 0.8s |
| M01-003 | 30.081 | degC | 4.6s | 22ms | 2.9s | 0.3s |
| M01-005 | 1.307 | bar | 3.9s | 23ms | 2.2s | 0.2s |
| M01-006 | 34.276 | degC | 5.1s | 32ms | 3.4s | 0.2s |
| M01-007 | -0.581 | bar | 17.7s | 34ms | 16.2s | 0.3s |
| M01-008 | 32.137 | degC | 4.2s | 31ms | 2.7s | 0.2s |
| M01-009 | 5.530 | bar | 4.1s | 22ms | 2.8s | 0.2s |
| M01-010 | 6.008 | bar | 4.9s | 24ms | 2.4s | 0.3s |
| M01-011 | 1.330 | bar | 6.8s | 23ms | 5.5s | 0.2s |
| M01-012 | —(OCR 15.8s后失败) | — | 9.4s | 37ms | 8.1s | 0.2s |

**成功 11/12**(与离线批处理 m01_full8 结果一致);**瓶颈=OCR 多 pass**(空/低分时触发旋转+4变体=多次 engine 调用,单帧 2~16s)。

## 4. 分阶段耗时(均值,有仪表帧)

| 阶段 | 均值 |
|---|---|
| YOLO 检测(GPU) | 22-35 ms |
| 指针分割掩膜(onnx CPU) | 200-300 ms |
| **OCR(RapidOCR,多pass)** | **2000~16000 ms** ← 瓶颈 |
| 几何+映射+单位 | 其余 |

**FPS(有仪表)≈ 0.15~0.5**(受 OCR 支配)。

## 5. Demo 显示要素(截图验证)

窗口实时显示 ✅:仪表框 / 指针方向线(品红)/ 数值 / 单位 / family / confidence / mapping方法 / FPS+分阶段耗时;`q`/ESC 退出。
- 截图:`outputs/pointer_keypoints/demo_sample_unit.jpg`(M01-001:READ 7.906 bar,[ransac conf=0.65],family=circle,FPS 0.1 + 6346ms/帧)
- 图片回放:`--image-dir` 正常;摄像头模式:`--camera 0` 正常启动。

## 6. 启动命令

```powershell
.\scripts\run_camera_demo.ps1                          # 摄像头0, 640x480
.\scripts\run_camera_demo.ps1 -Camera 0 -Width 1280 -Height 720
.\scripts\run_camera_demo.ps1 -ImageDir assets\demo_images
# 等价直接:
python -m style_reader.demo_camera --camera 0 --width 640 --height 480
python -m style_reader.demo_camera --image-dir assets\demo_images
```

## 7. RK3576 优化必要性(结论)

| 项 | Windows GPU | RK3576(arm64 CPU)预期 | 建议 |
|---|---|---|---|
| YOLO | 22-35ms | >100-300ms(CPU) | **RKNN 量化(NPU,后续)** |
| 掩膜 | 200-300ms | 更慢 | 保留(差压关键)或降分辨率 |
| **OCR** | **2-16s(多pass)** | 更慢 | ①**单pass主检测+仅在低置信才旋转**(40%→1/3时间);②快速det/rec模型;③**模板命中跳过OCR**;④帧率上限/抽帧 |
| 实时性 | ≤0.5fps | 不达标 | **推荐:回放/定距采集+离线读数**;或 RKNN+单pass后 1-2fps |

**结论:识别正确性在 Windows 验证通过(11/12);实时性瓶颈=OCR多pass;RK3576 需(1) RKNN 化 YOLO/掩膜 (2) OCR 单pass化 (3) 必要时抽帧/回放兜底**——以上为后续优化,不属本轮(算法冻结)。
