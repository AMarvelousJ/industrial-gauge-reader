# 指针式仪表读数开源方案调研

核验日期：2026-08-21。仓库状态、默认分支提交时间和许可证均通过 GitHub 仓库页、提交 Atom feed 与仓库内许可证文件现场核验；没有许可证文件的项目不复制代码。

## 结论

最值得复用的是 ETH Zurich 的完整流水线设计，以及 Gauge-Pointer-Reading 的预训练指针分割与椭圆刻度弧几何。本项目已采用可审计的 `冻结 YOLO 仪表定位 → 预训练指针掩膜/OpenCV 回退 → RapidOCR → 圆周刻度拟合 → 数值和单位`；第三方模型仅推理，没有训练或修改。透视椭圆归一化仍是下一阶段重点。

## 仓库对比

| 仓库 | 许可证 | 最后提交活动 | 可直接借鉴 | 适用边界 |
| --- | --- | --- | --- | --- |
| [ethz-asl/analog_gauge_reader](https://github.com/ethz-asl/analog_gauge_reader) | MIT；仓库有 `LICENSE.md` | 2024-05-13，158 commits | `geometry/ellipse.py`、`warp_ellipse.py` 的椭圆与透视处理；`ocr/ocr_reading.py` 的数字框投影/读数组织；`angle_reading_fit` 的角度-数值拟合；逐阶段 `result.json`/`error.json` 诊断结构 | 功能最完整，但原环境锁定 Python 3.8、PyTorch 2.0、MMOCR/MMDetection，不能整包直接搬到当前轻量环境 |
| [MaomaoMAo-17/Gauge-Pointer-Reading](https://github.com/MaomaoMAo-17/Gauge-Pointer-Reading) | MIT；仓库有 `LICENSE` | 2025-05-09，8 commits | `predict.py` 中刻度点聚合、`cv2.fitEllipse`、按最大角度间隙确定刻度弧首尾、指针角度映射为弧上比例 | 需要其 YOLOv11 分割模型；只处理一根指针和一段刻度弧，起止规则仍带经验性 |
| [dgomes/analog-gauge-reader](https://github.com/dgomes/analog-gauge-reader) | MIT（Intel 2014-2017）；仓库有 `LICENSE` | 2019-10-03 | Hough 圆、Hough 线、中心到指针端点角度、手工起止角/量程映射；适合作为纯 OpenCV 基线和单元测试参照 | 长期未更新，固定半径和手工标定较多；不适合作为陌生仪表泛化方案 |
| [mc260/meter-vision](https://github.com/mc260/meter-vision) | **未找到仓库 LICENSE/LICENSE.md**；README 的许可文字不能替代许可证文件 | 2026-08-04，近期活跃 | 仅借鉴结构：中心/指针/刻度关键点、透视鲁棒的分段角度插值、FastAPI 与 MJPEG 演示层 | 未明确授权前不复制代码；固定关键点/固定量程也不适合未知刻度数 |

`Gauge-Pointer-Reading` 已连同原始 MIT LICENSE 克隆到 `third_party/Gauge-Pointer-Reading/`，并下载仓库 README 指向的 `scale_segment.pt` 做兼容性实测。审计脚本为 `python -m style_reader.evaluate_third_party_segmentation`；它只统计预训练模型在 20 图上是否同时产生 pointer 与 scale 掩膜，不训练模型。实测结果：pointer 覆盖率 `90%`，scale 覆盖率仅 `35%`，两者同时满足、可直接进入上游椭圆几何的比例也是 `35%`。因此该预训练分割模型不能直接作为 20 图通用解，但它的指针分割可作为 OpenCV 指针候选的互补分支，椭圆几何仍值得复用。

## 本项目采用与暂缓的模块

### 立即实现

1. 使用用户提供的 `best.pt` 做冻结单类检测，只输出仪表 ROI；不训练或微调 YOLO。
2. ROI 灰度化、CLAHE、反色二值图和 Canny 边缘。
3. Hough 圆候选估计盘面中心/半径；失败时使用 ROI 几何中心并降低置信度。
4. 概率 Hough 线段按“靠近中心、长度、径向一致性、是否到达外环”评分，选择指针候选。
5. 统一角度定义：表盘正上方为 `0°`，顺时针增加。
6. 输出原图检测框、ROI、圆心、指针端点、角度、候选线、失败原因及可视化图片。

### 已补齐

1. RapidOCR 数字框、旋转共识补识和数字位置投影，拟合 `value=f(angle)`。
2. 单/双刻度半径聚类、跨 0°/360° 展开、伪数字 RANSAC 离群剔除。
3. 以审计后的 MD 人工读数评测误差，独立报告 coverage 和 accuracy。
4. 黑针预训练分割优先、Hough/径向暗度回退；红色读数标记使用 HSV 细长分量。

### 仍需继续

1. 参考 Gauge-Pointer-Reading 与 ETHZ 加入椭圆/透视归一化。
2. 对方形表、偏心轴和遮挡轴心做专门几何模型。
3. 扩充独立训练/验证集，验证未知品牌和未知量程，而不是继续调 20 张闭集图片。

## 复用原则

- 当前实现为独立重写，只采用公开算法思路，没有复制第三方源码。
- 后续若复制 MIT 项目代码，必须保留其版权与许可证文本。
- `meter-vision` 在许可证缺失状态下只作概念参考。
- 任何读数都必须保留中间几何证据；低置信度输出失败状态，不能伪造数值。
