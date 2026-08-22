# 工业单针轴心/针尖关键点 V1

该模块训练独立的两关键点姿态模型，不训练或修改现有仪表定位`best.pt`。两个关键点依次为轴心`pivot`和主测量针针尖`pointer_tip`。

## 1. 审核队列

当前已经生成`outputs/pointer_keypoint_review_v1`：240张目标样本加80张备用样本，原80条人工审核记录按`record_id`合并保留。打开新的审核队列：

```powershell
.\scripts\run_keypoint_annotation_ui.ps1
```

默认使用`127.0.0.1:8766`，避免与旧的8765端口冲突。每条训练样本需要：

- `scope_status=in_scope`且审核状态为`accepted/corrected`；
- 轴心与人工针尖坐标；
- 仪表族、物理仪表ID、采集条件、训练轨道、来源组、品牌和型号；
- 原有读数、单位、量程等完成字段。

暂缓类型选择对应`deferred_*`范围状态并保持暂存。训练轨道的70/30预填值只是候选分配，不是人工真值。

## 2. 数据导出

先检查是否达到240条完整真值：

```powershell
.\.venv\Scripts\python.exe -m pointer_keypoints.dataset `
  --review-csv outputs\pointer_keypoint_review_v1\review.csv `
  --manifest outputs\pointer_keypoint_review_v1\review_manifest.json `
  --validate-only
```

准备完成后去掉`--validate-only`。默认生成180/30/30的训练、验证、冻结测试拆分，业务优先/泛化保护为168/72。拆分按物理仪表、来源组、重复簇和品牌型号的连通组隔离。仅训练集额外生成30%的模糊、反光和弱光派生图。

## 3. 训练、阈值校准和冻结测试

```powershell
.\.venv\Scripts\python.exe -m pointer_keypoints.train

.\.venv\Scripts\python.exe -m pointer_keypoints.evaluate `
  --weights runs\pointer_keypoints\industrial_single_pointer_v1\weights\best.pt `
  --split val --calibrate `
  --output outputs\pointer_keypoints\val_calibration.json

.\.venv\Scripts\python.exe -m pointer_keypoints.evaluate `
  --weights runs\pointer_keypoints\industrial_single_pointer_v1\weights\best.pt `
  --split test `
  --threshold-file outputs\pointer_keypoints\val_calibration.json `
  --output outputs\pointer_keypoints\frozen_test.json
```

默认使用`yolov8n-pose.pt`、384输入、150轮、patience 30和固定种子20260822。若训练目录已存在会拒绝覆盖。

## 4. 读数链集成

```powershell
.\scripts\run_style_reader.ps1 `
  -PointerKeypoints runs\pointer_keypoints\industrial_single_pointer_v1\weights\best.pt `
  -KeypointThresholdFile outputs\pointer_keypoints\val_calibration.json
```

没有提供关键点权重时读数器保持原有行为。关键点低置信或长度/边界不合理时记录拒绝原因并回退旧几何路径。`predictions.json` 1.0接口不变，诊断中增加关键点、置信度、派生角度和`geometry_source`。
