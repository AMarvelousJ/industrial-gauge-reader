# YOLO 模型资产审计

审计日期：2026-08-21  
审计范围：用户提供的 `best.pt`、现有 YOLO 工程配置、训练结果与检测裁剪接口。

## 1. 资产身份与完整性

用户提供的 `best.pt` 已作为项目的规范检测器资产保存于：

```text
D:\tiaozhanbei\yolo\runs\detect\meter_yolov8n_final\weights\best.pt
```

该文件的 SHA256 为：

```text
A0447F659564955C0FFBCD7BD68394745C9C3CA5686117EFBB41A631CE79E1A1
```

项目推理链路引用的正是上述规范路径，不存在另行微调或替换后的检测权重。因此，用户 `best.pt` 与项目实际使用的检测权重是同一模型资产，SHA256 一致。后续即使复制或改名，也只能在 SHA256 仍等于上述值时称为同一资产。

## 2. 能力边界

现有 YOLO 是单类目标检测器，而不是仪表样式分类器或仪表读数模型：

```yaml
names:
  0: meter
```

它只负责定位图像中的仪表并输出 `meter` 框、置信度和裁剪区域。它不能直接输出 M01、M02 等样式，也不能输出表盘示值。检测 mAP 不得作为样式分类准确率或读数准确率报告。

本项目对该检测器采用“冻结推理”策略：

- 只加载上述 `best.pt` 执行推理；
- 不调用检测器训练或微调流程；
- 推理时使用 evaluation mode 和 no-grad/inference mode；
- 不用 Markdown 测试图片的人工框代替检测结果；
- 检测失败必须按失败计入端到端评测，不得静默回退到 GT 框或整图。

独立的样式分类器可以训练，但不能反向更新或覆盖该 YOLO 权重。

## 3. 已有检测指标

`runs/detect/meter_yolov8n_final/results.csv` 显示，120 轮训练中验证集 `mAP50-95` 最佳点在 epoch 95：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.98648 |
| Recall | 0.96117 |
| mAP50 | 0.99178 |
| mAP50-95 | 0.91986 |

这些数值来自训练过程的 **validation split**。现有 `runs/detect/runs/evaluate/meter_yolov8n_final_test/` 仅保留可视化图片，没有独立的 `results.csv` 或其他机器可审计指标文件。因此，不得将上表描述为独立 test 指标，也不得据此声称对新仪表样式的泛化准确率。

## 4. 推荐冻结推理链路

推荐使用以下固定顺序：

```text
输入图片
  -> 冻结 best.pt 检测 meter
  -> 每图选最高置信度框（max-det=1）
  -> 在框四周保留 5% padding
  -> 裁剪 ROI
  -> 样式分类器
  -> 逐输入记录预测或 detector_miss
```

现有裁剪脚本可直接复用：

```powershell
.\.venv\Scripts\python.exe scripts\predict_crop.py `
  --weights runs\detect\meter_yolov8n_final\weights\best.pt `
  --source <输入图片或目录> `
  --output <输出目录> `
  --conf 0.25 `
  --max-det 1 `
  --padding 0.05 `
  --device 0
```

程序内调用时，应复用一个 `YOLO(best.pt)` 实例，以流式方式推理，并取 `boxes.conf.argmax()` 对应的框。不要为每张图片重复加载权重。

## 5. Manifest 与缺检计数合同

当前 `scripts/predict_crop.py` 的 `detections.jsonl` 只为实际检出的框写记录。若某张输入图片零检出，它不会自然产生一行记录；仅统计该文件的行数会静默漏掉失败样本并虚高后续准确率。

端到端评测 manifest 必须满足“一张输入至少一条状态记录”，至少包含：

- `source`：输入图片；
- `detector_status`：`ok` 或 `detector_miss`；
- `detector_confidence`：成功时的最高框置信度；
- `bbox_xyxy` 与 `crop`：成功时的框和裁剪路径；
- `prediction`：成功进入分类器后的预测；
- `ground_truth`：评测标签；
- `is_correct`：最终是否正确。

每次报告必须同时给出：

```text
input_count
detected_count
detector_miss_count = input_count - detected_count
detection_coverage = detected_count / input_count
correct_count
end_to_end_accuracy = correct_count / input_count
```

`detector_miss` 必须计为端到端错误。只有同时报告 coverage 时，才可以另行提供“已检出样本上的分类准确率”，且不能用它替代端到端准确率。

## 6. 许可证与权重分发提醒

该 `.pt` 模型资产中包含 Ultralytics 与 `AGPL-3.0` 标识，工程也依赖 Ultralytics 推理代码。若比赛部署包、镜像、安装程序或仓库需要分发权重及相关推理程序，应在分发前完成许可证审查，确认源码提供、许可证文本、修改披露及网络服务等适用义务，或取得适合闭源分发的商业许可。

不要删除模型内的许可证元数据，也不要在未完成审查时把“只分发权重”视为当然不受许可证约束。本节是工程风险提醒，不构成法律意见。
