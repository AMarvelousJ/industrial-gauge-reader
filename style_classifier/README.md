# 仪表盘样式分类器

该模块把原始 `../all_set/Mxx/images` 的 `Mxx` 作为类别，模型输入仅为图像像素。训练前会修复清单里可确认的多余零路径候选，并按 SHA256 把 20 张唯一测试图及其副本全部留出。同哈希跨样式冲突会整组丢弃，相同哈希不会跨训练/验证集。

```powershell
# 从 yolo 目录执行
# 一条命令：训练分类器、冻结 YOLO 推理、80% 门槛评测、回归测试
.\scripts\run_style_pipeline.ps1

# 已有分类权重时，只重新做端到端评测与测试
.\scripts\run_style_pipeline.ps1 -SkipTrain

# 等价的分步命令
.\.venv\Scripts\python.exe -m style_classifier.train --epochs 12
.\.venv\Scripts\python.exe -m style_classifier.predict_manifest `
  --detector-weights runs/detect/meter_yolov8n_final/weights/best.pt
.\.venv\Scripts\python.exe -m style_classifier.evaluate --target 0.80
```

主要输出：

- `outputs/style_classifier/best.pt`：最佳分类模型。
- `outputs/style_classifier/training_report.json`：训练/验证曲线及数据量。
- `outputs/style_classifier/manifest_audit.json`：MD 重复、缺失及建议路径。
- `outputs/style_classifier/dataset_hash_audit.json`：测试排除、同图去重及跨类冲突证据。
- `outputs/style_classifier/manifest_eval/predictions.json`：逐行预测和唯一图片准确率。
- `outputs/style_classifier/manifest_eval/predictions.csv`：方便人工检查的表格。
- `outputs/style_classifier/final_evaluation.json`：机器可读验收结论。
- `outputs/style_classifier/final_report.md`：人可读最终报告。

分类器训练默认使用配套 YOLO 人工标注框裁出最大仪表区域；该裁剪只提供位置，不提供样式类别。MD 测试则强制使用用户现有的冻结 `best.pt` 检测器推理裁剪，绝不读取测试 GT 框。检测失败会明确记录为 `detector_miss`、计入错误并降低 coverage，不会回退到 GT 或整图。此模块从不训练或微调 YOLO。评测同时报告修复后 20 张唯一图准确率、4 类 macro recall、逐类召回和原始 26 行准确率。
