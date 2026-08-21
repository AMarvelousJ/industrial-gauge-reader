# 冻结 YOLO + 仪表盘样式识别

## 当前验收结果

本次任务**不会训练或微调 YOLO**。用户提供的 `best.pt` 与工程内
`runs/detect/meter_yolov8n_final/weights/best.pt` 的 SHA256 完全一致，
只作为冻结的单类仪表检测器。其裁剪结果交给独立 ResNet18 分类器判断
`M01`–`M10`、`M12` 样式。

一条命令完成分类器训练、冻结 YOLO 端到端评测、80% 门槛检查和回归测试：

```powershell
.\scripts\run_style_pipeline.ps1
```

已存在 `outputs/style_classifier/best.pt` 时可快速复核：

```powershell
.\scripts\run_style_pipeline.ps1 -SkipTrain
```

当前固定 Markdown 清单按路径纠错后为 20 张唯一图片，端到端正确 19 张，
准确率 95%，检测覆盖率 100%，四类宏平均召回 91.67%。这是固定清单上的
封闭集结果；不是对未知来源新仪表的泛化准确率。详见
`outputs/style_classifier/final_report.md` 和 `docs/data_audit.md`。

## 原有单类仪表检测工程

本工程训练一个统一的 `meter` 目标类别。M01–M12 表示不同仪表外观，
但它们都作为“仪表”参与训练；模型负责定位并裁剪仪表，读数识别由下一个模型完成。

## 1. 环境

已创建隔离环境：

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

如果 PowerShell 禁止激活脚本，可以始终直接使用：

```powershell
.\.venv\Scripts\python.exe <命令>
```

安装固定版本的 CUDA PyTorch 与 Ultralytics：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 准备数据

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

该命令不会移动或修改 `all_set`。它会：

- 检查图片、标签、类别与归一化框坐标；
- 排除“有图片但无标签”的样本，避免被误当成负样本；
- 按 M01–M12 样式分层，以固定随机种子生成 80%/10%/10% 划分；
- 生成 `dataset/meter.yaml`、`dataset/splits/*.txt` 与审计报告。

当前原始数据包含 1,040 张图片，其中 1,035 张有合法标签。M12 的
`M12_P02_D_0001` 至 `M12_P02_D_0005` 共 5 张缺少标签，会被安全排除。

## 3. 训练检测器

默认参数针对 RTX 3060 Laptop 6GB 显存：

```powershell
.\.venv\Scripts\python.exe scripts\train_detector.py
```

默认关闭图片缓存，训练过程不会在 `all_set` 的原始图片旁创建 `.npy`
缓存文件。如确认允许写入缓存，可显式传入 `--cache disk`。

常用覆盖参数：

```powershell
.\.venv\Scripts\python.exe scripts\train_detector.py `
  --model yolov8n.pt `
  --epochs 120 `
  --imgsz 640 `
  --batch 8 `
  --device 0
```

显存不足时将 `--batch` 调为 4 或 2。最佳权重默认位于：

```text
runs/detect/meter_yolov8n/weights/best.pt
```

中断后恢复训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_detector.py `
  --model runs\detect\meter_yolov8n\weights\last.pt `
  --resume
```

## 4. 标框并裁剪给读数模型

```powershell
.\.venv\Scripts\python.exe scripts\predict_crop.py `
  --weights runs\detect\meter_yolov8n\weights\best.pt `
  --source path\to\images
```

输出位于 `outputs/meter_detection/`：

- `annotated/`：画出检测框的图片；
- `crops/`：默认额外保留 5% 边缘的仪表 ROI；
- `detections.jsonl`：置信度、原始框、裁剪框和裁剪文件路径，可直接传给下游。

## 关于“多样式”和“多类别”

当前目标是找出所有仪表并交给同一个后续读数阶段，因此采用单类检测最合适。
多样式能力来自训练集中覆盖不同外观、角度、背景和光照，而不取决于类别数量。
只有当后续读数算法需要根据 M01、M02 等样式选择不同量程或不同模型时，
才应把样式改成多个检测类别或增加独立的样式分类器。
