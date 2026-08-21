# 指针式仪表读数最小基线

该模块解决的是“实际读数”，不再把 M01-M12 目录当作目标。流水线为：冻结 YOLO 仪表框 → RapidOCR 数字/单位 → MIT 开源预训练指针分割（无掩膜时回退 OCR 遮罩后的 OpenCV/HSV）→ 单/双量程圆周 RANSAC → 指针角 → 物理读数。刻度不足时仅补跑 90°/270° OCR，并只接受两个方向在同一位置重复出现的数字；只有两个可信刻度时使用带边界诊断的圆周插值。

```powershell
.\scripts\run_style_reader.ps1 -OutputDir outputs/style_reader/latest
```

命令会依次生成 `results.json`、严格预测契约 `predictions.json`、`evaluation.json`、`report.md` 和逐图读数总览图。`ground_truth_reading` 只供独立评测器使用，算法明确标注 `ground_truth_used_by_algorithm=false`。当前审计后的 20 图协议结果为 16/20（80%），覆盖率 20/20；这只是小型闭集基线，不代表比赛要求的未知仪表泛化率。

用户提供的检测器和第三方分割器都只调用 `predict`，本模块没有任何训练、微调或权重写入代码。
