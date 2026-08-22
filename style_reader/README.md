# 多形态指针式仪表读数 V1

该模块解决的是“实际读数”，不再把 M01-M12 目录当作目标。流水线为：冻结 YOLO 仪表框 → 圆/椭圆/矩形边界判断 → 高置信椭圆仿射归一化或矩形单应性 → RapidOCR 数字/单位 → 主测量指针语义选择 → OCR数字与刻度候选关联诊断 → 已验证圆周映射 → 物理读数。红色设定或峰值标记保留在诊断候选中，但不能覆盖主测量针。

正视圆表保持原有稳定路径；只有轴比、重投影误差和边界置信度同时达标时，椭圆仿射或矩形分支才会接管。椭圆分支是受阈值约束的仿射启发式，不等同于完整相机标定或一般投影单应性。刻度候选映射在缺少主刻度、刻度环和单位一致性证据前保持影子诊断，不覆盖正式读数。每张结果都包含 `dial_geometry`、`normalization`、`pointer_candidates`、`selected_pointer_role`、`tick_mapping` 和 `stage_diagnostics`。

```powershell
.\scripts\run_style_reader.ps1 -OutputDir outputs/style_reader/latest
```

命令会依次生成 `results.json`、严格预测契约 `predictions.json`、`evaluation.json`、`report.md` 和逐图读数总览图。`ground_truth_reading` 只供独立评测器使用，算法明确标注 `ground_truth_used_by_algorithm=false`。V1在当前审计20图上为17/20（85%），严格18图口径为15/18（83.3%），覆盖率均为100%；这只是小型闭集回归结果，不代表比赛要求的陌生仪表泛化率。

942张原始图片的自动预标和120张人工审核包由下面的独立命令生成；它不读取读数Markdown，40张冻结集在人工审核完成前不会产生准确率：

```powershell
.\.venv\Scripts\python.exe -m data_premark --source all_set --output outputs/data_premark_v1
```

用户提供的检测器和第三方分割器都只调用 `predict`，本模块没有任何训练、微调或权重写入代码。
