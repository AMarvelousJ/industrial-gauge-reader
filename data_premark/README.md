# 多形态仪表数据预标与审核包

这个模块只负责数据抽样、近重复隔离、自动几何预标和人工审核材料，不训练或修改任何模型。

## 运行

```powershell
.\.venv\Scripts\python.exe -m data_premark `
  --source all_set `
  --output outputs/data_premark_v1 `
  --per-shape 30 `
  --validation-per-shape 10 `
  --seed 20260822
```

输出包括：

- `frozen_private/preannotations_all.jsonl`：942张图片的自动预标记录，不放在公开开发集根目录；
- `review_manifest.json`和`review.csv`：仅80张开发集；
- `frozen_private/review_frozen_manifest.json`和`review_frozen.csv`：物理隔离的40张冻结集；
- `frozen_private/combined_machine_manifest.json`：120张机器可读索引，不作为人工真值填写入口；
- `preannotation.schema.json`：与审核包一起交付的1.0.0 JSON Schema；
- `review.md`、`selected_contact_sheet.jpg`和`thumbnails/`：审核说明与可视化；
- `audit.json`：数据量、重复簇、分层拆分和泄漏审计。

## 数据边界

- 发现逻辑只访问 `all_set/M*/images` 和同名YOLO定位标签，不访问任何Markdown文件。
- YOLO标签只为几何分析提供已有表盘框；形态、轴心、指针候选和刻度弧均由图像启发式算法生成。
- 该ROI的来源明确记为 `existing_yolo_label_for_annotation_only`，冻结端到端预测不得使用它，必须从图像/模型自行定位。
- SHA-256完全相同或pHash/颜色直方图高度相似的图片归入同一重复簇；一个簇最多选一个审核样本，因此不会跨 `dev` 与 `frozen_validation`。
- 聚类是单链接并查集，可能因相似图片链产生过度合并；阈值和证据计数记录在 `audit.json`，不能把簇数量当作真实仪表类型数。
- `reading`、`unit`、`range_min`、`range_max`、`minor_division`只能由独立人工审核填写。

## 审核原则

自动预标是减少人工操作的候选，不是真值。审核者需要逐一确认形态、边界、轴心、主测量指针、红色设定/峰值指针、刻度弧和量程。冻结验证集审核完成后应与开发流程隔离。

JSON记录接口由 `schemas/preannotation.schema.json` 固定为1.0.0。

## 本地可视化标注器

不要手工计算归一化轴心和指针角度。下面的命令只加载80张开发集并在浏览器打开本地页面；开发模式会拒绝冻结manifest：

```powershell
.\scripts\run_annotation_ui.ps1
```

页面中点击“轴心”后在原图上点轴心，再选择自动候选针，或切换“针尖”模式从轴心向针尖拖动。角度自动按右0°、下90°计算。`接受并下一张`与`修正并下一张`要求读数、单位、量程等字段完整；无法判断的样本使用`暂存待复核`并在备注中说明，不能把`no_output`作为完成状态。

服务默认只监听`127.0.0.1:8765`，图片只能按开发manifest中的`record_id`访问。保存仅更新审核列，使用同目录临时文件和原子替换，并在覆盖前保留`review.csv.bak`。如果CSV在页面外被修改，服务会拒绝覆盖并要求重新启动。

## 审核验收

审核者分别在开发集 `review.csv` 和私有冻结集 `frozen_private/review_frozen.csv` 中填写：`review_status`（`accepted`或`corrected`）、`review_shape`、归一化轴心 `pivot_x/pivot_y`、主测量指针角色和 `pointer_angle_deg`。角度采用图像坐标：向右为0°、向下为90°。读数、单位、量程必须填写；`minor_division`可留空，此时使用满量程的1%作为容差。

```powershell
.\.venv\Scripts\python.exe -m data_premark.review_acceptance `
  --review-csv outputs\data_premark_v1\review.csv `
  --manifest outputs\data_premark_v1\review_manifest.json `
  --frozen-review-csv outputs\data_premark_v1\frozen_private\review_frozen.csv `
  --frozen-manifest outputs\data_premark_v1\frozen_private\review_frozen_manifest.json
```

审核未完成或字段缺失时，命令输出 `status: not_ready` 并返回退出码2，不生成指标。未传预测文件时，只计算80张开发集的自动形态宏召回率、轴心归一化误差和主指针角度误差；冻结集只做完整度计数，不进入指标。

最终预测JSON直接采用读数器的1.0接口，并且必须恰好包含40个冻结 `sample_id`（其值等于审核记录的 `record_id`），不得混入开发集ID。`status=ok` 时必须提供有限数值和受支持单位；评测器会先换算到真值单位再计算误差。为兼容早期工具，也接受 `record_id/reading` 字段：

```json
{
  "predictions": [
    {"sample_id": "0123456789abcdef", "status": "ok", "value": 0.425, "unit": "MPa"}
  ]
}
```

```powershell
.\.venv\Scripts\python.exe -m data_premark.review_acceptance `
  --predictions outputs\data_premark_v1\frozen_predictions.json `
  --output outputs\data_premark_v1\acceptance.json
```

工具只输出冻结集聚合结果，不回显逐条真值。40张中至少34张误差不超过一个最小分度（缺失时不超过1%满量程）才通过。
