# 仪表盘样式分类数据审计

审计日期：2026-08-21  
审计方式：只读检查；未重命名、移动、编辑或删除任何源图片、YOLO 标签和 Markdown 标注。

## 1. 结论摘要

1. 权威训练数据源应使用 `D:\tiaozhanbei\all_set`。该目录包含 1,040 张图片，其中 1,035 张有配对 YOLO 框；`D:\tiaozhanbei\yolo\dataset\audit_report.json` 也明确把它记录为 `source`。
2. `D:\tiaozhanbei\yolo\all_set` 是不完整副本，仅有 947 张图片。它比权威源少 93 张图片，但相应标签仍在，因此出现 93 个孤儿标签。它只应用于读取位于其中的 `仪表盘读数标注.md`，不应作为分类训练源。
3. Markdown 标题声称有 30 张图片，实际只有 26 个表格数据行、20 个唯一图片路径；原路径只能直接找到 15 个唯一文件。5 个不同的错误路径（共 6 行）都是编号多写了一个前导零，按本文映射后，20 个唯一文件全部存在。
4. 20 张唯一评测图只覆盖 4 个样式：M01=14、M03=3、M05=2、M06=1。恒定预测 M01 已能得到 70% 唯一图片准确率，因此 80% 目标必须按 20 张唯一图计算，并同时报告逐类召回率和 macro recall。
5. Markdown 中有 3 张图片被重复列出且具有互相矛盾的读数。样式标签仍可由路径中的 `Mxx` 得到，但当前第二列不能作为可靠的读数识别真值。
6. 完整源数据中有 10 组精确重复 SHA256：5 组为同类重复，另 5 组是 M01/M03 跨类同图冲突。旧的随机 train/val/test 切分中有 5 个重复组跨集合，不能作为无泄漏的样式分类评测。
7. 现有 YOLO 标签的类别全部是 `Dashboard`，训练后的检测器是单类 `meter` 定位器；M01、M02 等只存在于目录名中。检测器指标不能当作样式分类准确率。

## 2. 审计范围与标签定义

### 2.1 涉及路径

| 用途 | 路径 | 结论 |
| --- | --- | --- |
| 权威图片源 | `D:\tiaozhanbei\all_set` | 分类训练与重新划分应从此处读取 |
| Markdown 所在副本 | `D:\tiaozhanbei\yolo\all_set` | 仅 947 图，不完整，不用于训练 |
| 评测清单 | `D:\tiaozhanbei\yolo\all_set\仪表盘读数标注.md` | 26 行、20 张唯一图，需只读路径解析修复 |
| 旧检测数据审计 | `D:\tiaozhanbei\yolo\dataset\audit_report.json` | 记录完整源为 `D:/tiaozhanbei/all_set` |
| 旧随机切分 | `D:\tiaozhanbei\yolo\dataset\splits\*.txt` | 存在精确重复跨集合泄漏 |

### 2.2 M 样式的实际定义

项目内没有一份权威文件给 M01–M12 提供自然语言语义。当前唯一可审计定义是“图片所属的 `Mxx` 父目录”。训练阶段可以从父目录生成监督标签，但预测阶段不得把路径、文件名或父目录作为模型输入特征。

数据实际包含 11 个样式：M01–M10（缺少 M11）以及 M12。所有 `labels/classes.txt` 的内容都只有 `Dashboard`。

| 样式 | 文件名前缀 | 完整源图片数 | 完整源 YOLO 配对数 |
| --- | --- | ---: | ---: |
| M01 | `M01_P01_W` | 200 | 200 |
| M02 | `M02_P02_D` | 80 | 80 |
| M03 | `M03_P04_D` | 88 | 88 |
| M04 | `M04_P06_D`、`M04_P06_W` | 100 | 100 |
| M05 | `M05_P05_W` | 80 | 80 |
| M06 | `M06_P03_D` | 85 | 85 |
| M07 | `M07_P05_D` | 85 | 85 |
| M08 | `M08_P04_W` | 59 | 59 |
| M09 | `M09_P06_W` | 101 | 101 |
| M10 | `M10_P03_D`、`M10_P03_W` | 82 | 82 |
| M12 | `M12_P02_D` | 80 | 75 |
| **总计** |  | **1,040** | **1,035** |

M12 缺少框标注的是 `M12_P02_D_0001.jfif` 至 `M12_P02_D_0005.jfif`。它们仍可用于整图分类，但不能直接使用人工 YOLO 框生成训练 ROI。

## 3. Markdown 清单审计

### 3.1 可复核统计

| 指标 | 原始清单 | 应用确定性路径建议后 |
| --- | ---: | ---: |
| 文件物理行数 | 32 | 32 |
| 表格数据行 | 26 | 26 |
| 唯一路径 | 20 | 20 |
| 可找到的数据行 | 20 | 26 |
| 可找到的唯一图片 | 15 | 20 |
| 缺失数据行 | 6 | 0 |
| 重复行数（数据行减唯一图） | 6 | 6 |

路径建议只在清单解析结果中使用，不能回写用户 Markdown 或改名源图片。

### 3.2 路径修复表

| Markdown 原路径 | 建议解析为 | 涉及行数 | 原因 |
| --- | --- | ---: | --- |
| `M01/images/M01_P01_W_00019.jpg` | `M01/images/M01_P01_W_0019.jpg` | 2 | 五位编号多一个前导零 |
| `M01/images/M01_P01_W_00022.jpg` | `M01/images/M01_P01_W_0022.jpg` | 1 | 五位编号多一个前导零 |
| `M03/images/M03_P04_D_00058.jpg` | `M03/images/M03_P04_D_0058.jpg` | 1 | 五位编号多一个前导零 |
| `M03/images/M03_P04_D_00060.jpg` | `M03/images/M03_P04_D_0060.jpg` | 1 | 五位编号多一个前导零 |
| `M03/images/M03_P04_D_00063.jpg` | `M03/images/M03_P04_D_0063.jpg` | 1 | 五位编号多一个前导零 |

只允许在满足以下条件时应用建议：原路径不存在、候选编号可确定地从五位归一为四位、候选文件真实存在。否则应报告 `missing`，不能做模糊猜测。

### 3.3 重复行和冲突读数

| 修复后的唯一图片 | 出现次数 | Markdown 中的互相冲突读数 |
| --- | ---: | --- |
| `M01/images/M01_P01_W_0003.jpg` | 5 | `29.9 ℃`、`0.098 bar`、`6.2 bar`、`1.35 bar`、`1.22 bar` |
| `M01/images/M01_P01_W_0019.jpg` | 2 | `188 bar`、`124 ℃` |
| `M05/images/M05_P05_W_0067.jpg` | 2 | `21 Pa`、`42.5 Pa` |

这 3 张图在样式分类中只能各计一次。其 `Mxx` 样式没有行内冲突，但第二列读数不可用于训练或宣称读数准确率。

## 4. 20 张唯一图的冻结评测口径

### 4.1 样式分布

| 样式 | 唯一图片数 | 占比 |
| --- | ---: | ---: |
| M01 | 14 | 70% |
| M03 | 3 | 15% |
| M05 | 2 | 10% |
| M06 | 1 | 5% |
| **总计** | **20** | **100%** |

原始 26 行的分布为 M01=19、M03=3、M05=3、M06=1。按行评测会让同一张 M01 图片一次预测被重复计算 5 次，并把恒猜 M01 的基线抬到 19/26=73.08%，因此原始行准确率只能作为兼容性附表，不能作为主指标。

### 4.2 主指标

- 评测单位：路径归一后、按实际文件 SHA256 再去重的 20 张唯一图片。
- 输入边界：模型只能读取图片像素；路径中的 `Mxx` 只能用于生成 ground truth。
- 预测类别空间：完整的 11 个 M 样式，而不是只限制在清单出现的 4 类。
- 检测失败：记为错误并报告 `detector_miss`，不能回退使用测试集人工框或从文件名推断类别。
- 主指标：top-1 unique-image accuracy。
- 目标门槛：至少 16/20 正确，即 80%。
- 必报辅助指标：M01/M03/M05/M06 各自 recall、4 类 macro recall、coverage、混淆矩阵或等价逐类表。
- 必报产物：模型路径、执行命令、20 张逐图预测清单、置信度、失败原因和评测时间。

由于 M06 只有 1 张、M05 只有 2 张，macro recall 的方差很大；报告时必须同时给出分子/分母，不能只给百分比。

### 4.3 与旧检测切分的关系

修复后的 20 张唯一图全部出现在旧的 detector split 中：16 张位于 `train.txt`，4 张位于 `test.txt`。按原始 26 行计是 21 行 train、5 行 test。

若冻结的单类检测器只用于裁剪，并且新样式分类器完全排除这 20 张图片及其哈希副本，可以将结果描述为“冻结检测器上的闭集样式分类结果”。但由于检测器训练阶段见过其中 16 张图片，不能把整个端到端结果描述为严格的全流水线未见图泛化。要得到严格泛化结论，必须从排除这 20 张图片的 detector 数据重新训练定位器。

## 5. 精确重复、跨类冲突和旧切分泄漏

完整数据的 1,040 张图片只有 1,030 个唯一 SHA256，共 10 个二元重复组。

### 5.1 跨样式完全相同图片

以下 5 组像素文件完全相同，却分别被放进 M01 与 M03。样式分类不可能同时正确满足两个标签，因此整个哈希组都应从训练与验证中删除，等待人工修订类别本体。

| SHA256 前缀 | M01 文件 | M03 文件 | 旧 split |
| --- | --- | --- | --- |
| `47A402D94FF1` | `M01_P01_W_0142.jpg` | `M03_P04_D_0020.jpg` | train / val |
| `6EF5A89D507A` | `M01_P01_W_0080.jpg` | `M03_P04_D_0028.jpg` | test / train |
| `72CE6303DF83` | `M01_P01_W_0082.jpg` | `M03_P04_D_0033.jpg` | train / val |
| `9B700F5F249F` | `M01_P01_W_0181.jpg` | `M03_P04_D_0026.jpg` | val / train |
| `E78E69ED9012` | `M01_P01_W_0148.jpg` | `M03_P04_D_0021.jpg` | train / train |

### 5.2 同样式完全重复图片

| 样式 | 文件 1 | 文件 2 | 旧 split |
| --- | --- | --- | --- |
| M09 | `M09_P06_W_0026.jpg` | `M09_P06_W_0049.jpg` | train / train |
| M09 | `M09_P06_W_0020.jpg` | `M09_P06_W_0076.jpg` | test / train |
| M04 | `M04_P06_W_0008.png` | `M04_P06_W_0050.png` | train / train |
| M05 | `M05_P05_W_0012.jpg` | `M05_P05_W_0078.jpg` | train / train |
| M04 | `M04_P06_W_0032.jpg` | `M04_P06_W_0100.jpg` | train / train |

同样式重复可以保留哈希组中排序确定的一个代表，也可以把整个组绑定到同一 split；不能让相同哈希跨 train/val/test。

旧切分共有 5 个精确重复组跨集合，其中 4 个还是跨样式冲突。旧 detector mAP 可用于回顾原检测实验，但不能作为新样式分类器的无泄漏验证依据。

## 6. 推荐的无精确泄漏构建流程

1. 从 Markdown 只读解析 26 行，并应用第 3.2 节的确定性候选映射。
2. 以修复后的 20 个唯一图片 SHA256 构建冻结测试哈希集合。
3. 从 `D:\tiaozhanbei\all_set` 枚举 11 个 M 样式；排除测试哈希的所有副本，而不是只排除同路径文件。
4. 对剩余数据按 SHA256 分组。跨样式哈希组全部删除并记录；同样式哈希组只保留一个确定性代表。
5. 再将 SHA256 组按样式分层切为 train/val，保证一个哈希组不会跨集合。
6. 最好进一步使用感知哈希、来源 URL 或连续相似序列建立近重复组，再以组为单位切分。SHA256 只能发现字节完全相同，无法发现缩放、重新压缩或轻微裁剪后的同图。
7. 在最终报告中同时保存测试哈希、训练/验证清单、随机种子、类别映射和冲突审计。

排除 20 张冻结测试图、删除 10 张跨样式冲突图、对 5 组同类重复各移除一个副本后，剩余 1,005 张可用于分类训练/验证，其中 1,000 张有人工 YOLO 框：

| 样式 | 剩余分类图片 | 其中有框 |
| --- | ---: | ---: |
| M01 | 181 | 181 |
| M02 | 80 | 80 |
| M03 | 80 | 80 |
| M04 | 98 | 98 |
| M05 | 77 | 77 |
| M06 | 84 | 84 |
| M07 | 85 | 85 |
| M08 | 59 | 59 |
| M09 | 99 | 99 |
| M10 | 82 | 82 |
| M12 | 80 | 75 |
| **总计** | **1,005** | **1,000** |

## 7. 图片编码兼容性风险

两份数据目录都包含同一批编码异常文件。在权威源 `D:\tiaozhanbei\all_set` 和不完整副本 `D:\tiaozhanbei\yolo\all_set` 中，均有 51 张扩展名与文件实际编码不一致：49 张内容为 PNG 但扩展名为 `.jpg`，2 张内容为 WebP 但扩展名为 `.jpg`。

两个 WebP 文件是：

- `M10/images/M10_P03_W_0023.jpg`
- `M10/images/M10_P03_W_0082.jpg`

它们以 `RIFF....WEBP` 文件头开头，并非空文件；不支持 WebP 的解码器会把它们报告为损坏图片。训练和推理加载器应按文件内容解码，并在启动前验证 Pillow/OpenCV 构建是否支持 WebP。不得通过静默跳过来改变每类样本量。

## 8. 复核命令

以下命令均为只读，从 `D:\tiaozhanbei\yolo` 执行。

### 8.1 查看 Markdown 的实际行号

```powershell
$md = 'D:\tiaozhanbei\yolo\all_set\仪表盘读数标注.md'
$lines = Get-Content -LiteralPath $md -Encoding UTF8
for ($i = 0; $i -lt $lines.Count; $i++) {
  '{0,3}: {1}' -f ($i + 1), $lines[$i]
}
```

### 8.2 比较两份数据源的图片数量

```powershell
$ext = '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif'
Get-ChildItem -LiteralPath 'D:\tiaozhanbei\all_set' -Recurse -File |
  Where-Object { $_.Extension.ToLowerInvariant() -in $ext } |
  Measure-Object

Get-ChildItem -LiteralPath 'D:\tiaozhanbei\yolo\all_set' -Recurse -File |
  Where-Object { $_.Extension.ToLowerInvariant() -in $ext } |
  Measure-Object
```

### 8.3 核对权威源和旧切分声明

```powershell
Get-Content -LiteralPath 'D:\tiaozhanbei\yolo\dataset\audit_report.json' -Encoding UTF8
Get-Content -LiteralPath 'D:\tiaozhanbei\yolo\dataset\splits\train.txt' -Encoding UTF8 | Measure-Object
Get-Content -LiteralPath 'D:\tiaozhanbei\yolo\dataset\splits\val.txt' -Encoding UTF8 | Measure-Object
Get-Content -LiteralPath 'D:\tiaozhanbei\yolo\dataset\splits\test.txt' -Encoding UTF8 | Measure-Object
```

### 8.4 运行纯临时目录的数据规则单测

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\style_classifier\test_manifest.py `
  tests\style_classifier\test_hash_split.py -q
```

这些测试只在 pytest 的 `tmp_path` 中创建临时文件，用于验证 Markdown 缺失/重复报告、五位编号候选修复、测试哈希排除、同类去重和跨类冲突删除，不会修改 `all_set`。

## 9. 结果命名边界

若今晚的分类器达到 16/20，可以准确表述为：

> 在冻结 Markdown 清单的 20 张唯一图片上，样式分类 top-1 accuracy 达到 80% 或以上；分类器训练阶段按 SHA256 排除了这些测试图及其副本，并删除了跨样式精确冲突。该结果是当前冻结单类检测器上的闭集样式分类结果。

在没有按来源/近重复组建立新的外部测试集、且没有从 detector 训练中排除这 20 张图之前，不应表述为“陌生仪表泛化准确率 80%”。
