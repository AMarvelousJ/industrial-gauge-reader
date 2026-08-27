# 模型资产

本目录通过 Git LFS 保存可直接运行主推理链所需的模型。克隆仓库后执行：

```powershell
git lfs install
git lfs pull
python scripts\verify_setup.py
```

| 文件 | 作用 | SHA256 | 说明 |
|---|---|---|---|
| `meter_detector.pt` | 单类 `meter` 定位和 ROI 裁剪 | `A0447F659564955C0FFBCD7BD68394745C9C3CA5686117EFBB41A631CE79E1A1` | 冻结，禁止训练覆盖 |
| `scale_segment.pt` | 指针像素分割及 PCA 方向/轴心证据 | `ED02E20A11E4B4D86A40220A75E0146726E2E67F132F7FEB768EE0948C234007` | 第三方 MIT 资产 |
| `pointer_keypoints.pt` | `pivot`、`pointer_tip` 两关键点 | `80C079BCFF67B920B2F7A711245070F246651B4CBCDFC86DCB8B416DD7C03540` | 当前只作诊断证据 |

样式分类器的 `best.pt` 没有放入这里，因为 `style_classifier/` 是旧的 M01-M12 分组实验，当前正式读数链不依赖它。
