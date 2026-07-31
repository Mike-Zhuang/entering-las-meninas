# 单图输出与重建比较指南

本目录中的图片不是从组合图裁切出来的，而是由 `export-individual-images.py` 从原始图像、模型张量和中间激活直接导出。因此，它们与组合图使用相同的数据，同时保留了各自的原始像素尺寸。

## 目录结构

```text
individual/
├── 01-resolution/
├── 02-training-patterns/
│   ├── all-800/
│   ├── displayed-samples/
│   └── manifest.json
├── 03-reconstruction-comparison/
├── 04-feature-maps/
│   ├── early-layer/
│   └── bottleneck/
├── 05-artwork-comparison/
│   ├── own-work/
│   └── las-meninas/
├── 06-forward-pass/
├── 07-network-structure/
└── 08-training/
```

## 怎样比较输入与重建？

`03-reconstruction-comparison/` 提供几种互补的方法。

### 1. 单独打开输入和重建

- `own-work-input-128x128.png`
- `own-work-reconstruction-128x128.png`

两张图片具有完全相同的 128 × 128 尺寸和像素对齐。适合放大观察文字、人物、房间框架和红色系绳。

### 2. 并排比较

`own-work-input-vs-reconstruction-side-by-side.png`

适合观察整体构图是否保留。重点比较房间边界、中央矩形、人物位置和大面积明暗。

### 3. 50% 透明叠加

`own-work-overlay-input-50-reconstruction-50.png`

输入与重建各占 50%。如果结构完全对齐，叠加结果会比较稳定；模糊、位置偏移和新增纹理会形成重影。

### 4. 闪烁比较

`own-work-input-reconstruction-blink.gif`

GIF 每 0.9 秒在输入和重建之间切换。人的视觉对突然变化非常敏感，因此它特别适合发现线条消失、边缘位移和亮度改变。

### 5. RGB 绝对差异

`own-work-absolute-difference-rgb-amplified-4x.png`

计算每个 RGB 像素的绝对差值，并放大四倍便于观看。越亮表示输入与重建之间的颜色差异越大。

### 6. 带标尺的差异热图

- `own-work-absolute-difference-heatmap.png`：无标题纯热图；
- `own-work-difference-with-color-scale.png`：包含颜色标尺、MSE和MAE。

热图适合确定误差集中在哪里。它不告诉我们变化是“变亮”还是“变暗”，只表示差异大小。

### 7. 红色系绳差异

`own-work-red-dominance-input-minus-output.png`

暖色表示输入比重建更红，冷色表示重建比输入更红。它用于区分真实系绳被削弱和模型在其他区域加入红色倾向这两种现象。

### 8. 数值比较

`comparison-metrics.json` 保存：

- 均方误差（MSE）；
- 平均绝对误差（MAE）；
- 输入与输出的红色优势能量；
- 红色能量输出/输入比。

数值用于记录整体误差，不能代替视觉比较，也不能衡量艺术质量。

## 特征图命名

`early-layer/own-work-early-map-07-of-12.png` 表示第一层卷积的第 7 个通道，共 12 个通道。

`bottleneck/own-work-bottleneck-channel-03-of-32.png` 表示瓶颈的第 3 个通道，共 32 个通道。

所有单通道热图都独立归一化，因此适合观察每张图内部的空间分布，不适合仅凭颜色亮度跨图片比较绝对激活值。

## 重新导出

模型训练完成后运行：

```bash
python export-individual-images.py
```

脚本不会重新训练模型，只会读取现有权重并重新生成确定性的训练图案和单图输出。
