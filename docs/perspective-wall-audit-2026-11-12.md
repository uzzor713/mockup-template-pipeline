# 测试墙平面透视审计（2026-11-12）

本页保留修复前的墙面快照和人工判定清单。当天的批量复测与本地修复结果见 [平面透视批量修复](perspective-plane-batch-2026-11-12.md)。

## 范围与结论

- 测试墙快照：5,347 个模板；本地有对应素材的 3,912 个模板。缺失的 1,435 个以服装等曲面品类为主，未纳入本次平面四角审计。
- 本地素材中有 1,691 个 `surfaceFit=plane` 区域。438 个区域的蒙版轮廓在几何上满足四边形条件。
- 其中 95 个区域、94 个模板的图案透视与蒙版四角偏差超过阈值。逐张对照原照片后，23 个模板的蒙版边界并非实体边界或所在纸张并非单平面，不能用该指标定性；其余 72 个区域、71 个模板的透视过强，列在下方。
- 这 71 个是按本标准可确认的清单，不代表其余 1,253 个无法量测四角的平面区域都正确。正式墙接口本次连接失败，因此版本核对仅针对测试墙。
- `canvas-022` 和 `cd-sleeve-003` 已在测试墙发布修正版，不在当前故障清单中；前者现在的量测偏差为 0.004。

## 判定标准

1. 模板当前的 `pipeline.surfaceFit` 必须是 `plane`；可印区外轮廓近似为凸四边形，四角近似后的面积差不超过 2%，边长至少 100 像素，且上下边、左右边的偏斜不超过各自尺寸的四分之一。还要在原照片中核实四角对应真实可见的平面边界；人为设定的内部印刷框不能当作摄影透视的依据。弯曲、折叠、遮挡严重和不规则轮廓不作单平面判断。
2. 从可印区四角计算照片应有的平面单应透视；从实际 `surface` 位移图拟合图案使用的单应透视。位移拟合的第 90 百分位残差必须不超过 4 个原图像素。
3. 在 Logo 框投影的水平或垂直跨度内（上限为可印区跨度），比较两者的投影分母梯度：`偏差 = |h_图案 - h_照片| × 可见跨度`。任一方向达到 0.050 才列入。0.050 约对应小角度下两端尺度变化相差 5 个百分点；它是几何偏差，不是主观观感分数。
4. 每一条的 `print` 和 `surface` 文件 SHA-256 均与对应测试墙版本的收件箱文件一致。全部候选都检查了蒙版叠加原照片，预览图按分组抽查；小 Logo 已按 Logo 框跨度缩小量测范围。

下面的 `偏差` 即第 3 条指标；版本号是测试墙当前版本。

## 透视过强（71）

### album-cover（9）

- `album-cover-025` · v2 · 区域 1 · 偏差 0.138（y）
- `album-cover-028` · v2 · 区域 1 · 偏差 0.105（y）
- `album-cover-010` · v2 · 区域 1 · 偏差 0.104（y）
- `album-cover-020` · v3 · 区域 1 · 偏差 0.079（x）
- `album-cover-036` · v2 · 区域 1 · 偏差 0.065（x）
- `album-cover-004` · v3 · 区域 1 · 偏差 0.063（x）
- `album-cover-035` · v2 · 区域 1 · 偏差 0.060（x）
- `album-cover-009` · v2 · 区域 1 · 偏差 0.056（x）
- `album-cover-013` · v3 · 区域 1 · 偏差 0.053（x）

### book（12）

- `book-hardcover-book-032` · v3 · 区域 1 · 偏差 0.085（y）
- `book-hardcover-book-001` · v3 · 区域 1 · 偏差 0.080（y）
- `book-square-book-032` · v2 · 区域 1 · 偏差 0.075（x）
- `book-hardcover-book-013` · v3 · 区域 1 · 偏差 0.072（y）
- `book-paperback-book-024` · v2 · 区域 1 · 偏差 0.070（y）
- `book-cover-004` · v2 · 区域 1 · 偏差 0.061（x）
- `book-cover-033` · v2 · 区域 1 · 偏差 0.061（x）
- `book-paperback-book-017` · v2 · 区域 1 · 偏差 0.058（x）
- `book-square-book-021` · v2 · 区域 1 · 偏差 0.057（x）
- `book-paperback-book-010` · v2 · 区域 1 · 偏差 0.056（x）
- `book-hardcover-book-026` · v3 · 区域 1 · 偏差 0.052（x）
- `book-square-book-008` · v2 · 区域 1 · 偏差 0.051（x）

### box（2）

- `mailer-box-025` · v3 · 区域 1 · 偏差 0.088（y）
- `cereal-box-037` · v3 · 区域 1 · 偏差 0.052（x）

### business-card（8）

- `business-card-016` · v1 · 区域 1 · 偏差 0.096（y）
- `business-card-020` · v2 · 区域 1 · 偏差 0.094（y）
- `business-card-030` · v2 · 区域 1 · 偏差 0.094（x）
- `business-card-032` · v2 · 区域 1 · 偏差 0.076（y）
- `business-card-004` · v2 · 区域 1 · 偏差 0.075（y）
- `business-card-039` · v2 · 区域 1 · 偏差 0.066（y）
- `business-card-038` · v3 · 区域 1 · 偏差 0.063（x）
- `business-card-025` · v2 · 区域 1 · 偏差 0.054（y）

### canvas（12）

- `canvas-039` · v2 · 区域 1 · 偏差 0.096（x）
- `canvas-020` · v3 · 区域 1 · 偏差 0.087（x）
- `canvas-026` · v2 · 区域 1 · 偏差 0.081（x）
- `canvas-011` · v2 · 区域 1 · 偏差 0.080（x）
- `canvas-015` · v3 · 区域 1 · 偏差 0.067（x）
- `canvas-001` · v2 · 区域 1 · 偏差 0.067（x）
- `canvas-021` · v2 · 区域 1 · 偏差 0.063（x）
- `canvas-040` · v3 · 区域 1 · 偏差 0.058（x）
- `canvas-010` · v3 · 区域 1 · 偏差 0.058（x）
- `canvas-002` · v2 · 区域 1 · 偏差 0.055（x）
- `canvas-012` · v2 · 区域 1 · 偏差 0.050（x）
- `canvas-013` · v2 · 区域 1 · 偏差 0.050（x）

### cd（7）

- `cd-sleeve-007` · v3 · 区域 1 · 偏差 0.111（y）
- `cd-cover-015` · v2 · 区域 1 · 偏差 0.069（x）
- `cd-sleeve-001` · v2 · 区域 1 · 偏差 0.068（x）
- `cd-case-021` · v2 · 区域 1 · 偏差 0.065（y）
- `cd-sleeve-011` · v3 · 区域 1 · 偏差 0.060（x）
- `cd-cover-029` · v2 · 区域 1 · 偏差 0.059（y）
- `cd-cover-023` · v2 · 区域 1 · 偏差 0.053（x）

### flyer（1）

- `flyer-017` · v2 · 区域 1 · 偏差 0.054（x）

### frame（3）

- `picture-frame-023` · v2 · 区域 1 · 偏差 0.073（x）
- `picture-frame-026` · v2 · 区域 1 · 偏差 0.053（x）
- `picture-frame-014` · v2 · 区域 1 · 偏差 0.051（x）

### greeting-card（4）

- `greeting-card-postcard-040` · v2 · 区域 1 · 偏差 0.114（y）
- `greeting-card-postcard-038` · v2 · 区域 1 · 偏差 0.075（y）
- `greeting-card-postcard-010` · v3 · 区域 1 · 偏差 0.053（x）
- `greeting-card-postcard-036` · v2 · 区域 1 · 偏差 0.051（x）

### magazine（11）

- `magazine-cover-027` · v3 · 区域 1 · 偏差 0.150（y）
- `magazine-cover-008` · v2 · 区域 1 · 偏差 0.117（y）
- `magazine-cover-004` · v2 · 区域 1 · 偏差 0.110（x）
- `magazine-cover-010` · v3 · 区域 1 · 偏差 0.078（y）
- `magazine-cover-023` · v3 · 区域 1 · 偏差 0.076（x）
- `magazine-cover-026` · v2 · 区域 1 · 偏差 0.071（x）
- `magazine-cover-012` · v3 · 区域 1 · 偏差 0.065（x）
- `magazine-cover-019` · v2 · 区域 1 · 偏差 0.062（x）
- `magazine-cover-032` · v3 · 区域 1 · 偏差 0.059（x）
- `magazine-cover-007` · v2 · 区域 1 · 偏差 0.057（x）
- `magazine-cover-001` · v2 · 区域 1 · 偏差 0.054（x）

### notebook（2）

- `printful-036` · v2 · 区域 1 · 偏差 0.064（x）
- `notebook-spiral-notebook-020` · v2 · 区域 1 · 偏差 0.061（x）

## 需按实体外框复核（23）

这些模板仅达到“图案与蒙版四角不一致”的数值门槛，不能据此判定真实摄影透视的强弱。海报的可印区多是板面内部预设范围，折页和报纸则不是单一平面。

- `brochure-trifold-brochure-025` · v3 · 旧指标 0.057（不作故障判定）
- `magazine-newspaper-026` · v3 · 旧指标 0.053（不作故障判定）
- `magazine-newspaper-039` · v2 · 旧指标 0.059（不作故障判定）
- `poster-11x17-poster-007` · v2 · 旧指标 0.056（不作故障判定）
- `poster-11x17-poster-014` · v4 · 旧指标 0.068（不作故障判定）
- `poster-11x17-poster-037` · v3 · 旧指标 0.056（不作故障判定）
- `poster-canvas-poster-004` · v3 · 旧指标 0.059（不作故障判定）
- `poster-canvas-poster-009` · v3 · 旧指标 0.111（不作故障判定）
- `poster-canvas-poster-039` · v2 · 旧指标 0.075（不作故障判定）
- `poster-mdf-poster-009` · v3 · 旧指标 0.057（不作故障判定）
- `poster-mdf-poster-018` · v1 · 旧指标 0.118（不作故障判定）
- `poster-mdf-poster-023` · v2 · 旧指标 0.060（不作故障判定）
- `poster-mdf-poster-035` · v3 · 旧指标 0.056（不作故障判定）
- `poster-minimal-poster-006` · v3 · 旧指标 0.079（不作故障判定）
- `poster-minimal-poster-008` · v2 · 旧指标 0.057（不作故障判定）
- `poster-minimal-poster-022` · v3 · 旧指标 0.053（不作故障判定）
- `poster-minimal-poster-039` · v2 · 旧指标 0.051（不作故障判定）
- `poster-movie-poster-003` · v1 · 旧指标 0.052（不作故障判定）
- `poster-movie-poster-005` · v2 · 旧指标 0.050（不作故障判定）
- `poster-street-poster-039` · v3 · 旧指标 0.057（不作故障判定）
- `poster-wall-poster-022` · v3 · 旧指标 0.052（不作故障判定）
- `poster-wall-poster-023` · v2 · 旧指标 0.056（不作故障判定）
- `poster-wall-poster-031` · v2 · 旧指标 0.060（不作故障判定）

`poster-mdf-poster-018` 的图案与黑色板面外框不贴合，但旧指标称其“透视过弱”没有依据；要按板面的真实四角校验，后来按商品外缘重算修好，见 [平面透视批量修复](perspective-plane-batch-2026-11-12.md)。
