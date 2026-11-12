# 平面透视批量检测与修复（2026-11-12）

## 发布状态

- 2026-11-12 修复进度表 `plane-perspective` 的 104 个全部发布到测试库，同批发布的还有 `plane-unfit` 补的 4 个、贴纸 `sticker-target` 3 个。
- `poster-canvas-poster-019`、`poster-mdf-poster-030`、`poster-street-poster-017` 原来拉伸检查没过、没上过墙，这次作为新模板放在收件箱 `2026-11-12/print/`。
- 正式库未改，同步由 `fixes.py sync` 出总结交给管正式库的同事。

## 检测口径

- 测试墙共有 5,347 张模板；本机 `output/` 有 4,031 张，对其中 1,704 个标记为 `surfaceFit=plane` 的区域做检测。本机没有物料的墙上模板不在本轮自动扫描范围。
- 可印区外轮廓须近似凸四边形：轮廓与四角多边形的面积差不超过 2%，边长至少 100 像素，并满足近直立矩形的边线限制。位移图反求单应变换后，第 90 百分位拟合残差不超过 4 原图像素。
- 在 Logo 框可见跨度内，比较图案和照片四角的投影分母梯度；`max(|h_图案 - h_照片| × 跨度)` 达到 0.050 才记为几何偏差。可印区面积少于商品区 85% 的，可能只是内部印刷框，转人工核对实体边界，不据此判定透视错。
- 这是筛查指标，不能只凭数值区分纸面正面、侧厚、画框或遮挡。批量自动重算仅用于先前已逐张对照照片确认的清单；原审计见 [墙面快照](perspective-wall-audit-2026-11-12.md)。

## 本轮结果

- 修复前：对齐 245 个区域、偏差 84 个、印刷框需复核 112 个、无法取得可信四角 1,263 个。
- 已确认清单中，64 张模板的 65 个偏差区域自动重算并通过 17 项物料检查；复测后这 65 个区域全部低于 0.050，记录到四位小数均为 0.0000。旧偏差中位数 0.0658，最大值 0.1466。唱片封面、精装书、画布、CD 套、杂志封面已抽看新旧合成图。
- 修复后全量复测：对齐 310、偏差 19、印刷框需复核 112、无可信四角 1,263。`poster-street-poster-017` 虽然照片四角清楚，但重算因拉伸/8 位位移约束只保留 60% 透视，仍超阈值，这一轮没改；下一轮按商品外缘重算，接受只留部分透视（见下）。
- 原人工确认的 71 张中，64 张本地修好，`canvas-020` 已作为两张试点之一发布，另 6 张（精装书 001、谷物盒 037、邮寄盒 025、相框 014/023/026）要先确认实体正面边界。`poster-mdf-poster-018` 属于原先 23 张内部印刷框待复核项，已单独用板面外缘修好并发布。

## 尚有数值偏差的 19 张

这些是**筛查结果**，不等于 19 张都已确认故障。海报的内部印刷框、折页和报纸尤其不能直接套用四角规则；`canvas-006`、`canvas-036` 的厚边/遮挡需人工判正面。

- `brochure-trifold-brochure-025` 0.0569；`magazine-newspaper-026` 0.0519；`magazine-newspaper-039` 0.0590
- `canvas-006` 0.0526；`canvas-036` 0.0514
- `poster-11x17-poster-007` 0.0562；`poster-11x17-poster-037` 0.0545
- `poster-canvas-poster-004` 0.0590；`poster-canvas-poster-009` 0.1056；`poster-canvas-poster-039` 0.0759
- `poster-mdf-poster-009` 0.0549；`poster-mdf-poster-023` 0.0605；`poster-mdf-poster-035` 0.0543
- `poster-minimal-poster-022` 0.0518；`poster-movie-poster-005` 0.0506
- `poster-street-poster-017` 0.1327；`poster-street-poster-039` 0.0558
- `poster-wall-poster-022` 0.0518；`poster-wall-poster-023` 0.0578

## 按商品外缘复筛

可印区四角只在可印区就是这一面的时候可信;海报四周留白、日历画面、相框画芯这类可印区只是面上的一块印刷框,按它量出的透视差不能作数。这一轮改用商品外缘当四角,把本机全部按平面贴的区域再筛一遍。

- 口径:商品外缘取得到可信四角(同上),商品的点到印面平面的距离 p90 至多商品尺寸的 5%(保证外缘和印面在同一个平面上);按外缘量的透视差 ≥ 0.050,且按可印区四角也没对上的,收作候选。可印区四角已对上、只是外缘不一致的 14 个不收,差别多来自框边和厚边。
- 1,704 个区域:外缘对齐 319、偏差 56、外缘取不到可信四角 1,328(盒子、带书脊的书、被遮挡的等)。偏差的 56 个里 42 个进候选,逐张叠图看商品外缘和可印区的四边形。
- 修好 38 个,都过了 17 项检查、按平面贴:
  - 改按商品外缘(例外清单写 `planeOutline: product`)25 个:海报 22 个(MDF、画布、街头、极简、墙面、11×17)、`book-cover-010`、`calendar-039`、`pod-012`。
  - 外缘带着画框、框边,按可印区四角重算 13 个:相框 6 个(016/023/030/032/035/039)、`poster-11x17-poster-014`、墙面海报 3 个(011/022/031)、电影海报 2 个(003/005)、`canvas-006`。
  - 透视陡、位移图存成 8 位放不下完整透视的,只留一部分:`poster-mdf-poster-030` 留 80%(透视差 0.1315→0.0844)、`poster-street-poster-017` 留 60%(0.093→0.0775)。比原来近,但还没到 0.050。
- 不修的 4 个:`brochure-trifold-brochure-025`、`magazine-newspaper-039` 是折着的,不是一个平面;`cd-case-013`、`cd-case-036` 的外缘带着盒子侧面。
- 上一轮剩下的 19 个里,`poster-canvas-poster-009`、`poster-canvas-poster-039`、`magazine-newspaper-026`、`canvas-036`、`poster-street-poster-039` 商品外缘取不到四角,这个办法量不了。要确认实体正面的 6 个里 `picture-frame-023` 这一轮按可印区四角修好;`hardcover-001`、`cereal-box-037`、`mailer-box-025`、`picture-frame-014`、`picture-frame-026` 外缘也取不到四角,仍待看照片。
- 顺带重跑了 4 个本机产出还是平面贴法之前的模板(补进 `plane-unfit`):`poster-canvas-poster-026`、`notebook-spiral-notebook-032`、`poster-movie-poster-009` 现在按平面贴;`calendar-025` 立着的日历和桌上一摞卡片并成了一个可印区、离平面 20%,改成只取日历重新分割(补进 `multi-item`)。

修复进度表里这个问题记作 `plane-perspective`,共 104 个模板:上一轮 64 个、这一轮 38 个,加上先发过的 `poster-mdf-poster-018`、`canvas-020`。

## 重跑与数据

- 检测：`.venv/bin/python -I scripts/plane_perspective_scan.py --out fixes/plan/perspective-scan-after-2026-11-12.csv`
- 自动重算入口：`.venv/bin/python -I scripts/plane_perspective_repair.py`；候选列表在 `fixes/perspective-plane-confirmed-2026-11-12.txt`，失败会恢复原物料。
- 本机明细：`fixes/plan/perspective-scan-2026-11-12.csv`、`fixes/plan/perspective-2026-11-12-repair.csv`、`fixes/plan/perspective-scan-after-2026-11-12.csv`。`fixes/plan/` 不进仓库，保存了原物料备份和逐张日志。
