# 模板上传

给把模板传到 vorla 的人看:批量跑完以后,哪些模板上传、怎么整理、传到哪、传完做什么。上传要求以 vorla 给的《Mockup 模板素材上传说明(1.7.2)》为准(B-纸品 生图包根目录的 `mockup-template上传说明.md`),入库和线上路径由后端做。

## 一、哪些模板上传

只传通过的,也就是批次汇总表 `batches/<批次>/report.csv` 里 `status` 为 `pass` 的(自动校验 15 项全过,不在跳过之列),并且在抽查页 `review.html` 看过。没过、出错、跳过的不传,修好重跑通过后再传。

## 二、整理

```bash
.venv/bin/python -I inbox.py batches/b-print inbox/2026-10-28
```

`inbox.py` 把这一批里全过的模板整理到输出目录,结构就是上传说明要的样子:

```
inbox/2026-10-28/
  manifest.csv
  poster-11x17-poster-004/
    template.json
    cover.webp        展示图
    white.png         2K 白模图
    base.webp
    base-css.webp
    product-1.png
    print-1.png
    surface-1.png
    folds-1.png
  ...
```

| 文件 | 从哪来 |
| --- | --- |
| 目录名 | `template.json` 里的 `id` |
| `template.json`、底图、各区域的蒙版和流向图 | `output/<id>/` 原样复制,不改名、不转格式、不重新压缩 |
| `cover.webp`(展示图) | 生图包里带印花的那张 `<key>.png`,按源图同样的比例裁、放大到模板尺寸 |
| `white.png`(白模图) | 源图 `<key>-blank-2x.png` 裁好、缩放到模板尺寸后存 PNG,和底图、蒙版逐像素对齐 |
| `manifest.csv` | 每个模板一行:`id`;`categories` 是生图包里的页面文件夹名(如 `11x17-poster`、`canvas-poster`、`business-card`);`internal_name` 由页面名和编号得出(如 `11x17 Poster 004`) |

`checks.json` 只用来筛选,不传。

所有模板都是 2048 × 2560(横版 2560 × 2048),两种流向图是 1/4 尺寸。源图裁成 4:5 / 5:4 后不是这个尺寸的(如方形书 2048 × 2048 的源图),`batch.py` 先缩放到这个尺寸再跑各步骤。

一个模板约 9MB(白模图约 6MB),1000 个约 9GB。

## 三、上传

- 传到 bucket `vorla-assets` 的 `mockup-template-inbox/<日期>/`,账号是单独开的,只能往这个目录传。
- 整个输出目录原样传上去,文件保持私有,不设公共读:

```bash
ossutil cp -r inbox/2026-10-28/ oss://vorla-assets/mockup-template-inbox/2026-10-28/
```

- 不要传到 `mockup-templates/`(线上正在用,同名文件会覆盖线上图),也不要传到 `exports/` 开头的目录(1 天后自动删除)。

## 四、传完以后

告诉 han 模板数量和总大小(`inbox.py` 最后一行会打出来)。后端逐个校验文件齐全、尺寸、区域,不合格的列出来退回。

## 五、分几次传

B-纸品 这种分批跑的,每次整理到新的日期目录,用 `--uploaded` 带上之前传过的 `manifest.csv`,已经传过的 `id` 跳过:

```bash
.venv/bin/python -I inbox.py batches/b-print inbox/2026-10-30 --uploaded inbox/2026-10-28/manifest.csv
```

## 六、线上读取时注意

模板上线后,`base.webp`、`surface-n.png`、`folds-n.png` 只能读原文件,不能经过 OSS 图片处理(缩放、转格式):底图的透明度存的是明暗,流向图的像素值是位移数据,处理过就不准了。模板墙用的 `cover.webp` 可以随意缩放;`base-css.webp`、`product-n.png`、`print-n.png` 缩放时要输出无损透明度(转 PNG)。
