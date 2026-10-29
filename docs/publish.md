# 模板发布:编排和上传 OSS

给发布模板的人和 vorla 后端看:批量跑完以后,哪些模板发布、每个模板传哪些文件、OSS 上怎么放、库里记什么。物料格式和合成公式见 vorla 仓库 `docs/plans/v1.7.2-requirement-261019/14-backend-sync.md`,那里第八节定了「发布由后端负责,路径带版本」,这里落到具体做法。

## 一、哪些模板发布

- 批次目录 `batches/<批次>/report.csv` 里 `status` 为 `pass` 的,也就是 `checks.json` 里 15 项全过,并且在抽查页 `review.html` 看过。
- `fail`、`error`、`skip` 的不发。`fail` 里只差拉伸一项的(8 位流向图精度,设计稿上的直线有细微锯齿)发不发,见第七节。
- 同一个 `id` 只对应一张源图:B-纸品 和抽样批次用同一张图时 `id` 相同,是同一个模板,不重复发。

## 二、每个模板传哪些文件

全部在 `output/<id>/` 里,一个模板约 2.5MB,1400 个约 3.5GB。

| 文件 | 传 | 谁用 |
| --- | --- | --- |
| `cover.webp` | 传 | 模板墙封面 |
| `base.webp` | 传 | 编辑页、服务端导出 |
| `base-css.webp` | 传 | 模板墙的 CSS 叠图 |
| `product-n.png`、`print-n.png` | 传 | 模板墙、编辑页、导出 |
| `surface-n.png`、`folds-n.png` | 传 | 编辑页、导出 |
| `template.json` | 传 | 流向图的位移幅度、明暗倍数、Logo 框只在这里 |
| `checks.json` | 不传 | 流水线自己的校验结果,只用来筛选 |

`work/<id>/`(中间结果)、`preview/<id>/`(预览图)、源图都不传。

## 三、OSS 上怎么放

路径按模板 `id` 和版本分目录,文件名照原样:

```
mockup-templates/{id}/v{version}/cover.webp
mockup-templates/{id}/v{version}/base.webp
mockup-templates/{id}/v{version}/template.json
...
```

例:`mockup-templates/poster-11x17-poster-004/v1/base.webp`。

- **`version` 取 `template.json` 里的值**,现在都是 1。模板重跑后要换内容时,版本加 1、写新目录,不覆盖旧文件,库里改指新版本;CDN 不用刷新缓存。旧目录等没有引用了再删。
- **路径里不放品类**:品类会调整(比如布面海报从 `poster` 分出预设),放进路径就得搬文件。品类和页面记在库里。
- **响应头**:`Content-Type` 按扩展名(`image/webp`、`image/png`、`application/json`);因为每个版本的目录写完就不再改,`Cache-Control: public, max-age=31536000, immutable`。
- `mockup-templates/<品类>/` 下现在放的是源图(如 `mockup-templates/boxy-tee/mockup-boxy-tee-001-blank.webp`),和模板目录同在一个前缀下。模板 `id` 都带编号,不会和这些品类目录重名;想分开的话模板换个前缀,见第七节。

用 ossutil 上传一个模板的例子(以后端实际用的工具为准):

```bash
ossutil cp -r output/poster-11x17-poster-004/ oss://<bucket>/mockup-templates/poster-11x17-poster-004/v1/ \
  --exclude "checks.json" --meta "Cache-Control:public, max-age=31536000, immutable"
```

## 四、OSS 图片处理(缩放、转格式)能用在哪

| 文件 | 能不能走图片处理 | 原因 |
| --- | --- | --- |
| `cover.webp` | 能,随便缩放、转格式 | 只是展示 |
| `base-css.webp`、`product-n.png`、`print-n.png`(模板墙) | 能缩放,输出要保留无损透明度(转 PNG) | 透明度存的是明暗和边缘覆盖度,有损编码会让叠出来的颜色和边缘不准 |
| `base.webp` | 不能,编辑页和导出读原文件 | 透明度存的是明暗,经过处理会被有损编码 |
| `surface-n.png`、`folds-n.png` | 不能 | 像素值是位移数据,不是图像 |
| `template.json` | 不涉及 | |

模板墙缩略图的例子:`cover.webp?x-oss-process=image/resize,w_480`;`base-css.webp?x-oss-process=image/resize,w_480/format,png`。

## 五、库里记什么

每个模板一行:

| 字段 | 来源 |
| --- | --- |
| `id`、`version` | `template.json` |
| `category`、`tags` | `template.json`;`tags` 里有 Channel、Seasonal 页面的 `channel/amazon`、`seasonal/christmas` 这类标签 |
| 页面 | 现在没有,见第七节 |
| 物料目录 | `mockup-templates/{id}/v{version}/` |
| 配置 | `template.json` 去掉 `pipeline` 字段后整个存成 JSON,列表和详情接口直接带出 |
| 发布时间、上下线状态 | 发布时写;模板墙按发布时间倒序 |

接口要带哪些字段,见对接说明第五节:模板墙要 `cover`、`base-css` 和各区域的 `product`、`print`、`logo`;编辑页要全部物料地址和配置。

## 六、每批的流程

1. 批量跑完,得到 `batches/<批次>/report.csv` 和抽查页。
2. 抽查,有问题的走例外清单重跑(见 [operator-guide.md](operator-guide.md))。
3. 列出要发布的模板:`report.csv` 里 `pass` 的 `id`,去掉已经按同一版本发布过的。B-纸品 这种分批跑的,每次只发新增和改了版本的。
4. 按第三节上传,再按第五节写库、上线。

流水线这边还没有发布脚本;要的话可以加一个,读 `report.csv` 和已发布清单,列出这批要传的 `id`、版本、文件,并直接调 ossutil 上传,写库仍由后端做。

## 七、待定

- **页面归属**:`template.json` 只记小品类(`category`,如 `poster`),不记页面(如 `11x17-poster`、`canvas-poster`),页面现在只体现在 `id` 里。模板墙要按页面列的话,流水线可以把页面写进 `tags`(如 `page/canvas-poster`),和 Channel、Seasonal 的做法一样。
- **只差拉伸一项的模板发不发**:B-纸品 里书封最多。
- **路径前缀和 bucket**:模板目录和源图是否分开前缀。
- **下线和删旧版本**:谁来做、旧版本留多久。
