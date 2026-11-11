# 模板上传

给把模板传到 vorla 的人看:批量跑完以后,哪些模板上传、怎么整理、传到哪、传完做什么。上传要求以 vorla 给的《Mockup 模板素材上传说明(1.7.2)》为准(B-纸品 生图包根目录的 `mockup-template上传说明.md`),入库和线上路径由后端做。

## 一、哪些模板上传

只传通过的,也就是批次汇总表 `batches/<批次>/report.csv` 里 `status` 为 `pass` 的(自动校验 17 项全过,不在跳过之列),并且在抽查页 `review.html` 看过。没过、出错、跳过的不传,修好重跑通过后再传。

已经在测试站模板墙上的,`inbox.py` 不再当新模板整理;重跑过、要换文件的见[八、改已经发布的模板](#八改已经发布的模板)。

## 二、整理

```bash
.venv/bin/python -I scripts/inbox.py batches/b-print inbox/2026-10-28
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
.venv/bin/python -I scripts/inbox.py batches/b-print inbox/2026-10-30 --uploaded inbox/2026-10-28/manifest.csv
```

## 六、线上读取时注意

模板上线后,`base.webp`、`surface-n.png`、`folds-n.png` 只能读原文件,不能经过 OSS 图片处理(缩放、转格式):底图的透明度存的是明暗,流向图的像素值是位移数据,处理过就不准了。模板墙用的 `cover.webp` 可以随意缩放;`base-css.webp`、`product-n.png`、`print-n.png` 缩放时要输出无损透明度(转 PNG)。

## 七、发布到测试站

收件箱里的模板由 vorla 仓库的发布脚本 `apps/web/scripts/mockup-templates-publish.ts` 发布(用法写在脚本开头):校验通过的模板复制原图到私有目录、生成模板墙和编辑页的预览、按 slug 写库。默认连 `.env.local` 里的测试库 `vorla_test`;写正式库要显式传 `DATABASE_URL`,每次单独授权。

```bash
cd ~/Projects/vorla/apps/web
pnpm --filter @vorla/web exec tsx --conditions=react-server scripts/mockup-templates-publish.ts            # 只检查,不写 OSS、不写库
pnpm --filter @vorla/web exec tsx --conditions=react-server scripts/mockup-templates-publish.ts --apply    # 发布
... --apply --only id1,id2 --concurrency 2                                                                    # 只发这几个
```

- 先跑只检查,看报告里的「不通过」:品类页还没上线的、同一个 id 出现两个目录的(两份都会被挡掉)、`template.json` 解析失败的、Logo 框超出 0–1 的。报告在 `scripts/output/mockup-publish/<时间>/`。
- 发布可以重复跑:同版本、源文件没变的跳过文件(报告记为 unchanged),照样写库。中途网络超时、写库报错的,把失败的 id 用 `--only` 再发一次;写库报错时报告里只有 SQL,看不到原因,降低 `--concurrency` 重发。
- 默认并发 4,每个模板下载约 3MB,网络慢时约每分钟 20 个;只改了 `template.json`(如 Logo 框)的也要复制原图、重出预览。

## 八、改已经发布的模板

改了已发布模板的任何文件(包括只改 `template.json` 里的 Logo 框),`template.json` 的 `version` 都要加 1 再传到收件箱:发布后的公开文件按版本放在 `v<版本>/` 下、缓存一年不过期,脚本发现同版本的源文件变了会拒绝(「已发布过,这些文件变了但版本号没升」)。新版本发在新目录,旧版本目录留着,模板墙按库里的版本换地址。

重跑过的模板用 `--update` 整理,版本号由脚本按模板墙写:

```bash
.venv/bin/python -I scripts/inbox.py batches/<批次> inbox/update-<日期> --update --only 重跑过的.txt
```

- 拉测试站模板墙,上过墙的 `version` 写成墙上的版本(`coverImage` 里的 `v<版本>`)加 1,没上过的保持原样。
- 列 OSS 收件箱(约 1 分钟),按这个 id 在收件箱里现在的目录放:输出目录下的相对路径和收件箱里一样,只放 `template.json` 和它列的文件,展示图、白模图、`manifest.csv` 不动。
- 收件箱里还没有的列出来,按新模板不带 `--update` 整理;同一个 id 在收件箱里有两个目录的也列出来,发布脚本会两份都挡掉,先删掉多的一份。
- 小品类和测试库里不一样的列出来(预设改过小品类时会有):发布后测试库跟着改,同步正式库时也会改。
- 最后打出两条命令:整个输出目录传到收件箱根目录,盖在原处(`ossutil cp -r -f ... < /dev/null`:不加 `-f` 时同名文件逐个问是否覆盖,并从标准输入读回答,写在脚本里会把后面的命令当成回答读掉,文件跳过不传);再用发布脚本 `--apply --batch mockup-template-inbox/ --only` 发这几个。前缀给收件箱根目录,哪个日期目录里的都找得到;不加 `--files-only`,测试库里的那一行和文件一起更新。

[修复进度表](fixes.md)的 `fixes.py publish` 按批次挑出该发布的,打出这条命令。

核对时 `scripts/wall.py` 打出模板墙的总数和各版本的个数,给一个路径时每个模板一行存成 CSV(id、版本、显示名、小品类、标签、最后发布的时间);加 `--prod` 拉正式站(读正式库)。接口是 `https://vorla.drfans.com/api/trpc/mockup.listTemplates`,分页参数 `{"json":{"page":1,"pageSize":48}}`,`materials.regions` 里是各区域的 Logo 框;直接请求 `assets.vorla.ai` 上的图要带浏览器的 User-Agent,不带会被 CDN 拦成 403。

## 九、同步到正式库

正式库和测试库的模板文件在 OSS 同一个 bucket,发布到测试库时已经发好,同步只改正式库 `template` 表里的行。写正式库由管正式库的同事做:[修复进度表](fixes.md#流程)的 `fixes.py sync` 列出进了测试库、正式库还是旧版本的模板,写成同步总结交给他们。
