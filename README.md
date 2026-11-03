# Mockup 模板素材管线

把一张商品照片加工成 Mockup 模板要用的素材:半透明底图、每件商品的改色范围和可印区域、流向图、配置。每个模板写一份配置(`specs/<模板>.json`),脚本本身不含具体模板的内容;样本是一个两件商品(T 恤 + 牛仔裤)的模板,配置是 `specs/sample-tee-jeans.json`。

编辑页的合成方式:每件商品先填上用户选的颜色,设计稿按所属商品的可印区域裁切并按流向图位移(大曲面层固定,褶皱层乘 Warping),最后在线性光下把底图按透明度盖在最上面(颜色那项乘上明暗倍数 `shadeGain`),明暗、纹理和光泽都在底图里。

## 运行

需要 Python 3.12。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# MODEL_BACKEND=local 时另装:
# .venv/bin/pip install -r requirements-models.txt
# .venv/bin/pip install --no-deps -r requirements-moge.txt

SPEC=specs/sample-tee-jeans.json
.venv/bin/python -I fetch_sources.py $SPEC   # 下载配置里 fetch 列出的素材
.venv/bin/python -I run_segment.py $SPEC     # SAM 3 按文字提示分割商品、部件和遮挡物,ViTMatte 求边缘透明度
.venv/bin/python -I intrinsic.py $SPEC       # Marigold 本征分解
.venv/bin/python -I normals.py $SPEC         # Marigold 法线
.venv/bin/python -I geometry.py $SPEC        # MoGe-2 的 3D 坐标和相机内参
.venv/bin/python -I pipeline.py $SPEC        # 生成模板文件,并跑自动校验
.venv/bin/python -I preview.py $SPEC         # 合成预览图
```

模型都在 Replicate 上跑,源图会上传过去。token 写进仓库根目录的 `.env.local`(不进仓库),一行 `REPLICATE_API_TOKEN=...`,或者设同名的环境变量。`.env` 里的 `REPLICATE_MIN_INTERVAL` 是本机所有进程合起来两次预测之间至少隔的秒数(`--jobs` 并发时也按这个节拍排队):账号余额低时 Replicate 每分钟只让建 6 次预测,要设 11;充值后每分钟 600 次(响应头 `ratelimit-remaining`),设 0。超了会返回 429,`remote.py` 等一会儿重试;Replicate 偶发的服务器错误(5xx)和断连也重试。同一张源图在本机只上传一次,分割、本征分解、法线、几何几步共用(上传记录在系统临时目录的 `replicate-uploads/`,文件在 Replicate 存 24 小时,快过期的重新上传)。

分割用的 SAM 3 始终走 Replicate;抠图、本征分解、法线、几何四个模型在哪跑由 `MODEL_BACKEND` 决定,默认值写在 `.env`(进仓库),是 `replicate`:调 Replicate 上的社区封装(版本固定在 `remote.py`),本机只要 `requirements.txt` 和 token。`local` 在本机跑,要装上面的模型依赖、有 Apple 芯片,一个模板 1.5–2.5 分钟,留作对照和开发,批量不用。本机要改就在 `.env.local` 写一行 `MODEL_BACKEND=local`,或设同名环境变量;优先级是环境变量 > `.env.local` > `.env`。两种结果的差别:MoGe-2 逐位相同、抠图几乎一样;Marigold 两个封装不能固定随机种子和处理分辨率、只出 8 位图,换算回来和本机的相关系数本征分解 0.94–0.995(残差那层最低)、法线 0.96–0.98,每次运行结果也会略有不同。

每个模板的中间结果在 `work/<id>/`,模板文件在 `output/<id>/`,预览图在 `preview/<id>/`,`<id>` 是配置里的 `id`。

`MODEL_BACKEND=local` 时四个模型脚本首次运行会从 Hugging Face 下载模型,`run_segment.py` 约 0.1GB(用到手点提示时另加 SAM 2.1 约 0.3GB),`intrinsic.py`、`normals.py` 各约 2.6GB,`geometry.py` 约 1.3GB。模型下载好后,在 16GB 内存的 M1 上 `run_segment.py` 约 75 秒(`REPLICATE_MIN_INTERVAL=11` 时,大半是调 SAM 3 之间的间隔)、`intrinsic.py` 约 22 秒、`normals.py` 约 20 秒、`geometry.py` 约 10 秒,内存峰值约 6–7GB。Marigold 是扩散模型,本机跑时随机种子固定,重复运行结果一致。`replicate` 时每个模型的推理只要几秒,但这些封装用的人少,闲置后再调要冷启动,每个模型一到五分钟,一个模板一个个接着跑要十一二分钟,所以批量要开 `--jobs` 让各模型一直有请求;按推理时长计费,一个模板约 1–2 美分。

ViTMatte、两个 Marigold 和 MoGe-2 在 Replicate 上用的是个人上传的社区封装,`remote.py` 锁了版本号,作者改不了已锁定的版本,但可以整个删掉;删了就切回 `local`,或者把同样的权重部署到自己的账号。SAM 3 的封装只收文字提示,不收框和点:文字分不开的东西(如画框和画心)分不出来。

源图要用空白底(商品上没有印花和文字):本征分解会把花纹从底图里去掉,但印花边缘的明暗起伏留在纹理里,换色后看得出图案的轮廓。

`sources/`、`work/`、`output/`、`preview/` 都不进仓库,按上面的步骤在本地生成。样本用的竞品素材只作内部示例,不能用于产品。

### 批量

同一品类的商品、部件、可印规则和遮挡物写成预设 `presets/<预设>.json`,`batch.py` 按预设给每张源图生成配置,跑完上面除 `fetch_sources.py` 以外的六步。源图按品类分好文件夹时直接给文件夹:

```bash
.venv/bin/python -I batch.py 源图文件夹 --dry-run   # 先只生成配置、列出各品类张数,确认文件夹都对上了预设
.venv/bin/python -I batch.py 源图文件夹             # --jobs N 同时跑几个模板,--local N 至多几个同时在本机算,--force 全部重跑,--no-preview 不出预览
```

`--jobs`:`MODEL_BACKEND=local` 时每个模板峰值约 6–7GB 内存,16GB 的机器开 1 个;`replicate` 时本机只等网络,可以多开,各模型一直有请求就不会闲置下来、反复冷启动。前四步等 Replicate 时每个进程占约 150MB;`pipeline.py`、`logo.py`、`preview.py` 在本机算,每个峰值约 2GB、要一两核,`--local N`(默认 2)让同一时间至多 N 个模板在跑这几步,其余排队,`--jobs` 开大时内存不会跟着涨。一个模板的这几步约 40 秒,`--local 2` 每分钟约能过 2 个模板,`--jobs` 开到十几个才会在这里排队。

- 每个子文件夹是一个品类,文件夹名是预设名(vorla 的小品类 slug,如 `t-shirt`、`coffee-bag`),或写在预设的 `folders` 里(如 `"folders": ["T恤", "boxy-tee"]`);里面(含更深的子文件夹)的 png / jpg / webp 都算,以 `_` 开头的文件夹不算。对不上预设的文件夹、不在品类文件夹里的图会列出来并停下,不会悄悄跳过。
- id 是品类文件夹名(不是纯英文时换成预设名)加文件名:文件名去掉 `mockup-` 前缀和 `-blank` 后缀,转小写,字母数字以外的连成 `-`;文件名本来就以文件夹名开头的不重复加。`t-shirt/001.png` → `t-shirt-001`,`boxy-tee/mockup-boxy-tee-020-blank.webp` → `boxy-tee-020`。生成的清单存在 `batches/<文件夹名>/list.csv`。
- 个别源图要改设置时加 `--overrides`,给例外清单 CSV 或放例外清单的文件夹(里面的 csv 按文件名顺序),可以写多次,后面的覆盖前面的。每行一张,`id` 对上 `list.csv` 里的,空着的列不改:

| 列 | 内容 |
| --- | --- |
| `product` | 这张的商品提示词,只能用在一件商品的预设上 |
| `parts` | 这张的不可印部件提示词,`;` 隔开,写 `-` 表示没有;同上只能用在一件商品的预设上 |
| `occluders`、`crop`、`preset` | 同下面清单的列 |
| `logo` | 这张的默认 Logo 放法,填 `logo.py` 里 `PLACES` 的名字:`left-chest` 左胸、`chest` 胸口正中,Gemini 判断前襟判错时按张改正;同样只能用在一件商品的预设上 |
| `skip` | 填 `1` 跳过这张,不跑也不进抽查页,`report.csv` 里记为 `skip` |
| `issue`、`note` | 抽查页导出的问题类型和备注,只给人看 |

```csv
id,issue,note,product,parts,occluders,crop,preset,skip
coffee-bag-07,部件不对,顶上封口条也印上了,,top of the bag,,,,
t-shirt-001,遮挡物不对,,,,hair;hand;arm,,,
poster-017,其他,画框也被改色,,,,,,1
```

  每次运行都按源图和预设重新生成配置,改过的源图没带上它的例外清单就会回到预设的设置、重算一遍,所以例外清单要一直留着、每次都带上;放进一个文件夹、每次 `--overrides 那个文件夹` 最省事。

源图没按文件夹分时,也可以给一份清单 CSV,每行一张:

```csv
source,preset,id,occluders,crop
https://assets.vorla.ai/mockup-templates/boxy-tee/mockup-boxy-tee-001-blank.webp,t-shirt,,,
/Users/me/mockups/02 Studio club sweatshirt.png,sweatshirt,,hair;hand;arm,
```

| 列 | 内容 |
| --- | --- |
| `source`、`preset` | 源图的地址或本地路径、预设名,必填;地址的源图下载到 `sources/<预设>/`,本地的直接引用 |
| `id` | 可选,不写由文件名得出,规则同上(不加文件夹名) |
| `occluders` | 可选,遮挡物提示词,`;` 隔开,替换预设里的(同名的沿用预设里的 `matteBand`);写 `-` 表示没有遮挡物 |
| `crop` | 可选,`x0 y0 x1 y1`(源图像素),不写按预设的比例规则裁;裁好后不是 2048 × 2560(横版 2560 × 2048)的,配置里记上 `size`,各步骤先缩放到这个尺寸 |
| `tags` | 可选,标签,`;` 隔开,写进配置和 `template.json`,如 Channel、Seasonal 页面的 `channel/amazon`、`seasonal/christmas` |

预设里除了配置的 `category`、`occluders`、`products` 和文件夹别名 `folders`,还有裁切规则:竖图裁成 4:5、横图 5:4,正方形按 `square`(默认 `[4, 5]`);多出的宽度两边平分,多出的高度按 `trim`(`bottom` / `top` / `center`,默认 `center`)从哪边裁。

预设按 vorla mockups 页面主树的 69 个小品类各一份,文件名和 `category` 都是小品类的 slug;Seasonal、Channel 两条横轴是标签,模板仍归在某个小品类下(按画面里实际的商品选预设),不单独设预设,清单的 `tags` 列写上 `seasonal/<页面>` 或 `channel/<页面>`。一件商品一个区域,`tracksuit` 是上衣、裤子两个区域;画面里是两件同类商品的页面另设预设,文件名用页面 slug,`category` 仍是小品类:`front-and-back-hoodie`、`front-and-back-t-shirt`(正面、背面两个区域)、`mug-pair`(左右两只杯子),翻开的书用 `open-book`(内页可印),`cd-case`(CD 盒和封面卡,cd-case、cd-cover 页面用)、`cd-sleeve`(纸套)的 `category` 是 `cd`,`can-pair` 是两罐并排,`canvas-poster`(挂着的布面海报,备选说法 fabric、cloth、curtain)的 `category` 是 `poster`;`circle-sticker`(圆形贴纸,备选说法 circle、round label)的 `category` 是 `sticker`;`bottle`、`candle` 可编辑的是标签。`t-shirt`、`sweatshirt`、`tote-bag`、`cap`、`pillow`、`bottle`、`coffee-bag`、`notebook`、`poster`、`paper-bag` 各拿一张图跑通过,其余的提示词没拿图试过,全量跑之前每类先抽几张跑一遍,看抽查页;提示词分不出来的用 `experiments/sam3_replicate.py` 试别的说法。

结果在 `batches/<文件夹名或清单文件名>/`:`specs/` 是生成的配置,`logs/<id>.log` 是各步输出,`report.csv` 是每个模板的结果(`pass` / `fail` 带没通过的校验项 / `error` 带出错的步骤和最后一行输出 / `skip`),`review.html` 是抽查页。重跑同一条命令时,各步记下上次用到的配置字段、源图内容和 `MODEL_BACKEND`(`work/<id>/steps.json`),没变的跳过;改商品的 `logo` 设置(预设里的,或例外清单的 `logo` 列)只重跑 `logo.py` 和预览;改遮挡物、商品或部件提示词只重跑分割和之后两步,三个模型脚本只看源图内容和裁切,源图挪了文件夹也不重跑。

#### 抽查和调整

`review.html` 是总览,列出各品类的出错、没通过、通过张数,点进品类页(`review/<品类>.html`);品类页里出错、没通过的全列,通过的按 id 打散后先列 48 张当抽样,其余点开再看。每张卡片是分割叠图(绿 = 商品,蓝 = 不可印部件,红 = 遮挡物)和蒙版示意(黄线 = 可印区域,红框 = 默认 Logo)的缩略图,点开看原图。跑的过程中抽查页每分钟更新一次。

跑批量的人和调参数的人不是同一个时:

1. 跑的人在品类页点「有问题」,选问题类型、可以写备注(标记存在这台电脑的浏览器里),看完在页面底部「导出例外清单」;
2. 跑的人 `.venv/bin/python -I batch.py 源图文件夹 --pack 例外清单.csv`,把这几张的源图(放在和原来一样的品类文件夹下)、分割叠图和蒙版示意(`_review/`)连同清单打成 zip,存到 `batches/<文件夹名>/`,发给调参数的人;
3. 调参数的人解压,`batch.py 解压出的文件夹 --dry-run` 得到同样的 id,用 `experiments/sam3_replicate.py batches/<zip 名>/specs/<id>.json "说法一" "说法二"` 试提示词(每个说法分出来的样子画在 `work/<id>/experiments/sam3/sheet.png`),填好例外清单;可以带上清单对解压出的文件夹真跑一遍,看抽查页确认;实在分不好的填 `skip`;
4. 清单发回去,跑的人放进例外清单文件夹,`batch.py 源图文件夹 --overrides 例外清单文件夹`,只重算这几张。

提示词改了、对整个品类都更好的,改预设而不是一张张写例外清单。

## 脚本

| 文件 | 做什么 |
| --- | --- |
| `config.py` | 读模板配置、源图、设置和密钥(环境变量、`.env.local`、`.env`),给出每个模板的 `work/`、`output/`、`preview/` 目录,各脚本共用 |
| `remote.py` | Replicate 的调用(上传、预测、取结果),以及 `MODEL_BACKEND=replicate` 时四个模型的封装版本和输出换算 |
| `fetch_sources.py` | 按配置的 `fetch` 下载素材,并核对是否与生成样本时的文件一致 |
| `segment.py` | 分割的函数:SAM 3 按文字提示(走 Replicate)或 SAM 2.1 按外接框和提示点,分出遮挡物、商品和不可印部件;商品的提示词分出几个实例时(如书架上的书、背景里路人的包),把各实例编号画在图上,和原图一起交给 Gemini(`google/gemini-3-flash`,走 Replicate,一次约 0.1 美分)挑出要展示的那几个,回答看不懂时全留,说一个都不是时换备选说法(`alternatives`),挑的结果写在日志里;SAM 3 的置信度阈值默认 0.5,所有说法都分不出时(空白商品没印花、特征少)依次放低到 0.3、0.2 再把各个说法试一遍,低阈值下只分出一个实例也交给 Gemini 确认,日志里写明最后用的说法和阈值;上层商品的蒙版吞了下层的(同色的整套运动服,外套的说法把裤子一起分成一个实例:下层蒙版八成以上落在上层里,上层没有八成落在下层里)时,从上层减掉下层,日志里写明;ViTMatte 在商品和背景的交界求商品的透明度,在商品和遮挡物的交界求遮挡物的透明度(`MODEL_BACKEND=replicate` 时走 Replicate)。不读写本地文件 |
| `run_segment.py` | 按配置调 `segment.py`,存每件商品的透明度和部件蒙版、遮挡物透明度,以及人工检查用的叠图 `segment-check.png` |
| `intrinsic.py` | 本征分解:Marigold 把照片拆成固有色、光照和残差,给 `pipeline.py` 出底图的明暗和光泽,去掉商品上的花纹 |
| `normals.py` | 法线:Marigold 估计每个像素的表面朝向,给 `pipeline.py` 出流向图里的褶皱 |
| `geometry.py` | 几何:MoGe-2 估计每个像素的 3D 坐标和相机内参,给 `pipeline.py` 出流向图里的大曲面和透视 |
| `pipeline.py` | 改色范围、可印区域、底图(含模板墙用的 `base-css`)、流向图、`template.json`、自动校验 |
| `logo.py` | 按可印区和商品的 `logo` 设置算默认 Logo 框(上衣的前襟交给 Gemini 判断),`pipeline.py` 出模板时调用;单独跑时只重写 `template.json` 里的 Logo 框和对应的校验项,不重出底图和流向图 |
| `preview.py` | 只读模板文件合成预览,和编辑页用同一套公式 |
| `batch.py` | 按品类文件夹(或清单)和品类预设批量生成配置、跑完各步,出汇总表和抽查页;按例外清单打包问题源图 |
| `review.py` | 批量的抽查页:按品类分页、缩略图、标记问题并导出例外清单 |
| `inbox.py` | 把一批里自动校验全过的模板按 vorla 的上传说明整理成上传用的文件夹(模板文件、展示图 `cover.webp`、白模图 `white.png`、`manifest.csv`),见 [docs/publish.md](docs/publish.md) |

`experiments/` 里是试验脚本,不在上面的流程里,结果写到 `work/<id>/experiments/`:

| 文件 | 做什么 |
| --- | --- |
| `sam3_replicate.py` | SAM 3 按文字提示分割,每个实例分开存下来,用来给新模板挑提示词;token 和 `run_segment.py` 一样 |
| `detect_boxes.py` | 开放词汇检测模型(默认 Grounding DINO)按文字找出各部分的外接框 |
| `sam21_auto.py` | 检测出的框直接当 SAM 2.1 的提示,和手点提示的结果对比 |
| `seam_snap.py` | 以 `sam21_auto.py` 的袖子为大致范围,沿照片里的接缝(细暗线)重新切分袖子和身体 |
| `compare_mockey.py` | 同一张铺满商品的网格,按竞品的位移算法(本地复现,和竞品截图对过)和我们的模板分别形变,并排对比 |
| `compare_blend.py` | 模板墙用 img 直接叠图(读 `base-css.webp`)和线性光合成的对比,附编辑页带曲面褶皱的效果和胸口原尺寸细节;第二个参数是设计稿 PNG |

## 配置

`specs/<模板>.json`,路径都相对仓库根目录:

| 字段 | 内容 |
| --- | --- |
| `id`、`category`、`tags` | 模板 id(也是 `work/`、`output/`、`preview/` 下的目录名)、品类、可选的标签列表,写进 `template.json` |
| `photo` | 源图。带透明度的叠在白色上;透明度不到 1 的像素视为不是原始照片(如从竞品底图还原的商品区),这些像素落在商品区外时,底图里换成周围的背景色 |
| `crop` | 可选,源图先按 `[x0, y0, x1, y1]`(像素)裁出这一块再用,比例不对时用;各脚本和封面都用裁好的图 |
| `size` | 可选,`[宽, 高]`,裁好后再缩放到这个尺寸;`batch.py` 在裁好的图不是 2048 × 2560 / 2560 × 2048 时写上(vorla 后端按这个尺寸收模板) |
| `fetch` | 可选,要下载的素材:`{路径: {url, sha256}}` |
| `occluders` | 遮挡物的提示列表。一个提示是 `{text: 名词短语}`(SAM 3,图里所有符合的实例合在一起,如 `"bag strap"`),或 `{box: [x0, y0, x1, y1], points: [[x, y, 1 正 / 0 负], ...]}`(SAM 2.1,文字分不准时手点兜底);`note` 只作说明;`matteBand` 可选,和商品交界的待求带半宽 / 宽度,头发这类边上有细丝的放宽(如 `0.015`),不写为 0.004 |
| `products` | 商品列表,由上层到下层,每件:`id`(文件名用)、`key`、`name`(写进 `template.json`)、`text` 或 `box` 和 `points`(分割提示,格式同上)、可选的 `parts`(不可印部件的提示列表,如袖子、领口)、`print`、可选的 `binaryEdge`、`recolorable`(默认 `true`)、`reference`;可选的 `alternatives`:备选的说法列表,`text` 一个实例都分不出、或 Gemini 说分出的都不是商品时按顺序换着试(空白商品上同一品类不同页面能认的说法不同),各个说法都在阈值 0.5 试过才放低阈值,所以备选说法在 0.5 分得出时优先于 `text` 在低阈值分出;可选的 `instance`:文字提示挑剩的实例里只取从左往右第几个(从 0 数),同一张图里两件同类商品时各写一件 |
| `products[].print` | 可印区域规则:商品区减去 `parts`,`inset: {top, bottom, left, right}` 按宽度的比例从商品和背景的交界往里收,只写需要的那几侧(部件、遮挡物和上层商品的边上不收),`rows: [从, 到]` 只留这段高度(占商品高度的比例),`largest: true` 只留最大的一块(部件把可印区切开时,如围裙的腰带),`face: true` 只留正对的那一面(法线和可印区法线中位数的夹角在 35° 以内、连成一片的最大一块;盒子、书、本子、纸袋这类硬的多面商品,侧面、书脊不印,剩下的不到原来 30% 时不收窄) |
| `products[].logo` | 可选,默认 Logo 框的放法:不写放在可印区正中;`top` 是框中心在可印区从上往下的位置(上衣写 0.32,胸口);`side: "left"` 和 `top` 一起写,放在左胸;`closure` 让 Gemini 判断照片是不是这件的正面、前襟有没有拉链或扣子,是就合上里面的设置;`fill: true` 铺满可印区(平面印刷品);`freeform: true` 可印区是异形时框不转(贴纸、袜子、钥匙扣);`width` 改框的大小(默认 0.4)。预设里上衣写 `top`,其中卫衣、夹克、球衣、polo 衫、运动套装的上衣另写 `closure: {side: "left"}`;平面印刷品写 `fill`,贴纸另写 `freeform`;`split` 写单只的说法,框只放在 SAM 3 分出的最大那只上(袜子、牛仔裤、运动裤) |
| `products[].binaryEdge` | 源图里这件商品的轮廓是台阶(如从竞品底图还原的),抠图救不回来:按二值来源处理,平滑轮廓后重出软边 |
| `products[].reference` | 可选,对比用的参考蒙版 `{product, print}`,每项 `{file, transparent}`;`transparent` 取图里半透明的部分,否则取白色 × 透明度。`pipeline.py` 最后打印重合度,不算校验 |
| `preview` | 预览的裁切范围 `crop`(占宽高的比例)和边缘放大的取样位置 `spots`(`[名称, x, y]`,模板像素) |
| `competitor` | 可选,只给 `experiments/compare_mockey.py` 用:对应的商品 `product`、竞品位移图 `displacement`、舞台缩放 `stageScale`、默认 Warping `defaultWarping`、局部放大范围 `zoom` |

## 产出

`output/<id>/` 里是模板本身,以样本为例:

| 文件 | 内容 |
| --- | --- |
| `cover.webp` | 模板封面(原图) |
| `base.webp` | 底图:T 恤、牛仔裤区域半透明,在线性光下合成时 `shadeGain` × (1 − 透明度) = 明暗 × 纹理、颜色 × 透明度 = 光泽 × 纹理,不含花纹;背景、皮肤、包带不透明 |
| `base-css.webp` | 模板墙 CSS 叠图用的底图:由 `base.webp` 换算成按 sRGB 普通叠加的格式,选白色、纯黑时和线性光结果一致,中间颜色有几个到十几个色阶的偏差 |
| `product-1.png` / `product-2.png` | T 恤 / 牛仔裤的改色范围(白色 + 透明度),编号按配置里商品的顺序 |
| `print-1.png` / `print-2.png` | T 恤 / 牛仔裤的可印区域(白色 + 透明度) |
| `surface-1.png` / `surface-2.png` | T 恤 / 牛仔裤的大曲面层流向图,固定叠上,不随 Warping 变:R / G = x / y 方向的位移,中灰 = 不位移,通道值 0 / 255 对应 ∓`surfaceRangePx`;B 未用;1/4 尺寸(400×500),使用时双线性放大 |
| `folds-1.png` / `folds-2.png` | T 恤 / 牛仔裤的褶皱层流向图,格式同上,值是 Warping 100% 时的位移,通道值对应 ∓`foldsRangePx`,使用时乘 Warping |
| `template.json` | 品类 `category` 和标签 `tags`、文件名(含 `baseCss`)、部件名、默认 Logo 框、合成方式 `blend: "linear"` 和明暗倍数 `shadeGain`、两层流向图的 `displacement.surfaceRangePx`、`foldsRangePx`、生成参数 |
| `checks.json` | 自动校验结果(15 项) |

把模板放进 vorla 的 lab 页或编辑页看效果时,选 `output/<id>/` 里的全部文件:lab 按文件名认 `base.*`、`product-n.*`、`print-n.*`、`surface-n.*`、`folds-n.*`,位移幅度只在 `template.json` 里,要一起选。`work/<id>/` 里是中间结果,lab 认不出。

`preview/<id>/` 里是示意图:

- `preview.png`:只读模板文件合成的效果。从左到右依次是封面、白 / 白、红 / 藏青、黑 / 黑(Warping 50%)、芥末黄 / 橄榄绿(Warping 100%),颜色按商品顺序。
- `edges.png`:配置里的几处边缘按像素放大 3 倍,左起封面、红 / 黑、黑 / 芥末黄。样本是袖口、肩线、T 恤压牛仔裤、包带、裤裆、裤脚压鞋。
- `masks.png`:各件商品区依次着绿、蓝、紫,黄线为可印区域,红框为默认 Logo 位置。

## 怎么生成的

以样本为例:

- 分割:SAM 3 按文字提示分出包和包带(遮挡物)、T 恤、袖子(提示词 `short sleeve`)、领口和牛仔裤,T 恤减去遮挡物;袖子沿落肩接缝分开,领口是那圈罗纹。商品蒙版只补面积不到整件 0.2% 的小洞,包带、提手和包身围出来的背景留在外面;部件伸出商品蒙版的部分(如提手顶上)并进商品,绳子这类细的部件不做去毛刺的开运算,免得整条被削掉。ViTMatte 在商品和背景的交界几个像素内求透明度;商品和遮挡物的交界两边都待求,把遮挡物当前景抠,商品在那里的覆盖度 = 1 − 遮挡物透明度,遮挡物下面商品接着铺。源图里的 T 恤是从竞品底图还原出来的,轮廓本身是台阶,抠图救不回来,所以 T 恤按二值来源处理:平滑轮廓后重出软边。牛仔裤和 T 恤的交界不算边缘,牛仔裤在 T 恤下面接着铺。
- 可印区域按配置里的规则取,只从商品和背景的交界往里收;袖子、包带和上层商品的边上不收,设计稿铺到被挡住的地方,由底图或上层商品盖住,商品和包带之间几像素宽的缝算包带。T 恤减去袖子和领口罗纹,只在下摆往里收 1.2% 宽度(竞品下摆留边的中位数),肩线和左右两侧不收;牛仔裤四周往里收 1% 宽度(T 恤下摆盖住的地方不收),取裤腰以下到大腿中段。袖子蒙版和轮廓、包带之间夹出的零碎小块去掉。和竞品模板自带的蒙版相比,T 恤商品区重合度 0.994、可印区域 0.960。
- 底图:Marigold 把照片拆成固有色、光照和残差,两者的比例按每件商品拟合。底图是「这件商品如果是白色」的样子,分三部分:明暗(光照)、光泽(残差,高光这类不随颜色变的反光)和纹理(照片里比长边 / 256 更细的起伏)。明暗和纹理乘在用户选的颜色上,光泽乘上纹理后叠加在最上面,所以深色衣服也有反光和布纹。残差只取 0.6 当光泽,其余并入明暗:选白色时亮度不变,纯色和黑色的暗部和竞品对齐(样本 T 恤上纯红 223, 42, 42,竞品 224, 43, 43)。做旧洗白、染料这些花纹留在固有色里,改色时不会带出来。「白色」按白布料的固有色算(样本白 T 恤在本征分解里约 0.87),选白色时和照片一样亮,不额外提亮。深色商品(实心部分照片亮度中位数低于线性 0.05,约 sRGB 0.25)例外:照片里光照和固有色分不开,算出的光照常偏暗,选白色会发灰,所以保留明暗起伏,把明暗的 90% 分位抬到白布料的亮度;这类照片噪点占比大,纹理只保留一半,抬亮后不显颗粒。深色布、光线强时亮处的明暗会超过 1(样本的牛仔裤约 11% 的面积),底图里存的是明暗 ÷ 倍数(`shadeGain`,样本 1.652),渲染时颜色那项乘回倍数,超过 1 的部分按实际颜色截:选白色照样是白,深色、彩色的亮处不跟着压暗。商品边缘的半透明像素里,背后那份颜色(背景、皮肤、头发)按前景色估计解出来,改色后边缘不留原色。
- 流向图分两层,分开存:
  - 大曲面:MoGe-2 给出每个像素的 3D 坐标,由此算出屏幕上每像素在曲面上实际对应多长(以商品中位深度处正对镜头时为 1),在商品区内平滑约 32px(宽度的 2%):身体、裤腿转向侧面的地方压缩,轮廓内一二十像素压得最多,至多 2 倍;离镜头近的地方略放大,透视也在里面。再用最小二乘积分成位移。这层按形状算出来就是实际的样子,固定叠上,不随 Warping 变。
  - 褶皱:起伏取 Marigold 法线的 z 分量,也就是表面正对镜头的程度。位移沿它的坡度,褶皱横着就上下压缩、竖着就左右压缩、斜着就按对应角度压缩。只看表面朝向,和照片的光线方向、花纹都无关。坡度系数每件商品各自定:Warping 100% 时可印区域里 20% 的面积刚好顶到下面的上限;系数有上限,几乎没褶的商品不会把噪点放大成假褶皱。
  - Warping 只缩放褶皱层。Warping 0–100% 之间两层合起来,各方向每像素前进量都限制在 0.5–2.5(拉伸至多 2 倍、压缩至多 2.5 倍),超出的地方局部减弱褶皱那层,深褶处设计稿不会被拉成一片或翻折;大曲面层的位移幅度大、8 位量化误差也大,减弱时多留出这部分余量。大曲面层本身不减弱,人穿着的衣服侧面、袖口这些陡边上会超出一点;设计稿只画在可印区里,自动校验只查可印区,允许其中至多 0.5% 的面积超出。
  - 位移会沿曲面一路累积,渲染时减去设计稿中心处的位移(两层各自减),中心不动、两边按曲面压扁。
  - 设计稿中心可以拖出商品。商品外不让位移归零,而是铺成边上的值、远一点再抹平:要整体减掉的「中心处位移」在商品外不再猛变,拖出去时设计稿跟着手走,不会反着往商品里跑。商品分成几块(罐子和盖子、敞开夹克的两片前襟)而只印其中一块时,流向图只铺在可印区所在的那几块上,其余几块当作商品外。自动校验从可印区中心沿各方向逐像素往外拖,确认不往回走。
- 默认 Logo 框(`template.json` 里各区域的 `logo`:`cx`、`cy`、`width`、`height` 按模板宽高归一,`rotation` 是度,屏幕上顺时针为正):
  - 居中的、放胸口的是正方形,边长为可印区最宽处(衣服是袖口到袖口)的 40%,至多最大内切圆半径的 1.4 倍。放胸口的先放在正中,再往上挪到中心在可印区从上往下 32% 处(竞品 Mockey 的上衣在约 32%),框上沿碰到领口就停;背面照同样的规则落在上背部。
  - 卫衣、夹克、球衣、polo 衫、运动套装的上衣,`logo.py` 把封面和描出可印区的封面交给 Gemini(和分割挑实例用的同一个,一次约 0.1 美分),问是不是正面、前襟有没有拉链或扣子;是就放左胸,背面照和没有前襟的仍放胸口。回答存在 `work/<id>/logo-closure.json`,图没变不再问;没问到(调用失败、回答看不懂)时放胸口。判错的在例外清单的 `logo` 列改正。
  - 左胸指穿衣人的左胸,正面照里在画面右侧:框中心和放胸口的一样高,横向在门襟(那一高度可印区的中线)到右端的中点,边长为门襟到右端宽度的 40%。
  - 袜子、牛仔裤、运动裤:一双袜子、两条裤腿挨着或叠着,可印区连成一块,按整块放会压在两只的交界、裆部或两腿之间。预设写 `split`(单只的说法,袜子 `sock`、裤子 `pant leg`),`logo.py` 让 SAM 3 按它分出单只(一次约 0.1 美分),框只放在可印区里最大的那只上,再按 `top` 往上挪:袜子 0.3(袜筒),裤子 0.25(大腿上部),框上沿碰到袜口、腰头就停。分出的单只存在 `work/<id>/logo-split*`,图和说法没变不再分;盖住可印区 10% 以上的不到两只、调用失败时按整块放。袜子的单只是 L 形,外接矩形是斜的,另写 `freeform` 不转。
  - 钥匙扣的链子、挂扣、拉链头算不可印部件,框放在挂件本体上;挂件是六边形这类异形的写 `freeform` 不转。
  - 平面印刷品铺满可印区;一张图里几张卡片、几页纸分开时只铺满最大的一块。正对着的矩形、圆形和异形贴纸取上下、左右两端围成的框,设计稿盖满不留白,形状外面的部分被可印区裁掉。透视下的梯形、斜放的平行四边形(可印区的凸包是四边形、不到外接框的 90%)外接框会比封面大一圈,设计稿四边被裁,改取里面放得下的最大矩形,方向取放得下最大矩形的那个(窄长的四边形两条短边斜得厉害,四条边平均的方向会偏);手指、夹子挡住边角的小缺口先补成凸包。两张卡叠成一块(可印区不到凸包的 90%)时框落在放得下最大矩形的那张上;书签压在书封上这类大块遮挡也会被当成这种情况,框偏小。
  - 框跟着可印区摆放的方向转:可印区的轮廓能近似成四边形时(海报、卡片),取四条边方向的平均,偏 3° 以上就转;衣服取最小外接矩形的方向,偏 10° 以上才转。人穿着站直时躯干轮廓左右不对称,会算出至多约 9° 的倾角,平放、搭着的衣服歪 10° 以上的都是真歪。圆形、椭圆形的可印区看不出方向,不转。贴纸的可印区不是四边形、面积不到最小外接矩形的 85% 时(爱心、闪电、云朵这类异形),也看不出方向,不转;圆角、被手指挡住一点的矩形贴纸照常转。
- 模型只用在分割、边缘透明度、本征分解、法线和 3D 坐标,其余步骤都是固定代码。

## 和正式素材的差别

- 尺寸是 1600×2000(4:5),正式素材为 2048×2560。
- 源图里的牛仔裤是做旧蓝色。本征分解去掉了大片的洗白,改成浅色后还留有细碎的云纹,是从照片取纹理时带进来的洗水纹。黑色水洗、多色印花等其他花纹还没验证过,正式源图仍以白色或浅灰素面为主。
- 源图是竞品底图叠在白色上还原的,T 恤的轮廓由竞品的分割决定,我们的分割在这张图上只能沿着它走;分割和边缘透明度的效果要换自有源图再验证。
- T 恤的边缘在封面原图里就是台阶,改色后的边缘是平滑的,不改色时显示的封面仍是原样。

## 模型和许可

`MODEL_BACKEND=replicate` 时 ViTMatte、两个 Marigold 和 MoGe-2 改调 Replicate 上的社区封装:`superhighfives/vitmatte-trimap`、`jasonod888/marigold-intrinsics`、`jasonod888/marigold-normalsv2`、`jasonod888/moge2`,权重和下表相同,封装本身没写许可。

| 模型 | 用在 | 许可 |
| --- | --- | --- |
| SAM 3(Replicate 上的 `mattsays/sam3-image`,权重 `facebook/sam3`) | 按文字提示分割商品、部件和遮挡物 | SAM License(Meta 自定的条款),商用前要确认;走 Replicate 时源图会上传过去 |
| SAM 2.1(`facebook/sam2.1-hiera-base-plus`) | 按外接框和提示点分割(手点兜底) | Apache-2.0 |
| ViTMatte small(`hustvl/vitmatte-small-composition-1k`) | 边缘透明度 | 模型标注 Apache-2.0;训练数据 Composition-1k 出自 Adobe 的抠图数据集,商用前要确认该数据集的条款 |
| Marigold 本征分解(`prs-eth/marigold-iid-lighting-v1-1`) | 底图的明暗和光泽,去掉花纹 | OpenRAIL++,可商用,附使用用途限制 |
| Marigold 法线(`prs-eth/marigold-normals-v1-1`) | 流向图里的褶皱 | OpenRAIL++,可商用,附使用用途限制 |
| MoGe-2(`Ruicheng/moge-2-vitl-normal`) | 流向图里的大曲面和透视 | MIT(代码和模型) |
| pymatting(`estimate_foreground_ml`,算法库,不是模型) | 商品边缘像素背后的颜色 | MIT |
