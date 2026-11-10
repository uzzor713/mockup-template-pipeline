# 批量

`scripts/batch.py` 按品类预设给每张源图生成配置,跑完各步,出汇总表和抽查页。操作步骤见 [operator-guide.md](operator-guide.md),这里是规则和参数。

```bash
.venv/bin/python -I scripts/batch.py <源图文件夹 | 清单.csv> [选项]
```

| 选项 | 作用 |
| --- | --- |
| `--dry-run` | 只生成配置,列出各品类张数和真跑时每一步要跑几个模板,不跑 |
| `--only id,... \| id 清单.txt` | 只跑这几个 id(清单一行一个,`#` 开头的行不算),例外清单照样整份带上;汇总表和抽查页里其他模板保留上次的结果 |
| `--jobs N` | 同时跑几个模板 |
| `--local N` | 同一时间至多几个模板在跑本机计算的几步(默认 2),见[并发](#并发) |
| `--overrides 例外清单` | 例外清单 CSV 或放清单的文件夹,可以写多次,见[例外清单](#例外清单) |
| `--pack 例外清单.csv` | 把清单里这几张的源图和抽查图打包,见[抽查和调整](#抽查和调整) |
| `--force` | 全部重跑 |
| `--no-preview` | 不出预览 |

## 输入

### 按品类分好的文件夹

- 每个子文件夹是一个品类。文件夹名是预设名(vorla 的小品类 slug,如 `t-shirt`、`coffee-bag`),或写在预设的 `folders` 里(如 `"folders": ["T恤", "boxy-tee"]`)。
- 子文件夹里(含更深的子文件夹)的 png / jpg / webp 都算;以 `_` 开头的文件夹不算。
- 对不上预设的文件夹、不在品类文件夹里的图会列出来并停下,不会悄悄跳过。

**id 的取法**:品类文件夹名(不是纯英文时换成预设名)+ 文件名。

- 文件名去掉 `mockup-` 前缀和 `-blank` 后缀,转小写,字母数字以外的连成 `-`;
- 文件名本来就以文件夹名开头的不重复加。

| 源图 | id |
| --- | --- |
| `t-shirt/001.png` | `t-shirt-001` |
| `boxy-tee/mockup-boxy-tee-020-blank.webp` | `boxy-tee-020` |

生成的清单存在 `batches/<文件夹名>/list.csv`。

### 清单 CSV

源图没按文件夹分时,给一份清单,每行一张:

```csv
source,preset,id,occluders,crop
https://assets.vorla.ai/mockup-templates/boxy-tee/mockup-boxy-tee-001-blank.webp,t-shirt,,,
/Users/me/mockups/02 Studio club sweatshirt.png,sweatshirt,,hair;hand;arm,
```

| 列 | 内容 |
| --- | --- |
| `source` | 必填,源图的地址或本地路径。地址的下载到 `sources/<预设>/`,本地的直接引用 |
| `preset` | 必填,预设名 |
| `id` | 可选,不写由文件名得出,规则同上(不加文件夹名) |
| `occluders` | 可选,遮挡物提示词,`;` 隔开,替换预设里的(同名的沿用预设里的 `matteBand`);写 `-` 表示没有 |
| `crop` | 可选,`x0 y0 x1 y1`(源图像素),不写按预设的比例规则裁 |
| `tags` | 可选,标签,`;` 隔开,如 `channel/amazon`、`seasonal/christmas` |

## 例外清单

个别源图要改设置时用 `--overrides`。每行一张,`id` 对上 `list.csv` 里的,空着的列不改;给文件夹时里面的 csv 按文件名顺序读,后面的覆盖前面的。

```csv
id,issue,note,product,parts,occluders,crop,preset,skip
coffee-bag-07,部件不对,顶上封口条也印上了,,top of the bag,,,,
t-shirt-001,遮挡物不对,,,,hair;hand;arm,,,
poster-017,其他,画框也被改色,,,,,,1
```

| 列 | 内容 |
| --- | --- |
| `product` | 商品提示词 \*\* |
| `parts` | 不可印部件提示词,`;` 隔开,写 `-` 表示没有 \*\* |
| `occluders`、`crop`、`preset` | 同清单 CSV 的列 |
| `logo` | 默认 Logo 放法,填 `scripts/logo.py` 里 `PLACES` 的名字:`left-chest` 左胸、`chest` 胸口正中。Gemini 判断前襟判错时用 * |
| `solid` | 填 `1`:蒙版里的洞全补上。SAM 在画布、盒子、唱片这类空白面上漏掉大块时用 |
| `hull` | 填 `1`:蒙版换成最大一块的凸包。轮廓是凸的、缺口开在轮廓上补不成洞时用(唱片被纸套切掉一块、立着的台历) |
| `largest` | 填 `1`:分出的实例里只取面积最大的一个。几件同类商品分开摆、只印一件时用(桌上散开的几张牌、墙上几张海报) |
| `skip` | 填 `1`:不跑,也不进抽查页,`report.csv` 里记为 `skip` |
| `issue`、`note` | 抽查页导出的问题类型和备注,只给人看 |

\* 只能用在一件商品的预设上。

\*\* 只能用在一件商品、或几件商品说法相同(左右两件)的预设上,几件一起改。

> 每次运行都按源图和预设重新生成配置。改过的源图没带上它的例外清单,就会回到预设的设置、重算一遍。所以例外清单要一直留着,放进一个文件夹,每次都 `--overrides 那个文件夹`。

## 预设

`presets/<预设>.json`,字段同[模板配置](spec.md)的 `category`、`occluders`、`products`,另有:

| 字段 | 内容 |
| --- | --- |
| `folders` | 品类文件夹的别名 |
| `square` | 正方形源图裁成什么比例,默认 `[4, 5]` |
| `trim` | 多出的高度从哪边裁:`bottom` / `top` / `center`(默认) |

裁切规则:竖图裁成 4:5,横图 5:4,多出的宽度两边平分。裁好后不是 2048 × 2560(横版 2560 × 2048)的,配置里记上 `size`,各步骤先缩放到这个尺寸。

### 品类和标签

- vorla mockups 页面主树的每个小品类一份预设,文件名和 `category` 都是小品类的 slug。
- Seasonal、Channel 两条横轴是标签,不单独设预设:模板按画面里实际的商品选预设,清单的 `tags` 写 `seasonal/<页面>` 或 `channel/<页面>`。
- 主树里没有对应小品类的商品(`pet-tag`、`compact-mirror`、`ceramic-tile`、`ring-dish`、`bookmark`),预设的 `category` 是 `null`,模板只挂在横轴页面上,清单必须写 `tags`。
- 书签压在书上的照片用 `bookmark`,不用 `book`:书本类预设会选中下面的书封。书签上系的丝带算遮挡物,不跟着改色。
- 一件商品一个区域;`tracksuit` 是上衣、裤子两个区域,摊开的书和杂志是左右两页两个区域。`bottle`、`candle` 可编辑的是标签。

### 特殊预设

画面里是两件同类商品、或同一品类有特殊摆法的页面另设预设:

| 预设 | 画面 | `category` |
| --- | --- | --- |
| `front-and-back-hoodie`、`front-and-back-t-shirt` | 正面、背面两个区域 | `hoodie`、`t-shirt` |
| `mug-pair` | 左右两只杯子 | `mug` |
| `socks-pair` | 一双袜子分开摆,左右各一件 | `socks` |
| `can-pair`、`coaster-pair`、`frame-pair`、`cap-pair`、`tube-pair` | 两罐、两个杯垫、两幅挂画、两顶帽子、两个纸筒并排,左右各一件 | 同单件的预设 |
| `canvas-pair` | 两块无框画布挂墙(挂画页面),说法同 `canvas`,边上不往里收 | `frame` |
| `open-book`、`magazine-spread` | 摊开的书、杂志(magazine-ad、magazine-spread 页面),左右两页各一个区域,说法 page | `book`、`magazine` |
| `business-card-pair`、`greeting-card-pair`、`id-card-pair`、`playing-cards-pair` | 两张名片、贺卡、证件卡、扑克牌分开摆(并排、两只手各拿一张、一摞和一张),左右各一个区域。三张以上分开摆的用单件的预设,例外清单写 `largest`,只印最大的一张 | 同单件的预设 |
| `cd-case` | CD 盒和封面卡(cd-case、cd-cover 页面) | `cd` |
| `cd-sleeve` | CD 纸套 | `cd` |
| `zip-hoodie` | 敞开的拉链卫衣,可印区按左右两片前襟连着算 | `hoodie` |
| `canvas-poster` | 挂着的布面海报,备选说法 fabric、cloth、curtain | `poster` |
| `circle-sticker` | 圆形贴纸,备选说法 circle、round label | `sticker` |

### 哪些预设拿图试过

`t-shirt`、`sweatshirt`、`tote-bag`、`cap`、`pillow`、`bottle`、`coffee-bag`、`notebook`、`poster`、`paper-bag` 各拿一张图跑通过,其余的提示词没拿图试过。全量跑之前每类先抽几张跑一遍、看抽查页;分不出来的用 `experiments/sam3_replicate.py` 试别的说法。

提示词改了、对整个品类都更好的,改预设,不要一张张写例外清单。

## 并发

| 阶段 | 资源 | 怎么控制 |
| --- | --- | --- |
| 前四步(分割、本征分解、法线、几何) | `replicate` 时只等网络,每个进程约 150MB;`local` 时每个模板峰值 6–7GB | `--jobs`:`replicate` 可以多开,各模型一直有请求就不会反复冷启动;`local` 在 16GB 的机器上开 1 |
| 后三步(`pipeline.py`、`logo.py`、`preview.py`) | 本机算,每个峰值约 2GB、一两核,一个模板约 40 秒 | `--local N`:至多 N 个同时跑,其余排队,`--jobs` 开大时内存不跟着涨 |

`--local 2` 每分钟约能过 2 个模板,`--jobs` 开到十几个才会在后三步排队。

## 结果和重跑

结果在 `batches/<文件夹名或清单文件名>/`:

| 文件 | 内容 |
| --- | --- |
| `specs/` | 生成的配置 |
| `logs/<id>.log` | 各步输出 |
| `report.csv` | 每个模板的结果:`pass` / `fail`(带没通过的校验项)/ `error`(带出错的步骤和最后一行输出)/ `skip` |
| `review.html` | 抽查页,跑的过程中每分钟更新 |
| `command.json` | 最近一次运行的输入和例外清单,[修复进度表](fixes.md)按它打出重跑命令 |

重跑同一条命令时,各步记下上次用到的配置字段、源图内容和 `MODEL_BACKEND`(`work/<id>/steps.json`),没变的跳过:

| 改了什么 | 重跑哪几步 |
| --- | --- |
| 商品的 `logo` 设置(预设或例外清单的 `logo` 列) | `logo.py` 和预览 |
| 遮挡物、商品或部件的提示词 | 分割和之后三步(`pipeline.py`、`logo.py`、预览) |
| 商品的备选说法 `alternatives`、`print` 设置 | `pipeline.py`、`logo.py` 和预览,不重新分割 |
| 源图内容或裁切 | 全部 |
| 某一步的代码:`scripts/batch.py` 里 `STEPS` 这一步的版本号加 1 | 这一步,和 `pipeline.py`、`logo.py`、预览里排在它后面的 |
| 源图换了文件夹 | 不重跑 |

分割不看 `alternatives` 和 `print`:`print` 只给 `pipeline.py` 用;备选说法只在主说法分不出时才试,上次分成功的加了也一样,上次失败的没记比较值、照样会重跑。上次是放低阈值才分出来的,加了备选说法可能改用备选说法分,要重分时对这个模板单独跑 `run_segment.py`。早先记下的比较值含这两项,含不含都算对得上。

改了某一步的代码、同样的配置算出的结果会变时,要给这一步的版本号加 1,不然重跑会跳过它。改 `pipeline.py` 加它的版本号,只重算本机的三步,前面花钱的四步照样跳过。

真跑前先 `--dry-run`:最后一行列出真跑时每一步要跑几个模板。前四步(分割、本征分解、法线、几何)走 `MODEL_BACKEND`,用 Replicate 时按次计费。只想重算本机的几步却出现了前四步,是预设或例外清单在上次跑完后变过,先查清楚再跑。

`steps.json` 还记着每一步跑完的时间(`ran`,UTC)、没变跳过时核对的时间(`checked`)和所在的批次(`batch`),修复进度表按它认哪些模板修好后重跑过、该用哪一批的命令重跑。

## 抽查和调整

- `review.html` 是总览:各品类出错、没通过、通过的张数,点进品类页 `review/<品类>.html`。
- 品类页里出错、没通过的全列;通过的按 id 打散后先列 48 张当抽样,其余点开再看。
- 每张卡片两幅缩略图,点开看原图:
  - 分割叠图:绿 = 商品,蓝 = 不可印部件,红 = 遮挡物;
  - 蒙版示意:黄线 = 可印区域,红框 = 默认 Logo。

跑批量的人和调参数的人不是同一个时:

1. 跑的人在品类页点「有问题」,选问题类型、写备注(标记存在这台电脑的浏览器里),看完点页面底部「导出例外清单」。
2. 跑的人 `scripts/batch.py 源图文件夹 --pack 例外清单.csv`:这几张的源图(放在原来的品类文件夹下)、分割叠图和蒙版示意(`_review/`)连同清单打成 zip,存到 `batches/<文件夹名>/`,发给调参数的人。
3. 调参数的人解压,`scripts/batch.py 解压出的文件夹 --dry-run` 得到同样的 id,再用 `experiments/sam3_replicate.py batches/<zip 名>/specs/<id>.json "说法一" "说法二"` 试提示词(每个说法的结果画在 `work/<id>/experiments/sam3/sheet.png`),填好例外清单。可以带上清单真跑一遍、看抽查页确认;实在分不好的填 `skip`。
4. 清单发回去,跑的人放进例外清单文件夹,`scripts/batch.py 源图文件夹 --overrides 例外清单文件夹`,只重算这几张。
