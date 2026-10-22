# Mockup 模板素材管线

把一张商品照片加工成 Mockup 模板要用的素材:半透明底图、每件商品的改色范围和可印区域、流向图、配置。每个模板写一份配置(`specs/<模板>.json`),脚本本身不含具体模板的内容;样本是一个两件商品(T 恤 + 牛仔裤)的模板,配置是 `specs/sample-tee-jeans.json`。

编辑页的合成方式:每件商品先填上用户选的颜色,设计稿按所属商品的可印区域裁切并按流向图位移(大曲面层固定,褶皱层乘 Warping),最后在线性光下把底图按透明度盖在最上面,明暗、纹理和光泽都在底图里。

## 运行

需要 Python 3.12。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-models.txt   # run_segment.py、intrinsic.py、normals.py、geometry.py 要用;其余脚本只需 requirements.txt
.venv/bin/pip install --no-deps -r requirements-moge.txt   # geometry.py 要用

SPEC=specs/sample-tee-jeans.json
.venv/bin/python -I fetch_sources.py $SPEC   # 下载配置里 fetch 列出的素材
.venv/bin/python -I run_segment.py $SPEC     # SAM 2.1 分割商品、部件和遮挡物,ViTMatte 求边缘透明度
.venv/bin/python -I intrinsic.py $SPEC       # Marigold 本征分解
.venv/bin/python -I normals.py $SPEC         # Marigold 法线
.venv/bin/python -I geometry.py $SPEC        # MoGe-2 的 3D 坐标和相机内参
.venv/bin/python -I pipeline.py $SPEC        # 生成模板文件,并跑自动校验
.venv/bin/python -I preview.py $SPEC         # 合成预览图
```

每个模板的中间结果在 `work/<id>/`,模板文件在 `output/<id>/`,预览图在 `preview/<id>/`,`<id>` 是配置里的 `id`。

四个模型脚本首次运行会从 Hugging Face 下载模型,`run_segment.py` 约 0.4GB,`intrinsic.py`、`normals.py` 各约 2.6GB,`geometry.py` 约 1.3GB。模型下载好后,在 16GB 内存的 M1 上 `run_segment.py` 约 30 秒、`intrinsic.py` 约 22 秒、`normals.py` 约 20 秒、`geometry.py` 约 10 秒,内存峰值约 6–7GB。Marigold 是扩散模型,随机种子固定,重复运行结果一致。

`sources/`、`work/`、`output/`、`preview/` 都不进仓库,按上面的步骤在本地生成。样本用的竞品素材只作内部示例,不能用于产品。

## 脚本

| 文件 | 做什么 |
| --- | --- |
| `config.py` | 读模板配置和源图,给出每个模板的 `work/`、`output/`、`preview/` 目录,各脚本共用 |
| `fetch_sources.py` | 按配置的 `fetch` 下载素材,并核对是否与生成样本时的文件一致 |
| `segment.py` | 分割的函数:SAM 2.1 按外接框和提示点分出遮挡物、商品和不可印部件,ViTMatte 在商品和背景、遮挡物的交界求透明度。不读写文件 |
| `run_segment.py` | 按配置调 `segment.py`,存每件商品的透明度和部件蒙版、遮挡物蒙版,以及人工检查用的叠图 `segment-check.png` |
| `intrinsic.py` | 本征分解:Marigold 把照片拆成固有色、光照和残差,给 `pipeline.py` 出底图的明暗和光泽,去掉商品上的花纹 |
| `normals.py` | 法线:Marigold 估计每个像素的表面朝向,给 `pipeline.py` 出流向图里的褶皱 |
| `geometry.py` | 几何:MoGe-2 估计每个像素的 3D 坐标和相机内参,给 `pipeline.py` 出流向图里的大曲面和透视 |
| `pipeline.py` | 改色范围、可印区域、底图、流向图、`template.json`、自动校验 |
| `preview.py` | 只读模板文件合成预览,和编辑页用同一套公式 |

`experiments/` 里是试验脚本,不在上面的流程里,结果写到 `work/<id>/experiments/`:

| 文件 | 做什么 |
| --- | --- |
| `sam3_replicate.py` | 调 Replicate 上的 SAM 3 按文字提示分割,需要 `REPLICATE_API_TOKEN`,会把源图上传到 Replicate |
| `detect_boxes.py` | 开放词汇检测模型(默认 Grounding DINO)按文字找出各部分的外接框 |
| `sam21_auto.py` | 检测出的框直接当 SAM 2.1 的提示,和手点提示的结果对比 |
| `seam_snap.py` | 以 `sam21_auto.py` 的袖子为大致范围,沿照片里的接缝(细暗线)重新切分袖子和身体 |

## 配置

`specs/<模板>.json`,路径都相对仓库根目录:

| 字段 | 内容 |
| --- | --- |
| `id`、`category` | 模板 id(也是 `work/`、`output/`、`preview/` 下的目录名)、品类,写进 `template.json` |
| `photo` | 源图。带透明度的叠在白色上;透明度不到 1 的像素视为不是原始照片(如从竞品底图还原的商品区),这些像素落在商品区外时,底图里换成周围的背景色 |
| `fetch` | 可选,要下载的素材:`{路径: {url, sha256}}` |
| `occluders` | 遮挡物的提示列表,每项 `{box: [x0, y0, x1, y1], points: [[x, y, 1 正 / 0 负], ...]}`,`note` 只作说明 |
| `products` | 商品列表,由上层到下层,每件:`id`(文件名用)、`key`、`name`(写进 `template.json`)、`box` 和 `points`(分割提示)、可选的 `parts`(不可印部件的提示列表,如袖子)、`print`、可选的 `binaryEdge`、`recolorable`(默认 `true`)、`reference` |
| `products[].print` | 可印区域规则:商品区减去 `parts`,`inset: [x, y]` 按宽度的比例从商品和背景的交界往里收(部件、遮挡物和上层商品的边上不收),`rows: [从, 到]` 只留这段高度(占商品高度的比例) |
| `products[].binaryEdge` | 源图里这件商品的轮廓是台阶(如从竞品底图还原的),抠图救不回来:按二值来源处理,平滑轮廓后重出软边 |
| `products[].reference` | 可选,对比用的参考蒙版 `{product, print}`,每项 `{file, transparent}`;`transparent` 取图里半透明的部分,否则取白色 × 透明度。`pipeline.py` 最后打印重合度,不算校验 |
| `preview` | 预览的裁切范围 `crop`(占宽高的比例)和边缘放大的取样位置 `spots`(`[名称, x, y]`,模板像素) |

## 产出

`output/<id>/` 里是模板本身,以样本为例:

| 文件 | 内容 |
| --- | --- |
| `cover.webp` | 模板封面(原图) |
| `base.webp` | 底图:T 恤、牛仔裤区域半透明,在线性光下合成时 1 − 透明度 = 明暗 × 纹理、颜色 × 透明度 = 光泽 × 纹理,不含花纹;背景、皮肤、包带不透明 |
| `product-1.png` / `product-2.png` | T 恤 / 牛仔裤的改色范围(白色 + 透明度),编号按配置里商品的顺序 |
| `print-1.png` / `print-2.png` | T 恤 / 牛仔裤的可印区域(白色 + 透明度) |
| `surface-1.png` / `surface-2.png` | T 恤 / 牛仔裤的大曲面层流向图,固定叠上,不随 Warping 变:R / G = x / y 方向的位移,中灰 = 不位移,通道值 0 / 255 对应 ∓`surfaceRangePx`;B 未用;1/4 尺寸(400×500),使用时双线性放大 |
| `folds-1.png` / `folds-2.png` | T 恤 / 牛仔裤的褶皱层流向图,格式同上,值是 Warping 100% 时的位移,通道值对应 ∓`foldsRangePx`,使用时乘 Warping |
| `template.json` | 部件名、默认 Logo 框、合成方式 `blend: "linear"`、两层流向图的 `displacement.surfaceRangePx`、`foldsRangePx`、生成参数 |
| `checks.json` | 自动校验结果(12 项) |

`preview/<id>/` 里是示意图:

- `preview.png`:只读模板文件合成的效果。从左到右依次是封面、白 / 白、红 / 藏青、黑 / 黑(Warping 50%)、芥末黄 / 橄榄绿(Warping 100%),颜色按商品顺序。
- `edges.png`:配置里的几处边缘按像素放大 3 倍,左起封面、红 / 黑、黑 / 芥末黄。样本是袖口、肩线、T 恤压牛仔裤、包带、裤裆、裤脚压鞋。
- `masks.png`:各件商品区依次着绿、蓝、紫,黄线为可印区域,红框为默认 Logo 位置。

## 怎么生成的

以样本为例:

- 分割:本地运行的 SAM 2.1 以外接框和几个提示点为输入,分出包带和包(遮挡物)、T 恤、两只袖子和牛仔裤,T 恤减去遮挡物;ViTMatte 在商品和背景、遮挡物的交界几个像素内求透明度。源图里的 T 恤是从竞品底图还原出来的,轮廓本身是台阶,抠图救不回来,所以 T 恤按二值来源处理:平滑轮廓后重出软边。牛仔裤和 T 恤的交界不算边缘,牛仔裤在 T 恤下面接着铺。
- 可印区域按配置里的规则取,只从商品和背景的交界往里收;袖子、包带和上层商品的边上不收,设计稿铺到被挡住的地方,由底图或上层商品盖住,商品和包带之间几像素宽的缝算包带。T 恤减去袖子,领口、下摆按竖直方向往里收 1% 宽度,左右两侧不收;牛仔裤四周往里收 1% 宽度(T 恤下摆盖住的地方不收),取裤腰以下到大腿中段。袖子蒙版和轮廓、包带之间夹出的零碎小块去掉。和竞品模板自带的蒙版相比,T 恤商品区重合度 0.993、可印区域 0.912。
- 底图:Marigold 把照片拆成固有色、光照和残差,两者的比例按每件商品拟合。底图是「这件商品如果是白色」的样子,分三部分:明暗(光照)、光泽(残差,高光这类不随颜色变的反光)和纹理(照片里比长边 / 256 更细的起伏)。明暗和纹理乘在用户选的颜色上,光泽乘上纹理后叠加在最上面,所以深色衣服也有反光和布纹。做旧洗白、染料这些花纹留在固有色里,改色时不会带出来。「白色」按白布料的固有色算(样本白 T 恤在本征分解里约 0.87),选白色时和照片一样亮,不额外提亮。白色的结果不超过 1,亮处超出的部分从明暗里扣。
- 流向图分两层,分开存:
  - 大曲面:MoGe-2 给出每个像素的 3D 坐标,由此算出屏幕上每像素在曲面上实际对应多长(以商品中位深度处正对镜头时为 1),在商品区内平滑约 32px(宽度的 2%):身体、裤腿转向侧面的地方压缩,轮廓内一二十像素压得最多,至多 2 倍;离镜头近的地方略放大,透视也在里面。再用最小二乘积分成位移。这层按形状算出来就是实际的样子,固定叠上,不随 Warping 变。
  - 褶皱:起伏取 Marigold 法线的 z 分量,也就是表面正对镜头的程度。位移沿它的坡度,褶皱横着就上下压缩、竖着就左右压缩、斜着就按对应角度压缩。只看表面朝向,和照片的光线方向、花纹都无关。坡度系数每件商品各自定:Warping 100% 时可印区域里 20% 的面积刚好顶到下面的上限;系数有上限,几乎没褶的商品不会把噪点放大成假褶皱。
  - Warping 只缩放褶皱层。Warping 0–100% 之间两层合起来,各方向每像素前进量都限制在 0.5–2.5(拉伸至多 2 倍、压缩至多 2.5 倍),超出的地方局部减弱褶皱那层,深褶处设计稿不会被拉成一片或翻折;大曲面层的位移幅度大、8 位量化误差也大,减弱时多留出这部分余量。
  - 位移会沿曲面一路累积,渲染时减去设计稿中心处的位移(两层各自减),中心不动、两边按曲面压扁。
- 模型只用在分割、边缘透明度、本征分解、法线和 3D 坐标,其余步骤都是固定代码。

## 和正式素材的差别

- 尺寸是 1600×2000(4:5),正式素材为 2048×2560。
- 源图里的牛仔裤是做旧蓝色。本征分解去掉了大片的洗白,改成浅色后还留有细碎的云纹,是从照片取纹理时带进来的洗水纹。黑色水洗、多色印花等其他花纹还没验证过,正式源图仍以白色或浅灰素面为主。
- 源图是竞品底图叠在白色上还原的,T 恤的轮廓由竞品的分割决定,我们的分割在这张图上只能沿着它走;分割和边缘透明度的效果要换自有源图再验证。
- T 恤的边缘在封面原图里就是台阶,改色后的边缘是平滑的,不改色时显示的封面仍是原样。

## 模型和许可

| 模型 | 用在 | 许可 |
| --- | --- | --- |
| SAM 2.1(`facebook/sam2.1-hiera-base-plus`) | 分割商品、部件和遮挡物 | Apache-2.0 |
| ViTMatte small(`hustvl/vitmatte-small-composition-1k`) | 边缘透明度 | 模型标注 Apache-2.0;训练数据 Composition-1k 出自 Adobe 的抠图数据集,商用前要确认该数据集的条款 |
| Marigold 本征分解(`prs-eth/marigold-iid-lighting-v1-1`) | 底图的明暗和光泽,去掉花纹 | OpenRAIL++,可商用,附使用用途限制 |
| Marigold 法线(`prs-eth/marigold-normals-v1-1`) | 流向图里的褶皱 | OpenRAIL++,可商用,附使用用途限制 |
| MoGe-2(`Ruicheng/moge-2-vitl-normal`) | 流向图里的大曲面和透视 | MIT(代码和模型) |
