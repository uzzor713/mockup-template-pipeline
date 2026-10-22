# Mockup 模板素材管线

把一张商品照片加工成 Mockup 模板要用的素材:半透明底图、每件商品的改色范围和可印区域、起伏图、配置。样本是一个两件商品(T 恤 + 牛仔裤)的模板。

编辑页的合成方式:每件商品先填上用户选的颜色,设计稿按所属商品的可印区域裁切并按流向图位移(大曲面层固定,褶皱层乘 Warping),最后在线性光下把底图按透明度盖在最上面,明暗、纹理和光泽都在底图里。

## 运行

需要 Python 3.12。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-models.txt   # segment.py、intrinsic.py、normals.py、geometry.py 要用;其余脚本只需 requirements.txt
.venv/bin/pip install --no-deps -r requirements-moge.txt   # geometry.py 要用

.venv/bin/python -I fetch_sources.py   # 下载竞品素材到 sources/
.venv/bin/python -I segment.py         # SAM 2.1 分割 T 恤、袖子、遮挡物和牛仔裤,ViTMatte 求边缘透明度 → work/ 下的透明度和蒙版
.venv/bin/python -I intrinsic.py       # Marigold 本征分解 → work/albedo.png、shading.png、residual.png
.venv/bin/python -I normals.py         # Marigold 法线 → work/normals.png
.venv/bin/python -I geometry.py        # MoGe-2 的 3D 坐标和相机内参 → work/points.npy、geometry.json
.venv/bin/python -I pipeline.py        # 生成 output/,并跑自动校验
.venv/bin/python -I preview.py         # 合成 preview.png、edges.png、masks.png
```

四个模型脚本首次运行会从 Hugging Face 下载模型,`segment.py` 约 0.4GB,`intrinsic.py`、`normals.py` 各约 2.6GB,`geometry.py` 约 1.3GB。模型下载好后,在 16GB 内存的 M1 上 `segment.py` 约 20 秒、`intrinsic.py` 约 22 秒、`normals.py` 约 20 秒、`geometry.py` 约 10 秒,内存峰值约 6–7GB。Marigold 是扩散模型,随机种子固定,重复运行结果一致。

`sources/`、`work/`、`output/` 和根目录的预览图都不进仓库,按上面的步骤在本地生成。竞品素材只作内部示例,不能用于产品。

## 脚本

| 文件 | 做什么 |
| --- | --- |
| `fetch_sources.py` | 下载竞品 Mockey 免费模板 04674 的底图(叠在白色上当源图)和可印区域蒙版(只用来对比),并核对是否与生成样本时的文件一致 |
| `segment.py` | 分割:SAM 2.1 按外接框和提示点分出包带和包(遮挡物)、T 恤、两只袖子和牛仔裤(牛仔裤的外接框由颜色加形态学的方法给出),ViTMatte 在商品和背景、遮挡物的交界求透明度,结果给 `pipeline.py` 用;对比图 `work/tee-compare.png`(T 恤和竞品模板自带的商品区)、`work/jeans-compare.png`(牛仔裤的两种方法) |
| `intrinsic.py` | 本征分解:Marigold 把照片拆成固有色、光照和残差,给 `pipeline.py` 出底图的明暗和光泽,去掉商品上的花纹 |
| `normals.py` | 法线:Marigold 估计每个像素的表面朝向,给 `pipeline.py` 出起伏图里的褶皱 |
| `geometry.py` | 几何:MoGe-2 估计每个像素的 3D 坐标和相机内参,给 `pipeline.py` 出起伏图里的大曲面和透视 |
| `pipeline.py` | 底图、流向图、配置、自动校验 |
| `preview.py` | 只读 `output/` 里的文件合成预览,和编辑页用同一套公式 |

## 产出

`output/` 里是模板本身:

| 文件 | 内容 |
| --- | --- |
| `cover.webp` | 模板封面(原图) |
| `base.webp` | 底图:T 恤、牛仔裤区域半透明,在线性光下合成时 1 − 透明度 = 明暗 × 纹理、颜色 × 透明度 = 光泽 × 纹理,不含花纹;背景、皮肤、包带不透明 |
| `product-1.png` / `product-2.png` | T 恤 / 牛仔裤的改色范围(白色 + 透明度) |
| `print-1.png` / `print-2.png` | T 恤 / 牛仔裤的可印区域(白色 + 透明度) |
| `surface-1.png` / `surface-2.png` | T 恤 / 牛仔裤的大曲面层流向图,固定叠上,不随 Warping 变:R / G = x / y 方向的位移,中灰 = 不位移,通道值 0 / 255 对应 ∓`surfaceRangePx`;B 未用;1/4 尺寸(400×500),使用时双线性放大 |
| `folds-1.png` / `folds-2.png` | T 恤 / 牛仔裤的褶皱层流向图,格式同上,值是 Warping 100% 时的位移,通道值对应 ∓`foldsRangePx`,使用时乘 Warping |
| `template.json` | 部件名、默认 Logo 框、合成方式 `blend: "linear"`、两层流向图的 `displacement.surfaceRangePx`、`foldsRangePx`、生成参数 |
| `checks.json` | 自动校验结果(12 项) |

示意图:

- `preview.png`:只读上面这些文件合成的效果。从左到右依次是封面、白 / 白、红 / 藏青、黑 / 黑(Warping 50%)、芥末黄 / 橄榄绿(Warping 100%)。
- `edges.png`:六处边缘(袖口、肩线、T 恤压牛仔裤、包带、裤裆、裤脚压鞋)按像素放大 3 倍,左起封面、红 / 黑、黑 / 芥末黄。
- `masks.png`:绿色为 T 恤商品区,蓝色为牛仔裤商品区,黄线为可印区域,红框为默认 Logo 位置。

## 怎么生成的

- 分割:本地运行的 SAM 2.1 以外接框和几个提示点为输入,分出包带和包(遮挡物)、T 恤、两只袖子和牛仔裤,T 恤减去遮挡物;ViTMatte 在商品和背景、遮挡物的交界几个像素内求透明度。源图里的 T 恤是从竞品底图还原出来的,轮廓本身是台阶,抠图救不回来,所以 T 恤按二值来源处理:平滑轮廓后重出软边。牛仔裤和 T 恤的交界不算边缘,牛仔裤在 T 恤下面接着铺。
- 可印区域按规则取:T 恤减去袖子,领口、下摆按竖直方向往里收 1% 宽度,左右两侧和包带边上不收,设计稿一直铺到轮廓,被包带挡住的部分由底图盖住;牛仔裤取大腿上段。`pipeline.py` 最后会打印和竞品模板自带蒙版的重合度,样本 T 恤商品区 0.993、可印区域 0.904。
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
| SAM 2.1(`facebook/sam2.1-hiera-base-plus`) | 分割 T 恤、袖子、遮挡物和牛仔裤 | Apache-2.0 |
| ViTMatte small(`hustvl/vitmatte-small-composition-1k`) | 边缘透明度 | 模型标注 Apache-2.0;训练数据 Composition-1k 出自 Adobe 的抠图数据集,商用前要确认该数据集的条款 |
| Marigold 本征分解(`prs-eth/marigold-iid-lighting-v1-1`) | 底图的明暗和光泽,去掉花纹 | OpenRAIL++,可商用,附使用用途限制 |
| Marigold 法线(`prs-eth/marigold-normals-v1-1`) | 起伏图里的褶皱 | OpenRAIL++,可商用,附使用用途限制 |
| MoGe-2(`Ruicheng/moge-2-vitl-normal`) | 起伏图里的大曲面和透视 | MIT(代码和模型) |
