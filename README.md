# Mockup 模板素材管线

把一张商品照片加工成 Mockup 模板要用的素材:半透明底图、每件商品的改色范围和可印区域、起伏图、配置。样本是一个两件商品(T 恤 + 牛仔裤)的模板。

编辑页的合成方式:每件商品先填上用户选的颜色,设计稿按所属商品的可印区域裁切并按流向图位移,最后把底图按透明度盖在最上面,明暗都在底图里。

## 运行

需要 Python 3.12。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-models.txt   # segment.py、intrinsic.py 要用;其余脚本只需 requirements.txt

.venv/bin/python -I fetch_sources.py   # 下载竞品素材到 sources/
.venv/bin/python -I segment.py         # SAM 2.1 分割牛仔裤、ViTMatte 求边缘透明度 → work/jeans-alpha.png
.venv/bin/python -I intrinsic.py       # Marigold 本征分解 → work/albedo.png、shading.png、residual.png
.venv/bin/python -I pipeline.py        # 生成 output/,并跑自动校验
.venv/bin/python -I preview.py         # 合成 preview.png、edges.png、masks.png
```

两个模型脚本首次运行会从 Hugging Face 下载模型,`segment.py` 约 0.4GB,`intrinsic.py` 约 2.6GB。模型下载好后,在 16GB 内存的 M1 上 `segment.py` 约 16 秒、`intrinsic.py` 约 22 秒,内存峰值约 6–7GB。Marigold 是扩散模型,随机种子固定,重复运行结果一致。

`sources/`、`work/`、`output/` 和根目录的预览图都不进仓库,按上面的步骤在本地生成。竞品素材只作内部示例,不能用于产品。

## 脚本

| 文件 | 做什么 |
| --- | --- |
| `fetch_sources.py` | 下载竞品 Mockey 免费模板 04674 的底图和可印区域蒙版,并核对是否与生成样本时的文件一致 |
| `segment.py` | 分割牛仔裤:颜色加形态学的方法给出外接框,SAM 2.1 出蒙版,ViTMatte 在边界两侧求透明度,结果给 `pipeline.py` 用;颜色加形态学和 SAM 的对比图在 `work/jeans-compare.png` |
| `intrinsic.py` | 本征分解:Marigold 把照片拆成固有色、光照和残差,给 `pipeline.py` 去掉商品上的花纹 |
| `pipeline.py` | 底图、流向图、配置、自动校验 |
| `preview.py` | 只读 `output/` 里的文件合成预览,和编辑页用同一套公式 |

## 产出

`output/` 里是模板本身:

| 文件 | 内容 |
| --- | --- |
| `cover.webp` | 模板封面(原图) |
| `base.webp` | 底图:T 恤、牛仔裤区域换成半透明灰,明暗都在这层里,不含花纹;背景、皮肤、包带不透明 |
| `product-1.png` / `product-2.png` | T 恤 / 牛仔裤的改色范围(白色 + 透明度) |
| `print-1.png` / `print-2.png` | T 恤 / 牛仔裤的可印区域(白色 + 透明度) |
| `displacement-1.png` / `displacement-2.png` | T 恤 / 牛仔裤的流向图:R / G = Warping 100% 时 x / y 方向的位移,中灰 = 不位移,通道值 0 / 255 对应 ∓`rangePx`;B 未用;1/4 尺寸(400×500),使用时双线性放大 |
| `template.json` | 部件名、默认 Logo 框、流向图的 `displacement.rangePx`、生成参数 |
| `checks.json` | 自动校验结果(12 项) |

示意图:

- `preview.png`:只读上面这些文件合成的效果。从左到右依次是封面、白 / 白、红 / 藏青、黑 / 黑(Warping 50%)、芥末黄 / 橄榄绿(Warping 100%)。
- `edges.png`:六处边缘(袖口、肩线、T 恤压牛仔裤、包带、裤裆、裤脚压鞋)按像素放大 3 倍,左起封面、红 / 黑、黑 / 芥末黄。
- `masks.png`:绿色为 T 恤商品区,蓝色为牛仔裤商品区,黄线为可印区域,红框为默认 Logo 位置。

## 怎么生成的

- 分割:T 恤取自竞品模板里已分割好的结果,竞品的边缘是二值台阶,平滑轮廓后重出软边;牛仔裤用本地运行的 SAM 2.1 分割(以外接框和几个提示点为输入),再用 ViTMatte 在边界两侧几个像素内求透明度。牛仔裤和 T 恤的交界不算边缘,牛仔裤在 T 恤下面接着铺。牛仔裤的可印区域按规则取大腿上段。
- 明暗:Marigold 把照片拆成固有色、光照和残差。底图和流向图用的是「这件商品如果是白色」的亮度(光照 + 残差,两者的比例按每件商品拟合),做旧洗白、染料这些花纹留在固有色里,改色时不会带出来,也不会被当成褶皱。模型按长边 768 处理,更细的纹理从照片补回。
- 模型只用在分割、边缘透明度和本征分解,其余步骤都是固定代码。
- 流向图:位移沿明暗起伏的坡度,褶皱横着就上下压缩、竖着就左右压缩、斜着就按对应角度压缩。Warping 100% 时各方向每像素前进量限制在 0.5–2.5(拉伸至多 2 倍、压缩至多 2.5 倍),超出的地方局部减弱,深褶处设计稿不会被拉成一片或翻折。坡度系数每件商品各自定:Warping 100% 时可印区域里 20% 的面积刚好顶到这个上限,所以 100% 在任何模板上都是不翻折的前提下最强的效果;系数有上限,几乎没褶的商品不会把噪点放大成假褶皱。

## 和正式素材的差别

- 尺寸是 1600×2000(4:5),正式素材为 2048×2560。
- 源图里的牛仔裤是做旧蓝色。本征分解去掉了大片的洗白,改成白色后还留有细碎的云纹,是从照片补回细纹理时带进来的洗水纹。黑色水洗、多色印花等其他花纹还没验证过,正式源图仍以白色或浅灰素面为主。
- T 恤的边缘在封面原图里就是台阶,改色后的边缘是平滑的,不改色时显示的封面仍是原样。

## 模型和许可

| 模型 | 用在 | 许可 |
| --- | --- | --- |
| SAM 2.1(`facebook/sam2.1-hiera-base-plus`) | 分割牛仔裤 | Apache-2.0 |
| ViTMatte small(`hustvl/vitmatte-small-composition-1k`) | 边缘透明度 | 模型标注 Apache-2.0;训练数据 Composition-1k 出自 Adobe 的抠图数据集,商用前要确认该数据集的条款 |
| Marigold 本征分解(`prs-eth/marigold-iid-lighting-v1-1`) | 去掉花纹 | OpenRAIL++,可商用,附使用用途限制 |
