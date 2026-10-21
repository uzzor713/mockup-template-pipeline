# Mockup 模板素材管线

把一张商品照片加工成 Mockup 模板要用的素材:半透明底图、每件商品的改色范围和可印区域、起伏图、配置。样本是一个两件商品(T 恤 + 牛仔裤)的模板。

编辑页的合成方式:每件商品先填上用户选的颜色,设计稿按所属商品的可印区域裁切并按起伏图位移,最后把底图按透明度盖在最上面,明暗都在底图里。

## 运行

需要 Python 3.12。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-sam.txt   # 不重跑 SAM 分割的话,装 requirements.txt 就够

.venv/bin/python -I fetch_sources.py   # 下载竞品素材到 sources/
.venv/bin/python -I segment.py         # SAM 2.1 分割牛仔裤 → work/jeans-mask.png;首次运行会从 Hugging Face 下载模型
.venv/bin/python -I pipeline.py        # 生成 output/,并跑自动校验
.venv/bin/python -I preview.py         # 合成 preview.png、edges.png、masks.png
```

`sources/`、`work/`、`output/` 和根目录的预览图都不进仓库,按上面的步骤在本地生成。竞品素材只作内部示例,不能用于产品。

## 脚本

| 文件 | 做什么 |
| --- | --- |
| `fetch_sources.py` | 下载竞品 Mockey 免费模板 04674 的底图和可印区域蒙版,并核对是否与生成样本时的文件一致 |
| `segment.py` | 分割牛仔裤:传统的颜色加形态学方法和 SAM 2.1 各做一遍,对比图在 `work/jeans-compare.png`;SAM 的结果给 `pipeline.py` 用 |
| `pipeline.py` | 边缘细化、底图、起伏图、配置、自动校验 |
| `preview.py` | 只读 `output/` 里的文件合成预览,和编辑页用同一套公式 |

## 产出

`output/` 里是模板本身:

| 文件 | 内容 |
| --- | --- |
| `cover.webp` | 模板封面(原图) |
| `base.webp` | 底图:T 恤、牛仔裤区域换成半透明灰,明暗都在这层里;背景、皮肤、包带不透明 |
| `product-1.png` / `product-2.png` | T 恤 / 牛仔裤的改色范围(白色 + 透明度) |
| `print-1.png` / `print-2.png` | T 恤 / 牛仔裤的可印区域(白色 + 透明度) |
| `displacement.png` | 起伏图,每件商品一个通道(R = T 恤、G = 牛仔裤、B 未用),中灰 = 不位移;1/4 尺寸(400×500),使用时双线性放大 |
| `template.json` | 部件名、默认 Logo 框、生成参数 |
| `checks.json` | 自动校验结果(11 项) |

示意图:

- `preview.png`:只读上面这些文件合成的效果。从左到右依次是封面、白 / 白、红 / 藏青、黑 / 黑、芥末黄 / 橄榄绿(Warping 100%)。
- `edges.png`:六处边缘(袖口、肩线、T 恤压牛仔裤、包带、裤裆、裤脚压鞋)按像素放大 3 倍,左起封面、红 / 黑、黑 / 芥末黄。
- `masks.png`:绿色为 T 恤商品区,蓝色为牛仔裤商品区,黄线为可印区域,红框为默认 Logo 位置。

## 怎么生成的

- 分割:T 恤取自竞品模板里已分割好的结果,竞品的边缘是二值台阶,平滑轮廓后重出软边;牛仔裤用本地运行的 SAM 2.1 分割(以外接框和几个提示点为输入),再在边界两侧按颜色细化软边。牛仔裤的可印区域按规则取大腿上段。
- 其余步骤(边缘细化、底图、起伏图、配置、校验)都是固定代码,没有模型调用。

## 和正式素材的差别

- 尺寸是 1600×2000(4:5),正式素材为 2048×2560。
- 源图里的牛仔裤是做旧蓝色,洗白处被当成了高光,改成白色时看起来像银色。正式源图的商品应为白色或浅灰素面。
- T 恤的边缘在封面原图里就是台阶,改色后的边缘是平滑的,不改色时显示的封面仍是原样。
