# Mockup 模板素材管线

把一张空白底商品照片加工成 vorla Mockup 模板:半透明底图、每件商品的改色范围和可印区域、流向图、`template.json`。

## 目录

| 目录 | 内容 | 进仓库 |
| --- | --- | --- |
| `scripts/` | 管线脚本,命令都在仓库根目录执行 | 是 |
| `presets/` | 品类预设,每个小品类一份 | 是 |
| `experiments/` | 试验脚本,不在流程里 | 是 |
| `docs/` | 文档 | 是 |
| `requirements/` | 依赖 | 是 |
| `sources/` | 源图、清单、例外清单 | 否 |
| `batches/<批次>/` | 每批的配置、日志、汇总表、抽查页 | 否 |
| `work/<id>/` | 每个模板的中间结果 | 否 |
| `output/<id>/` | 模板文件 | 否 |
| `preview/<id>/` | 预览图 | 否 |
| `inbox/` | 整理好、待上传 OSS 的模板 | 否 |

## 安装

需要 Python 3.12。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/base.txt
```

Replicate 的 token 写进仓库根目录的 `.env.local`(不进仓库),一行 `REPLICATE_API_TOKEN=...`。模型默认都在 Replicate 上跑,本机不需要显卡;要在本机跑见 [docs/models.md](docs/models.md)。

## 常用命令

```bash
# 批量:源图按品类分文件夹,文件夹名是预设名
.venv/bin/python -I scripts/batch.py 源图 --dry-run                    # 只生成配置,核对各品类张数
.venv/bin/python -I scripts/batch.py 源图 --jobs 16 --overrides 例外清单  # 正式跑;重跑同一条命令只算变了的
.venv/bin/python -I scripts/batch.py 源图 --pack 例外清单.csv          # 把有问题的几张打包给调参数的人

# 上传:整理通过的模板
.venv/bin/python -I scripts/inbox.py batches/<批次> inbox/<日期>
```

单独重跑一个模板的某一步,给它的配置:

```bash
SPEC=batches/<批次>/specs/<id>.json
.venv/bin/python -I scripts/run_segment.py $SPEC   # 分割
.venv/bin/python -I scripts/intrinsic.py $SPEC     # 本征分解
.venv/bin/python -I scripts/normals.py $SPEC       # 法线
.venv/bin/python -I scripts/geometry.py $SPEC      # 3D 坐标
.venv/bin/python -I scripts/pipeline.py $SPEC      # 生成模板文件、自动校验
.venv/bin/python -I scripts/logo.py $SPEC          # 只重算默认 Logo 框
.venv/bin/python -I scripts/preview.py $SPEC       # 预览图
```

## 脚本

| 脚本 | 做什么 |
| --- | --- |
| `batch.py` | 按品类预设给每张源图生成配置、跑完各步,出汇总表和抽查页 |
| `review.py` | 抽查页,`batch.py` 调用 |
| `inbox.py` | 把通过的模板整理成上传用的文件夹;重跑过的已发布模板换文件、升版本 |
| `wall.py` | 拉测试站或正式站的模板墙:每个模板的版本、显示名、小品类、标签、最后发布的时间 |
| `fetch_sources.py` | 按配置的 `fetch` 下载素材并核对 sha256 |
| `run_segment.py` | 分割:商品、不可印部件、遮挡物的蒙版和边缘透明度 |
| `intrinsic.py` | 本征分解:固有色、光照、残差 |
| `normals.py` | 法线 |
| `geometry.py` | 3D 坐标和相机内参 |
| `pipeline.py` | 改色范围、可印区域、底图、流向图、`template.json`、自动校验 |
| `logo.py` | 默认 Logo 框 |
| `preview.py` | 只读模板文件合成预览 |
| `segment.py`、`remote.py`、`config.py` | 分割函数、Replicate 调用、配置和密钥读取,各脚本共用 |

试验脚本在 `experiments/`,结果写到 `work/<id>/experiments/`:

| 脚本 | 做什么 |
| --- | --- |
| `sam3_replicate.py` | 一次试几个 SAM 3 说法,给新品类挑提示词 |
| `detect_boxes.py` | 开放词汇检测(默认 Grounding DINO)按文字找各部分的外接框 |
| `sam21_auto.py` | 检测出的框直接当 SAM 2.1 的提示,和手点对比 |
| `seam_snap.py` | 沿照片里的接缝重新切分袖子和身体 |
| `compare_blend.py` | 模板墙 CSS 叠图和线性光合成的对比 |

## 产出

`output/<id>/` 是模板本身,第 n 件商品按配置里的顺序编号:

| 文件 | 内容 |
| --- | --- |
| `cover.webp` | 封面(原图) |
| `base.webp` | 底图:商品区半透明,含明暗、纹理和光泽,不含花纹 |
| `base-css.webp` | 模板墙 CSS 叠图用的底图,按 sRGB 普通叠加;选白色、纯黑时和线性光一致 |
| `product-n.png` | 改色范围(白色 + 透明度) |
| `print-n.png` | 可印区域(白色 + 透明度) |
| `surface-n.png` | 大曲面层流向图,固定叠上 |
| `folds-n.png` | 褶皱层流向图,乘 Warping |
| `template.json` | 品类、标签、文件名、部件名、默认 Logo 框、合成方式、位移幅度 |
| `checks.json` | 自动校验结果(15 项) |

- 流向图:R / G = x / y 方向的位移,中灰不位移,0 / 255 对应 ∓`surfaceRangePx`(褶皱层 ∓`foldsRangePx`,为 Warping 100% 时的值);B 未用;1/4 尺寸,使用时双线性放大。
- 放进 vorla 的 lab 页或编辑页看效果时,选 `output/<id>/` 里的全部文件:lab 按文件名认图,位移幅度只在 `template.json` 里。

`preview/<id>/` 是示意图:

- `preview.png`:合成效果,从左到右是封面、白 / 白、红 / 藏青、黑 / 黑(Warping 50%)、芥末黄 / 橄榄绿(Warping 100%),颜色按商品顺序;
- `edges.png`:配置里 `preview.spots` 的几处边缘放大 3 倍;
- `masks.png`:各件商品区着绿、蓝、紫,黄线是可印区域,红框是默认 Logo。

## 文档

| 文档 | 给谁看 |
| --- | --- |
| [operator-guide.md](docs/operator-guide.md) | 跑批量的人:一步步操作 |
| [batch.md](docs/batch.md) | 批量的规则:文件夹和 id、清单和例外清单、预设、并发、重跑 |
| [spec.md](docs/spec.md) | 模板配置的字段 |
| [how-it-works.md](docs/how-it-works.md) | 分割、可印区、底图、流向图、Logo 框各自怎么算 |
| [models.md](docs/models.md) | 模型在哪跑、Replicate 设置、耗时、许可 |
| [publish.md](docs/publish.md) | 上传到 vorla |
| [batch-plan.md](docs/batch-plan.md) | 8000 张的批量方案和费用 |
