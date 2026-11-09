# 模型和运行环境

模型在哪跑、怎么设置 Replicate、耗时和费用、许可。批量的费用估算见 [batch-plan.md](batch-plan.md)。

## 在哪跑

| 模型 | 用在 | 在哪跑 |
| --- | --- | --- |
| SAM 3 | 分割(文字提示) | 始终走 Replicate |
| Gemini(`google/gemini-3-flash`) | 分割挑实例、判断上衣前襟 | 始终走 Replicate,一次约 0.1 美分 |
| ViTMatte | 边缘透明度 | 看 `MODEL_BACKEND` |
| Marigold 本征分解 | 底图 | 看 `MODEL_BACKEND` |
| Marigold 法线 | 褶皱层流向图 | 看 `MODEL_BACKEND` |
| MoGe-2 | 大曲面层流向图 | 看 `MODEL_BACKEND` |

`MODEL_BACKEND` 有两个值:

| 值 | 做法 | 用途 |
| --- | --- | --- |
| `replicate`(默认) | 调 Replicate 上的社区封装,版本固定在 `scripts/remote.py`;本机只装 `requirements/base.txt` | 批量 |
| `local` | 本机跑,另装 `requirements/models.txt` 和 `requirements/moge.txt`,要 Apple 芯片 | 对照和开发 |

默认值写在 `.env`(进仓库)。本机要改就在 `.env.local` 写一行 `MODEL_BACKEND=local`,或设同名环境变量;优先级是环境变量 > `.env.local` > `.env`。

## Replicate 设置

- **token**:写进仓库根目录的 `.env.local`(不进仓库),一行 `REPLICATE_API_TOKEN=...`,或设同名环境变量。源图会上传到 Replicate。
- **限流**:`.env` 里的 `REPLICATE_MIN_INTERVAL` 是本机所有进程合起来两次预测之间至少隔几秒,`--jobs` 并发时也按这个节拍排队。
  - 账号余额低时每分钟只能建 6 次预测,设 `11`;
  - 充值后每分钟 600 次(看响应头 `ratelimit-remaining`),设 `0`。
- **重试**:429(超限)、5xx 和断连都等一会儿重试。
- **上传复用**:同一张源图在本机只上传一次,分割、本征分解、法线、几何几步共用。上传记录在系统临时目录的 `replicate-uploads/`;文件在 Replicate 存 24 小时,快过期的重新上传。
- **社区封装的风险**:ViTMatte、两个 Marigold 和 MoGe-2 用的是个人上传的封装。版本号锁死了,作者改不了已锁定的版本,但可以整个删掉;删了就切回 `local`,或把同样的权重部署到自己的账号。
- **SAM 3 的限制**:封装只收文字提示,不收框和点。文字分不开的东西(如画框和画心)分不出来。

## 两种后端的差别

| 模型 | `replicate` 和 `local` 比 |
| --- | --- |
| MoGe-2 | 逐位相同(关掉 fp16) |
| ViTMatte | 几乎一样;长边超过 2048 时在缩小的图上抠,再放大回原图 |
| Marigold 本征分解 | 不能固定随机种子和处理分辨率,只出 8 位图;换算后相关系数 0.94–0.995(残差那层最低) |
| Marigold 法线 | 同上;相关系数 0.96–0.98 |

`local` 时 Marigold 的随机种子固定,重复运行结果一致;`replicate` 时每次略有不同。

## 耗时和费用

**`replicate`**:每个模型的推理只要几秒,但封装用的人少,闲置后再调要冷启动,每个模型一到五分钟。一个模板一步步接着跑要十一二分钟,所以批量要开 `--jobs`,让各模型一直有请求。按推理时长计费,一个模板约 1–2 美分。

**`local`**(16GB 内存的 M1,模型已下载):

| 脚本 | 首次下载 | 耗时 |
| --- | --- | --- |
| `run_segment.py` | 约 0.1GB(用到手点提示时另加 SAM 2.1 约 0.3GB) | 约 75 秒,大半是 `REPLICATE_MIN_INTERVAL=11` 时调 SAM 3 之间的间隔 |
| `intrinsic.py` | 约 2.6GB | 约 22 秒 |
| `normals.py` | 约 2.6GB | 约 20 秒 |
| `geometry.py` | 约 1.3GB | 约 10 秒 |

一个模板合计 1.5–2.5 分钟,内存峰值约 6–7GB。模型从 Hugging Face 下载。

## 许可

| 模型 | 用在 | 许可 |
| --- | --- | --- |
| SAM 3(Replicate 上的 `mattsays/sam3-image`,权重 `facebook/sam3`) | 按文字提示分割 | SAM License(Meta 自定的条款),商用前要确认 |
| SAM 2.1(`facebook/sam2.1-hiera-base-plus`) | 按框和点分割(手点兜底) | Apache-2.0 |
| ViTMatte small(`hustvl/vitmatte-small-composition-1k`) | 边缘透明度 | 模型标注 Apache-2.0;训练数据 Composition-1k 出自 Adobe 的抠图数据集,商用前要确认该数据集的条款 |
| Marigold 本征分解(`prs-eth/marigold-iid-lighting-v1-1`) | 底图的明暗和光泽 | OpenRAIL++,可商用,附使用用途限制 |
| Marigold 法线(`prs-eth/marigold-normals-v1-1`) | 褶皱层 | OpenRAIL++,可商用,附使用用途限制 |
| MoGe-2(`Ruicheng/moge-2-vitl-normal`) | 大曲面层和透视 | MIT(代码和模型) |
| pymatting(`estimate_foreground_ml`,算法库) | 商品边缘像素背后的颜色 | MIT |

`replicate` 时用的社区封装是 `superhighfives/vitmatte-trimap`、`jasonod888/marigold-intrinsics`、`jasonod888/marigold-normalsv2`、`jasonod888/moge2`,权重和上表相同,封装本身没写许可。
