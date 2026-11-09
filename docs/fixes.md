# 修复进度

已知的模板问题和受影响的模板记在 `fixes/` 的两张表里。每个模板修到哪一步不手填,由 `scripts/fixes.py` 按本机的运行记录和两边的模板墙(测试站读测试库,正式站读正式库)算出来。一个模板有几个问题时,等这几个问题都修好再一起重跑,只升一次版本、发布一次。修好的模板连文件带库里那一行发布到测试库,最后整理一份清单交给管正式库的同事同步。

```bash
.venv/bin/python -I scripts/fixes.py                    # 各问题、各进度的个数
.venv/bin/python -I scripts/fixes.py show <问题 | id>    # 这个问题下每个模板的进度,或这个模板每个问题的进度
.venv/bin/python -I scripts/fixes.py fixed <问题>         # 修好了:记下现在的时间
.venv/bin/python -I scripts/fixes.py rerun               # 挑出该重跑的,打出重跑命令
.venv/bin/python -I scripts/fixes.py publish             # 挑出重跑过、该发布到测试库的,打出整理命令
.venv/bin/python -I scripts/fixes.py sync                # 进了测试库、正式库还是旧版本的,写一份同步总结
```

## 两张表

`fixes/issues.csv`,一行一个问题:

| 列 | 内容 |
| --- | --- |
| `issue` | 问题的短名,如 `white-gray` |
| `title` | 中文名 |
| `step` | 修在哪一步(`run_segment.py`、`pipeline.py`、`logo.py` 等),按这一步跑完的时间认重跑 |
| `fixed` | 修好的时间(UTC),由 `fixes.py fixed` 填;空着是还没修 |
| `example` | 一个例子的 id |
| `note` | 原因,以及受影响的模板是怎么筛出来的 |

`fixes/templates.csv`,一行一个模板 × 问题:`id`、`issue`、`found`(发现的日期)、`note`(筛出它的指标,如「区域 2 白色中位亮度 0.638」)。新发现一个问题时,先在 `issues.csv` 加一行,再把受影响的模板加进来。

## 进度

| 进度 | 怎么算 |
| --- | --- |
| 待修 | 这个问题的 `fixed` 还空着 |
| 代码已改 | 问题修好了,但这个模板的 `step` 这一步还没在修好之后跑过 |
| 已重跑 | 这一步在修好之后跑过(`work/<id>/steps.json` 的 `ran`);或修好之前试跑时已经是现在的配置,修好后跑重跑命令时没变跳过(`checked`) |
| 已入测试库 | 重跑以后,测试库里这一行又更新过(测试站模板墙的 `updatedAt` 晚于重跑的时间) |
| 已同步正式库 | 正式库里这个模板的版本赶上了测试库(正式站模板墙的版本号) |

- 一个模板的进度取它各个问题里最慢的。
- 本机没有的模板(跑批量的那边跑的,`work/` 里没有)没有重跑记录:测试库更新的时间晚于修好的时间就算已入测试库,否则一直停在代码已改。

`fixes.py` 还列出重跑、发布前要先处理的:

| 标记 | 意思 |
| --- | --- |
| 不在本机 | `work/` 里没有,要交给跑批量的那边重跑 |
| 源图不在 | 配置里的源图找不到,先找回源图 |
| 不知道归哪一批 | 找不到这个模板的批次和配置 |
| 步骤不全 | `steps.json` 里没有 `pipeline.py` 或 `logo.py` 的记录,多是早期的批次,重跑时会补上 |
| 没通过检查 | 自动校验有不过的,重跑可以,发布前要先看 |

## 流程

1. **修**:改代码、预设或例外清单,拿几个例子小批试过,再 `fixes.py fixed <问题>`。只改了代码、配置没变的,同时把 `scripts/batch.py` 里 `STEPS` 那一步的版本号加 1(见 [batch.md](batch.md#结果和重跑)),不然重跑时会跳过这一步。
2. **重跑**:`fixes.py rerun` 挑出所有问题都修好、还有没重跑的模板。还有别的问题没修好的先不跑,等齐了一起跑。
   - 按模板所属的批次写进 `fixes/plan/rerun-<批次>.txt`,打出 `batch.py ... --only` 命令。批次取 `steps.json` 里的 `batch`;更早的取 `batches/*/logs/<id>.log` 最后写过的那一批。
   - 命令里的输入和例外清单取自 `batches/<批次>/command.json`;没有的按 `sources/<批次>.csv`、`batches/<批次>.csv`、`sources/<批次>/` 和 `sources/<批次>-overrides` 猜,打出来时标明。
   - 本机跑不了的分开列:`rerun-_not-local.txt`(交给跑批量的那边)、`rerun-_no-photo.txt`(源图不在)、`rerun-_no-batch.txt`(不知道归哪一批)。
   - 先带 `--dry-run` 跑一遍,看真跑时每一步要跑几个。只改了本机的几步却出现前四步(按次计费),说明预设或例外清单在上次跑完后变过,先查清楚再跑。
3. **抽查**:看这一批的 `review.html`。
4. **发布到测试库**:`fixes.py publish` 挑出所有问题都重跑过、通过检查、还没进测试库的模板,写进 `fixes/plan/publish-<批次>.txt`,打出 `inbox.py --update` 命令。之后的上传、发布见 [publish.md](publish.md#八改已经发布的模板)。发布脚本不加 `--files-only`,文件和测试库里的那一行(版本、Logo 框、小品类等)一起更新,模板墙上看得到才算这一步做完。
5. **同步正式库**:本机的问题都修完、发布到测试库以后,`fixes.py sync` 挑出所有问题都进了测试库、正式库还是旧版本的模板,写成 `fixes/plan/sync-<日期>.md` 交给管正式库的同事(id 清单另存 `fixes/plan/sync.txt`)。

同步总结给管正式库的人看,写了:

- 怎么同步:文件两边共用 OSS 同一个 bucket,不用再传,只改正式库 `template` 表里这几行;用发布脚本带正式库的 `DATABASE_URL` 和 `--only` 跑,或者按 slug 从测试库复制。
- 按问题的模板数。
- 每个模板一行:正式库和测试库的版本、改了什么(图片、Logo 框、小品类、标签、显示名,按两边模板墙上这一行的差别和问题修在哪一步得出)、修的问题。

正式库同步以后再跑 `fixes.py`,这些模板就到了已同步正式库。

`fixes/plan/` 每次运行时重新生成,不进仓库。
