"""
修复进度表:fixes/issues.csv 是已知的问题,fixes/templates.csv 是哪些模板有哪个问题(一行一个模板 × 问题)。
进度不手填,按本机的 work/<id>/steps.json 和两边的模板墙(测试站读测试库,正式站读正式库)算:
- 待修:issues.csv 里这个问题的 fixed 还空着;
- 代码已改:fixed 写了修好的时间,模板的这一步(issues.csv 的 step)还没在这之后重跑过;
- 已重跑:这一步在 fixed 之后跑过(steps.json 的 ran),或 fixed 之前试跑时已是现在的配置、之后跑重跑命令时没变跳过(checked);
- 已入测试库:重跑以后测试库里这一行又更新过(测试站墙上的 updatedAt 晚于重跑的时间);
- 已同步正式库:正式库里这个模板的版本赶上了测试库。
本机没有的模板(work/ 里没有,在跑批的人那边)没有重跑的记录,测试库更新晚于 fixed 就算已入测试库。
一个模板的进度取它各个问题里最慢的。所有问题都到了「代码已改」才进重跑清单,几个问题一起只重跑、升版本、发布一次。

用法:.venv/bin/python -I scripts/fixes.py                    # 各问题、各进度的个数
      .venv/bin/python -I scripts/fixes.py show <问题 | id>    # 这个问题下每个模板的进度,或这个模板每个问题的进度
      .venv/bin/python -I scripts/fixes.py fixed <问题>         # 修好了:fixed 写成现在
      .venv/bin/python -I scripts/fixes.py rerun               # 该重跑的按批次写进 fixes/plan/rerun-<批次>.txt,打出命令
      .venv/bin/python -I scripts/fixes.py publish             # 重跑过、通过了、还没入测试库的按批次写进 fixes/plan/publish-<批次>.txt,打出命令
      .venv/bin/python -I scripts/fixes.py sync                # 进了测试库、正式库还是旧版本的,写一份给管正式库的人的总结(用 ossutil 列收件箱,约 1 分钟)

模板归哪一批:steps.json 记下的 batch(batch.py 跑过一步就记);更早的取 batches/*/logs/<id>.log 最后写过的那一批。
重跑的命令按 batches/<批次>/command.json(batch.py 每次运行时写);没有的按 sources/<批次>.csv、batches/<批次>.csv、
sources/<批次>/ 和 sources/<批次>-overrides 猜,打出来时标明,先 --dry-run 看要跑哪几步。
"""
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config
import oss_inbox
import wall

ROOT = config.ROOT
FIXES = ROOT / 'fixes'
PLAN = FIXES / 'plan'
STAGES = ['待修', '代码已改', '已重跑', '已入测试库', '已同步正式库']
STEPS = ['run_segment.py', 'intrinsic.py', 'normals.py', 'geometry.py', 'pipeline.py', 'logo.py', 'preview.py']
ISSUE_FIELDS = ['issue', 'title', 'step', 'fixed', 'example', 'note']
# 本机重跑不了的,单独存一份清单:文件名后缀
HELD = {'不在本机': 'not-local', '不知道归哪一批': 'no-batch', '源图不在': 'no-photo'}
# 同步总结里「改了什么」:库里这一行两边不一样的字段
ROW_FIELDS = {'name': '显示名', 'tags': '标签', 'logos': 'Logo 框'}
PUBLISH = 'pnpm --filter @vorla/web exec tsx --conditions=react-server scripts/mockup-templates-publish.ts'


def read(name):
    with open(FIXES / name, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def when(text):
    return datetime.fromisoformat(text) if text else None


def owners():
    """{id: 批次}:每个模板最后写过日志的那一批"""
    latest = {}
    for log in ROOT.glob('batches/*/logs/*.log'):
        mtime = log.stat().st_mtime
        if log.stem not in latest or mtime > latest[log.stem][0]:
            latest[log.stem] = (mtime, log.parent.parent.name)
    return {key: batch for key, (_, batch) in latest.items()}


class Ledger:
    def __init__(self):
        self.issues = {row['issue']: row for row in read('issues.csv')}
        self.rows = read('templates.csv')
        unknown = sorted({row['issue'] for row in self.rows} - set(self.issues))
        if unknown:
            raise SystemExit(f'templates.csv 里的问题不在 issues.csv:{", ".join(unknown)}')
        self.wall, self.prod = wall.fetch('test'), wall.fetch('prod')
        logged = owners()
        self.state, self.batch = {}, {}
        for key in {row['id'] for row in self.rows}:
            path = ROOT / 'work' / key / 'steps.json'
            self.state[key] = json.loads(path.read_text()) if path.exists() else None
            self.batch[key] = (self.state[key] or {}).get('batch') or logged.get(key)

    def stage(self, key, issue):
        """(进度, 重跑的时间)"""
        fixed = when(self.issues[issue]['fixed'])
        if fixed is None:
            return '待修', None
        published = when(self.wall[key]['updated']) if key in self.wall else None
        state, step = self.state[key], self.issues[issue]['step']
        ran = None if state is None else when(state.get('ran', {}).get(step))
        # 修好之前就试跑过、配置没再变的,修好后跑重跑命令时这一步跳过,只记下核对的时间(checked)
        checked = None if state is None else when(state.get('checked', {}).get(step))
        latest = max([t for t in (ran, checked) if t is not None], default=None)
        if state is not None and (latest is None or latest < fixed):
            return '代码已改', None
        if not (published and published > (ran or fixed)):
            return ('代码已改' if state is None else '已重跑'), ran
        return ('已同步正式库' if self.synced(key) else '已入测试库'), ran

    def synced(self, key):
        return key in self.prod and self.prod[key]['version'] >= self.wall[key]['version']

    def by_template(self):
        """{id: {问题: 进度}}"""
        table = defaultdict(dict)
        for row in self.rows:
            table[row['id']][row['issue']] = self.stage(row['id'], row['issue'])[0]
        return table

    def flags(self, key):
        """重跑、发布前要先处理的:不在本机、源图不在、没跑完、没通过检查"""
        if self.state[key] is None:
            return '不在本机'
        batch = self.batch[key]
        spec_path = ROOT / 'batches' / str(batch) / 'specs' / f'{key}.json'
        if batch is None or not spec_path.exists():
            return '不知道归哪一批'
        if not (ROOT / config.load(spec_path)['photo']).exists():
            return '源图不在'
        if not all(step in self.state[key] for step in ('pipeline.py', 'logo.py')):
            return '步骤不全'
        checks = ROOT / 'output' / key / 'checks.json'
        if not checks.exists() or not all(json.loads(checks.read_text()).values()):
            return '没通过检查'
        return ''


def slowest(stages):
    return min(stages.values(), key=STAGES.index)


def summary(ledger):
    table = ledger.by_template()
    print(f'修复进度:{len(table)} 个模板,{len(ledger.rows)} 行(模板 × 问题);测试库 {len(ledger.wall)} 个,正式库 {len(ledger.prod)} 个')
    for issue, info in ledger.issues.items():
        counts = Counter(stages[issue] for stages in table.values() if issue in stages)
        fixed = f"{info['fixed'][:16].replace('T', ' ')} 修好" if info['fixed'] else ''
        where = ', '.join(filter(None, [info['step'], fixed]))
        print(f"  {issue}  {info['title']}{f'({where})' if where else ''}:"
              + ' · '.join(f'{s} {counts[s]}' for s in STAGES if counts[s]))
    counts = Counter(slowest(stages) for stages in table.values())
    print('按模板(取最慢的问题):' + ' · '.join(f'{s} {counts[s]}' for s in STAGES if counts[s]))
    flags = Counter(ledger.flags(key) for key in table)
    notes = [f'{name} {n} 个' for name, n in flags.items() if name]
    if notes:
        print('要先处理的:' + ','.join(notes))


def show(ledger, target):
    rows = [row for row in ledger.rows if target in (row['id'], row['issue'])]
    if not rows:
        raise SystemExit(f'进度表里没有 {target}')
    for row in sorted(rows, key=lambda r: (r['issue'], r['id'])):
        stage, ran = ledger.stage(row['id'], row['issue'])
        on_wall = ledger.wall.get(row['id'])
        print('  '.join(filter(None, [
            row['id'], ledger.issues[row['issue']]['title'], stage,
            f'重跑于 {ran:%Y-%m-%d %H:%M}' if ran else '',
            (f"测试库 v{on_wall['version']}" + (f" 正式库 v{ledger.prod[row['id']]['version']}" if row['id'] in ledger.prod else ' 正式库没有'))
            if on_wall else '没上墙',
            ledger.batch[row['id']] or '', ledger.flags(row['id']), row.get('note', '')])))


def mark_fixed(issue):
    rows = read('issues.csv')
    match = [row for row in rows if row['issue'] == issue]
    if not match:
        raise SystemExit(f"issues.csv 里没有 {issue},有:{', '.join(row['issue'] for row in rows)}")
    if not match[0]['step']:
        raise SystemExit(f'issues.csv 里 {issue} 的 step 还空着,先写上修在哪一步,按这一步跑完的时间认重跑')
    match[0]['fixed'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(FIXES / 'issues.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ISSUE_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    step = match[0]['step']
    print(f"{match[0]['title']}:{match[0]['fixed']} 修好")
    print(f'改的是 {step} 的代码、配置没变时,把 scripts/batch.py 的 STEPS 里 {step} 的版本号加 1,重跑时才会重算这一步;'
          '改的是预设或例外清单的,配置变了自然重跑')


def command(batch):
    """(输入, 例外清单, 是不是猜的)"""
    path = ROOT / 'batches' / batch / 'command.json'
    if path.exists():
        saved = json.loads(path.read_text())
        return saved['input'], saved['overrides'], False
    inputs = [name for name in (f'sources/{batch}.csv', f'batches/{batch}.csv', f'sources/{batch}') if (ROOT / name).exists()]
    overrides = [name for name in (f'sources/{batch}-overrides',) if (ROOT / name).exists()]
    return (inputs[0] if inputs else None), overrides, True


def write_plan(kind, groups):
    PLAN.mkdir(parents=True, exist_ok=True)
    for old in PLAN.glob(f'{kind}-*.txt'):
        old.unlink()
    for name, ids in groups.items():
        (PLAN / f'{kind}-{name}.txt').write_text(''.join(f'{key}\n' for key in sorted(ids)))


def rerun(ledger):
    """所有问题都修好了、还有没重跑的"""
    table = ledger.by_template()
    ready = [key for key, stages in table.items() if '待修' not in stages.values() and '代码已改' in stages.values()]
    waiting = sum(1 for stages in table.values() if '待修' in stages.values() and '代码已改' in stages.values())
    groups, held = defaultdict(list), defaultdict(list)
    for key in ready:
        flag = ledger.flags(key)
        if flag in HELD:
            held[flag].append(key)
        else:
            groups[ledger.batch[key]].append(key)
    write_plan('rerun', {**groups, **{f'_{HELD[flag]}': ids for flag, ids in held.items()}})
    print(f'该重跑 {len(ready)} 个;还有别的问题没修好、先不跑的 {waiting} 个')
    for batch, ids in sorted(groups.items()):
        source, overrides, guessed = command(batch)
        if source is None:
            print(f'  {batch} {len(ids)} 个:找不到这一批的输入,手动对上')
            continue
        line = f".venv/bin/python -I scripts/batch.py {source}{''.join(f' --overrides {o}' for o in overrides)} --only fixes/plan/rerun-{batch}.txt --jobs 16 --local 4"
        print(f"  {batch} {len(ids)} 个{'(输入和例外清单是猜的)' if guessed else ''}:\n    {line} --dry-run\n    {line}")
    for flag, ids in held.items():
        print(f'  {flag} {len(ids)} 个,在 fixes/plan/rerun-_{HELD[flag]}.txt')
    if ready:
        print('先 --dry-run 看「要跑的步骤」:出现前四步(花钱)而这次只改了本机步骤的,是预设或例外清单在上次跑完后变过,先查清楚再跑')


def publish(ledger):
    """所有问题都重跑过、通过了、还没进测试库的"""
    table = ledger.by_template()
    ready = [key for key, stages in table.items()
             if all(s in STAGES[2:] for s in stages.values()) and '已重跑' in stages.values()]
    groups, held = defaultdict(list), defaultdict(list)
    for key in ready:
        flag = ledger.flags(key)
        (held[flag] if flag else groups[ledger.batch[key]]).append(key)
    write_plan('publish', groups)
    print(f'重跑过、该发布到测试库的 {len(ready)} 个')
    for batch, ids in sorted(groups.items()):
        print(f'  {batch} {len(ids)} 个:\n    .venv/bin/python -I scripts/inbox.py batches/{batch} inbox/update-{time.strftime("%Y%m%d")}-{batch} '
              f'--update --only fixes/plan/publish-{batch}.txt')
    for flag, ids in held.items():
        print(f"  {flag} {len(ids)} 个,先不发:{', '.join(sorted(ids)[:20])}{' …' if len(ids) > 20 else ''}")
    if groups:
        print('inbox.py --update 最后打出上传和发布的命令;上传、发布前先看抽查页。发布不加 --files-only,文件和测试库的那一行一起更新')


def sync(ledger):
    """所有问题都进了测试库、正式库还是旧版本的:写一份给管正式库的人的总结"""
    table = ledger.by_template()
    done = {key for key, stages in table.items() if all(s in STAGES[3:] for s in stages.values())}
    ready = sorted(key for key in done if '已入测试库' in table[key].values())
    print(f'进了测试库、要同步正式库的 {len(ready)} 个;已经同步的 {len(done) - len(ready)} 个;'
          f'还没修好或没进测试库的 {len(table) - len(done)} 个')
    PLAN.mkdir(parents=True, exist_ok=True)
    for old in PLAN.glob('sync*'):
        old.unlink()
    if not ready:
        return
    rows, by_issue, new = [], Counter(), 0
    for key in ready:
        test, prod = ledger.wall[key], ledger.prod.get(key)
        issues = sorted(table[key])
        if prod is None:
            # 正式库还没有的:管正式库的人没见过旧样子,修的问题对他没有意义,写小品类(新出现在哪一页)
            new += 1
            rows.append(f"| {key} | 没有 | v{test['version']} | 新模板,小品类 {test['category'] or '无'} | — |")
            continue
        by_issue.update(issues)
        changed = ['图片'] if any(ledger.issues[i]['step'] != 'logo.py' for i in issues) else []
        changed += [label for field, label in ROW_FIELDS.items() if test[field] != prod[field]]
        if test['category'] != prod['category']:
            changed.append(f"小品类 {prod['category'] or '无'} → {test['category'] or '无'}")
        rows.append(f"| {key} | v{prod['version']} | v{test['version']} | "
                    f"{'、'.join(changed)} | {'、'.join(ledger.issues[i]['title'] for i in issues)} |")
    # 发布脚本按 --batch 前缀扫收件箱:同一个 id 扫到几个目录时几份都挡掉,这些从新目录所在的那一层单独发
    found = oss_inbox.inbox_dirs()
    lost = [key for key in ready if key not in found]
    if lost:
        raise SystemExit(f'OSS 收件箱里没有这 {len(lost)} 个的目录,发布脚本发不了:{", ".join(lost)}')
    groups = defaultdict(list)
    for key in ready:
        groups[oss_inbox.prefix(found[key])].append(key)
    commands = ['export DATABASE_URL=<正式库>']
    for batch, ids in sorted(groups.items()):
        if len(groups) > 1:
            commands += ['', f'# {len(ids)} 个,' + (f'只扫 {batch}' if batch else '扫整个收件箱')]
        commands += [f"IDS={','.join(ids)}",
                     f'{PUBLISH} --batch mockup-template-inbox/{batch} --only $IDS            # 只检查,不写库',
                     f'{PUBLISH} --batch mockup-template-inbox/{batch} --only $IDS --apply    # 写正式库']
    doubled = sum(len(ids) for batch, ids in groups.items() if batch)
    passes = '、'.join(f'{len(ids)}' for _, ids in sorted(groups.items()))
    (PLAN / 'sync.txt').write_text(''.join(f'{key}\n' for key in ready))
    path = PLAN / f'sync-{time.strftime("%Y%m%d")}.md'
    path.write_text('\n'.join([
        '# Mockup 模板同步到正式库',
        '',
        f"{time.strftime('%Y-%m-%d')} · {len(ready)} 个模板要从测试库同步到正式库:"
        f"{len(ready) - new} 个在测试库更新到了新版本,正式库还是旧版本"
        + (f';{new} 个正式库还没有,同步后会新出现在模板墙上。' if new else '。'),
        '',
        '模板文件两边共用 OSS 同一个 bucket,新版本已经发在 `mockup-templates/<id>/v<版本>/` 下,不用再传。'
        '要改的只有正式库 `template` 表里这几行(按 slug)。',
        '',
        '## 怎么同步',
        '',
        '在 vorla 仓库的 `apps/web` 下运行模板发布脚本,`DATABASE_URL` 指正式库'
        + (f'。分 {len(groups)} 组跑:{doubled} 个在 OSS 收件箱里还另有一份不全的旧目录,整个收件箱一起扫时'
           '发布脚本报「同名目录出现多次」、两份都不发,这几个从新目录所在的那一层扫:' if doubled else ':'),
        '',
        '```bash',
        *commands,
        '```',
        '',
        f'- 只检查时应全部通过,通过的个数依次是 {passes},不通过 0。`--only` 里的 id 在扫的范围里找不到时不报错、直接不算,'
        '要对一下个数;个别报「读取失败」的多是网络超时,再跑一次。',
        '- `--apply` 时模板文件都已经发过,发布结果应全是 unchanged,只写库。',
        '- 写的是这几行的显示名、封面、宽高比、小品类、标签和 `mockup_materials`;正式库已有的行发布时间不变,模板在墙上的位置不动'
        + (',新模板的发布时间是写库的时间。' if new else '。'),
        '- 不用脚本的话,按 slug 从测试库的 `template` 表把这几列复制过来也一样' + (',新模板整行复制。' if new else '。'),
        '',
        '## 更新的模板按问题' if new else '## 按问题',
        '',
        '| 问题 | 模板数 |',
        '| --- | --- |',
        *(f"| {ledger.issues[i]['title']} | {n} |" for i, n in sorted(by_issue.items(), key=lambda item: -item[1])),
        '',
        '## 模板',
        '',
        '| id | 正式库 | 测试库 | 改了什么 | 修的问题 |',
        '| --- | --- | --- | --- | --- |',
        *rows,
        '',
    ]))
    print(f'总结在 {path.relative_to(ROOT)},id 清单在 {(PLAN / "sync.txt").relative_to(ROOT)}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == 'fixed':
        mark_fixed(args[1])
    elif len(args) == 2 and args[0] == 'show':
        show(Ledger(), args[1])
    elif len(args) <= 1 and (args or [''])[0] in ('', 'rerun', 'publish', 'sync'):
        {'': summary, 'rerun': rerun, 'publish': publish, 'sync': sync}[(args or [''])[0]](Ledger())
    else:
        raise SystemExit(__doc__)
