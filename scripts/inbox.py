"""
按 vorla 的模板上传说明,把一批里通过的模板(report.csv 里 status 为 pass,自动校验全过、不在跳过之列)整理成上传用的文件夹:每个模板一个目录(目录名是 id),
放模板文件(template.json 里列的,原样复制)、展示图 cover.webp、白模图 white.png,根目录写 manifest.csv。
整个文件夹原样传到 OSS 的收件目录(如 vorla-assets/mockup-template-inbox/<日期>/),checks.json 不传。
用法:.venv/bin/python -I scripts/inbox.py batches/<批次> <输出目录> [--uploaded 已传过的 manifest.csv ...] [--only id,... | id 清单.txt]
      .venv/bin/python -I scripts/inbox.py batches/<批次> <输出目录> --update [--only ...]   # 已经传过的模板重跑后换上新的模板文件

源图按生图包的布局找另外两张:<页面>/<key>/<key>-blank-2x.png 是源图,同一目录的 <key>.png 是带印花的展示图,
<页面> 文件夹名是 manifest 的小品类(categories)。展示图和源图构图不一定对得上,按源图同样的比例裁、放大到模板尺寸。
白模图是源图裁好、缩放到模板尺寸后的样子,和底图、蒙版逐像素对齐。
--uploaded 给已经传过的 manifest.csv,里面的 id 跳过,只整理新增的(批次分几次传时用)。已经在测试站模板墙上的也跳过,改了要用 --update。
--only 只整理这几个 id(逗号隔开,或一行一个 id 的文件,# 开头的行不算)。

--update:已经传进 OSS 收件箱的模板重跑以后,只换模板文件(template.json 和它列的文件),展示图、白模图和 manifest.csv 不动。
按 OSS 收件箱里这个 id 现在的目录放(输出目录下的相对路径和收件箱里一样),整个输出目录传到收件箱根目录就盖在原处;
上过模板墙的,template.json 的 version 写成墙上的版本加 1(发布脚本见同版本的文件变了会拒绝),没上墙的保持原样。
收件箱里还没有的列出来,按新模板不带 --update 整理。要用 ossutil(PATH 里或 ~/bin/ossutil)列收件箱,约 1 分钟。
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config
import wall
from oss_inbox import INBOX, complete, inbox_dirs

ROOT = config.ROOT
COVER_QUALITY = 85  # 和 pipeline.py 写 cover.webp 时一样
BLANK_SUFFIX = '-blank-2x'


def template_files(template):
    """template.json 里列出的模板文件(不含 cover.webp,它换成展示图)"""
    files = [template['files']['base'], template['files']['baseCss']]
    for region in template['regions']:
        files += [region[name] for name in ('product', 'print', 'surface', 'folds')]
    return files


def internal_name(page, key):
    """11x17-poster、11x17-poster-004 → 11x17 Poster 004"""
    number = re.search(r'(\d+)$', key)
    words = [w if any(c.isdigit() for c in w) else w.capitalize() for w in page.split('-')]
    return ' '.join(words + ([number.group(1)] if number else []))


def cover(spec, display, size):
    """展示图按源图同样的比例裁(源图坐标换到展示图上),放大到模板尺寸"""
    image = Image.open(display).convert('RGB')
    if 'crop' in spec:
        source_width, source_height = Image.open(ROOT / spec['photo']).size
        kx, ky = image.width / source_width, image.height / source_height
        x0, y0, x1, y1 = spec['crop']
        image = image.crop((round(x0 * kx), round(y0 * ky), round(x1 * kx), round(y1 * ky)))
    return image.resize(size, Image.LANCZOS)


def only_ids(value):
    path = Path(value).expanduser()
    lines = path.read_text().splitlines() if path.is_file() else value.split(',')
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith('#')}


def update(specs):
    """重跑过的模板换上新的模板文件,按收件箱里原来的目录放"""
    found = inbox_dirs()
    on_wall = wall.fetch()
    done, bumped, missing, doubled, recategorized = [], [], [], [], []
    for spec in specs:
        if spec['id'] not in found:
            missing.append(spec['id'])
            continue
        places = found[spec['id']]
        # 同一个 id 有两个目录时发布脚本两份都挡掉;先放进有展示图和白模图的那份,另一份要删
        folder = complete(places)
        if len(places) > 1:
            doubled.append(f"{spec['id']}({' / '.join(place for place, _ in places)})")
        source = ROOT / 'output' / spec['id']
        template = json.loads((source / 'template.json').read_text())
        if spec['id'] in on_wall:
            template['version'] = on_wall[spec['id']]['version'] + 1
            before = on_wall[spec['id']]['category']
            if template.get('category') != before:
                recategorized.append(f"{spec['id']}({before or '无'} → {template.get('category') or '无'})")
            bumped.append(spec['id'])
        target = args.out / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))
        for name in template_files(template):
            shutil.copy2(source / name, target / name)
        done.append(spec['id'])
    total = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
    print(f'{len(done)} 个模板,共 {total / 1e9:.2f} GB,在 {args.out};上过模板墙、版本号加 1 的 {len(bumped)} 个')
    if missing:
        print(f'收件箱里没有 {len(missing)} 个,按新模板不带 --update 整理:{", ".join(missing)}')
    if doubled:
        print(f'收件箱里有两个目录的 {len(doubled)} 个,发布脚本会两份都挡掉,先删掉不全的那个:\n  ' + '\n  '.join(doubled))
    if recategorized:
        print(f'小品类和测试库里不一样的 {len(recategorized)} 个,发布后测试库跟着改,同步正式库时也会改:\n  ' + '\n  '.join(recategorized))
    if done:
        # 盖在原处时 ossutil 逐个问是否覆盖、从标准输入读回答,-f 直接覆盖,< /dev/null 防止它吃掉脚本里后面的命令
        print(f'传:ossutil cp -r -f {args.out}/ {INBOX} -j 16 < /dev/null')
        # 收件箱根目录作前缀,哪个日期目录里的都找得到;不加 --files-only,文件和测试库的那一行一起更新
        print(f'发布(vorla/apps/web):pnpm --filter @vorla/web exec tsx --conditions=react-server scripts/mockup-templates-publish.ts '
              f'--apply --batch {INBOX.removeprefix("oss://vorla-assets/")} --only {",".join(done)}')


parser = argparse.ArgumentParser()
parser.add_argument('batch', type=Path, help='batches/<批次>')
parser.add_argument('out', type=Path, help='输出目录')
parser.add_argument('--uploaded', type=Path, action='append', default=[], help='已经传过的 manifest.csv')
parser.add_argument('--only', help='只整理这几个 id:逗号隔开,或一行一个 id 的文件')
parser.add_argument('--update', action='store_true', help='已经传过的模板换上新的模板文件,按收件箱里原来的目录放')
args = parser.parse_args()

uploaded = {row['id'] for path in args.uploaded for row in csv.DictReader(open(path, encoding='utf-8-sig'))}
report = args.batch / 'report.csv'
if not report.exists():
    raise SystemExit(f'没有 {report}:批次跑完才有汇总表')
passed = {row['id'] for row in csv.DictReader(open(report, encoding='utf-8-sig')) if row['status'] == 'pass'}
specs = [config.load(path) for path in sorted((args.batch / 'specs').glob('*.json'))]
if args.only:
    wanted = only_ids(args.only)
    unknown = wanted - {spec['id'] for spec in specs}
    if unknown:
        raise SystemExit(f'--only 里的 id 不在这一批:{", ".join(sorted(unknown))}')
    specs = [spec for spec in specs if spec['id'] in wanted]
failed = [spec['id'] for spec in specs if spec['id'] not in passed]
if failed:
    print(f'没通过或跳过 {len(failed)} 个,不整理' + (f':{", ".join(failed)}' if args.only else ''))
specs = [spec for spec in specs if spec['id'] in passed]
args.out.mkdir(parents=True, exist_ok=True)
if args.update:
    update(specs)
    raise SystemExit
on_wall = wall.fetch()
rows, skipped = [], {'已传过': [], '已上模板墙(改了用 --update)': [], '找不到展示图': []}
for spec in specs:
    folder = ROOT / 'output' / spec['id']
    if spec['id'] in uploaded:
        skipped['已传过'].append(spec['id'])
        continue
    if spec['id'] in on_wall:
        skipped['已上模板墙(改了用 --update)'].append(spec['id'])
        continue
    source = Path(spec['photo'])
    key = source.stem.removesuffix(BLANK_SUFFIX)
    display = source.with_name(f'{key}.png')
    if source.stem == key or not display.exists():
        skipped['找不到展示图'].append(spec['id'])
        continue
    template = json.loads((folder / 'template.json').read_text())
    size = (template['size']['width'], template['size']['height'])
    target = args.out / spec['id']
    target.mkdir(exist_ok=True)
    for name in ['template.json', *template_files(template)]:
        shutil.copy2(folder / name, target / name)
    cover(spec, display, size).save(target / 'cover.webp', quality=COVER_QUALITY)
    white, _ = config.photo(spec)
    Image.fromarray(np.round(white * 255).astype(np.uint8)).save(target / 'white.png', optimize=True)
    page = source.parent.parent.name
    rows.append({'id': spec['id'], 'categories': page, 'internal_name': internal_name(page, key)})

with open(args.out / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'categories', 'internal_name'])
    writer.writeheader()
    writer.writerows(rows)
total = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
print(f'{len(rows)} 个模板,共 {total / 1e9:.2f} GB,在 {args.out}')
for reason, ids in skipped.items():
    if ids:
        print(f'{reason} {len(ids)} 个' + (f':{", ".join(ids[:20])}{" …" if len(ids) > 20 else ""}' if reason == '找不到展示图' else ''))
