"""
批量出模板:每张源图按品类预设(presets/<预设>.json)生成配置,依次跑分割、本征分解、法线、几何、
pipeline.py 和 preview.py,最后汇总结果。
用法:.venv/bin/python -I batch.py <按品类分好的文件夹 | 清单.csv> [--overrides 例外.csv] [--dry-run] [--jobs N] [--force]

按品类分好的文件夹:每个子文件夹是一个品类,文件夹名是预设名或预设里 folders 列的别名,里面(含更深的子文件夹)
的 png / jpg / webp 都算。对不上预设的文件夹会列出来并停下。id 默认是品类文件夹名(不是纯英文时换成预设名)加文件名(规则同下),
文件名本来就以文件夹名开头的不重复加(t-shirt/001.png → t-shirt-001,boxy-tee/mockup-boxy-tee-020-blank.webp → boxy-tee-020)。
个别源图要改设置时用 --overrides 给一份例外清单,按 id 对上,列同下面的 preset、occluders、crop。
生成的清单存一份在 batches/<名称>/list.csv,可以从里面抄 id。

清单(CSV,第一行是列名):
- source:源图的地址或本地路径,必填;
- preset:预设名,必填;
- id:可选,不写由文件名得出:去掉 mockup- 前缀和 -blank 后缀,转小写,字母数字以外的连成 -
  (mockup-boxy-tee-020-blank.webp → boxy-tee-020,01 Slow days tee.png → 01-slow-days-tee);
- occluders:可选,遮挡物提示词,多个用 ; 隔开,写了就替换预设里的(写 - 表示没有遮挡物);
- crop:可选,x0 y0 x1 y1(源图像素),写了就不按预设的比例规则裁。

结果都在 batches/<文件夹名或清单文件名>/:specs/ 是生成的配置,logs/ 是每个模板的运行日志,report.csv 是汇总,
review.html 是抽查页(没通过的排前面)。--dry-run 只生成配置、列出各品类的张数,不跑。改了清单重跑同一条命令,只重算受影响的模板和步骤:
改遮挡物只重跑分割和之后两步,三个模型脚本只看源图(按文件内容)和裁切;改 MODEL_BACKEND 全部重跑。--force 全部重跑。
"""
import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config

ROOT = config.ROOT
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
# 每一步和它用到的配置字段(另加源图的内容和 MODEL_BACKEND):都没变、前一次成功过就跳过;None 表示用到整份配置
STEPS = [
    ('run_segment.py', ['photo', 'crop', 'occluders', 'products']),
    ('intrinsic.py', ['photo', 'crop']),
    ('normals.py', ['photo', 'crop']),
    ('geometry.py', ['photo', 'crop']),
    ('pipeline.py', None),
    ('preview.py', None),
]


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def template_id(source):
    return slug(Path(source.split('?')[0]).stem.removeprefix('mockup-').removesuffix('-blank'))


def scan(root):
    """按品类分好的文件夹 → 清单的各行"""
    names = {}  # 文件夹名 → 预设名
    for path in sorted((ROOT / 'presets').glob('*.json')):
        for alias in [path.stem, *json.loads(path.read_text()).get('folders', [])]:
            names[alias] = path.stem
    rows, unknown = [], []
    loose = [p.name for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if loose:
        unknown.append(f'(不在品类文件夹里)({len(loose)} 张)')
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in sub.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith('.'))
        if not images:
            continue
        if sub.name not in names:
            unknown.append(f'{sub.name}({len(images)} 张)')
            continue
        prefix = slug(sub.name) if sub.name.isascii() else names[sub.name]  # 中文文件夹名转不出 id,用预设名
        for image in images:
            relative = image.relative_to(sub)
            own = slug('-'.join([*relative.parent.parts, template_id(relative.name)]))
            spec_id = own if own == prefix or own.startswith(prefix + '-') else f'{prefix}-{own}'
            rows.append({'source': str(image), 'preset': names[sub.name], 'id': spec_id})
    if unknown:
        raise SystemExit('这些文件夹对不上预设,在 presets/ 里加预设或在预设的 folders 里加上文件夹名:\n  '
                         + '\n  '.join(unknown) + f'\n现有预设:{", ".join(sorted(set(names.values())))}')
    return rows


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def crop_box(width, height, preset):
    """竖图裁成 4:5、横图 5:4,正方形按预设的 square(默认 4:5);多出的宽度两边平分,多出的高度按 trim 从哪边裁(默认 center)"""
    aw, ah = (4, 5) if height > width else (5, 4) if width > height else preset.get('square', (4, 5))
    k = min(width // aw, height // ah)
    w, h = aw * k, ah * k
    x0 = (width - w) // 2
    y0 = {'bottom': 0, 'top': height - h, 'center': (height - h) // 2}[preset.get('trim', 'center')]
    return [x0, y0, x0 + w, y0 + h]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path):
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def make_spec(row):
    """清单的一行 → 模板配置"""
    source, name = row['source'].strip(), row['preset'].strip()
    preset = json.loads((ROOT / 'presets' / f'{name}.json').read_text())
    remote = source.startswith(('http://', 'https://'))
    if remote:
        photo = f"sources/{name}/{Path(source.split('?')[0]).name}"
        download(source, ROOT / photo)
    else:
        # 本地源图直接引用,配置里写绝对路径
        photo = str(Path(source).expanduser().resolve())
        if not Path(photo).is_file():
            raise SystemExit(f'找不到源图:{source}')
    width, height = Image.open(ROOT / photo).size
    crop = [int(v) for v in row['crop'].split()] if row.get('crop', '').strip() else crop_box(width, height, preset)
    occluders = preset['occluders']
    if row.get('occluders', '').strip():
        words = [w.strip() for w in row['occluders'].split(';') if w.strip() and w.strip() != '-']
        # 换成清单里的词时,预设里同名提示的其他设置(如头发的 matteBand)保留
        known = {o['text']: o for o in preset['occluders'] if 'text' in o}
        occluders = [known.get(w, {'text': w}) for w in words]
    spec = {'id': row.get('id', '').strip() or template_id(source), 'category': preset['category'], 'photo': photo}
    if crop != [0, 0, width, height]:
        spec['crop'] = crop
    if remote:
        spec['fetch'] = {photo: {'url': source, 'sha256': sha256(ROOT / photo)}}
    spec |= {'occluders': occluders, 'products': preset['products']}
    return spec


def digest(spec, fields, photo_sha):
    part = spec if fields is None else {k: spec.get(k) for k in fields}
    return hashlib.sha1(json.dumps([part, photo_sha, config.backend()], sort_keys=True).encode()).hexdigest()


def run(spec_path, log_path, force):
    """跑一个模板,返回 (状态, 说明, 秒数)"""
    spec = config.load(spec_path)
    state_path = config.folder(spec, 'work') / 'steps.json'
    state = {} if force or not state_path.exists() else json.loads(state_path.read_text())
    photo_sha = sha256(ROOT / spec['photo'])
    t0 = time.time()
    rerun = False  # 前面有一步重跑过,用到整份配置的步骤(读前面各步的结果)也要重跑
    with open(log_path, 'a') as log:
        for script, fields in STEPS:
            key = digest(spec, fields, photo_sha)
            if state.get(script) == key and not (rerun and fields is None):
                continue
            rerun = True
            log.write(f'\n==== {time.strftime("%F %T")} {script}\n')
            log.flush()
            result = subprocess.run([sys.executable, '-I', str(ROOT / script), str(spec_path)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                state.pop(script, None)
                state_path.write_text(json.dumps(state, indent=2))
                lines = Path(log_path).read_text().strip().splitlines()
                return 'error', f'{script}: {lines[-1] if lines else result.returncode}', time.time() - t0
            state[script] = key
            state_path.write_text(json.dumps(state, indent=2))
    checks = json.loads((ROOT / 'output' / spec['id'] / 'checks.json').read_text())
    failed = [name for name, ok in checks.items() if not ok]
    return ('fail' if failed else 'pass'), '; '.join(failed), time.time() - t0


def write_review(folder, results):
    """抽查页:没通过的排前面,每个模板一张分割叠图和一张蒙版示意"""
    order = {'error': 0, 'fail': 1, 'pass': 2}
    cards = []
    for r in sorted(results, key=lambda r: (order[r['status']], r['id'])):
        images = ''.join(f'<a href="{src}"><img src="{src}" loading="lazy" alt="{alt}"></a>' for src, alt in (
            (f"../../work/{r['id']}/segment-check.png", '分割叠图'), (f"../../preview/{r['id']}/masks.png", '蒙版示意')))
        note = f'<p class="note">{html.escape(r["note"])}</p>' if r['note'] else ''
        cards.append(f'<li class="card {r["status"]}"><p class="name">{html.escape(r["id"])} · {r["status"]}</p>{note}'
                     f'<div class="images">{images}</div></li>')
    counts = ' · '.join(f'{s} {sum(r["status"] == s for r in results)}' for s in order)
    (folder / 'review.html').write_text(f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(folder.name)} 抽查</title>
<style>
body {{ margin: 0; padding: 16px; font: 14px/1.5 system-ui, sans-serif; background: #f4f4f4; color: #222; }}
ul {{ list-style: none; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
.card {{ background: #fff; border-radius: 8px; padding: 10px; border-top: 4px solid #3a3; }}
.card.fail {{ border-color: #e90; }} .card.error {{ border-color: #d33; }}
.name {{ margin: 0; font-weight: 600; }} .note {{ margin: 4px 0 0; color: #a40; font-size: 12px; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }}
img {{ width: 100%; display: block; background: #ddd; }}
</style></head><body>
<h1>{html.escape(folder.name)} 抽查</h1>
<p>{counts}。左:分割(绿 = 商品,蓝 = 不可印部件,红 = 遮挡物);右:蒙版示意(黄线 = 可印区域,红框 = 默认 Logo)。点图看原图。</p>
<ul>{''.join(cards)}</ul>
</body></html>
''')


parser = argparse.ArgumentParser()
parser.add_argument('input', help='按品类分好的文件夹,或清单 CSV')
parser.add_argument('--overrides', help='例外清单 CSV,按 id 改 preset、occluders、crop')
parser.add_argument('--dry-run', action='store_true', help='只生成配置、列出各品类的张数,不跑')
parser.add_argument('--jobs', type=int, default=1, help='同时跑几个模板;MODEL_BACKEND=local 时每个峰值约 6–7GB 内存')
parser.add_argument('--force', action='store_true', help='忽略上次的结果,全部重跑')
args = parser.parse_args()

source = Path(args.input).expanduser()
if source.is_dir():
    rows = scan(source.resolve())
else:
    rows = [row for row in read_csv(source) if row.get('source', '').strip()]
if args.overrides:
    by_id = {row['id']: row for row in rows if row.get('id')}
    for change in read_csv(args.overrides):
        if change.get('id', '').strip() not in by_id:
            raise SystemExit(f"例外清单里的 id 对不上:{change.get('id')}")
        by_id[change['id'].strip()].update({k: v for k, v in change.items() if k != 'id' and v and v.strip()})
folder = ROOT / 'batches' / (source.name if source.is_dir() else source.stem)
(folder / 'specs').mkdir(parents=True, exist_ok=True)
(folder / 'logs').mkdir(exist_ok=True)
if source.is_dir():
    with open(folder / 'list.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'preset', 'id', 'occluders', 'crop'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

with ThreadPoolExecutor(8) as pool:
    specs = list(pool.map(make_spec, rows))
ids = [s['id'] for s in specs]
duplicates = sorted({i for i in ids if ids.count(i) > 1})
if duplicates:
    raise SystemExit(f'清单里 id 重复:{", ".join(duplicates)}')
paths = []
for spec in specs:
    path = folder / 'specs' / f"{spec['id']}.json"
    text = json.dumps(spec, ensure_ascii=False, indent=2) + '\n'
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    paths.append(path)
print(f'{len(paths)} 个模板,配置在 {folder.relative_to(ROOT)}/specs/')
if args.dry_run:
    for name in sorted({r['preset'] for r in rows}):
        print(f"  {name}: {sum(r['preset'] == name for r in rows)} 张")
    raise SystemExit

results = []


def one(path):
    spec_id = path.stem
    status, note, seconds = run(path, folder / 'logs' / f'{spec_id}.log', args.force)
    results.append({'id': spec_id, 'status': status, 'note': note, 'seconds': round(seconds)})
    print(f'[{len(results)}/{len(paths)}] {status:5} {seconds:6.0f}s  {spec_id}' + (f'  {note}' if note else ''), flush=True)
    write_review(folder, results)


with ThreadPoolExecutor(args.jobs) as pool:
    list(pool.map(one, paths))

with open(folder / 'report.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'status', 'note', 'seconds'])
    writer.writeheader()
    writer.writerows(sorted(results, key=lambda r: r['id']))
print(' · '.join(f'{s} {sum(r["status"] == s for r in results)}' for s in ('pass', 'fail', 'error')))
print(f'汇总 {(folder / "report.csv").relative_to(ROOT)},抽查页 {(folder / "review.html").relative_to(ROOT)}')
