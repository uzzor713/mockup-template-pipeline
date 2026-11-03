"""
批量出模板:每张源图按品类预设(presets/<预设>.json)生成配置,依次跑分割、本征分解、法线、几何、
pipeline.py 和 preview.py,最后汇总结果。
用法:.venv/bin/python -I batch.py <按品类分好的文件夹 | 清单.csv> [--overrides 例外.csv 或文件夹 ...] [--dry-run] [--jobs N] [--local N] [--force] [--no-preview]
      .venv/bin/python -I batch.py <按品类分好的文件夹> --pack 例外.csv   # 把这几张打包给调参数的人

按品类分好的文件夹:每个子文件夹是一个品类,文件夹名是预设名或预设里 folders 列的别名,里面(含更深的子文件夹)
的 png / jpg / webp 都算,以 _ 开头的文件夹不算。对不上预设的文件夹会列出来并停下。id 默认是品类文件夹名(不是纯英文时换成预设名)加文件名(规则同下),
文件名本来就以文件夹名开头的不重复加(t-shirt/001.png → t-shirt-001,boxy-tee/mockup-boxy-tee-020-blank.webp → boxy-tee-020)。
个别源图要改设置时用 --overrides 给例外清单,按 id 对上,空着的列不改:
- product、parts:这张的商品提示词、不可印部件提示词(; 隔开,- 表示没有),只能用在一件商品的预设上;
- occluders、crop、preset:同下面清单的列;
- logo:这张的默认 Logo 放法,填 logo.py 里 PLACES 的名字(left-chest 左胸、chest 胸口正中,Gemini 判断前襟判错时改正),只能用在一件商品的预设上;
- skip:填 1 跳过这张,不跑也不进抽查页;
- issue、note:抽查页导出的问题类型和备注,只给人看。
--overrides 可以写多次,也可以给文件夹(里面的 csv 按文件名顺序),后面的覆盖前面的。每次运行都按清单重新生成配置,
改过的源图没带上它的例外清单就会回到预设的设置,所以例外清单要一直留着、每次都带上。
--pack 按例外清单把这几张的源图(放在和原来一样的品类文件夹下)、分割叠图和蒙版示意(_review/)连同清单打成 zip,
存到 batches/<名称>/;对方解压后对它跑 batch.py,id 和原来一样,试好提示词把清单填好发回来。
生成的清单存一份在 batches/<名称>/list.csv,可以从里面抄 id。

清单(CSV,第一行是列名):
- source:源图的地址或本地路径,必填;
- preset:预设名,必填;
- id:可选,不写由文件名得出:去掉 mockup- 前缀和 -blank 后缀,转小写,字母数字以外的连成 -
  (mockup-boxy-tee-020-blank.webp → boxy-tee-020,01 Slow days tee.png → 01-slow-days-tee);
- occluders:可选,遮挡物提示词,多个用 ; 隔开,写了就替换预设里的(写 - 表示没有遮挡物);
- crop:可选,x0 y0 x1 y1(源图像素),写了就不按预设的比例规则裁;
- tags:可选,标签,多个用 ; 隔开,写进 template.json(如 Channel、Seasonal 页面的 channel/amazon、seasonal/christmas)。

结果都在 batches/<文件夹名或清单文件名>/:specs/ 是生成的配置,logs/ 是每个模板的运行日志,report.csv 是汇总,
review.html 是抽查页(按品类分页,见 review.py)。--dry-run 只生成配置、列出各品类的张数,不跑。改了清单重跑同一条命令,只重算受影响的模板和步骤:
改遮挡物只重跑分割和之后两步,三个模型脚本只看源图(按文件内容)和裁切;改 MODEL_BACKEND 全部重跑。--force 全部重跑。
--no-preview 不出预览(只用来抽查,上传用不到),这次没出的下次不带它跑时补上。
"""
import argparse
import contextlib
import csv
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config
import logo
import review

ROOT = config.ROOT
REVIEW_EVERY = 60  # 抽查页最多隔这么多秒更新一次,跑完再写一次
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
SIZE = (2048, 2560)  # 竖版模板的尺寸,横版宽高对调:vorla 后端按这个尺寸收模板,源图裁好后不是这个尺寸的缩放过去
LATER = {'size'}  # 后加的配置字段,没写时不算进下面的比较,加字段之前做过的模板不用重跑
# 每一步和它用到的配置字段(另加源图的内容和 MODEL_BACKEND;源图路径不算,挪了文件夹不重跑):都没变、前一次成功过就跳过;
# None 表示用到整份配置。商品的 logo 设置只有 USES_LOGO 里的步骤算进比较,改了只重出 Logo 框和预览
USES_LOGO = {'logo.py', 'preview.py'}
# 在本机算的步骤:每个峰值约 2GB 内存、一两核,同一时间至多 --local 个模板在跑,其余排队;前四步大半时间在等 Replicate,不限。
# pipeline.py 里问 Gemini、分单只(logo.py)也占着名额,几秒到一分钟
LOCAL_STEPS = {'pipeline.py', 'logo.py', 'preview.py'}
STEPS = [
    ('run_segment.py', ['crop', 'size', 'occluders', 'products']),
    ('intrinsic.py', ['crop', 'size']),
    ('normals.py', ['crop', 'size']),
    ('geometry.py', ['crop', 'size']),
    ('pipeline.py', None),
    ('logo.py', None),
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
    loose = [p.name for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith('.')]
    if loose:
        unknown.append(f'(不在品类文件夹里)({len(loose)} 张)')
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(('_', '.'))):
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
            rows.append({'source': str(image), 'preset': names[sub.name], 'id': spec_id, 'relative': image.relative_to(root).as_posix()})
    if unknown:
        raise SystemExit('这些文件夹对不上预设,在 presets/ 里加预设或在预设的 folders 里加上文件夹名:\n  '
                         + '\n  '.join(unknown) + f'\n现有预设:{", ".join(sorted(set(names.values())))}')
    return rows


def pack(root, rows, issues):
    """例外清单里这几张的源图、分割叠图和蒙版示意连同清单打成 zip,源图放在和原来一样的品类文件夹下"""
    by_id = {row['id']: row for row in rows}
    marked = [m for m in read_csv(issues) if m.get('id', '').strip()]
    missing = [m['id'] for m in marked if m['id'].strip() not in by_id]
    if missing:
        raise SystemExit(f'例外清单里的 id 对不上:{", ".join(missing)}')
    name = f"{root.name}-issues-{time.strftime('%Y%m%d-%H%M')}"
    path = ROOT / 'batches' / root.name / f'{name}.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as package:
        package.write(issues, f'{name}/{issues.name}')
        for m in marked:
            row = by_id[m['id'].strip()]
            package.write(row['source'], f"{name}/{row['relative']}")
            for kind, folder_name, file in (('segment', 'work', 'segment-check.png'), ('masks', 'preview', 'masks.png')):
                image = ROOT / folder_name / row['id'] / file
                if image.exists():
                    package.write(image, f"{name}/_review/{row['id']}-{kind}.png")
    print(f'{len(marked)} 张,打包到 {path}')


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


def words(value):
    """; 隔开的提示词;- 表示一个都没有"""
    return [w.strip() for w in value.split(';') if w.strip() and w.strip() != '-']


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
        # 换成清单里的词时,预设里同名提示的其他设置(如头发的 matteBand)保留
        known = {o['text']: o for o in preset['occluders'] if 'text' in o}
        occluders = [known.get(w, {'text': w}) for w in words(row['occluders'])]
    products = preset['products']
    if row.get('product', '').strip() or row.get('parts', '').strip():
        if len(products) != 1:
            raise SystemExit(f"{row.get('id') or source}:预设 {name} 有 {len(products)} 件商品,不能按单张改 product、parts")
        item = dict(products[0])
        if row.get('product', '').strip():
            item['text'] = row['product'].strip()
        if row.get('parts', '').strip():
            item['parts'] = [{'text': w} for w in words(row['parts'])]
            if not item['parts']:
                del item['parts']
        products = [item]
    if row.get('logo', '').strip():
        if len(products) != 1:
            raise SystemExit(f"{row.get('id') or source}:预设 {name} 有 {len(products)} 件商品,不能按单张改 logo")
        products = [{**products[0], 'logo': logo.PLACES[row['logo'].strip()]}]
    spec = {'id': row.get('id', '').strip() or template_id(source), 'category': preset['category'], 'photo': photo}
    if words(row.get('tags', '')):
        spec['tags'] = words(row['tags'])
    if crop != [0, 0, width, height]:
        spec['crop'] = crop
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    size = list(SIZE if h > w else SIZE[::-1])
    if [w, h] != size and w * size[1] == h * size[0]:
        spec['size'] = size
    if remote:
        spec['fetch'] = {photo: {'url': source, 'sha256': sha256(ROOT / photo)}}
    spec |= {'occluders': occluders, 'products': products}
    return spec


def digest(spec, fields, photo_sha, logo=False):
    if not logo:
        spec = {**spec, 'products': [{k: v for k, v in item.items() if k != 'logo'} for item in spec['products']]}
    part = {k: v for k, v in spec.items() if k not in ('photo', 'fetch')} if fields is None else {k: spec.get(k) for k in fields if k in spec or k not in LATER}
    return hashlib.sha1(json.dumps([part, photo_sha, config.backend()], sort_keys=True).encode()).hexdigest()


def run(spec_path, log_path, force, skip=()):
    """跑一个模板,返回 (状态, 说明, 秒数);skip 里的步骤这次不跑,也不记下"""
    spec = config.load(spec_path)
    state_path = config.folder(spec, 'work') / 'steps.json'
    state = {} if force or not state_path.exists() else json.loads(state_path.read_text())
    photo_sha = sha256(ROOT / spec['photo'])
    t0 = time.time()
    rerun = False  # 前面有一步重跑过,用到整份配置的步骤(读前面各步的结果)也要重跑
    with open(log_path, 'a') as log:
        for script, fields in STEPS:
            if script in skip:
                continue
            key = digest(spec, fields, photo_sha, script in USES_LOGO)
            if state.get(script) == key and not (rerun and fields is None):
                continue
            rerun = True
            with local_slots if script in LOCAL_STEPS else contextlib.nullcontext():
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


parser = argparse.ArgumentParser()
parser.add_argument('input', help='按品类分好的文件夹,或清单 CSV')
parser.add_argument('--overrides', action='append', default=[], help='例外清单 CSV 或放例外清单的文件夹,可写多次')
parser.add_argument('--pack', help='把这份例外清单里的源图、分割叠图和蒙版示意打成 zip,不跑')
parser.add_argument('--dry-run', action='store_true', help='只生成配置、列出各品类的张数,不跑')
parser.add_argument('--jobs', type=int, default=1, help='同时跑几个模板;MODEL_BACKEND=local 时每个峰值约 6–7GB 内存')
parser.add_argument('--local', type=int, default=2, help='同一时间至多几个模板跑本机计算的步骤(pipeline.py、logo.py、preview.py,各约 2GB)')
parser.add_argument('--force', action='store_true', help='忽略上次的结果,全部重跑')
parser.add_argument('--no-preview', action='store_true', help='不出预览,下次不带它跑时补上')
args = parser.parse_args()
local_slots = threading.Semaphore(max(1, args.local))

source = Path(args.input).expanduser()
if source.is_dir():
    rows = scan(source.resolve())
else:
    rows = [row for row in read_csv(source) if row.get('source', '').strip()]
if args.pack:
    if not source.is_dir():
        raise SystemExit('--pack 要给按品类分好的文件夹')
    pack(source.resolve(), rows, Path(args.pack).expanduser())
    raise SystemExit
by_id = {row['id']: row for row in rows if row.get('id')}
for item in args.overrides:
    item = Path(item).expanduser()
    for change in [c for f in (sorted(item.glob('*.csv')) if item.is_dir() else [item]) for c in read_csv(f)]:
        if not change.get('id', '').strip():
            continue
        if change['id'].strip() not in by_id:
            raise SystemExit(f"例外清单里的 id 对不上:{change['id']}")
        by_id[change['id'].strip()].update({k: v for k, v in change.items() if k != 'id' and v and v.strip()})
skipped = [row for row in rows if row.get('skip', '').strip() not in ('', '0')]
rows = [row for row in rows if row not in skipped]
if skipped:
    print(f"例外清单里跳过 {len(skipped)} 张:{', '.join(row.get('id') or row['source'] for row in skipped)}")
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
lock = threading.Lock()  # 几个模板同时跑完时,结果和抽查页一个个写
last_review = 0.0


def one(path):
    global last_review
    spec = config.load(path)
    status, note, seconds = run(path, folder / 'logs' / f"{spec['id']}.log", args.force, {'preview.py'} if args.no_preview else ())
    review.thumbnails(folder, spec['id'])
    with lock:
        results.append({'id': spec['id'], 'category': spec['category'], 'source': Path(spec['photo']).name,
                        'status': status, 'note': note, 'seconds': round(seconds)})
        print(f"[{len(results)}/{len(paths)}] {status:5} {seconds:6.0f}s  {spec['id']}" + (f'  {note}' if note else ''), flush=True)
        if time.time() - last_review > REVIEW_EVERY:
            review.write(folder, results)
            last_review = time.time()


with ThreadPoolExecutor(args.jobs) as pool:
    list(pool.map(one, paths))
review.write(folder, results)

with open(folder / 'report.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'category', 'status', 'note', 'seconds'], extrasaction='ignore')
    writer.writeheader()
    writer.writerows(sorted(results + [{'id': row.get('id') or template_id(row['source']), 'category': row['preset'], 'status': 'skip',
                                        'note': row.get('note', '')} for row in skipped], key=lambda r: r['id']))
print(' · '.join(f'{s} {sum(r["status"] == s for r in results)}' for s in ('pass', 'fail', 'error')) + (f' · skip {len(skipped)}' if skipped else ''))
print(f'汇总 {(folder / "report.csv").relative_to(ROOT)},抽查页 {(folder / "review.html").relative_to(ROOT)}')
