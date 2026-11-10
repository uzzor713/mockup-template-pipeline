"""
批量出模板:每张源图按品类预设(presets/<预设>.json)生成配置,依次跑分割、本征分解、法线、几何、
pipeline.py 和 preview.py,最后汇总结果。
用法:.venv/bin/python -I scripts/batch.py <按品类分好的文件夹 | 清单.csv> [--overrides 例外.csv 或文件夹 ...] [--only id,... | id 清单.txt] [--dry-run] [--jobs N] [--local N] [--force] [--no-preview]
      .venv/bin/python -I scripts/batch.py <按品类分好的文件夹> --pack 例外.csv   # 把这几张打包给调参数的人

按品类分好的文件夹:每个子文件夹是一个品类,文件夹名是预设名或预设里 folders 列的别名,里面(含更深的子文件夹)
的 png / jpg / webp 都算,以 _ 开头的文件夹不算。对不上预设的文件夹会列出来并停下。id 默认是品类文件夹名(不是纯英文时换成预设名)加文件名(规则同下),
文件名本来就以文件夹名开头的不重复加(t-shirt/001.png → t-shirt-001,boxy-tee/mockup-boxy-tee-020-blank.webp → boxy-tee-020)。
个别源图要改设置时用 --overrides 给例外清单,按 id 对上,空着的列不改:
- product、parts:这张的商品提示词、不可印部件提示词(; 隔开,- 表示没有),只能用在一件商品、或几件商品说法相同(左右两只杯子)的预设上,几件一起改;
- occluders、crop、preset:同下面清单的列;
- logo:这张的默认 Logo 放法,填 logo.py 里 PLACES 的名字(left-chest 左胸、chest 胸口正中,Gemini 判断前襟判错时改正),只能用在一件商品的预设上;
- solid:填 1 时这张的商品蒙版里的洞全补上(商品配置的 solid),SAM 在空白面上漏掉大块时用;
- hull:填 1 时这张的商品蒙版换成最大一块的凸包(商品配置的 hull),轮廓是凸的、缺口开在边上时用;
- largest:填 1 时这张只取分出的实例里面积最大的一个(商品配置的 largest),几件同类商品分开摆、只印一件时用;
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
改了某一步的代码、同样的配置算出的结果会变时,把下面 STEPS 里这一步的版本号加 1:之后跑到的模板重算这一步和排在它后面的本机步骤,
别的步骤照样跳过(改本机的三步不会重跑前面花钱的四步)。
--only 只跑这几个 id(逗号隔开,或一行一个 id 的文件,# 开头的行不算),例外清单照样整份带上;汇总表和抽查页里其他模板保留上次的结果。
--no-preview 不出预览(只用来抽查,上传用不到),这次没出的下次不带它跑时补上。
每个模板的 work/<id>/steps.json 记下各步上次的比较值、跑完的时间(ran,UTC)、没变跳过时核对的时间(checked)和所在的批次(batch),
修复进度表(fixes.py)按它认哪些模板重跑过。
"""
import argparse
import contextlib
import csv
import hashlib
import itertools
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
# 每一步、它用到的配置字段(另加源图的内容和 MODEL_BACKEND;源图路径不算,挪了文件夹不重跑)和代码版本:都没变、前一次成功过就跳过;
# None 表示用到整份配置。商品的 logo 设置只有 USES_LOGO 里的步骤算进比较,改了只重出 Logo 框和预览。
# 代码版本:改了这一步的脚本、同样的配置算出的结果会变时加 1(只是整理代码、结果不变的不加);0 不算进比较,有版本号之前做过的模板不用重跑
USES_LOGO = {'logo.py', 'preview.py'}
# 商品设置里不算进分割比较的:print、surface 分割不读,只给 pipeline.py;alternatives 只在主说法分不出时换着试,上次分成功的加了也一样
# (上次失败的没记比较值,照样重跑;上次放低阈值才分出的,加了备选说法可能改用备选说法,要重分单独跑 run_segment.py)。
# 早先的比较值含或不含这几项,都算对得上,不为这个重跑、重新计费
SEGMENT_IGNORES = ('alternatives', 'print', 'surface')
# 在本机算的步骤:每个峰值约 2GB 内存、一两核,同一时间至多 --local 个模板在跑,其余排队;前四步大半时间在等 Replicate,不限。
# pipeline.py 里问 Gemini、分单只(logo.py)也占着名额,几秒到一分钟
LOCAL_STEPS = {'pipeline.py', 'logo.py', 'preview.py'}
STEPS = [
    ('run_segment.py', ['crop', 'size', 'occluders', 'products'], 0),
    ('intrinsic.py', ['crop', 'size'], 0),
    ('normals.py', ['crop', 'size'], 0),
    ('geometry.py', ['crop', 'size'], 0),
    ('pipeline.py', None, 5),
    ('logo.py', None, 0),
    ('preview.py', None, 0),
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


def row_id(row):
    return row.get('id', '').strip() or template_id(row['source'].strip())


def only_ids(value):
    """--only:逗号隔开的 id,或一行一个 id 的文件(# 开头的行不算)"""
    path = Path(value).expanduser()
    lines = path.read_text().splitlines() if path.is_file() else value.split(',')
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith('#')}


def shown(path):
    """仓库里的路径写成相对仓库根目录的,其余写绝对路径"""
    path = Path(path).expanduser().resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


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
        if len({item.get('text') for item in products}) != 1:
            raise SystemExit(f"{row.get('id') or source}:预设 {name} 的 {len(products)} 件商品说法不同,不能按单张改 product、parts")
        changed = []
        for item in products:
            item = dict(item)
            if row.get('product', '').strip():
                item['text'] = row['product'].strip()
            if row.get('parts', '').strip():
                item['parts'] = [{'text': w} for w in words(row['parts'])]
                if not item['parts']:
                    del item['parts']
            changed.append(item)
        products = changed
    if row.get('logo', '').strip():
        if len(products) != 1:
            raise SystemExit(f"{row.get('id') or source}:预设 {name} 有 {len(products)} 件商品,不能按单张改 logo")
        products = [{**products[0], 'logo': logo.PLACES[row['logo'].strip()]}]
    if row.get('solid', '').strip() not in ('', '0'):
        products = [{**item, 'solid': True} for item in products]
    if row.get('hull', '').strip() not in ('', '0'):
        products = [{**item, 'hull': True} for item in products]
    if row.get('largest', '').strip() not in ('', '0'):
        products = [{**item, 'largest': True} for item in products]
    spec = {'id': row.get('id', '').strip() or template_id(source), 'category': preset['category'], 'photo': photo}
    if words(row.get('tags', '')):
        spec['tags'] = words(row['tags'])
    if spec['category'] is None and not any(t.split('/')[0] in ('seasonal', 'channel') for t in spec.get('tags', [])):
        raise SystemExit(f"{spec['id']}:预设 {name} 不归商品小品类,清单的 tags 要写 seasonal/<页面> 或 channel/<页面>")
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


def digest(spec, fields, photo_sha, logo=False, version=0):
    if not logo:
        spec = {**spec, 'products': [{k: v for k, v in item.items() if k != 'logo'} for item in spec['products']]}
    part = {k: v for k, v in spec.items() if k not in ('photo', 'fetch')} if fields is None else {k: spec.get(k) for k in fields if k in spec or k not in LATER}
    return hashlib.sha1(json.dumps([part, photo_sha, config.backend(), *([version] if version else [])], sort_keys=True).encode()).hexdigest()


def keys(spec, script, fields, photo_sha, version):
    """这一步的比较值:第一个是这次跑完记下的;分割另算早先各件商品含 SEGMENT_IGNORES 里任几项时的,对上了也不重跑"""
    if script != 'run_segment.py':
        return [digest(spec, fields, photo_sha, script in USES_LOGO, version)]
    present = [(i, k) for i, item in enumerate(spec['products']) for k in SEGMENT_IGNORES if k in item]
    accepted = []
    for n in range(len(present) + 1):
        for kept in itertools.combinations(present, n):
            products = [{k: v for k, v in item.items() if k not in SEGMENT_IGNORES or (i, k) in kept}
                        for i, item in enumerate(spec['products'])]
            accepted.append(digest({**spec, 'products': products}, fields, photo_sha, version=version))
    return accepted


def load_state(spec, force):
    path = ROOT / 'work' / spec['id'] / 'steps.json'
    return path, ({} if force or not path.exists() else json.loads(path.read_text()))


def pending(spec, state, skip=()):
    """这次要跑的步骤和各自的比较值:比较值变了的,以及前面有一步要跑时之后用到整份配置的(读前面各步的结果)"""
    photo_sha = sha256(ROOT / spec['photo'])
    steps, rerun = [], False
    for script, fields, version in STEPS:
        if script in skip:
            continue
        accepted = keys(spec, script, fields, photo_sha, version)
        if state.get(script) in accepted and not (rerun and fields is None):
            continue
        rerun = True
        steps.append((script, accepted[0]))
    return steps


def run(spec_path, log_path, force, batch, skip=()):
    """跑一个模板,返回 (状态, 说明, 秒数);skip 里的步骤这次不跑,也不记下"""
    spec = config.load(spec_path)
    config.folder(spec, 'work')
    state_path, state = load_state(spec, force)
    ran = state.setdefault('ran', {})
    t0 = time.time()
    steps = pending(spec, state, skip)
    # 没变、这次跳过的步骤记下核对的时间:修复前就试跑过的模板,修好后跑重跑命令时也算重跑过
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    todo = {script for script, _ in steps}
    state['checked'] = {**state.get('checked', {}),
                        **{script: now for script, _, _ in STEPS if script in state and script not in todo and script not in skip}}
    state_path.write_text(json.dumps(state, indent=2))
    with open(log_path, 'a') as log:
        for script, key in steps:
            with local_slots if script in LOCAL_STEPS else contextlib.nullcontext():
                log.write(f'\n==== {time.strftime("%F %T")} {script}\n')
                log.flush()
                result = subprocess.run([sys.executable, '-I', str(Path(__file__).parent / script), str(spec_path)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['batch'] = batch
            if result.returncode != 0:
                state.pop(script, None)
                ran.pop(script, None)
                state_path.write_text(json.dumps(state, indent=2))
                lines = Path(log_path).read_text().strip().splitlines()
                return 'error', f'{script}: {lines[-1] if lines else result.returncode}', time.time() - t0
            state[script] = key
            ran[script] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            state_path.write_text(json.dumps(state, indent=2))
    checks = json.loads((ROOT / 'output' / spec['id'] / 'checks.json').read_text())
    failed = [name for name, ok in checks.items() if not ok]
    return ('fail' if failed else 'pass'), '; '.join(failed), time.time() - t0


parser = argparse.ArgumentParser()
parser.add_argument('input', help='按品类分好的文件夹,或清单 CSV')
parser.add_argument('--overrides', action='append', default=[], help='例外清单 CSV 或放例外清单的文件夹,可写多次')
parser.add_argument('--pack', help='把这份例外清单里的源图、分割叠图和蒙版示意打成 zip,不跑')
parser.add_argument('--only', help='只跑这几个 id:逗号隔开,或一行一个 id 的文件;汇总表和抽查页里其他模板保留上次的结果')
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
listed = rows
if args.only:
    wanted = only_ids(args.only)
    unknown = wanted - {row_id(row) for row in rows + skipped}
    if unknown:
        raise SystemExit(f'--only 里的 id 不在这一批:{", ".join(sorted(unknown))}')
    rows = [row for row in rows if row_id(row) in wanted]
    skipped = [row for row in skipped if row_id(row) in wanted]
if skipped:
    print(f"例外清单里跳过 {len(skipped)} 张:{', '.join(row.get('id') or row['source'] for row in skipped)}")
folder = ROOT / 'batches' / (source.name if source.is_dir() else source.stem)
(folder / 'specs').mkdir(parents=True, exist_ok=True)
(folder / 'logs').mkdir(exist_ok=True)
# 这一批的输入和例外清单,fixes.py 按它写重跑的命令
(folder / 'command.json').write_text(json.dumps({'input': shown(source), 'overrides': [shown(item) for item in args.overrides]},
                                                ensure_ascii=False, indent=2) + '\n')
if source.is_dir():
    with open(folder / 'list.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'preset', 'id', 'occluders', 'crop'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(listed)

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
    skip = {'preview.py'} if args.no_preview else ()
    with ThreadPoolExecutor(8) as pool:
        todo = list(pool.map(lambda spec: [script for script, _ in pending(spec, load_state(spec, args.force)[1], skip)], specs))
    print('真跑时要跑的步骤(前四步走 MODEL_BACKEND,replicate 时按次计费):'
          + (','.join(f'{script} {sum(script in steps for steps in todo)} 个' for script, _, _ in STEPS
                      if any(script in steps for steps in todo)) or '都不用跑'))
    raise SystemExit

REPORT = ['id', 'category', 'source', 'status', 'note', 'seconds']
results = []
lock = threading.Lock()  # 几个模板同时跑完时,结果和抽查页一个个写
last_review = 0.0
# --only 时这次没跑的模板沿用上次汇总表里的结果
done = {row_id(row) for row in rows + skipped}
kept = [{**r, 'source': r.get('source', '')} for r in read_csv(folder / 'report.csv') if r['id'] not in done] \
    if args.only and (folder / 'report.csv').exists() else []


def one(path):
    global last_review
    spec = config.load(path)
    status, note, seconds = run(path, folder / 'logs' / f"{spec['id']}.log", args.force, folder.name, {'preview.py'} if args.no_preview else ())
    review.thumbnails(folder, spec['id'])
    with lock:
        # 只挂节日 / 渠道的横轴模板 category 为 null,抽查页和汇总按标签的页面分组
        category = spec['category'] or '|'.join(t.split('/')[-1] for t in spec.get('tags', [])) or '-'
        results.append({'id': spec['id'], 'category': category, 'source': Path(spec['photo']).name,
                        'status': status, 'note': note, 'seconds': round(seconds)})
        print(f"[{len(results)}/{len(paths)}] {status:5} {seconds:6.0f}s  {spec['id']}" + (f'  {note}' if note else ''), flush=True)
        if time.time() - last_review > REVIEW_EVERY:
            review.write(folder, reviewed())
            last_review = time.time()


def reviewed():
    """抽查页列这次和沿用的结果,跳过的不进抽查页"""
    return [r for r in kept if r['status'] != 'skip'] + results


with ThreadPoolExecutor(args.jobs) as pool:
    list(pool.map(one, paths))
review.write(folder, reviewed())

with open(folder / 'report.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=REPORT, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(sorted(kept + results + [{'id': row_id(row), 'category': row['preset'], 'source': Path(row['source'].strip()).name,
                                               'status': 'skip', 'note': row.get('note', '')} for row in skipped], key=lambda r: r['id']))
print(' · '.join(f'{s} {sum(r["status"] == s for r in results)}' for s in ('pass', 'fail', 'error')) + (f' · skip {len(skipped)}' if skipped else ''))
print(f'汇总 {(folder / "report.csv").relative_to(ROOT)},抽查页 {(folder / "review.html").relative_to(ROOT)}')
