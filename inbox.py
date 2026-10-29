"""
按 vorla 的模板上传说明,把一批里自动校验全过的模板整理成上传用的文件夹:每个模板一个目录(目录名是 id),
放模板文件(template.json 里列的,原样复制)、展示图 cover.webp、白模图 white.png,根目录写 manifest.csv。
整个文件夹原样传到 OSS 的收件目录(如 vorla-assets/mockup-template-inbox/<日期>/),checks.json 不传。
用法:.venv/bin/python -I inbox.py batches/<批次> <输出目录> [--uploaded 已传过的 manifest.csv ...]

源图按生图包的布局找另外两张:<页面>/<key>/<key>-blank-2x.png 是源图,同一目录的 <key>.png 是带印花的展示图,
<页面> 文件夹名是 manifest 的小品类(categories)。展示图和源图构图不一定对得上,按源图同样的比例裁、放大到模板尺寸。
白模图是源图裁好、缩放到模板尺寸后的样子,和底图、蒙版逐像素对齐。
--uploaded 给已经传过的 manifest.csv,里面的 id 跳过,只整理新增的(批次分几次传时用)。
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


parser = argparse.ArgumentParser()
parser.add_argument('batch', type=Path, help='batches/<批次>')
parser.add_argument('out', type=Path, help='输出目录')
parser.add_argument('--uploaded', type=Path, action='append', default=[], help='已经传过的 manifest.csv')
args = parser.parse_args()

uploaded = {row['id'] for path in args.uploaded for row in csv.DictReader(open(path, encoding='utf-8-sig'))}
rows, skipped = [], {'没全过校验': [], '已传过': [], '找不到展示图': []}
args.out.mkdir(parents=True, exist_ok=True)
for spec_path in sorted((args.batch / 'specs').glob('*.json')):
    spec = config.load(spec_path)
    folder = ROOT / 'output' / spec['id']
    checks_path = folder / 'checks.json'
    if not checks_path.exists() or not all(json.loads(checks_path.read_text()).values()):
        skipped['没全过校验'].append(spec['id'])
        continue
    if spec['id'] in uploaded:
        skipped['已传过'].append(spec['id'])
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
