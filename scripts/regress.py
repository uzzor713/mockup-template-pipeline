"""
改 pipeline.py 这类本机步骤的代码以后,拿一组固定的模板对照改前改后:哪些文件变了、自动校验过没过。
  .venv/bin/python -I scripts/regress.py                # 改前取 HEAD,改后取现在的 scripts/
  .venv/bin/python -I scripts/regress.py --base <提交>   # 改前取别的提交
  .venv/bin/python -I scripts/regress.py <id> ...       # 另外指定模板(要在本机 work/ 里)

改前、改后各在一个临时目录里跑 pipeline.py:scripts/ 分别取自 --base 和现在的目录,work/<id>/ 是复制的,
其余(源图、预设、收件箱)连回仓库。仓库的 work/、output/ 不动;临时目录里没有 .env.local、也不带 Replicate 的密钥,
前面各步的结果和 Gemini 的回答都读 work/ 里存好的,哪一步要重新调模型就直接报错,不花钱。

每个模板打一行:改后没过的校验、变了的文件(template.json 写到是哪几项)。改动本该只碰某一类商品时,
别的类一个文件都不该变。改后有校验没过或跑出错,退出码 1。
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# 每种贴法都有:平面、平面退回曲面、圆柱、圆柱退回曲面、布料走通用曲面。没在本机的跳过
SAMPLES = {
    '按平面贴': ['poster-canvas-poster-014', 'book-cover-024', 'greeting-card-postcard-012'],
    '平面退回曲面': ['magazine-spread-007', 'brochure-trifold-brochure-003'],
    '按圆柱贴': ['soda-can-022', 'water-bottle-025', 'coffee-cup-012', 'beer-can-002'],
    '圆柱退回曲面': ['wine-bottle-003'],
    '布料': ['baby-tee-033', 'grey-hoodie-021', 'baggy-sweatpants-014', 'embroidered-hat-005'],
}
NOT_COPIED = {'scripts', 'work', 'output', 'preview', '.env.local', '.git', '.venv'}

parser = argparse.ArgumentParser()
parser.add_argument('ids', nargs='*', help='另外指定模板,不写用固定的那一组')
parser.add_argument('--base', default='HEAD', help='改前的 scripts/ 取自哪个提交')
parser.add_argument('--jobs', type=int, default=4, help='同时跑几个 pipeline.py(各约 2GB)')
parser.add_argument('--keep', action='store_true', help='跑完留着临时目录,打出路径')
args = parser.parse_args()


def spec_of(template):
    """这个模板最后一次跑用的配置;不在本机返回 None"""
    steps = ROOT / 'work' / template / 'steps.json'
    batch = json.loads(steps.read_text()).get('batch') if steps.exists() else None
    path = ROOT / 'batches' / str(batch) / 'specs' / f'{template}.json'
    return path if path.exists() else None


def make_root(path, scripts):
    """临时的仓库根目录:scripts 是函数,把那一版 scripts/ 放进去"""
    path.mkdir()
    for entry in ROOT.iterdir():
        if entry.name not in NOT_COPIED:
            (path / entry.name).symlink_to(entry)
    scripts(path)
    (path / 'work').mkdir()
    return path


def from_commit(path):
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', args.base, 'scripts'], check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(path)], input=archive, check=True)


def run(root, template, spec):
    """跑一个,返回 (各项校验, 输出文件 → 内容);出错时校验是 None、第二项是日志最后一行"""
    shutil.copytree(ROOT / 'work' / template, root / 'work' / template)
    env = {name: value for name, value in os.environ.items() if name != 'REPLICATE_API_TOKEN'}
    result = subprocess.run([sys.executable, '-I', str(root / 'scripts' / 'pipeline.py'), str(spec)], cwd=root, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        return None, (result.stdout.strip().splitlines() or [str(result.returncode)])[-1]
    out = root / 'output' / template
    return json.loads((out / 'checks.json').read_text()), {file.name: file.read_bytes() for file in sorted(out.iterdir())}


def leaves(value, prefix=''):
    """template.json 摊平成 路径 → 值"""
    if isinstance(value, dict):
        return {key: leaf for name, child in value.items() for key, leaf in leaves(child, f'{prefix}.{name}' if prefix else name).items()}
    return {prefix: value}


def changed(before, after):
    """变了的文件;template.json 写到是哪几项,checks.json 只在校验项里说"""
    names = []
    for name in sorted(set(before) | set(after)):
        if name == 'checks.json' or before.get(name) == after.get(name):
            continue
        if name == 'template.json' and name in before and name in after:
            old, new = leaves(json.loads(before[name])), leaves(json.loads(after[name]))
            name += '(' + '、'.join(key for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)) + ')'
        names.append(name + ('' if name.split('(')[0] in before else ' 新增') + ('' if name.split('(')[0] in after else ' 没了'))
    return names


samples = {'指定的': args.ids} if args.ids else SAMPLES
specs = {template: spec_of(template) for group in samples.values() for template in group}
temp = Path(tempfile.mkdtemp(prefix='regress-'))
roots = {'before': make_root(temp / 'before', from_commit),
         'after': make_root(temp / 'after', lambda path: shutil.copytree(ROOT / 'scripts', path / 'scripts', ignore=shutil.ignore_patterns('__pycache__')))}
with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
    runs = {(side, template): pool.submit(run, root, template, spec)
            for template, spec in specs.items() if spec for side, root in roots.items()}
bad = False
for group, templates in samples.items():
    print(f'\n{group}')
    for template in templates:
        if not specs[template]:
            print(f'  {template}: 不在本机,跳过')
            continue
        (checks0, files0), (checks1, files1) = (runs[side, template].result() for side in roots)
        if checks0 is None or checks1 is None:
            bad = bad or checks1 is None
            print(f'  {template}: ' + ';'.join(f'{label}出错 {files}' for label, checks, files in (('改前', checks0, files0), ('改后', checks1, files1)) if checks is None))
            continue
        failed = [name + ('' if checks0.get(name, True) else '(改前也没过)') for name, ok in checks1.items() if not ok]
        added = [name for name in checks1 if name not in checks0]
        bad = bad or bool(failed)
        files = changed(files0, files1)
        print(f'  {template}: ' + ('没过 ' + ';'.join(failed) if failed else '校验全过')
              + (f',多了 {len(added)} 项校验' if added else '') + ';' + ('变了 ' + '、'.join(files) if files else '文件都没变'))
if args.keep:
    print(f'\n临时目录 {temp}')
else:
    shutil.rmtree(temp)
sys.exit(1 if bad else 0)
