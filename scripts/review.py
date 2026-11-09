"""
批量的抽查页,batch.py 调用,写到 batches/<名称>/:
- review.html:总览,各品类的通过、没通过、出错张数,点进品类页;
- review/<品类>.html:出错、没通过的全列,通过的按 id 打散后先列 PASS_SAMPLE 张,其余点开再看;
- thumbs/:每个模板的分割叠图、蒙版示意缩略图,点缩略图看原图。
每张卡片可以标「有问题」,选问题类型、写备注;标记存在浏览器里(按批次分开),页面底部导出成例外清单 CSV。
清单交给调参数的人(batch.py --pack 把这几张的源图一起打包),补上要改的列后给 --overrides 用。
"""
import hashlib
import html

from PIL import Image

import config

THUMB = 320  # 缩略图宽度
PASS_SAMPLE = 48  # 品类页先列的通过模板数
ORDER = {'error': 0, 'fail': 1, 'pass': 2}
ISSUES = ('商品没分全或分多了', '部件不对', '遮挡物不对', '裁切不对', '其他')
IMAGES = (('segment', 'work', 'segment-check.png', '分割结果'), ('masks', 'preview', 'masks.png', '可印区域'))

STYLE = '''
:root { --bg: #f4f4f4; --card: #fff; --text: #222; --muted: #666; --line: #ddd; --pass: #3a3; --fail: #e90; --error: #d33; --mark: #2a62d6; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161616; --card: #222; --text: #eee; --muted: #aaa; --line: #3a3a3a; --mark: #6c9bff; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 16px 16px 72px; font: 14px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--text); }
h1 { font-size: 20px; margin: 0 0 4px; }
a { color: var(--mark); }
.muted { color: var(--muted); margin: 0 0 12px; }
table { border-collapse: collapse; width: 100%; max-width: 720px; background: var(--card); }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; }
th.n, td.n { text-align: right; font-variant-numeric: tabular-nums; }
.cards { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
.card { background: var(--card); border-radius: 8px; padding: 10px; border-top: 4px solid var(--pass); }
.card.fail { border-top-color: var(--fail); } .card.error { border-top-color: var(--error); }
.card.marked { outline: 3px solid var(--mark); }
.head { display: flex; gap: 8px; align-items: baseline; justify-content: space-between; }
.name { margin: 0; font-weight: 600; word-break: break-all; }
.note { margin: 4px 0 0; color: var(--fail); font-size: 12px; }
.src { margin: 2px 0 0; color: var(--muted); font-size: 12px; word-break: break-all; }
.images { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }
figure { margin: 0; }
figcaption { margin-top: 2px; color: var(--muted); font-size: 12px; text-align: center; }
.images img, .missing { width: 100%; display: block; aspect-ratio: 4 / 5; object-fit: contain; background: var(--line); }
.missing { display: grid; place-items: center; color: var(--muted); font-size: 12px; }
button { font: inherit; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--line); background: var(--card); color: var(--text); cursor: pointer; }
button[aria-pressed="true"] { background: var(--mark); border-color: var(--mark); color: #fff; }
.more { margin: 16px 0; }
.issue { display: none; gap: 6px; margin-top: 8px; }
.card.marked .issue { display: grid; grid-template-columns: auto 1fr; }
select, input { font: inherit; min-width: 0; padding: 3px 6px; border-radius: 6px; border: 1px solid var(--line); background: var(--bg); color: var(--text); }
.bar { position: fixed; left: 0; right: 0; bottom: 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 16px;
       background: var(--card); border-top: 1px solid var(--line); }
'''

# 标记存 localStorage(按批次分开):{id: {category, issue, note}};读写失败(隐私模式等)时页面照常用,只是记不住
SCRIPT = '''
const KEY = 'review:' + document.body.dataset.batch;
const load = () => {
  let marks = {};
  try { marks = JSON.parse(localStorage.getItem(KEY)) || {}; } catch {}
  // 早先只存了品类名
  for (const id in marks) if (typeof marks[id] === 'string') marks[id] = { category: marks[id], issue: '', note: '' };
  return marks;
};
const save = (marks) => { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch {} };
let marks = load();
function refresh() {
  document.querySelectorAll('[data-id]').forEach((card) => {
    const mark = marks[card.dataset.id];
    card.classList.toggle('marked', !!mark);
    card.querySelector('button').setAttribute('aria-pressed', !!mark);
    card.querySelector('select').value = mark ? mark.issue : '';
    card.querySelector('input').value = mark ? mark.note : '';
  });
  document.querySelectorAll('[data-count]').forEach((cell) => {
    cell.textContent = Object.values(marks).filter((m) => m.category === cell.dataset.count).length || '';
  });
  document.getElementById('total').textContent = Object.keys(marks).length;
}
document.addEventListener('click', (event) => {
  const card = event.target.closest('[data-id] .head button') && event.target.closest('[data-id]');
  if (card) {
    if (card.dataset.id in marks) delete marks[card.dataset.id];
    else marks[card.dataset.id] = { category: card.dataset.category, issue: '', note: '' };
    save(marks); refresh();
  }
});
document.addEventListener('change', (event) => {
  const card = event.target.closest('[data-id]');
  const mark = card && marks[card.dataset.id];
  if (mark) { mark[event.target.name] = event.target.value; save(marks); }
});
// 例外清单:issue、note 只给人看;product 到 skip 由调参数的人填,空着的不改
const COLUMNS = ['id', 'issue', 'note', 'product', 'parts', 'occluders', 'crop', 'preset', 'skip'];
const cell = (value) => /[",\\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
const csv = () => COLUMNS.join(',') + '\\n' + Object.keys(marks).sort().map((id) =>
  [id, marks[id].issue, marks[id].note, '', '', '', '', '', ''].map(cell).join(',') + '\\n').join('');
document.getElementById('download').onclick = () => {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob(['\\ufeff' + csv()], { type: 'text/csv' }));
  link.download = document.body.dataset.batch + '-overrides.csv';
  link.click();
  URL.revokeObjectURL(link.href);
};
document.getElementById('clear').onclick = () => {
  if (Object.keys(marks).length) { marks = {}; save(marks); refresh(); }
};
const more = document.getElementById('more');
if (more) more.onclick = () => { document.getElementById('rest').hidden = false; more.remove(); };
window.addEventListener('storage', () => { marks = load(); refresh(); });
refresh();
'''


OPTIONS = ''.join(f'<option>{issue}</option>' for issue in ISSUES)


def thumbnails(folder, spec_id):
    """抽查页用的缩略图;原图比缩略图新才重出"""
    (folder / 'thumbs').mkdir(exist_ok=True)
    for kind, root, name, _ in IMAGES:
        source, thumb = config.ROOT / root / spec_id / name, folder / 'thumbs' / f'{spec_id}-{kind}.jpg'
        if source.exists() and (not thumb.exists() or thumb.stat().st_mtime < source.stat().st_mtime):
            image = Image.open(source).convert('RGB')
            image.thumbnail((THUMB, THUMB * 2))
            image.save(thumb, quality=80)


def page(folder, title, body, back=''):
    bar = (f'<div class="bar">{back}已标记 <b id="total">0</b> 张'
           '<button id="download" type="button">导出例外清单</button><button id="clear" type="button">清空标记</button></div>')
    return (f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{STYLE}</style></head><body data-batch="{html.escape(folder.name)}">{body}{bar}<script>{SCRIPT}</script></body></html>\n')


def card(r):
    images = ''
    for kind, root, name, alt in IMAGES:
        if (config.ROOT / root / r['id'] / name).exists():
            image = (f'<a href="../../../{root}/{r["id"]}/{name}"><img src="../thumbs/{r["id"]}-{kind}.jpg" '
                     f'loading="lazy" alt="{alt}"></a>')
        else:
            image = '<div class="missing">还没生成</div>'
        images += f'<figure>{image}<figcaption>{alt}</figcaption></figure>'
    note = f'<p class="note">{html.escape(r["note"])}</p>' if r['note'] else ''
    return (f'<li class="card {r["status"]}" data-id="{html.escape(r["id"])}" data-category="{html.escape(r["category"])}">'
            f'<div class="head"><p class="name">{html.escape(r["id"])} · {r["status"]}</p>'
            f'<button type="button" aria-pressed="false">有问题</button></div>{note}'
            f'<div class="issue"><select name="issue" aria-label="问题类型"><option value="">问题类型</option>{OPTIONS}</select>'
            '<input name="note" placeholder="备注(可不写)" aria-label="备注"></div>'
            f'<p class="src">{html.escape(r["source"])}</p><div class="images">{images}</div></li>')


def write(folder, results):
    """results:每项 {id, category, source, status, note, seconds}"""
    (folder / 'review').mkdir(exist_ok=True)
    categories = sorted({r['category'] for r in results})
    count = lambda rows, status: sum(r['status'] == status for r in rows)
    legend = '左:分割(绿 = 商品,蓝 = 不可印部件,红 = 遮挡物);右:可印区域(黄线以内可印,红框 = 默认 Logo)。点缩略图看原图。'
    rows = ''
    for name in categories:
        group = [r for r in results if r['category'] == name]
        rows += (f'<tr><td><a href="review/{html.escape(name)}.html">{html.escape(name)}</a></td>'
                 + ''.join(f'<td class="n">{count(group, s) or ""}</td>' for s in ORDER)
                 + f'<td class="n">{len(group)}</td><td class="n" data-count="{html.escape(name)}"></td></tr>')
        # 出错、没通过的全列;通过的按 id 的哈希打散,先列一批当抽样,避免只看到文件名靠前的
        bad = sorted((r for r in group if r['status'] != 'pass'), key=lambda r: (ORDER[r['status']], r['id']))
        good = sorted((r for r in group if r['status'] == 'pass'), key=lambda r: hashlib.sha1(r['id'].encode()).hexdigest())
        body = (f'<h1>{html.escape(name)}</h1><p class="muted">{folder.name} · 出错 {count(group, "error")} · '
                f'没通过 {count(group, "fail")} · 通过 {len(good)}。{legend}</p>'
                f'<ul class="cards">{"".join(card(r) for r in bad + good[:PASS_SAMPLE])}</ul>')
        if len(good) > PASS_SAMPLE:
            body += (f'<button id="more" class="more" type="button">显示其余 {len(good) - PASS_SAMPLE} 张通过的</button>'
                     f'<ul id="rest" class="cards" hidden>{"".join(card(r) for r in good[PASS_SAMPLE:])}</ul>')
        (folder / 'review' / f'{name}.html').write_text(page(folder, f'{name} · {folder.name}', body, '<a href="../review.html">总览</a> · '))
    totals = ' · '.join(f'{label} {count(results, s)}' for s, label in zip(ORDER, ('出错', '没通过', '通过')))
    body = (f'<h1>{html.escape(folder.name)} 抽查</h1><p class="muted">共 {len(results)} 张 · {totals}。'
            '品类页里出错、没通过的全列,通过的先列一批抽样。看到有问题的点「有问题」,最后导出例外清单。</p>'
            '<table><thead><tr><th>品类</th>' + ''.join(f'<th class="n">{h}</th>' for h in ('出错', '没通过', '通过', '共', '已标记'))
            + '</tr></thead>'
            f'<tbody>{rows}</tbody></table>')
    (folder / 'review.html').write_text(page(folder, f'{folder.name} 抽查', body))
