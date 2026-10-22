"""
试验:同一张铺满商品的网格,分别按竞品(Mockey)和我们的模板形变,并排对比,重点看轮廓附近的压缩。
竞品的形变按它的实现在本地复现:PixiJS DisplacementFilter,位移 = Warping × 舞台缩放 × (位移图预乘透明度后的值 − 0.5),
x、y 同一个值。复现和竞品真实截图对过:网格从 Warping 1 到 15 整体移动 15.8px,复现算出 14.8px。
需要配置里的 competitor(位移图、对应的商品、局部放大的范围),竞品的可印区域取该商品 reference.print。
结果在 work/<模板 id>/experiments/:mockey-compare.png(整件)和 mockey-compare-zoom.png(局部)。
用法:.venv/bin/python -I experiments/compare_mockey.py specs/<模板>.json [模板目录,默认 output/<id>]
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))  # -I 运行时仓库根目录不在 sys.path 里
import config

STEP, MAJOR = 40, 200  # 网格线间距,每隔 MAJOR 一条红线
OURS_DEFAULT = 0.4  # 编辑页 Warping 的默认值

spec = config.load(sys.argv[1])
TEMPLATE = Path(sys.argv[2]) if len(sys.argv) > 2 else config.folder(spec, 'output')
OUT = config.folder(spec, 'work') / 'experiments'
OUT.mkdir(exist_ok=True)
competitor = spec['competitor']
index = [p['id'] for p in spec['products']].index(competitor['product'])

read = lambda path: np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255
template = json.loads((TEMPLATE / 'template.json').read_text())
W, H = template['size']['width'], template['size']['height']
region = template['regions'][index]
cover = read(TEMPLATE / 'cover.webp')[..., :3]
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)


def flow(name, range_px):
    return cv2.resize((read(TEMPLATE / name)[..., :2] - 0.5) * 2 * range_px, (W, H), interpolation=cv2.INTER_LINEAR)


surface = flow(region['surface'], template['displacement']['surfaceRangePx'])
folds = flow(region['folds'], template['displacement']['foldsRangePx'])
center = int(region['logo']['cy'] * H), int(region['logo']['cx'] * W)
ours_print = read(TEMPLATE / region['print'])[..., 3]


def ours(warp):
    """编辑页的公式:大曲面固定,褶皱乘 Warping,各自减去设计稿中心处的位移"""
    return surface - surface[center] + (folds - folds[center]) * warp


# 竞品:Pixi 上传贴图时预乘透明度,滤镜直接读 r 通道
displacement = read(config.ROOT / competitor['displacement'])
premultiplied = displacement[..., 0] * displacement[..., 3]
reference = read(config.ROOT / spec['products'][index]['reference']['print']['file'])
mockey_print = (reference[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)) * reference[..., 3]


def mockey(warping):
    offset = warping * competitor['stageScale'] * (premultiplied - 0.5)
    return np.dstack([offset, offset])


def grid(offset, printable):
    """设计稿在 屏幕坐标 + 位移 处取样"""
    u, v = xs + offset[..., 0], ys + offset[..., 1]
    near = lambda a, step: np.abs((a + step / 2) % step - step / 2) < 1.5
    line, major = near(u, STEP) | near(v, STEP), near(u, MAJOR) | near(v, MAJOR)
    amount = (line * printable)[..., None]
    return cover * (1 - amount) + np.where(major[..., None], [0.85, 0.1, 0.1], [0.1, 0.1, 0.1]) * amount


default = competitor['defaultWarping']
cases = [(f'Mockey, warping {default} (default)', mockey(default), mockey_print),
         ('Mockey, warping 100', mockey(100), mockey_print),
         ('Ours, warping 0%', ours(0), ours_print),
         (f'Ours, warping {round(OURS_DEFAULT * 100)}% (default)', ours(OURS_DEFAULT), ours_print),
         ('Ours, warping 100%', ours(1), ours_print)]
images = [(title, grid(offset, printable)) for title, offset, printable in cases]


def sheet(boxes, scale, name):
    rows = []
    for x0, y0, x1, y1 in boxes:
        tiles = []
        for title, image in images:
            tile = Image.fromarray(np.round(np.clip(image[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8))
            tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.LANCZOS)
            labeled = Image.new('RGB', (tile.width, tile.height + 22), 'white')
            labeled.paste(tile, (0, 22))
            ImageDraw.Draw(labeled).text((5, 5), title, fill='black')
            tiles.append(labeled)
        row = Image.new('RGB', (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), tiles[0].height), 'white')
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width + 8
        rows.append(row)
    out = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows) + 8 * (len(rows) - 1)), 'white')
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height + 8
    out.save(OUT / name)


py, px = np.nonzero(ours_print > 0.5)
margin = 60
sheet([(max(px.min() - margin, 0), max(py.min() - margin, 0), min(px.max() + margin, W), min(py.max() + margin, H))],
      0.4, 'mockey-compare.png')
sheet([tuple(box) for box in competitor.get('zoom', [])], 1.0, 'mockey-compare-zoom.png')

# 数值:可印区域两三行上,离左右边缘不同距离处的横向前进量(> 1 为压缩)和相对中心的竖直位移
for fraction in (0.45, 0.7):
    y = int(py.min() + fraction * (py.max() - py.min()))
    inside = np.nonzero((ours_print[y] > 0.5) & (mockey_print[y] > 0.5))[0]
    x0, x1 = inside.min(), inside.max()
    points = [x0 + 3, x0 + 10, x0 + 20, x0 + 40, x0 + 80, center[1], x1 - 80, x1 - 40, x1 - 20, x1 - 10, x1 - 3]
    print(f'y={y}, 可印区域 x {x0}..{x1}')
    print('  x                    ' + ' '.join(f'{p:6d}' for p in points))
    for title, offset, _ in cases:
        advance = 1 + np.gradient(offset[y, :, 0])
        print(f'  {title[:20]:20s} 前进 ' + ' '.join(f'{advance[p]:6.2f}' for p in points))
        print(f'  {"":20s} 下垂 ' + ' '.join(f'{offset[y, p, 1] - offset[y, center[1], 1]:6.1f}' for p in points))
print('wrote', OUT / 'mockey-compare.png', OUT / 'mockey-compare-zoom.png')
