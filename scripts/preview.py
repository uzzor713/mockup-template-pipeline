"""
只读 output/<id>/ 里的模板文件,按编辑页的公式合成预览:每件商品各自的颜色和设计稿 → 按流向图位移 → 底图按 alpha 在线性光下盖上。
另出边缘放大图和蒙版示意图,存到 preview/<id>/;裁切范围和边缘取样位置在模板配置的 preview 里。
用法:.venv/bin/python -I scripts/preview.py batches/<批次>/specs/<id>.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config

spec = config.load(sys.argv[1])
OUT, PREVIEW = config.folder(spec, 'output'), config.folder(spec, 'preview')
COLORS = {'White': '#FFFFFF', 'Red': '#C02828', 'Navy': '#1F2A5A', 'Black': '#1A1A1A', 'Mustard': '#D4A017', 'Olive': '#5B6B2F'}
TINTS = [[0.1, 0.75, 0.3], [0.2, 0.45, 1.0], [0.85, 0.35, 0.8]]  # 蒙版示意里各件商品的颜色


def read(name):
    return np.asarray(Image.open(OUT / name)).astype(np.float32) / 255


template = json.loads((OUT / 'template.json').read_text())
W, H = template['size']['width'], template['size']['height']
base = np.asarray(Image.open(OUT / 'base.webp').convert('RGBA')).astype(np.float32) / 255  # 整张不透明时 webp 不存透明度通道
cover = read('cover.webp')
# 流向图双线性放大回模板尺寸,与编辑页的贴图采样一致;通道值 0 / 1 对应 ∓rangePx
ranges = template['displacement']


def flow(name, range_px):
    return cv2.resize((read(name)[..., :2] - 0.5) * 2 * range_px, (W, H), interpolation=cv2.INTER_LINEAR)


surfaces = [flow(r['surface'], ranges['surfaceRangePx']) for r in template['regions']]
folds = [flow(r['folds'], ranges['foldsRangePx']) for r in template['regions']]
products = [read(r['product'])[..., 3] for r in template['regions']]
prints = [read(r['print'])[..., 3] for r in template['regions']]


def checker(size=8):
    cells = (np.indices((size, size)).sum(axis=0) % 2).astype(np.float32)
    rgb = np.where(cells[..., None] > 0, [0.95, 0.95, 0.95], [0.1, 0.3, 0.85]).astype(np.float32)
    return rgb


DESIGN = checker()


def pick(names):
    """每件商品一个颜色,按顺序取,商品比颜色多时重复最后一个;返回标题和色值"""
    names = [names[min(i, len(names) - 1)] for i in range(len(products))]
    return ' / '.join(names), [COLORS[name] for name in names]


def hex_rgb(value):
    return np.array([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def render(colors, warp):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    canvas = np.zeros((H, W, 3), np.float32)
    for product, color in zip(products, colors):
        canvas += product[..., None] * hex_rgb(color)
    # 颜色层按覆盖度归一:边缘过渡由底图 alpha 负责
    canvas /= np.maximum(sum(products), 1e-4)[..., None]
    for index, (region, printable) in enumerate(zip(template['regions'], prints)):
        # 设计稿取所属商品的流向图:大曲面固定,褶皱乘 Warping;x、y 各自位移,减去设计稿中心处的位移
        box = region['logo']
        c = int(box['cy'] * H), int(box['cx'] * W)
        offset = surfaces[index] - surfaces[index][c] + (folds[index] - folds[index][c]) * warp
        # 位移后的位置转到 Logo 框自己的坐标(rotation 是度,屏幕上顺时针为正)
        dx, dy = xs + offset[..., 0] - box['cx'] * W, ys + offset[..., 1] - box['cy'] * H
        t = np.radians(box.get('rotation', 0))
        bw, bh = box['width'] * W, box['height'] * H
        u = (dx * np.cos(t) + dy * np.sin(t)) / bw + 0.5
        v = (-dx * np.sin(t) + dy * np.cos(t)) / bh + 0.5
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        n = DESIGN.shape[0]
        sample = DESIGN[np.clip((v * n).astype(int), 0, n - 1), np.clip((u * n).astype(int), 0, n - 1)]
        cover_amount = (inside * printable)[..., None]
        canvas = canvas * (1 - cover_amount) + sample * cover_amount
    alpha = base[..., 3:]
    return to_srgb(to_linear(base[..., :3]) * alpha + template.get('shadeGain', 1) * to_linear(canvas) * (1 - alpha))


options = spec.get('preview', {})
x0, y0, x1, y1 = options.get('crop', (0, 0, 1, 1))
CROP = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
panels = [('Cover', cover[..., :3])]
for names, warp in ((['White'], 0.5), (['Red', 'Navy'], 0.5), (['Black'], 0.5), (['Mustard', 'Olive'], 1.0)):
    title, colors = pick(names)
    panels.append((f'{title}, Warping {round(warp * 100)}%', render(colors, warp)))
scale = 0.32
tiles = []
for title, image in panels:
    tile = Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).crop(CROP)
    tile = tile.resize((int(tile.width * scale), int(tile.height * scale)), Image.LANCZOS)
    labeled = Image.new('RGB', (tile.width, tile.height + 28), 'white')
    labeled.paste(tile, (0, 28))
    ImageDraw.Draw(labeled).text((6, 7), title, fill='black')
    tiles.append(labeled)
sheet = Image.new('RGB', (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), tiles[0].height), 'white')
x = 0
for tile in tiles:
    sheet.paste(tile, (x, 0))
    x += tile.width + 8
sheet.save(PREVIEW / 'preview.png')

# 边缘放大:每行一处边缘(配置里的取样位置,模板像素),按像素放大不插值
SPOTS = options.get('spots', [])
columns = [('Cover', cover[..., :3])] + [(title, render(colors, 0)) for title, colors in (pick(['Red', 'Black']), pick(['Black', 'Mustard']))]
R, ZOOM, GAP, HEAD = 50, 3, 6, 20
cell = 2 * R * ZOOM
edges = Image.new('RGB', (len(columns) * (cell + GAP) - GAP, max(1, len(SPOTS) * (cell + HEAD + GAP) - GAP)), 'white')
for row, (spot, sx, sy) in enumerate(SPOTS):
    for column, (title, image) in enumerate(columns):
        crop = np.round(np.clip(image[sy - R:sy + R, sx - R:sx + R], 0, 1) * 255).astype(np.uint8)
        x, y = column * (cell + GAP), row * (cell + HEAD + GAP)
        edges.paste(Image.fromarray(crop).resize((cell, cell), Image.NEAREST), (x, y + HEAD))
        ImageDraw.Draw(edges).text((x + 4, y + 4), f'{spot} - {title}', fill='black')
edges.save(PREVIEW / 'edges.png')

# 蒙版示意:商品区着色,可印区域描黄边,默认 Logo 框描红边
vis = cover[..., :3].copy()
for product, tint in zip(products, TINTS * len(products)):
    vis = vis * (1 - 0.45 * product[..., None]) + np.array(tint) * 0.45 * product[..., None]
vis8 = np.round(vis * 255).astype(np.uint8)
for printable in prints:
    contours, _ = cv2.findContours((printable > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis8, contours, -1, (255, 220, 0), 4)
for region in template['regions']:
    box = region['logo']
    corners = cv2.boxPoints(((box['cx'] * W, box['cy'] * H), (box['width'] * W, box['height'] * H), box.get('rotation', 0)))
    cv2.polylines(vis8, [np.round(corners).astype(np.int32)], True, (230, 30, 30), 4)
masks = Image.fromarray(vis8).crop(CROP)
masks.resize((int(masks.width * 0.4), int(masks.height * 0.4)), Image.LANCZOS).save(PREVIEW / 'masks.png')
print(f'wrote preview.png, edges.png, masks.png to {PREVIEW.relative_to(config.ROOT)}')
