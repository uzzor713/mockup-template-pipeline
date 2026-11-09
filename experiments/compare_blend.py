"""
模板墙两种合成方式与编辑页的对比图:
- img 叠图:流水线另出的 base-css.webp(按 sRGB 直接叠),浏览器用色块 + mask-image + 底图叠出来;
- 线性光:和编辑页、导出同一个公式,不做位移;
- 编辑页:线性光 + 大曲面 + 褶皱(Warping 默认 40%)。
每格标出和「线性光」那格的平均色差(商品区内,0–255)。
用法:.venv/bin/python -I experiments/compare_blend.py batches/<批次>/specs/<id>.json <设计稿 PNG>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))  # -I 运行时 scripts/ 不在 sys.path 里
import config

spec = config.load(sys.argv[1])
OUT = config.folder(spec, 'output')
DEST = config.folder(spec, 'work') / 'experiments'
DEST.mkdir(parents=True, exist_ok=True)
COLORS = [('White', '#FFFFFF'), ('Red', '#C02828'), ('Mustard', '#D4A017'), ('Olive', '#5B6B2F'), ('Navy', '#1F2A5A'), ('Black', '#1A1A1A')]
WARP = 0.4


def read(path):
    return np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def hex_rgb(value):
    return np.array([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


template = json.loads((OUT / 'template.json').read_text())
W, H = template['size']['width'], template['size']['height']
k = template.get('shadeGain', 1)
base = read(OUT / 'base.webp')
alpha, rgb_lin = base[..., 3:], to_linear(base[..., :3])
products = [read(OUT / r['product'])[..., 3] for r in template['regions']]
prints = [read(OUT / r['print'])[..., 3] for r in template['regions']]
ranges = template['displacement']


def flow(name, range_px):
    return cv2.resize((read(OUT / name)[..., :2] - 0.5) * 2 * range_px, (W, H), interpolation=cv2.INTER_LINEAR)


surfaces = [flow(r['surface'], ranges['surfaceRangePx']) for r in template['regions']]
folds = [flow(r['folds'], ranges['foldsRangePx']) for r in template['regions']]

# img 叠图用流水线落盘的 base-css.webp,和模板墙实际加载的一致
css = read(OUT / template['files']['baseCss'])
css_alpha, css_rgb = css[..., 3:], css[..., :3]

design = read(sys.argv[2])
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)


def color_layer(color, warp):
    """颜色 + 设计稿(sRGB 值),设计稿按默认框摆放、可印区域裁切;warp 为 None 时不做位移"""
    canvas = np.zeros((H, W, 3), np.float32)
    for product in products:
        canvas += product[..., None] * hex_rgb(color)
    canvas /= np.maximum(sum(products), 1e-4)[..., None]
    for index, (region, printable) in enumerate(zip(template['regions'], prints)):
        box = region['logo']
        bw, bh = box['width'] * W, box['height'] * H
        # Size 100%:设计稿按比例完整放进默认框
        scale = min(bw / design.shape[1], bh / design.shape[0])
        dw, dh = design.shape[1] * scale, design.shape[0] * scale
        px, py = xs, ys
        if warp is not None:
            c = int(box['cy'] * H), int(box['cx'] * W)
            offset = surfaces[index] - surfaces[index][c] + (folds[index] - folds[index][c]) * warp
            px, py = xs + offset[..., 0], ys + offset[..., 1]
        mx = ((px - (box['cx'] * W - dw / 2)) / scale).astype(np.float32)
        my = ((py - (box['cy'] * H - dh / 2)) / scale).astype(np.float32)
        sample = cv2.remap(design, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        amount = sample[..., 3:] * printable[..., None]
        canvas = canvas * (1 - amount) + sample[..., :3] * amount
    return canvas


def linear(canvas):
    return to_srgb(rgb_lin * alpha + k * to_linear(canvas) * (1 - alpha))


def css(canvas):
    return canvas * (1 - css_alpha) + css_rgb * css_alpha


inside = sum(products) > 0.5
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
SMALL = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 17)


def stats(image, reference):
    """商品区内和线性光的差:平均差、99% 的像素不超过多少、整体偏亮还是偏暗(0–255)"""
    diff = (image - reference)[inside]
    mean = np.abs(diff).mean() * 255
    worst = np.percentile(np.abs(diff).max(axis=1), 99) * 255
    shift = (diff @ LUMA).mean() * 255
    return f'平均差 {mean:.0f}\n99% ≤ {worst:.0f}\n亮度 {shift:+.0f}'


def to_tile(image, box, size):
    x0, y0, x1, y1 = box
    tile = Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)[y0:y1, x0:x1])
    return tile.resize(size, Image.LANCZOS) if size != tile.size else tile


crop = spec.get('preview', {}).get('crop', (0, 0, 1, 1))
CROP = int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)
TILE = 360
HEAD, GAP, LABEL = 48, 12, 170
columns = ['模板墙 · img 叠图', '模板墙 · 线性光(= 下载结果的颜色)', '编辑页 · 线性光 + 曲面褶皱(Warping 40%)']
tile_h = int(TILE * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0]))
sheet = Image.new('RGB', (LABEL + 3 * (TILE + GAP), HEAD + len(COLORS) * (tile_h + GAP)), 'white')
draw = ImageDraw.Draw(sheet)
for i, title in enumerate(columns):
    draw.text((LABEL + i * (TILE + GAP) + 4, 14), title, fill='black', font=SMALL)
renders = {}
for row, (name, color) in enumerate(COLORS):
    flat = color_layer(color, None)
    reference = linear(flat)
    images = [css(flat), reference, linear(color_layer(color, WARP))]
    renders[name] = images[:2]
    y = HEAD + row * (tile_h + GAP)
    draw.text((8, y + 8), f'{name}  {color}', fill='black', font=SMALL)
    summary = stats(images[0], reference)
    print(name, summary.replace('\n', ', '))
    draw.text((8, y + 40), 'img 叠图对线性光:\n' + summary, fill='#444444', font=SMALL, spacing=6)
    for i, image in enumerate(images):
        sheet.paste(to_tile(image, CROP, (TILE, tile_h)), (LABEL + i * (TILE + GAP), y))
sheet.save(DEST / 'compare-blend.png')

# 细节:胸口 Logo 附近按原尺寸裁出来,img 叠图和线性光并排
box = template['regions'][0]['logo']
cw, ch = int(W * 0.42), int(H * 0.26)
cx, cy = int(box['cx'] * W), int(box['cy'] * H)
DETAIL = cx - cw // 2, cy - ch // 2 - int(ch * 0.1), cx + cw // 2, cy + ch // 2 - int(ch * 0.1)
DETAIL_COLORS = ['Red', 'Navy', 'Black', 'Mustard']
detail = Image.new('RGB', (LABEL + 2 * (cw + GAP), HEAD + len(DETAIL_COLORS) * (ch + GAP)), 'white')
draw = ImageDraw.Draw(detail)
for i, title in enumerate(['模板墙 · img 叠图(原尺寸)', '线性光 = 下载结果(原尺寸)']):
    draw.text((LABEL + i * (cw + GAP) + 4, 14), title, fill='black', font=SMALL)
for row, name in enumerate(DETAIL_COLORS):
    y = HEAD + row * (ch + GAP)
    draw.text((8, y + 8), f'{name}  {dict(COLORS)[name]}', fill='black', font=SMALL)
    for i, image in enumerate(renders[name]):
        detail.paste(to_tile(image, DETAIL, (cw, ch)), (LABEL + i * (cw + GAP), y))
detail.save(DEST / 'compare-blend-detail.png')
print(f'wrote compare-blend.png, compare-blend-detail.png to {DEST.relative_to(config.ROOT)}')
