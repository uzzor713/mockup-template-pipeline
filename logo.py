"""
默认 Logo 框(template.json 里各区域的 logo):pipeline.py 出模板时调用;配置里商品的 logo 设置改了时单独跑,
只读各区域的可印区(print-n.png),重写 template.json 里的 logo 和 checks.json 里的「logo center inside print」,不重出底图和流向图。
商品的 logo 设置:
- 不写:框是正方形,放在可印区正中,边长为可印区最宽处的 width(默认 LOGO_WIDTH);
- top:上衣用,框往上挪到中心在可印区从上往下 top 处(胸口,背面是上背部),竞品 Mockey 的上衣在约 32%;框上沿碰到领口就停;
- side 为 left:左胸(穿衣人的左边,正面照里在画面右侧),带拉链、扣子的正面用,和 top 一起写;横向放在门襟(那一高度可印区的中线)
  到右端的中点,边长为门襟到右端宽度的 width;
- fill:平面印刷品(海报、书、杂志、卡片、传单等)用,框铺满可印区,形状和可印区一样。
框跟着可印区摆放的方向转(rotation,度,屏幕上顺时针为正)。
用法:.venv/bin/python -I logo.py specs/<模板>.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TILT_MIN = 3  # 可印区是四边形(海报、卡片)时,倾角(度)小于它算摆正,框不转
TILT_FREE = 10  # 可印区不是四边形(衣服等)时摆正的上限:人穿着站直时躯干轮廓不对称,会算出 3–9.4°;平放、搭着的歪 10° 以上的都是真歪
ROUND_SHAPE = 0.93  # 可印区轮廓和拟合椭圆的重合度(交 / 并)高于它算圆形、椭圆形,看不出方向,框不转(圆贴纸中位数 0.99,衣服、海报、卡片至多 0.91)
LOGO_WIDTH = 0.4  # 没写 width 时,框边长至多为可印区最宽处的这么多
LOGO_REACH = 1.4  # 框边长至多为可印区最大内切圆半径的这么多倍,窄长的可印区里框不至于太大
# 例外清单 logo 列可填的名字(按模板单独改放法,如同一品类里拍正面、带拉链的那几张)
PLACES = {'left-chest': {'top': 0.25, 'side': 'left'}}
NECK = 90  # 框横向范围里各列可印区上沿的这个分位当作领口:取最低处会被门襟、抽绳这类从领口往下的窄缺口带偏
CHECK = 'logo center inside print'


def tilt(inside):
    """可印区摆放的倾角(度,屏幕上顺时针为正,±45° 内)。最大一块的轮廓能近似成四边形时(透视下的海报、卡片)取四条边方向的平均,
    偏 TILT_MIN 以上就转;否则(衣服的躯干等)取最小外接矩形的方向,偏 TILT_FREE 以上才转。
    方向按 90° 周期平均,横放、竖放都对齐到最近的那个。圆形、椭圆形看不出方向,为 0"""
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) >= 5:
        filled = cv2.drawContours(np.zeros(inside.shape, np.uint8), [contour], -1, 1, -1)
        ellipse = cv2.ellipse(np.zeros(inside.shape, np.uint8), cv2.fitEllipse(contour), 1, -1)
        if (filled & ellipse).sum() > ROUND_SHAPE * (filled | ellipse).sum():
            return 0.0
    quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)[:, 0]
    corners = (quad if len(quad) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))).astype(np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    a = 4 * np.arctan2(edges[:, 1], edges[:, 0])
    angle = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 4)
    return 0.0 if abs(angle) < (TILT_MIN if len(quad) == 4 else TILT_FREE) else round(angle, 1)


def box(inside, settings):
    """一个区域的默认 Logo 框 {cx, cy, width, height, rotation},位置、宽高按模板宽高归一。
    在可印区自己的方向上(按 rotation 转正)量:fill 取最大一块上下、左右两端围成的框;否则竖向取上下两端之间 top 处,
    横向取那一高度左右两端的中点,落到可印区外面时(L 形、分成几块的)退回离边最远的点"""
    H, W = inside.shape
    rotation = tilt(inside)
    ys, xs = np.nonzero(inside)
    c, s = np.cos(np.radians(rotation)), np.sin(np.radians(rotation))
    across, along = xs * c + ys * s, -xs * s + ys * c
    if settings.get('fill'):
        # 分成几块的(几张名片、扑克牌)只铺满最大的那块
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8))
        keep = labels[ys, xs] == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        across, along = across[keep], along[keep]
        middle, level = (across.min() + across.max()) / 2, (along.min() + along.max()) / 2
        width, height = across.max() - across.min(), along.max() - along.min()
        x, y = middle * c - level * s, middle * s + level * c
    else:
        distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
        span = along.max() - along.min()
        left = settings.get('side') == 'left'

        def place(level):
            row = np.abs(along - level) <= max(2.0, 0.01 * span)
            middle = (across[row].min() + across[row].max()) / 2
            reach = across[row].max() - middle
            if left:
                middle += reach / 2
            x, y = int(round(middle * c - level * s)), int(round(middle * s + level * c))
            if not (0 <= y < H and 0 <= x < W and distance[y, x] > 0):
                y, x = np.unravel_index(np.argmax(distance), distance.shape)
            return middle, x, y, reach

        # 大小:居中的只看整块可印区,最宽处(衣服是袖口到袖口)的 width;左胸看胸口那一高度门襟到右端的宽度。都至多最大内切圆半径的 LOGO_REACH 倍。
        # 先放在正中(左胸先放在胸口);top 再把框往上挪到那里,框上沿碰到领口(框横向范围里可印区上沿的低处,见 NECK)就停
        middle, x, y, reach = place(along.min() + (settings['top'] if left else 0.5) * span)
        wide = reach if left else across.max() - across.min()
        width = height = min(settings.get('width', LOGO_WIDTH) * wide, LOGO_REACH * float(distance.max()))
        if 'top' in settings:
            under = np.abs(across - middle) <= width / 2
            columns = np.round(across[under] - across[under].min()).astype(int)
            tops = np.full(columns.max() + 1, np.inf)
            np.minimum.at(tops, columns, along[under])
            neck = float(np.percentile(tops[np.isfinite(tops)], NECK))
            middle, x, y, _ = place(min(max(along.min() + settings['top'] * span, neck + width / 2), along.min() + 0.5 * span))
    return {'cx': round(float(x) / W, 4), 'cy': round(float(y) / H, 4), 'width': round(float(width) / W, 4),
            'height': round(float(height) / H, 4), 'rotation': rotation}


def centered(prints, products, logos):
    """框的中心在可印区里(铺满的框中心可以落在可印区外,比如几张卡片之间)"""
    H, W = prints[0].shape
    return all(item.get('logo', {}).get('fill') or printable[min(int(l['cy'] * H), H - 1), min(int(l['cx'] * W), W - 1)] > 0.5
               for printable, item, l in zip(prints, products, logos))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
    import config

    spec = config.load(sys.argv[1])
    out = config.folder(spec, 'output')
    template = json.loads((out / 'template.json').read_text())
    prints = [np.asarray(Image.open(out / region['print']).getchannel('A')).astype(np.float32) / 255 for region in template['regions']]
    logos = [box(printable > 0.5, item.get('logo', {})) for printable, item in zip(prints, spec['products'])]
    for region, logo in zip(template['regions'], logos):
        region['logo'] = logo
    (out / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))
    checks = json.loads((out / 'checks.json').read_text())
    checks[CHECK] = centered(prints, spec['products'], logos)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2))
    for region in template['regions']:
        print(region['key'], region['logo'])
    print(('PASS ' if checks[CHECK] else 'FAIL ') + CHECK)
