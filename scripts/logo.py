"""
默认 Logo 框(template.json 里各区域的 logo):pipeline.py 出模板时调用;配置里商品的 logo 设置改了时单独跑,
只读各区域的可印区(print-n.png),重写 template.json 里的 logo 和 checks.json 里的「logo center inside print」,不重出底图和流向图。
商品的 logo 设置:
- 不写:框是正方形,放在可印区正中,边长为可印区最宽处的 width(默认 LOGO_WIDTH);
- top:上衣用,框往上挪到中心在可印区从上往下 top 处(胸口,背面是上背部),竞品 Mockey 的上衣在约 32%;框上沿碰到领口就停;
- side 为 left:左胸(穿衣人的左边,正面照里在画面右侧),和 top 一起写;横向放在门襟(那一高度可印区的中线)
  到右端的中点,边长为门襟到右端宽度的 width;
- closure:上衣用,问 Gemini 照片拍的是不是这件的正面、前襟有没有拉链或扣子,是就把 closure 里的设置合进来(预设里写左胸);
  回答存在 work/<id>/logo-closure.json,图和问题没变就不再问;
- fill:平面印刷品(海报、书、杂志、卡片、传单等)用,框铺满可印区;透视下的梯形、斜放的平行四边形取里面放得下的最大矩形,不往外包;
- freeform:贴纸用,可印区可能是爱心、闪电这类异形,看不出摆放方向:不是四边形、也不接近矩形(见 RECT_SHAPE)时框不转;
- split:一双袜子、两条裤腿这类挨在一起、可印区连成一块的用,写单只的说法(如 sock、pant leg),或几个说法的列表;SAM 3 按它分出单只,
  框只放在可印区里最大的那只上,再按 top 等设置放。列表按顺序试,分出两只就不再往下试;都不到两只时照常按整块放。
  分出的蒙版存在 work/<id>/logo-split*,图和说法没变就不再分。
框跟着可印区摆放的方向转(rotation,度,屏幕上顺时针为正)。
用法:.venv/bin/python -I scripts/logo.py batches/<批次>/specs/<id>.json(配置里写了 closure、split 时要 Replicate 的 token,同 run_segment.py)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TILT_MIN = 3  # 可印区是四边形(海报、卡片)时,倾角(度)小于它算摆正,框不转;按平面贴的(plane)不取整,框跟着面的实际倾角转
TILT_FREE = 10  # 可印区不是四边形(衣服等)时摆正的上限:人穿着站直时躯干轮廓不对称,会算出 3–9.4°;平放、搭着的歪 10° 以上的都是真歪
ROUND_SHAPE = 0.93  # 可印区轮廓和拟合椭圆的重合度(交 / 并)高于它算圆形、椭圆形,看不出方向,框不转(圆贴纸中位数 0.99,衣服、海报、卡片至多 0.91)
RECT_SHAPE = 0.85  # 写了 freeform 的,可印区面积占最小外接矩形低于它算异形,框不转(异形贴纸至多 0.82,圆角、被手指挡住一点的矩形贴纸 0.88 以上)
LOGO_WIDTH = 0.4 # 没写 width 时,框边长至多为可印区最宽处的这么多
LOGO_REACH = 1.4  # 框边长至多为可印区最大内切圆半径的这么多倍,窄长的可印区里框不至于太大
# 例外清单 logo 列可填的名字,Gemini 判断前襟判错时按模板改正:left-chest 左胸,chest 胸口正中
PLACES = {'left-chest': {'top': 0.32, 'side': 'left'}, 'chest': {'top': 0.32}}
CLOSURE_SIZE = 1024  # 交给 Gemini 的图,长边缩到这么大
CLOSURE_PROMPT = '''Both images are the same photo, used to make a mockup template for: {what}.
In the second image, the {what} in question is outlined in yellow.
1. Does the photo show the front of that garment (not its back or side)?
2. Does its front open down the middle with a zipper, buttons or snaps (e.g. a zip hoodie, a jacket, a baseball jersey,
   a polo shirt's button placket)? Count a closed zipper or buttoned placket too; a plain crew or V neck has none.
Answer with JSON only, like {{"front": true, "closure": "zipper"}}; closure is one of "zipper", "buttons", "snaps", "none".'''
SKEW_FILL = 0.9  # 铺满的框:可印区凸包不到它方向上外接框的这么多算透视、斜放,取最大内接矩形;否则取外接框
HULL_SOLID = 0.9  # 铺满的框:可印区面积占凸包至少这么多时按凸包算(边角被挡的缺口小),否则按可印区本身
INSCRIBED_SIZE = 512  # 铺满的框在这么大的格子上找最大内接矩形(模板长边 2560 时误差约 5 像素,只往里收)
NECK = 90  # 框横向范围里各列可印区上沿的这个分位当作领口:取最低处会被门襟、抽绳这类从领口往下的窄缺口带偏
NECK_ROUNDS = 4  # 框挪了再量领口,至多几次(一般两三次就不动了)
SPLIT_SHARE = 0.1  # split 分出的单只盖住可印区至少这么多才算一只(背景里别的袜子、裤子不算)
CHECK = 'logo center inside print'


def tilt(inside, freeform=False, plane=False):
    """可印区摆放的倾角(度,屏幕上顺时针为正,±45° 内)。最大一块的轮廓能近似成四边形时(透视下的海报、卡片)取四条边方向的平均,
    偏 TILT_MIN 以上就转,按平面贴的(plane)多小都转:流向图在可印区中心只拉伸不转,设计稿跟着面转全靠框,
    歪 1–3° 的盒子面吸成 0 时设计稿和面的边差 1–1.5°;否则(衣服的躯干等)取最小外接矩形的方向,偏 TILT_FREE 以上才转。
    方向按 90° 周期平均,横放、竖放都对齐到最近的那个。圆形、椭圆形看不出方向,为 0;freeform 时异形的也为 0"""
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) >= 5:
        filled = cv2.drawContours(np.zeros(inside.shape, np.uint8), [contour], -1, 1, -1)
        ellipse = cv2.ellipse(np.zeros(inside.shape, np.uint8), cv2.fitEllipse(contour), 1, -1)
        if (filled & ellipse).sum() > ROUND_SHAPE * (filled | ellipse).sum():
            return 0.0
    quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)[:, 0]
    rect = cv2.minAreaRect(contour)
    if freeform and len(quad) != 4 and cv2.contourArea(contour) < RECT_SHAPE * rect[1][0] * rect[1][1]:
        return 0.0
    corners = (quad if len(quad) == 4 else cv2.boxPoints(rect)).astype(np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    a = 4 * np.arctan2(edges[:, 1], edges[:, 0])
    angle = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 4)
    floor = (0 if plane else TILT_MIN) if len(quad) == 4 else TILT_FREE
    return 0.0 if abs(angle) < floor else round(angle, 1) + 0.0  # + 0.0:-0.0 记成 0.0


def inscribed(across, along, size=INSCRIBED_SIZE):
    """多边形(顶点在可印区自己方向上的坐标)里放得下的最大矩形:中心和宽高,同一坐标。
    缩到长边 size 上逐行累计高度、按直方图找最大矩形,只取完全在多边形里的格子,往里收不往外包"""
    x0, y0 = across.min(), along.min()
    k = size / max(across.max() - x0, along.max() - y0)
    canvas = np.zeros((size + 3, size + 3), np.uint8)
    points = np.round(np.stack([(across - x0) * k, (along - y0) * k], 1) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [points], 1, cv2.LINE_8, 4)
    canvas = cv2.erode(canvas, np.ones((3, 3), np.uint8))  # 格子压线的不算在里面
    heights = np.zeros(canvas.shape[1] + 1, int)
    best = (0, 0, 0, 0, 0)  # 面积, 左, 右(不含), 上, 下(不含)
    for row in range(canvas.shape[0]):
        heights[:-1] = np.where(canvas[row], heights[:-1] + 1, 0)
        stack = []
        for col, h in enumerate(heights):
            start = col
            while stack and stack[-1][1] >= h:
                start, top = stack.pop()
                if top * (col - start) > best[0]:
                    best = (top * (col - start), start, col, row + 1 - top, row + 1)
            stack.append((start, h))
    _, left, right, top, bottom = best
    return x0 + (left + right) / 2 / k, y0 + (top + bottom) / 2 / k, (right - left) / k, (bottom - top) / k


def box(inside, settings):
    """一个区域的默认 Logo 框 {cx, cy, width, height, rotation},位置、宽高按模板宽高归一。
    within(split 分出的那只)不为 None 时只在可印区和它重合的部分里放。在可印区自己的方向上(按 rotation 转正)量:fill 取最大一块上下、左右两端围成的框,透视、斜放的取里面的最大内接矩形;否则竖向取上下两端之间 top 处,
    横向取那一高度左右两端的中点,落到可印区外面时(L 形、分成几块的)退回离边最远的点"""
    H, W = inside.shape
    if settings.get('within') is not None:
        inside = inside & settings['within']
    plane, turn = settings.get('plane', False), settings.get('turn')
    # 按平面贴、透视按四角校准过的,pipeline.py 给出设计稿该跟着面转的角度(turn),直接用:斜得厉害的梯形四条边平均、放得下最大矩形的方向都会偏 2–5°
    rotation = tilt(inside, settings.get('freeform', False), plane) if turn is None else turn
    ys, xs = np.nonzero(inside)
    c, s = np.cos(np.radians(rotation)), np.sin(np.radians(rotation))
    across, along = xs * c + ys * s, -xs * s + ys * c
    if settings.get('fill'):
        # 分成几块的(几张名片、扑克牌)只铺满最大的那块。正对着的矩形、圆形和星形这类异形取上下、左右两端围成的框,
        # 设计稿铺满不留白,异形外面的部分本来就被可印区裁掉;透视下的梯形、斜放的平行四边形(凸包是四边形、
        # 不到外接框的 SKEW_FILL)取里面放得下的最大矩形,不往外包、设计稿四边不被裁掉。
        # 缺口小的(手指、夹子挡住边角)按凸包算;两张卡叠成 L 形这类缺口大的不补,框只落在一张上
        # (写了 freeform 的贴纸缺口大是花形、树形这类异形,照样取外接框)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8))
        largest = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        keep = largest[ys, xs] > 0
        across, along = across[keep], along[keep]
        contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        solid = cv2.contourArea(contour) >= HULL_SOLID * cv2.contourArea(hull)
        bound = (across.max() - across.min()) * (along.max() - along.min())
        quad = len(cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)) == 4
        if (not quad or cv2.contourArea(hull) >= SKEW_FILL * bound) if solid else settings.get('freeform', False):
            middle, level = (across.min() + across.max()) / 2, (along.min() + along.max()) / 2
            width, height = across.max() - across.min(), along.max() - along.min()
            x, y = middle * c - level * s, middle * s + level * c
        else:
            px, py = ((hull if solid else contour)[:, 0, i].astype(np.float64) for i in (0, 1))

            def fit(angle, size):
                c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                middle, level, width, height = inscribed(px * c + py * s, -px * s + py * c, size)
                return width * height, middle * c - level * s, middle * s + level * c, width, height

            # 方向取放得下最大矩形的那个:透视下窄长的四边形两条短边斜得厉害,四条边平均的方向会偏;
            # 连成一块的看不出整体方向,框跟着最大的那张走。先粗找方向再细算
            if turn is None:
                rotation = max(range(-45, 45), key=lambda angle: fit(angle, INSCRIBED_SIZE // 4)[0])
                rotation = 0.0 if abs(rotation) < (0 if plane else TILT_MIN) else float(rotation)
            _, x, y, width, height = fit(rotation, INSCRIBED_SIZE)
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
        # 先放在正中(左胸先放在胸口);top 再把框往上挪到那里,框上沿碰到领口(框横向范围里可印区上沿的低处,见 NECK)就停。
        # 领口在框挪到的那一高度、按它横向范围里的各列量,挪了再量,至多 NECK_ROUNDS 次:斜放的袜子正中那一高度在脚跟,
        # 那几列往上先碰到的是袜筒斜着的侧边,按它量框会停在脚跟
        middle, x, y, reach = place(along.min() + (settings['top'] if left else 0.5) * span)
        wide = reach if left else across.max() - across.min()
        width = height = min(settings.get('width', LOGO_WIDTH) * wide, LOGO_REACH * float(distance.max()))
        if 'top' in settings:
            target = level = along.min() + settings['top'] * span
            for _ in range(NECK_ROUNDS):
                middle, x, y, _ = place(level)
                under = np.abs(across - middle) <= width / 2
                columns = np.round(across[under] - across[under].min()).astype(int)
                tops = np.full(columns.max() + 1, np.inf)
                np.minimum.at(tops, columns, along[under])
                neck = float(np.percentile(tops[np.isfinite(tops)], NECK))
                moved = min(max(target, neck + width / 2), along.min() + 0.5 * span)
                if abs(moved - level) < 1:
                    break
                level = moved
            middle, x, y, _ = place(level)
    return {'cx': round(float(x) / W, 4), 'cy': round(float(y) / H, 4), 'width': round(float(width) / W, 4),
            'height': round(float(height) / H, 4), 'rotation': rotation}


def split_labels(cover, item, text, n, work):
    """SAM 3 按 text 分出的各只,像素值是第几只;第 n 个说法的存成 work/logo-split-<商品 key>[-n].png,图和说法对得上就不再调用。
    调用失败时返回 None"""
    name = item['key'] if n == 0 else f"{item['key']}-{n}"
    path, labels_path = work / 'logo-split.json', work / f'logo-split-{name}.png'
    done = json.loads(path.read_text()) if path.exists() else {}
    key = hashlib.sha1(text.encode() + cover.tobytes()).hexdigest()
    if done.get(name) == key and labels_path.exists():
        return np.asarray(Image.open(labels_path))
    from segment import Sam3Replicate  # 只有要分的时候才用到网络和 token
    import config
    try:
        masks = Sam3Replicate(**config.replicate()).instances(cover, text)
    except (RuntimeError, OSError, ValueError) as error:
        print(f'{item["key"]}:SAM 3 按 {text} 没分到({error})')
        return None
    labels = np.zeros(cover.shape[:2], np.uint8)
    for index, mask in enumerate(masks[:255], start=1):
        labels[cv2.resize(mask.astype(np.uint8), cover.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0] = index
    Image.fromarray(labels).save(labels_path)
    path.write_text(json.dumps({**done, name: key}, indent=2))
    return labels


def single(cover, printable, item, texts, work):
    """split:SAM 3 按说法分出单只,返回可印区里最大的那只的蒙版。texts 是一个说法或几个说法的列表,按顺序试,
    分出的盖住可印区 SPLIT_SHARE 以上的有两只就用这个说法;都不到两只、调用失败时返回 None"""
    for n, text in enumerate([texts] if isinstance(texts, str) else texts):
        labels = split_labels(cover, item, text, n, work)
        if labels is None:
            continue
        if labels.shape != printable.shape:
            labels = cv2.resize(labels, printable.shape[::-1], interpolation=cv2.INTER_NEAREST)
        shares = np.bincount(labels[printable], minlength=int(labels.max()) + 1)[1:] / max(printable.sum(), 1)
        count = int((shares >= SPLIT_SHARE).sum())
        print(f"{item['key']}:SAM 3 按 {text} 分出 {count} 只{',放在最大的那只上' if count >= 2 else ''}")
        if count >= 2:
            break
    else:
        print(f"{item['key']}:没分出两只,按整块放")
        return None
    # 图上已经印了图案的(收件箱里的封面),SAM 3 分出的单只常把图案那块挖掉,补上里面的洞,框不至于落进洞里退到别处
    chosen = (labels == 1 + np.argmax(shares)).astype(np.uint8)
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(chosen, contours, -1, 1, cv2.FILLED) > 0


def settle(spec, cover, prints, work):
    """各件商品的 logo 设置。写了 closure 的,把封面和描出可印区的封面交给 Gemini,照片里是正面、前襟有拉链或扣子就合上 closure 里的设置。
    回答看不懂、调用失败时不合(放胸口),不记下来,下次再问。写了 split 的,within 是框要放的那只(见 single)"""
    path = work / 'logo-closure.json'
    answers = json.loads(path.read_text()) if path.exists() else {}
    h, w = cover.shape[:2]
    k = min(1, CLOSURE_SIZE / max(h, w))
    small = cv2.resize(cover, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    result, client = [], None
    for printable, item in zip(prints, spec['products']):
        settings = dict(item.get('logo', {}))
        if 'split' in settings:
            settings['within'] = single(cover, printable, item, settings.pop('split'), work)
        closure = settings.pop('closure', None)
        if closure is not None:
            outlined = small.copy()
            contours, _ = cv2.findContours(cv2.resize(printable.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(outlined, contours, -1, (255, 220, 0), 4)
            prompt = CLOSURE_PROMPT.format(what=item['name'].lower())
            key = hashlib.sha1(prompt.encode() + small.tobytes() + outlined.tobytes()).hexdigest()
            if key not in answers:
                from remote import Replicate  # 只有要问 Gemini 时才用到网络和 token
                from segment import GEMINI
                import config
                try:
                    client = client or Replicate(**config.replicate())
                    output = client.predict(GEMINI, {'prompt': prompt, 'images': [client.upload(small), client.upload(outlined)],
                                                     'temperature': 0, 'max_output_tokens': 2000})
                    answer = ''.join(output) if isinstance(output, list) else str(output)
                    parsed = json.loads(re.search(r'\{.*\}', answer, re.S).group())
                    answers[key] = {'front': parsed['front'] is True, 'closure': str(parsed['closure']), 'product': item['key']}
                    path.write_text(json.dumps(answers, indent=2))
                except (RuntimeError, OSError, AttributeError, ValueError, KeyError, TypeError) as error:
                    print(f'{item["key"]}:问 Gemini 前襟没问到({error}),放胸口')
            answer = answers.get(key)
            if answer and answer['front'] and answer['closure'] != 'none':
                settings |= closure
            if answer:
                print(f"{item['key']}:Gemini 说{'是正面' if answer['front'] else '不是正面'},前襟 {answer['closure']}"
                      f"{',放左胸' if settings.get('side') == 'left' else ''}")
        result.append(settings)
    return result


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
    cover = np.asarray(Image.open(out / 'cover.webp').convert('RGB'))
    settings = settle(spec, cover, [printable > 0.5 for printable in prints], config.folder(spec, 'work'))
    fits, turns = (template.get('pipeline', {}).get(key) or [] for key in ('surfaceFit', 'planeTurn'))
    logos = [box(printable > 0.5, {**s, 'plane': index < len(fits) and fits[index] == 'plane', 'turn': turns[index] if index < len(turns) else None})
             for index, (printable, s) in enumerate(zip(prints, settings))]
    for region, logo in zip(template['regions'], logos):
        region['logo'] = logo
    (out / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))
    checks = json.loads((out / 'checks.json').read_text())
    checks[CHECK] = centered(prints, spec['products'], logos)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2))
    for region in template['regions']:
        print(region['key'], region['logo'])
    print(('PASS ' if checks[CHECK] else 'FAIL ') + CHECK)
