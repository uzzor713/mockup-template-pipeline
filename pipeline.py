"""
按素材管线生成一个两件商品(T 恤 + 牛仔裤)的模板样本。

分割(第 2 步)的来源:
- T 恤:竞品 Mockey 模板 04674 已分割好的结果(底图透明度 = 商品区,mask-design = 可印区域);
- 牛仔裤:本地 SAM 2.1(segment.py 产出 work/jeans-mask.png),可印区域按规则从商品区里取。
其余步骤都是固定代码。用法:.venv/bin/python -I pipeline.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).parent
SRC = ROOT / 'sources'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

BETA = 0.2  # 深色的保底亮度
PERCENTILE = 90  # 归一化参考:商品区亮度的第 90 百分位当作白
SIGMA_RATIO = 0.006  # 起伏图模糊半径 / 宽度
DISP_SCALE = 4  # 起伏图缩小倍数:按块求平均,抹掉 8 位台阶和纹理噪点;使用时双线性放大
SLOPE_BLUR = 2.0  # 求坡度前在缩小后的起伏上再模糊(格):坡度再求一次导就是形变,对噪点更敏感
SLOPE_GAIN = 4.7e-4  # 位移 / 宽度 = SLOPE_GAIN × 起伏对(坐标 / 宽度)的坡度;样本 T 恤在 Warping 100% 时约 2% 的面积顶到上限
STRETCH_RANGE = (0.5, 2.5)  # Warping 100% 时各方向每像素前进量的范围:拉伸至多 2 倍、压缩至多 2.5 倍
CAP_MARGIN = (0.05, 0.2)  # 减弱时比上面的范围再收一点,留给 8 位量化和双线性放大
SMOOTH = 2.0  # 二值轮廓的平滑半径(px),台阶越长要越大
BAND_RATIO = 0.004  # 边缘细化带半宽 / 宽度,要盖住分割模型的边界误差


def luma(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def soft(binary, smooth=SMOOTH):
    """二值区域 → 覆盖度:模糊抹平台阶,再拉回约 1px 宽的过渡"""
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), smooth)
    return np.clip(0.5 + (blurred - 0.5) * smooth * np.sqrt(2 * np.pi), 0, 1)


def local_mean(image, weight, sigma):
    """weight 区域内颜色的局部均值,延伸到区外;离区太远的取最近的区内像素"""
    total = cv2.GaussianBlur(weight, (0, 0), sigma)
    mean = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma) / np.maximum(total, 1e-6)[..., None]
    far = total < 1e-3
    if far.any():
        _, (iy, ix) = ndimage.distance_transform_edt(weight < 0.5, return_indices=True)
        mean[far] = image[iy[far], ix[far]]
    return mean


def refine(binary, photo, band, inward):
    """边缘细化:分割边界向外 band px、向内 inward px 内,把像素颜色投影到「区内颜色 → 区外颜色」的连线上求覆盖度。
    向内收得少:离边界远的商品高光会被当成背景"""
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    fg = cv2.erode(binary.astype(np.uint8), disk(inward)).astype(np.float32)
    bg = 1 - cv2.dilate(binary.astype(np.uint8), disk(band)).astype(np.float32)
    F, B = local_mean(photo, fg, band), local_mean(photo, bg, band)
    d = F - B
    norm2 = (d * d).sum(-1)
    alpha = np.clip(((photo - B) * d).sum(-1) / np.maximum(norm2, 1e-6), 0, 1)
    trust = np.clip(np.sqrt(norm2) / 0.15, 0, 1)  # 两侧颜色太近时投影不可信,退回分割结果
    alpha = cv2.medianBlur((trust * alpha + (1 - trust) * soft(binary)).astype(np.float32), 3)
    return np.where(fg > 0, 1, np.where(bg > 0, 0, alpha)).astype(np.float32)


def quantize(coverage):
    """按落盘的 8 位精度取整,后面的计算和读回的文件一致"""
    return np.round(coverage * 255) / 255


def largest(binary):
    labels, count = ndimage.label(binary)
    if not count:
        return binary
    sizes = ndimage.sum(binary, labels, range(1, count + 1))
    return labels == 1 + int(np.argmax(sizes))


def stretch_range(a, b, c, d):
    """雅可比 [[a, b], [c, d]] 的最小、最大奇异值 = 各方向每像素前进量的范围;翻折时最小值记为负"""
    det = a * d - b * c
    q = a * a + b * b + c * c + d * d
    disc = np.sqrt(np.maximum(q * q - 4 * det * det, 0))
    return np.sqrt(np.maximum(q - disc, 0) / 2) * np.sign(det), np.sqrt((q + disc) / 2)


def jacobian(ox, oy, cell):
    """位移场(每格 cell 个模板像素)→ 模板像素下的雅可比"""
    return (1 + np.gradient(ox, axis=1) / cell, np.gradient(ox, axis=0) / cell,
            np.gradient(oy, axis=1) / cell, 1 + np.gradient(oy, axis=0) / cell)


def cap(ox, oy, inside, low, high, rounds=10):
    """形变超限的地方局部减弱位移:每轮算出每格能保留的最大比例,往外扩两格再平滑后乘上去,直到不超限"""
    for _ in range(rounds):
        a, b, c, d = jacobian(ox, oy, DISP_SCALE)
        keep = np.zeros_like(ox)
        for f in np.linspace(0, 1, 41):
            smin, smax = stretch_range(1 + f * (a - 1), f * b, f * c, 1 + f * (d - 1))
            keep = np.where((smin >= low) & (smax <= high), f, keep)
        keep = np.where(inside, keep, 1).astype(np.float32)
        if keep.min() == 1:
            break
        keep = cv2.GaussianBlur(cv2.erode(keep, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        ox, oy = ox * keep, oy * keep
    return ox, oy


def save_mask(coverage, name):
    """白色 + alpha = 覆盖度"""
    alpha = np.round(coverage * 255).astype(np.uint8)
    rgba = np.dstack([np.full_like(alpha, 255)] * 3 + [alpha])
    Image.fromarray(rgba, 'RGBA').save(OUT / name, optimize=True)


# ---- 1. 源图:竞品底图叠在白色上 = 白色 T 恤的照片
base_src = np.asarray(Image.open(SRC / 'tee-base.webp').convert('RGBA')).astype(np.float32) / 255
H, W = base_src.shape[:2]
photo = base_src[..., :3] * base_src[..., 3:] + (1 - base_src[..., 3:])
Image.fromarray(np.round(photo * 255).astype(np.uint8)).save(OUT / 'cover.webp', quality=85)

# ---- 2. 分割
# 竞品的商品区是二值边缘,平滑轮廓后重出软边
tee_bin = base_src[..., 3] < 1
tee = quantize(soft(tee_bin))
print_src = np.asarray(Image.open(SRC / 'tee-print.webp').convert('RGBA')).astype(np.float32) / 255
# 有损 webp 的白色只到 246–255,拉回满值
print_tee = np.minimum(np.clip((luma(print_src[..., :3]) - 0.03) / 0.93, 0, 1) * print_src[..., 3], tee)

jeans_sam = np.asarray(Image.open(ROOT / 'work/jeans-mask.png').convert('L')) > 127
jeans_bin = ndimage.binary_fill_holes(largest(jeans_sam)) & ~tee_bin
band_px = max(1, round(BAND_RATIO * W))
jeans = refine(jeans_bin, photo, band_px, max(1, band_px // 3))
# 牛仔裤在 T 恤下面接着铺,交界处两者相加为 1;重叠处归上层的 T 恤
_, (iy, ix) = ndimage.distance_transform_edt(tee_bin, return_indices=True)
jeans = quantize(np.minimum(jeans[iy, ix], 1 - tee))
jeans_bin = jeans > 0.5

# 牛仔裤可印区域(规则):向内收 1% 宽度,取裤腰以下到大腿中段
ys = np.nonzero(jeans_bin)[0]
top, height = ys.min(), ys.max() - ys.min()
band = np.zeros_like(jeans_bin)
band[int(top + 0.08 * height):int(top + 0.45 * height)] = True
inner = cv2.erode(jeans_bin.astype(np.uint8), np.ones((33, 33), np.uint8)).astype(bool)
print_jeans = np.minimum(soft(inner & band), jeans)

products = [tee, jeans]
prints = [quantize(print_tee), quantize(print_jeans)]
names = ['T-shirt', 'Jeans']
keys = ['top', 'bottom']
coverage = np.minimum(sum(products), 1)

# ---- 3. 底图
# 边缘像素在照片里混着商品原色:亮度取商品实心部分的颜色,区外那份换成背景色
L = luma(photo)
L_norm = np.zeros_like(L)
for product in products:
    inside = product > 0.5
    ref = np.percentile(L[inside], PERCENTILE)
    solid = product == 1
    own = np.where(solid, L, luma(local_mean(photo, solid.astype(np.float32), 2)))
    L_norm += product * np.minimum(1, own / ref)
L_norm = np.where(coverage > 0, L_norm / np.maximum(coverage, 1e-6), 0)
a = 1 - (1 - BETA) * L_norm
g = BETA * L_norm / a
alpha = (1 - coverage) + coverage * a
# T 恤台阶上被平滑到区外的像素在原图里仍是白色,同样换成背景色
dirty = (coverage > 0) | tee_bin
background = np.where(dirty[..., None], local_mean(photo, (~dirty).astype(np.float32), 2), photo)
rgb = ((1 - coverage)[..., None] * background + (coverage * a * g)[..., None]) / alpha[..., None]
base = np.dstack([rgb, alpha])
Image.fromarray(np.round(base * 255).astype(np.uint8), 'RGBA').save(OUT / 'base.webp', quality=85, alpha_quality=100)

# ---- 4. 起伏图:每件商品一张流向图,R / G = Warping 100% 时 x / y 方向的位移,1/4 尺寸
# 位移沿起伏的坡度,褶皱横着就上下压缩、竖着就左右压缩;压缩或拉伸超限的地方局部减弱,深褶处不翻折。
# 交界两侧的起伏不连续,每件商品单独一张、各自向外延伸,缩小和插值时才不混在一起
sigma = SIGMA_RATIO * W
small_size = (W // DISP_SCALE, H // DISP_SCALE)
gain = SLOPE_GAIN * W * W / DISP_SCALE  # 缩小后每格的坡度 → 模板像素位移
flows, guards = [], []
for product in products:
    inside = product > 0.5
    weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), sigma)
    blurred = cv2.GaussianBlur(L_norm * inside, (0, 0), sigma) / np.maximum(weight, 1e-6)
    # 区外取最近的区内值再缩小,边上的坡度不会因为区外突变而跳
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    height = cv2.resize(blurred[iy, ix], small_size, interpolation=cv2.INTER_AREA)
    gy, gx = np.gradient(cv2.GaussianBlur(height, (0, 0), SLOPE_BLUR))
    # 沾到商品的格子再往外两格都守住范围:双线性放大时商品边上的像素会插到外面一格
    touched = cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0
    guarded = cv2.dilate(touched.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    low, high = STRETCH_RANGE[0] + CAP_MARGIN[0], STRETCH_RANGE[1] - CAP_MARGIN[1]
    ox, oy = cap(gain * gx, gain * gy, guarded, low, high)
    # 离商品 3σ 以外不动
    far = ndimage.distance_transform_edt(~inside) > 3 * sigma
    far_cells = cv2.resize(far.astype(np.float32), small_size, interpolation=cv2.INTER_AREA) == 1
    flows.append(np.dstack([np.where(far_cells, 0, o) for o in (ox, oy)]).astype(np.float32))
    guards.append(guarded)
# 通道值 0 / 255 对应 ∓range 像素,按守住范围的格子里的最大位移定,8 位精度用满;更外面的格子用不到,截掉
range_px = round(float(np.ceil(max(np.abs(f[g]).max() for f, g in zip(flows, guards)) * 10)) / 10, 1)
for index, flow in enumerate(flows, start=1):
    rgb = np.dstack([0.5 + np.clip(flow, -range_px, range_px) / (2 * range_px), np.full(flow.shape[:2], 0.5, np.float32)])
    Image.fromarray(np.round(rgb * 255).astype(np.uint8), 'RGB').save(OUT / f'displacement-{index}.png', optimize=True)

for index, (product, printable) in enumerate(zip(products, prints), start=1):
    save_mask(product, f'product-{index}.png')
    save_mask(printable, f'print-{index}.png')

# ---- 5. 配置:默认 Logo 框取可印区域里离边最远的点
regions = []
for index, (printable, name, key) in enumerate(zip(prints, names, keys), start=1):
    inside = printable > 0.5
    distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
    xs = np.nonzero(inside)[1]
    size = min(0.4 * (xs.max() - xs.min()), 1.4 * distance.max())
    regions.append({
        'key': key,
        'name': name,
        'product': f'product-{index}.png',
        'print': f'print-{index}.png',
        'displacement': f'displacement-{index}.png',
        'recolorable': True,
        'logo': {'cx': round(float(cx) / W, 4), 'cy': round(float(cy) / H, 4), 'width': round(float(size) / W, 4),
                 'height': round(float(size) / H, 4), 'rotation': 0},
    })
template = {
    'id': 'sample-tee-jeans',
    'version': 1,
    'size': {'width': W, 'height': H},
    'category': 't-shirt',
    'files': {'cover': 'cover.webp', 'base': 'base.webp'},
    'displacement': {'rangePx': range_px},
    'regions': regions,
    'pipeline': {'luminancePercentile': PERCENTILE, 'beta': BETA, 'blurSigma': round(float(sigma), 1),
                 'displacementScale': 1 / DISP_SCALE, 'slopeBlur': SLOPE_BLUR, 'slopeGain': SLOPE_GAIN,
                 'stretchRange': list(STRETCH_RANGE)},
}
(OUT / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))

# ---- 6. 自动校验(读回落盘的文件)
def read(name):
    return np.asarray(Image.open(OUT / name)).astype(np.float32) / 255

base_out = read('base.webp')
d = [read(f'displacement-{i}.png') for i in (1, 2)]
p = [read(f'product-{i}.png')[..., 3] for i in (1, 2)]
q = [read(f'print-{i}.png')[..., 3] for i in (1, 2)]
union = (p[0] > 0) | (p[1] > 0)


def neutral_far(flow, product):
    """离这件商品 3σ 以外的格子 R、G 都是 0.5"""
    far = ndimage.distance_transform_edt(~(product > 0.5)) > 3 * sigma + DISP_SCALE
    cells = cv2.resize(far.astype(np.float32), small_size, interpolation=cv2.INTER_AREA) == 1
    return bool((np.abs(flow[..., :2][cells] - 0.5) <= 1 / 255).all())


def within_stretch(flow, product):
    """按编辑页的方式双线性放大回模板尺寸,Warping 100% 时商品区内各方向每像素前进量都在范围内"""
    offset = cv2.resize((flow[..., :2] - 0.5) * 2 * range_px, (W, H), interpolation=cv2.INTER_LINEAR)
    smin, smax = stretch_range(*jacobian(offset[..., 0], offset[..., 1], 1))
    inside = product > 0.5
    return bool(smin[inside].min() >= STRETCH_RANGE[0] and smax[inside].max() <= STRETCH_RANGE[1])


checks = {
    'sizes equal': all(x.shape[:2] == (H, W) for x in [base_out, *p, *q]),
    'displacement 1/4 size, B unused': all(x.shape == (H // DISP_SCALE, W // DISP_SCALE, 3)
                                           and bool((x[..., 2] == 128 / 255).all()) for x in d),
    'aspect 4:5': W * 5 == H * 4,
    'products overlap < 0.1%': float(((p[0] > 0.5) & (p[1] > 0.5)).sum() / max(1, (p[1] > 0.5).sum())) < 0.001,
    'print outside product < 0.1%': all(float(((qq > 0.5) & (pp < 0.5)).sum() / max(1, (qq > 0.5).sum())) < 0.001 for pp, qq in zip(p, q)),
    'base opaque outside products': bool((base_out[..., 3][~union] == 1).all()),
    'base matches formula (≤ 2/255)': bool(np.abs(base_out[..., 3] - alpha).max() <= 2 / 255 + 1e-6),
    'displacement neutral far outside': all(neutral_far(dd, pp) for dd, pp in zip(d, p)),
    f'stretch {STRETCH_RANGE[0]}–{STRETCH_RANGE[1]} at Warping 100%': all(within_stretch(dd, pp) for dd, pp in zip(d, p)),
    'prints not empty': all((qq > 0.5).any() for qq in q),
    'print largest component ≥ 80%': all(largest(qq > 0.5).sum() / (qq > 0.5).sum() >= 0.8 for qq in q),
    'logo center inside print': all(q[i][int(r['logo']['cy'] * H), int(r['logo']['cx'] * W)] > 0.5 for i, r in enumerate(regions)),
}
(OUT / 'checks.json').write_text(json.dumps(checks, indent=2))
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
for f in sorted(OUT.iterdir()):
    print(f'{f.stat().st_size / 1024:8.1f} KB  {f.name}')
