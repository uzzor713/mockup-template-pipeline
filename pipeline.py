"""
按素材管线生成一个两件商品(T 恤 + 牛仔裤)的模板样本。

源图是竞品 Mockey 模板 04674 的底图叠在白色上,还原成白色 T 恤的照片。模型的结果由前面的脚本存在 work/:
- 分割(第 2 步):本地 SAM 2.1 分出 T 恤、袖子、遮挡物和牛仔裤,ViTMatte 求边缘透明度(run_segment.py 产出 work/tee-alpha.png、
  tee-parts.png(袖子)、jeans-alpha.png、occluders.png);可印区域按规则从商品区里取。
- 底图(第 3 步):Marigold 本征分解(intrinsic.py 产出 work/albedo.png、shading.png、residual.png)。
- 流向图(第 4 步):大曲面和透视取 MoGe-2 的 3D 坐标(geometry.py 产出 work/points.npy、geometry.json),
  褶皱取 Marigold 法线(normals.py 产出 work/normals.png)。
其余步骤都是固定代码。用法:.venv/bin/python -I pipeline.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).parent
SRC = ROOT / 'sources'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

WHITE_ALBEDO = 0.87  # 白布料在本征分解里的固有色亮度(样本白 T 恤实测 0.86–0.88):选白色时还原照片的亮度
TEXTURE_RATIO = 1 / 256  # 纹理半径 / 长边:照片里比它细的起伏算纹理,粗的算明暗
TEXTURE_CLIP = (-0.65, 0.44)  # 纹理(对数)的截断,个别像素不过亮过暗
SIGMA_RATIO = 0.006  # 起伏图模糊半径 / 宽度
DISP_SCALE = 4  # 起伏图缩小倍数:按块求平均,抹掉 8 位台阶和纹理噪点;使用时双线性放大
SLOPE_BLUR = 2.0  # 求坡度前在缩小后的起伏上再模糊(格):坡度再求一次导就是形变,对噪点更敏感
STRETCH_RANGE = (0.5, 2.5)  # Warping 0–100% 时各方向每像素前进量的范围:拉伸至多 2 倍、压缩至多 2.5 倍
CAP_MARGIN = (0.1, 0.2)  # 减弱时比上面的范围再收一点,留给 8 位量化和双线性放大
CAP_AREA = 0.2  # 每件商品的坡度系数按它定:Warping 100% 时可印区域里这么多面积刚好顶到上限
SLOPE_GAIN_MAX = 1.4e-3  # 坡度系数(位移 / 宽度 ÷ 起伏对(坐标 / 宽度)的坡度)的上限:几乎没褶的商品不把噪点放大成假褶皱
SURFACE_RATIO = 0.02  # 大曲面:度量的平滑半径 / 宽度
SURFACE_RANGE = (0.7, 2.0)  # 大曲面每像素前进量的范围:近处至多放大到 1 / 0.7,侧面至多压缩 2 倍
SURFACE_INSET = 4  # 商品边上几个像素的 3D 坐标混着背景,往里收这么多再算
SMOOTH = 2.0  # 二值轮廓的平滑半径(px),台阶越长要越大
PRINT_HEM = 0.01  # 上衣可印区域在领口、下摆往里收的距离 / 宽度:罗纹和折边上不印


def luma(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


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


def read_layer(name):
    """intrinsic.py 存的 16 位线性 RGB"""
    return cv2.imread(str(ROOT / 'work' / name), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535


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


def cap(ox, oy, inside, low, high, base, rounds=10):
    """形变超限的地方局部减弱位移 ox、oy(base 不动,叠在它上面算):每轮算出每格能保留的最大比例
    (0 到这个比例之间都不超限),往外扩两格再平滑后乘上去,直到不超限"""
    a0, b0, c0, d0 = jacobian(*base, DISP_SCALE)
    for _ in range(rounds):
        a, b, c, d = jacobian(ox, oy, DISP_SCALE)
        keep = np.zeros_like(ox)
        ok = np.ones(ox.shape, bool)
        for f in np.linspace(0, 1, 41):
            smin, smax = stretch_range(a0 + f * (a - 1), b0 + f * b, c0 + f * c, d0 + f * (d - 1))
            ok &= (smin >= low) & (smax <= high)
            keep = np.where(ok, f, keep)
        keep = np.where(inside, keep, 1).astype(np.float32)
        if keep.min() == 1:
            break
        keep = cv2.GaussianBlur(cv2.erode(keep, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        ox, oy = ox * keep, oy * keep
    return ox, oy


def max_gain(gx, gy, low, high):
    """位移 = g × 坡度时,每格各方向前进量还在 [low, high] 内的最大 g。
    雅可比 = I + g·M,M 近似对称(起伏的二阶导),前进量就是 1 + g × M 的特征值"""
    a, b, c, d = jacobian(gx, gy, DISP_SCALE)
    half_gap = np.sqrt(((a - d) / 2) ** 2 + ((b + c) / 2) ** 2)
    stretch, squeeze = (a + d - 2) / 2 + half_gap, (a + d - 2) / 2 - half_gap
    with np.errstate(divide='ignore'):
        return np.minimum(np.where(stretch > 0, (high - 1) / stretch, np.inf), np.where(squeeze < 0, (1 - low) / -squeeze, np.inf))


def surface_flow(points, focal, product, small_size, sigma):
    """大曲面:设计稿贴在曲面上时,屏幕上每像素对应曲面上多长(以商品中位深度处正对镜头时为 1),
    最小二乘积分成位移(缩小后的格子,模板像素)。侧面压缩、近处略放大,透视也在里面。
    先求每像素的度量再在商品区内平滑:直接平滑 3D 坐标会把轮廓附近的坡度压平"""
    solid = cv2.erode((product > 0.5).astype(np.uint8), np.ones((2 * SURFACE_INSET + 1,) * 2, np.uint8)) > 0
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    p = points[iy, ix]
    # 3D 坐标对模板像素的导数,除以正对镜头时每像素的长度
    unit = np.median(p[..., 2][solid]) / focal
    dx, dy = (np.gradient(p, axis=axis) / unit for axis in (1, 0))
    # 轮廓上混着背景深度的像素导数会爆,各分量先截到范围上限的平方
    top = SURFACE_RANGE[1] ** 2
    weight = solid.astype(np.float32)
    total = cv2.GaussianBlur(weight, (0, 0), sigma)
    xx, xy, yy = (cv2.resize(cv2.GaussianBlur(np.clip(c, -top, top) * weight, (0, 0), sigma) / np.maximum(total, 1e-6),
                             small_size, interpolation=cv2.INTER_AREA)
                  for c in ((dx * dx).sum(-1), (dx * dy).sum(-1), (dy * dy).sum(-1)))
    g = np.stack([np.stack([xx, xy], -1), np.stack([xy, yy], -1)], -2)
    values, vectors = np.linalg.eigh(g)
    scale = np.clip(np.sqrt(np.maximum(values, 0)), *SURFACE_RANGE)
    target = np.einsum('...ik,...k,...jk->...ij', vectors, scale, vectors) - np.eye(2)
    # 相邻格子的位移差 = 目标雅可比 − I(每格 DISP_SCALE 个像素),最小二乘求位移,商品区内均值为 0。
    # 商品外延伸一圈、目标为不变形,位移平滑地延伸出去,双线性放大时边上不跳
    inside = cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0.5
    target[~inside] = 0
    region = cv2.dilate((cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    index = -np.ones(inside.shape, int)
    index[region] = np.arange(region.sum())
    rows, cols, values_, rhs = [], [], [], [[], []]
    count = 0
    for axis, (dy_, dx_) in enumerate(((0, 1), (1, 0))):
        h, w = inside.shape[0] - dy_, inside.shape[1] - dx_
        pair = region[:h, :w] & region[dy_:, dx_:]
        i0, i1 = index[:h, :w][pair], index[dy_:, dx_:][pair]
        r = np.arange(count, count + len(i0))
        count += len(i0)
        rows += [r, r]
        cols += [i1, i0]
        values_ += [np.ones(len(r)), -np.ones(len(r))]
        for component in range(2):
            t = target[..., component, axis] * DISP_SCALE
            rhs[component].append((t[:h, :w][pair] + t[dy_:, dx_:][pair]) / 2)
    A = sparse.csr_matrix((np.concatenate(values_), (np.concatenate(rows), np.concatenate(cols))), shape=(count, int(region.sum())))
    normal = (A.T @ A + 1e-6 * sparse.eye(A.shape[1])).tocsc()
    flow = np.zeros(inside.shape + (2,), np.float32)
    for component in range(2):
        solution = spsolve(normal, A.T @ np.concatenate(rhs[component]))
        flow[..., component][region] = solution - solution[index[inside]].mean()
    _, (iy, ix) = ndimage.distance_transform_edt(~region, return_indices=True)
    return flow[iy, ix]


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
# 竞品底图半透明的部分在源图里是还原出来的白色,第 3 步换背景色时要用
restored = base_src[..., 3] < 1

# ---- 2. 分割(run_segment.py 的结果)
read_alpha = lambda name: np.asarray(Image.open(ROOT / 'work' / name)).astype(np.float32) / 65535
read_mask = lambda name: np.asarray(Image.open(ROOT / 'work' / name)) > 127
# 这张源图里 T 恤是还原出来的,轮廓是台阶,抠图模型救不回来:按二值来源处理,平滑轮廓后重出软边
tee_bin = read_alpha('tee-alpha.png') > 0.5
tee = quantize(soft(tee_bin))
sleeves, occluders = read_mask('tee-parts.png'), read_mask('occluders.png')
# T 恤可印区域(规则):减去袖子,领口、下摆按竖直方向往里收;左右两侧和遮挡物边上不收,
# 设计稿一直铺到轮廓,被遮挡物挡住的部分由底图盖住
hem = round(PRINT_HEM * W)
torso = cv2.erode(((tee_bin | occluders) & ~sleeves).astype(np.uint8), np.ones((2 * hem + 1, 1), np.uint8)) > 0
print_tee = np.minimum(soft(torso), tee)

jeans = read_alpha('jeans-alpha.png')
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
# 编辑页在线性光下合成:结果 = 商品颜色 × (1 − alpha) + 底图 × alpha。
# 商品区里 1 − alpha = 明暗 × 纹理,底图 × alpha = 光泽 × 纹理(光泽是不随颜色变的反光,纹理也作用在它上面)。
# 照片 ≈ 固有色 × s·光照 + r·残差,光照和残差只到相差一个比例,每件商品在实心部分拟合 s、r,花纹和染料留在固有色里;
# 明暗取白布料的固有色 × s·光照,光泽取 r·残差,纹理取照片里比纹理半径更细的起伏
layers = {name: cv2.resize(read_layer(f'{name}.png'), (W, H), interpolation=cv2.INTER_LINEAR)
          for name in ('albedo', 'shading', 'residual')}
photo_linear = to_linear(photo)
I = luma(photo_linear)
radius = TEXTURE_RATIO * max(W, H)
shade, gloss = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
for product in products:
    inside = product > 0.5
    solid = product == 1
    terms = np.stack([(layers['albedo'] * layers['shading'])[solid].ravel(), layers['residual'][solid].ravel()], 1)
    (s, r), *_ = np.linalg.lstsq(terms, photo_linear[solid].ravel(), rcond=None)
    weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), radius)

    def blur(x):
        return cv2.GaussianBlur(x * inside, (0, 0), radius) / np.maximum(weight, 1e-6)

    texture = np.exp(np.clip(np.log(np.maximum(I, 1e-4)) - np.log(np.maximum(blur(I), 1e-4)), *TEXTURE_CLIP))
    diffuse = WHITE_ALBEDO * blur(luma(s * layers['shading'])) * texture
    sheen = luma(max(r, 0) * layers['residual']) * texture
    # 边缘像素在照片里混着商品原色:取商品实心部分的值往外延伸,区外那份换成背景色
    for total, layer in ((shade, diffuse), (gloss, sheen)):
        total += product * np.where(solid, layer, local_mean(np.repeat(layer[..., None], 3, -1), solid.astype(np.float32), 2)[..., 0])
shade, gloss = (np.where(coverage > 0, x / np.maximum(coverage, 1e-6), 0) for x in (shade, gloss))
gloss = np.clip(gloss, 0, 1)
# 白色的结果不超过 1:超出的部分从明暗里扣,光泽留着,深色上主要看它
shade = np.clip(shade, 0, 1 - gloss)
alpha = 1 - coverage * shade
# 源图里还原出来的白色落在商品区外的部分,同样换成背景色
dirty = (coverage > 0) | restored
background = np.where(dirty[..., None], local_mean(photo, (~dirty).astype(np.float32), 2), photo)
premultiplied = (1 - coverage)[..., None] * to_linear(background) + (coverage * gloss)[..., None]
rgb = to_srgb(premultiplied / np.maximum(alpha, 1e-6)[..., None])
base = np.dstack([rgb, alpha])
Image.fromarray(np.round(base * 255).astype(np.uint8), 'RGBA').save(OUT / 'base.webp', quality=85, alpha_quality=100)

# ---- 4. 流向图:每件商品两张,R / G = x / y 方向的位移,1/4 尺寸
# - 大曲面(固定):按 MoGe-2 的 3D 坐标,设计稿像贴在曲面上,侧面压缩、近处略放大;
# - 褶皱(Warping 100% 时):起伏取法线的 z 分量(表面正对镜头的程度),位移沿它的坡度,褶皱横着就上下压缩、
#   竖着就左右压缩,和照片的光线方向无关;系数按每件商品自己的褶皱定,Warping 0–100% 时两层合起来超限的地方
#   局部减弱褶皱这层,深褶处不翻折。
# 位移会沿曲面累积,渲染端减去设计稿中心处的位移,中心不动。
# 交界两侧的起伏不连续,每件商品单独一张、各自向外延伸,缩小和插值时才不混在一起
facing = cv2.resize(read_layer('normals.png'), (W, H), interpolation=cv2.INTER_LINEAR)[..., 2] * 2 - 1
points = np.load(ROOT / 'work/points.npy')
focal = json.loads((ROOT / 'work/geometry.json').read_text())['intrinsics'][0][0] * W
sigma = SIGMA_RATIO * W
small_size = (W // DISP_SCALE, H // DISP_SCALE)
to_px = W * W / DISP_SCALE  # 系数 × to_px:缩小后每格的坡度 → 模板像素位移
low, high = STRETCH_RANGE[0] + CAP_MARGIN[0], STRETCH_RANGE[1] - CAP_MARGIN[1]
surfaces = [surface_flow(points, focal, product, small_size, SURFACE_RATIO * W) for product in products]
# 大曲面的 8 位量化让相邻格的位移差至多多出一级,两层叠起来减弱时再扣掉这部分
slack = 2 * max(np.abs(surface).max() for surface in surfaces) / 255 / DISP_SCALE
folds, guards, gains = [], [], []
for index, (product, printable) in enumerate(zip(products, prints)):
    inside = product > 0.5
    weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), sigma)
    blurred = cv2.GaussianBlur(facing * inside, (0, 0), sigma) / np.maximum(weight, 1e-6)
    # 区外取最近的区内值再缩小,边上的坡度不会因为区外突变而跳
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    height = cv2.resize(blurred[iy, ix], small_size, interpolation=cv2.INTER_AREA)
    gy, gx = np.gradient(cv2.GaussianBlur(height, (0, 0), SLOPE_BLUR))
    # 沾到商品的格子再往外两格都守住范围:双线性放大时商品边上的像素会插到外面一格
    touched = cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0
    guarded = cv2.dilate(touched.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    cells = cv2.resize(printable if (printable > 0.5).any() else product, small_size, interpolation=cv2.INTER_AREA) > 0.5
    gain = min(SLOPE_GAIN_MAX, float(np.percentile(max_gain(gx, gy, low, high)[cells], CAP_AREA * 100)) / to_px)
    gains.append(gain)
    ox, oy = cap(gain * to_px * gx, gain * to_px * gy, guarded, low + slack, high - slack,
                 (surfaces[index][..., 0], surfaces[index][..., 1]))
    # 离商品 3σ 以外不动
    far = ndimage.distance_transform_edt(~inside) > 3 * sigma
    far_cells = cv2.resize(far.astype(np.float32), small_size, interpolation=cv2.INTER_AREA) == 1
    surfaces[index] = np.where(far_cells[..., None], 0, surfaces[index]).astype(np.float32)
    folds.append(np.dstack([np.where(far_cells, 0, c) for c in (ox, oy)]).astype(np.float32))
    guards.append(guarded)


def save_flows(flows, prefix):
    """通道值 0 / 255 对应 ∓range 像素,按守住范围的格子里的最大位移定,8 位精度用满;更外面的格子用不到,截掉"""
    range_px = round(float(np.ceil(max(np.abs(f[g]).max() for f, g in zip(flows, guards)) * 10)) / 10, 1)
    for index, flow in enumerate(flows, start=1):
        rgb = np.dstack([0.5 + np.clip(flow, -range_px, range_px) / (2 * range_px), np.full(flow.shape[:2], 0.5, np.float32)])
        Image.fromarray(np.round(rgb * 255).astype(np.uint8), 'RGB').save(OUT / f'{prefix}-{index}.png', optimize=True)
    return range_px


surface_range = save_flows(surfaces, 'surface')
folds_range = save_flows(folds, 'folds')

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
        'surface': f'surface-{index}.png',
        'folds': f'folds-{index}.png',
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
    'blend': 'linear',
    'displacement': {'surfaceRangePx': surface_range, 'foldsRangePx': folds_range},
    'regions': regions,
    'pipeline': {'whiteAlbedo': WHITE_ALBEDO, 'textureRadius': round(float(radius), 1), 'textureClip': list(TEXTURE_CLIP),
                 'blurSigma': round(float(sigma), 1),
                 'displacementScale': 1 / DISP_SCALE, 'slopeBlur': SLOPE_BLUR, 'stretchRange': list(STRETCH_RANGE),
                 'capArea': CAP_AREA, 'slopeGainMax': SLOPE_GAIN_MAX, 'slopeGain': [float(f'{g:.3g}') for g in gains],
                 'surfaceSigma': round(SURFACE_RATIO * W, 1), 'surfaceRange': list(SURFACE_RANGE)},
}
(OUT / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))

# ---- 6. 自动校验(读回落盘的文件)
def read(name):
    return np.asarray(Image.open(OUT / name)).astype(np.float32) / 255

base_out = read('base.webp')
s = [read(f'surface-{i}.png') for i in (1, 2)]
f = [read(f'folds-{i}.png') for i in (1, 2)]
p = [read(f'product-{i}.png')[..., 3] for i in (1, 2)]
q = [read(f'print-{i}.png')[..., 3] for i in (1, 2)]
union = (p[0] > 0) | (p[1] > 0)


def neutral_far(flow, product):
    """离这件商品 3σ 以外的格子 R、G 都是 0.5"""
    far = ndimage.distance_transform_edt(~(product > 0.5)) > 3 * sigma + DISP_SCALE
    cells = cv2.resize(far.astype(np.float32), small_size, interpolation=cv2.INTER_AREA) == 1
    return bool((np.abs(flow[..., :2][cells] - 0.5) <= 1 / 255).all())


def within_stretch(surface, fold, product):
    """按编辑页的方式双线性放大回模板尺寸,Warping 0–100% 时商品区内各方向每像素前进量都在范围内"""
    surface, fold = (cv2.resize((x[..., :2] - 0.5) * 2 * r, (W, H), interpolation=cv2.INTER_LINEAR)
                     for x, r in ((surface, surface_range), (fold, folds_range)))
    inside = product > 0.5
    for warp in np.linspace(0, 1, 5):
        offset = surface + warp * fold
        smin, smax = stretch_range(*jacobian(offset[..., 0], offset[..., 1], 1))
        if smin[inside].min() < STRETCH_RANGE[0] or smax[inside].max() > STRETCH_RANGE[1]:
            return False
    return True


checks = {
    'sizes equal': all(x.shape[:2] == (H, W) for x in [base_out, *p, *q]),
    'flow maps 1/4 size, B unused': all(x.shape == (H // DISP_SCALE, W // DISP_SCALE, 3)
                                        and bool((x[..., 2] == 128 / 255).all()) for x in s + f),
    'aspect 4:5': W * 5 == H * 4,
    'products overlap < 0.1%': float(((p[0] > 0.5) & (p[1] > 0.5)).sum() / max(1, (p[1] > 0.5).sum())) < 0.001,
    'print outside product < 0.1%': all(float(((qq > 0.5) & (pp < 0.5)).sum() / max(1, (qq > 0.5).sum())) < 0.001 for pp, qq in zip(p, q)),
    'base opaque outside products': bool((base_out[..., 3][~union] == 1).all()),
    'base matches formula (≤ 2/255)': bool(np.abs(base_out[..., 3] - alpha).max() <= 2 / 255 + 1e-6),
    'flow maps neutral far outside': all(neutral_far(x, pp) for x, pp in zip(s + f, p + p)),
    f'stretch {STRETCH_RANGE[0]}–{STRETCH_RANGE[1]} at Warping 0–100%': all(within_stretch(ss, ff, pp) for ss, ff, pp in zip(s, f, p)),
    'prints not empty': all((qq > 0.5).any() for qq in q),
    'print largest component ≥ 80%': all(largest(qq > 0.5).sum() / (qq > 0.5).sum() >= 0.8 for qq in q),
    'logo center inside print': all(q[i][int(r['logo']['cy'] * H), int(r['logo']['cx'] * W)] > 0.5 for i, r in enumerate(regions)),
}
(OUT / 'checks.json').write_text(json.dumps(checks, indent=2))
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
# 和竞品模板自带的 T 恤商品区、可印区域对比,只作参考
reference = np.asarray(Image.open(SRC / 'tee-print.webp').convert('RGBA')).astype(np.float32) / 255
for name, ours, theirs in (('product-1', p[0] > 0.5, restored), ('print-1', q[0] > 0.5, luma(reference[..., :3]) * reference[..., 3] > 0.5)):
    print(f'IoU vs Mockey 04674 {name}: {(ours & theirs).sum() / (ours | theirs).sum():.3f}')
for f in sorted(OUT.iterdir()):
    print(f'{f.stat().st_size / 1024:8.1f} KB  {f.name}')
