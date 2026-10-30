"""
按模板配置(specs/*.json)生成模板文件到 output/<id>/,并跑自动校验。

模型的结果由前面的脚本存在 work/<id>/:
- 分割(第 2 步):SAM 3(文字提示)或 SAM 2.1(手点)分出的商品、不可印部件和遮挡物,ViTMatte 求的边缘透明度(run_segment.py);
  可印区域按配置里的规则从商品区里取。
- 底图(第 3 步):Marigold 本征分解(intrinsic.py 产出 albedo.png、shading.png、residual.png)。
- 流向图(第 4 步):大曲面和透视取 MoGe-2 的 3D 坐标(geometry.py 产出 points.z、geometry.json),
  褶皱取 Marigold 法线(normals.py 产出 normals.png)。
其余步骤都是固定代码。用法:.venv/bin/python -I pipeline.py specs/<模板>.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pymatting import estimate_foreground_ml
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config
from segment import drop_specks, grow

spec = config.load(sys.argv[1])
WORK, OUT = config.folder(spec, 'work'), config.folder(spec, 'output')

WHITE_ALBEDO = 0.87  # 白布料在本征分解里的固有色亮度(样本白 T 恤实测 0.86–0.88):选白色时还原照片的亮度
DARK_PHOTO = 0.05  # 商品实心部分照片亮度(线性)的中位数低于它算深色商品(sRGB 约 0.25)
DARK_BRIGHT = 0.9  # 深色商品抬亮时对齐的亮部百分位
DARK_TEXTURE = 0.5  # 深色商品的纹理(对数)保留这么多
GLOSS_KEEP = 0.6  # 残差里只有这一份当光泽叠在颜色上,其余并入明暗:白色亮度不变,深色、纯色的暗通道和竞品对齐(样本上约为白色亮度的 3%)
SHADE_HEADROOM = 99.9  # 明暗倍数让商品区这个百分位以内的明暗都放得下,更亮的极少数截掉:倍数越大,8 位 alpha 里明暗的精度越低
TEXTURE_RATIO = 1 / 256  # 纹理半径 / 长边:照片里比它细的起伏算纹理,粗的算明暗
TEXTURE_CLIP = (-0.65, 0.44)  # 纹理(对数)的截断,个别像素不过亮过暗
SIGMA_RATIO = 0.006  # 起伏图模糊半径 / 宽度
DISP_SCALE = 4  # 起伏图缩小倍数:按块求平均,抹掉 8 位台阶和纹理噪点;使用时双线性放大
SLOPE_BLUR = 2.0  # 求坡度前在缩小后的起伏上再模糊(格):坡度再求一次导就是形变,对噪点更敏感
STRETCH_RANGE = (0.5, 2.5)  # Warping 0–100% 时各方向每像素前进量的范围:拉伸至多 2 倍、压缩至多 2.5 倍
STRETCH_OUTLIER = 0.005  # 校验时可印区里至多这么多面积可以超出上面的范围:人穿着的衣服侧面、袖口这些陡边上,大曲面层自己就会超出一点
PRODUCT_MIN = 0.005  # 每件商品至少占画面这么多;源图里商品被去印花一起抹掉、只分出一点碎块时由校验标出来(胸牌约 1%)
FACE_ANGLE = 35  # print.face:法线和主朝向夹角在这么多度以内的算同一面;盒子相邻两面差 90°
FACE_MIN = 0.3  # print.face:正对的那一面至少要占原来可印区这么多才收窄;更小的不收,由校验标出来
CAP_MARGIN = (0.1, 0.2)  # 减弱时比上面的范围再收一点,留给 8 位量化和双线性放大
CAP_AREA = 0.2  # 每件商品的坡度系数按它定:Warping 100% 时可印区域里这么多面积刚好顶到上限
SLOPE_GAIN_MAX = 1.4e-3  # 坡度系数(位移 / 宽度 ÷ 起伏对(坐标 / 宽度)的坡度)的上限:几乎没褶的商品不把噪点放大成假褶皱
SURFACE_RATIO = 0.02  # 大曲面:度量的平滑半径 / 宽度
SURFACE_RANGE = (0.7, 2.0)  # 大曲面每像素前进量的范围:近处至多放大到 1 / 0.7,侧面至多压缩 2 倍
SURFACE_INSET = 4  # 商品边上几个像素的 3D 坐标混着背景,往里收这么多再算
SMOOTH = 2.0  # 二值轮廓的平滑半径(px),台阶越长要越大
GAP_RATIO = 0.004  # 商品和遮挡物之间的缝,两边都在这么远(× 宽度)以内的算遮挡物
FLOW_BLEND = 4  # 流向图商品外铺成边上的值,离守住范围的格子这么多格以外完全换成模糊过的值(格)


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


def local_mean(image, weight, sigma, levels=5):
    """weight 区域内颜色的局部均值,延伸到区外:离区越远取越大半径(逐级翻倍)的均值,不会拉出条纹;再远的取最近的区内像素"""
    mean = np.zeros_like(image)
    pending = np.ones(weight.shape, bool)
    for level in range(levels):
        total = cv2.GaussianBlur(weight, (0, 0), sigma * 2 ** level)
        ready = pending & (total >= 1e-3)
        blurred = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma * 2 ** level)
        mean[ready] = (blurred / np.maximum(total, 1e-6)[..., None])[ready]
        pending &= ~ready
    if pending.any():
        _, (iy, ix) = ndimage.distance_transform_edt(weight < 0.5, return_indices=True)
        mean[pending] = image[iy[pending], ix[pending]]
    return mean


def read_layer(name):
    """intrinsic.py 存的 16 位线性 RGB"""
    return cv2.imread(str(WORK / name), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535


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


def extend_outside(flow, inside):
    """inside 以外的格子换成离它最近的 inside 格的值,离 inside FLOW_BLEND 格以外渐变成模糊过的值。
    渲染端整体减掉设计稿中心处的位移;商品外的位移若还在变,中心拖出商品时这份变得比拖动还快,设计稿会反着往商品里跑。
    只铺最近值的话,凹进去的地方(领口、两腿之间)两条边的值在中线上会跳,所以远处再渐变成模糊值"""
    distance, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    nearest = flow[iy, ix]
    blurred = nearest
    for _ in range(3):  # 盒式模糊连做三遍,近似高斯
        blurred = cv2.blur(blurred, (2 * FLOW_BLEND + 1,) * 2, borderType=cv2.BORDER_REPLICATE)
    t = np.minimum(1, distance / FLOW_BLEND)[..., None]
    extended = nearest * (1 - t) + blurred * t
    extended[inside] = flow[inside]
    return extended.astype(np.float32)


def print_area(product, parts, cover, rule):
    """可印区域:商品区减去不可印部件,只在商品和背景的交界按 rule 的 inset 往里收(上下左右各占宽度的比例,
    如下摆留边只写 bottom),再只留 rows 给的高度区间(占商品高度的比例)。cover 是盖在商品上的遮挡物和上层商品,
    这些边上和部件边上不收,设计稿一直铺到被挡住的地方,由底图或上层商品盖住"""
    top, bottom, left, right = (round(rule.get('inset', {}).get(side, 0) * W) for side in ('top', 'bottom', 'left', 'right'))
    solid = product > 0.5
    gap = round(GAP_RATIO * W)
    covered = solid | cover | (grow(solid, gap) & grow(cover, gap))
    # 某一侧往里收 = 那一侧一定距离内有背景就去掉:核只往那一侧伸
    kernel = np.ones((top + bottom + 1, left + right + 1), np.uint8)
    region = (cv2.erode(covered.astype(np.uint8), kernel, anchor=(left, top)) > 0) & ~parts
    if 'rows' in rule:
        ys = np.nonzero(solid)[0]
        y0, height = ys.min(), ys.max() - ys.min()
        band = np.zeros_like(region)
        band[int(y0 + rule['rows'][0] * height):int(y0 + rule['rows'][1] * height)] = True
        region &= band
    # 部件蒙版没贴到轮廓、遮挡物的地方会夹出零碎的小块
    visible = region & solid
    region &= drop_specks(visible) | ~visible
    coverage = np.minimum(soft(region), product)
    if rule.get('largest'):
        # 部件把可印区切成几块时(如围裙的腰带)只留最大的一块。部件边上常留着一两像素的细缝连着各块,
        # 柔化后才断开,所以按柔化后的结果挑,再往外留出软边
        coverage = np.where(grow(largest(coverage > 0.5), round(SMOOTH) + 2), coverage, 0)
    return coverage


def main_face(printable, normals):
    """可印区里只留正对的那一面:法线和主朝向(可印区里法线的中位数)夹角在 FACE_ANGLE 以内、连成一片的最大一块。
    盒子、精装书、纸袋这类硬的多面商品,可印区跨过棱角时,侧面、书脊上的设计稿会挤成一条"""
    inside = printable > 0.5
    n = normals / np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-6)
    main = np.median(n[inside], axis=0)
    main /= max(float(np.linalg.norm(main)), 1e-6)
    face = largest(inside & (n @ main > np.cos(np.radians(FACE_ANGLE))))
    return quantize(np.minimum(printable, soft(face))) if face.sum() >= FACE_MIN * inside.sum() else printable


def carrier(product, printable):
    """流向图铺在哪:商品分成几块(罐子和盖子、敞开夹克的两片前襟)而可印区只在其中一块时,只留可印区所在的那几块。
    设计稿只画在可印区里,别的块也铺位移的话,设计稿中心拖过块之间的空隙时位移会来回跳"""
    solid = product > 0.5
    labels, _ = ndimage.label(solid)
    other = solid & ~np.isin(labels, np.unique(labels[solid & (printable > 0.5)]))
    return np.where(grow(other, 2), 0, product) if other.any() and (printable > 0.5).any() else product


def save_mask(coverage, name):
    """白色 + alpha = 覆盖度"""
    alpha = np.round(coverage * 255).astype(np.uint8)
    rgba = np.dstack([np.full_like(alpha, 255)] * 3 + [alpha])
    Image.fromarray(rgba, 'RGBA').save(OUT / name, optimize=True)


# ---- 1. 源图
photo, source_alpha = config.photo(spec)
H, W = photo.shape[:2]
Image.fromarray(np.round(photo * 255).astype(np.uint8)).save(OUT / 'cover.webp', quality=85)
# 源图里透明度不到 1 的像素不是原始照片(如从竞品底图还原的商品区),第 3 步换背景色时要用
restored = source_alpha < 1

# ---- 2. 分割(run_segment.py 的结果),商品由上层到下层
read_alpha = lambda name: np.asarray(Image.open(WORK / name)).astype(np.float32) / 65535
read_mask = lambda name: np.asarray(Image.open(WORK / name)) > 127
occluders = read_alpha('occluders.png') > 0.5
# 法线(normals.py)只给 print.face 挑正对的那一面用,先模糊去掉纹理的起伏
normals = cv2.GaussianBlur(cv2.resize(read_layer('normals.png'), (W, H), interpolation=cv2.INTER_LINEAR) * 2 - 1, (0, 0), SIGMA_RATIO * W)
products, prints = [], []
above, above_bin = np.zeros((H, W), np.float32), np.zeros((H, W), bool)  # 上层商品的覆盖度之和、二值范围
for item in spec['products']:
    segmented = read_alpha(f"{item['id']}-alpha.png")
    # 源图里轮廓是台阶的商品(如从竞品底图还原的),抠图模型救不回来:按二值来源处理,平滑轮廓后重出软边
    alpha = soft(segmented > 0.5) if item.get('binaryEdge') else segmented
    # 下层商品在上层下面接着铺,交界处相加为 1;重叠处归上层
    _, (iy, ix) = ndimage.distance_transform_edt(above_bin, return_indices=True)
    product = quantize(np.minimum(alpha[iy, ix], 1 - above))
    products.append(product)
    prints.append(quantize(print_area(product, read_mask(f"{item['id']}-parts.png"), occluders | above_bin, item.get('print', {}))))
    if item.get('print', {}).get('face') and (prints[-1] > 0.5).any():
        prints[-1] = main_face(prints[-1], normals)
    above, above_bin = np.minimum(above + product, 1), above_bin | (segmented > 0.5)
empty = [item['id'] for item, product in zip(spec['products'], products) if not (product > 0.5).any()]
if empty:
    raise SystemExit(f"分割没分出商品:{', '.join(empty)}(提示词分不出,或源图里没有这件商品)")
blank = [item['id'] for item, printable in zip(spec['products'], prints) if not (printable > 0.5).any()]
if blank:
    raise SystemExit(f"可印区是空的:{', '.join(blank)}(分出的商品太小,或被部件、内缩扣光了)")
coverage = np.minimum(sum(products), 1)

# ---- 3. 底图
# 编辑页在线性光下合成:结果 = 明暗倍数 × 商品颜色 × (1 − alpha) + 底图 × alpha。
# 商品区里 倍数 × (1 − alpha) = 明暗 × 纹理,底图 × alpha = 光泽 × 纹理(光泽是不随颜色变的反光,纹理也作用在它上面)。
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
    dark = np.median(I[solid]) < DARK_PHOTO
    if dark:
        # 深色照片里噪点占的比例大,抬亮后会显成颗粒:纹理减弱
        texture = texture ** DARK_TEXTURE
    diffuse = WHITE_ALBEDO * blur(luma(s * layers['shading'])) * texture
    # 深色商品照片里光照和固有色分不开,算出的光照常偏暗:保留明暗起伏,亮部抬到白布料在正常光下的亮度。
    # 反光强的深色商品照片全被残差解释,拟合的 s 会是 0 或负数,所以按光照层本身的起伏缩放(s 为正时结果相同)
    if dark and np.percentile(diffuse[solid], DARK_BRIGHT * 100) < WHITE_ALBEDO:
        light = blur(luma(layers['shading'])) * texture
        diffuse = light * WHITE_ALBEDO / max(float(np.percentile(light[solid], DARK_BRIGHT * 100)), 1e-4)
    sheen = luma(max(r, 0) * layers['residual']) * texture
    # 边缘像素在照片里混着商品原色:取商品实心部分的值往外延伸,区外那份换成背景色
    for total, layer in ((shade, diffuse), (gloss, sheen)):
        total += product * np.where(solid, layer, local_mean(np.repeat(layer[..., None], 3, -1), solid.astype(np.float32), 2)[..., 0])
shade, gloss = (np.where(coverage > 0, x / np.maximum(coverage, 1e-6), 0) for x in (shade, gloss))
gloss = np.clip(gloss, 0, 1)
shade, gloss = shade + (1 - GLOSS_KEEP) * gloss, GLOSS_KEEP * gloss
# 底图 RGB 不超过 1 要求 明暗 ÷ 倍数 + 光泽 ≤ 1,倍数按商品区要的取,不溢出的模板为 1。
# 浅色超出 1 的部分留到渲染时按实际颜色截,深色的亮处不跟着压暗
shade_gain = round(max(1.0, float(np.percentile((shade / np.maximum(1 - gloss, 1e-6))[coverage > 0.5], SHADE_HEADROOM))), 3)
shade = np.clip(shade, 0, shade_gain * (1 - gloss))
alpha = 1 - coverage * shade / shade_gain
# 商品边缘的像素 = 商品色 × 覆盖度 + 背后的颜色 × (1 − 覆盖度),背后那份(背景、皮肤、头发等)按抠图的前景色估计解出来;
# 源图里还原出来的像素带着还原时的白色,改取周围原始照片的局部均值
dirty = (coverage > 0) | restored
_, behind = estimate_foreground_ml(photo_linear.astype(np.float64), coverage.astype(np.float64), return_background=True)
background = np.where((coverage > 0)[..., None], to_srgb(np.clip(behind, 0, 1)), photo)
background = np.where(restored[..., None], local_mean(photo, (~dirty).astype(np.float32), 2), background)
premultiplied = (1 - coverage)[..., None] * to_linear(background) + (coverage * gloss)[..., None]
rgb = to_srgb(premultiplied / np.maximum(alpha, 1e-6)[..., None])
base = np.dstack([rgb, alpha])
Image.fromarray(np.round(base * 255).astype(np.uint8), 'RGBA').save(OUT / 'base.webp', quality=85, alpha_quality=100)

# 模板墙的 CSS 叠图用的底图:浏览器只能按 sRGB 值普通叠加,由落盘的 base 换算,选白色、纯黑时和线性光结果一致,中间颜色按 sRGB 插值
# 整张不透明时 webp 不存透明度通道,读回来要补上
shipped = np.asarray(Image.open(OUT / 'base.webp').convert('RGBA')).astype(np.float32) / 255
on_white = to_srgb(shade_gain * (1 - shipped[..., 3:]) + to_linear(shipped[..., :3]) * shipped[..., 3:])
on_black = to_srgb(to_linear(shipped[..., :3]) * shipped[..., 3:])
css_alpha = np.clip(1 - (on_white - on_black).mean(axis=2, keepdims=True), 1 / 255, 1)
css_base = np.dstack([np.clip(on_black / css_alpha, 0, 1), css_alpha])
Image.fromarray(np.round(css_base * 255).astype(np.uint8), 'RGBA').save(OUT / 'base-css.webp', quality=85, alpha_quality=100)

# ---- 4. 流向图:每件商品两张,R / G = x / y 方向的位移,1/4 尺寸
# - 大曲面(固定):按 MoGe-2 的 3D 坐标,设计稿像贴在曲面上,侧面压缩、近处略放大;
# - 褶皱(Warping 100% 时):起伏取法线的 z 分量(表面正对镜头的程度),位移沿它的坡度,褶皱横着就上下压缩、
#   竖着就左右压缩,和照片的光线方向无关;系数按每件商品自己的褶皱定,Warping 0–100% 时两层合起来超限的地方
#   局部减弱褶皱这层,深褶处不翻折。
# 位移会沿曲面累积,渲染端减去设计稿中心处的位移,中心不动;设计稿中心可以拖出商品,商品外按 extend_outside 铺平。
# 交界两侧的起伏不连续,每件商品单独一张、各自向外延伸,缩小和插值时才不混在一起
facing = cv2.resize(read_layer('normals.png'), (W, H), interpolation=cv2.INTER_LINEAR)[..., 2] * 2 - 1
points = config.load_points(WORK)
focal = json.loads((WORK / 'geometry.json').read_text())['intrinsics'][0][0] * W
sigma = SIGMA_RATIO * W
small_size = (W // DISP_SCALE, H // DISP_SCALE)
to_px = W * W / DISP_SCALE  # 系数 × to_px:缩小后每格的坡度 → 模板像素位移
low, high = STRETCH_RANGE[0] + CAP_MARGIN[0], STRETCH_RANGE[1] - CAP_MARGIN[1]
carriers = [carrier(product, printable) for product, printable in zip(products, prints)]
surfaces = [surface_flow(points, focal, product, small_size, SURFACE_RATIO * W) for product in carriers]
# 大曲面的 8 位量化让相邻格的位移差至多多出一级,两层叠起来减弱时再扣掉这部分
slack = 2 * max(np.abs(surface).max() for surface in surfaces) / 255 / DISP_SCALE
folds, guards, gains = [], [], []
for index, (product, printable) in enumerate(zip(carriers, prints)):
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
    # 守住范围的格子原样留着(商品边上的像素插值要用到它们),从它们往外铺
    surfaces[index] = extend_outside(surfaces[index], guarded)
    folds.append(extend_outside(np.dstack([ox, oy]).astype(np.float32), guarded))
    guards.append(guarded)


def save_flows(flows, prefix):
    """通道值 0 / 255 对应 ∓range 像素,按守住范围的格子里的最大位移定,8 位精度用满;更外面的格子用不到,截掉。
    每张先整体减去守住范围的格子里各方向最大、最小位移的中点:渲染端减去设计稿中心处的位移,整体平移不改变结果,
    范围小了每一级对应的位移就小,直线上的台阶跟着变小"""
    flows = [flow - (flow[g].max(axis=0) + flow[g].min(axis=0)) / 2 for flow, g in zip(flows, guards)]
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
for index, (printable, item) in enumerate(zip(prints, spec['products']), start=1):
    inside = printable > 0.5
    distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
    xs = np.nonzero(inside)[1]
    size = min(0.4 * (xs.max() - xs.min()), 1.4 * distance.max())
    regions.append({
        'key': item['key'],
        'name': item['name'],
        'product': f'product-{index}.png',
        'print': f'print-{index}.png',
        'surface': f'surface-{index}.png',
        'folds': f'folds-{index}.png',
        'recolorable': item.get('recolorable', True),
        'logo': {'cx': round(float(cx) / W, 4), 'cy': round(float(cy) / H, 4), 'width': round(float(size) / W, 4),
                 'height': round(float(size) / H, 4), 'rotation': 0},
    })
template = {
    'id': spec['id'],
    'version': 1,
    'size': {'width': W, 'height': H},
    'category': spec['category'],
    'tags': spec.get('tags', []),
    'files': {'cover': 'cover.webp', 'base': 'base.webp', 'baseCss': 'base-css.webp'},
    'blend': 'linear',
    'shadeGain': shade_gain,
    'displacement': {'surfaceRangePx': surface_range, 'foldsRangePx': folds_range},
    'regions': regions,
    'pipeline': {'whiteAlbedo': WHITE_ALBEDO, 'darkPhoto': DARK_PHOTO, 'darkBright': DARK_BRIGHT, 'darkTexture': DARK_TEXTURE,
                 'glossKeep': GLOSS_KEEP, 'textureRadius': round(float(radius), 1), 'textureClip': list(TEXTURE_CLIP),
                 'blurSigma': round(float(sigma), 1),
                 'displacementScale': 1 / DISP_SCALE, 'slopeBlur': SLOPE_BLUR, 'stretchRange': list(STRETCH_RANGE),
                 'capArea': CAP_AREA, 'slopeGainMax': SLOPE_GAIN_MAX, 'slopeGain': [float(f'{g:.3g}') for g in gains],
                 'surfaceSigma': round(SURFACE_RATIO * W, 1), 'surfaceRange': list(SURFACE_RANGE)},
}
(OUT / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))

# ---- 6. 自动校验(读回落盘的文件)
def read(name, mode=None):
    image = Image.open(OUT / name)
    return np.asarray(image.convert(mode) if mode else image).astype(np.float32) / 255

base_out = read('base.webp', 'RGBA')  # 整张不透明时 webp 不存透明度通道
css_out = read('base-css.webp', 'RGBA')
numbers = range(1, len(products) + 1)
s = [read(f'surface-{i}.png') for i in numbers]
f = [read(f'folds-{i}.png') for i in numbers]
p = [read(f'product-{i}.png')[..., 3] for i in numbers]
q = [read(f'print-{i}.png')[..., 3] for i in numbers]
union = np.any([pp > 0 for pp in p], axis=0)


def overlap(a, b):
    """两件商品重叠的面积占较小那件的比例"""
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max(1, min(a.sum(), b.sum())))


def offsets(surface, fold, warp):
    """按编辑页的方式双线性放大回模板尺寸,Warping 为 warp(0–1)时每像素的位移"""
    surface, fold = (cv2.resize((x[..., :2] - 0.5) * 2 * r, (W, H), interpolation=cv2.INTER_LINEAR)
                     for x, r in ((surface, surface_range), (fold, folds_range)))
    return surface + warp * fold


def drag_never_back(surface, fold, printable):
    """设计稿中心从可印区中心沿各方向一路拖出画面,画上去的位置(中心 + 中心处位移)不往回走;
    8 位量化本身有零点几像素的抖动,回退超过 1px 才算"""
    ys, xs = np.nonzero(printable > 0.5)
    cx, cy = xs.mean(), ys.mean()
    for warp in np.linspace(0, 1, 5):
        offset = offsets(surface, fold, warp)
        for angle in np.radians(np.arange(0, 360, 5)):
            d = np.array([np.cos(angle), np.sin(angle)])
            steps = np.arange(int(np.hypot(W, H)))
            px, py = cx + d[0] * steps, cy + d[1] * steps
            keep = (px >= 0) & (py >= 0) & (px <= W - 1) & (py <= H - 1)
            px, py = px[keep], py[keep]
            ox, oy = (ndimage.map_coordinates(offset[..., c], [py, px], order=1) for c in (0, 1))
            drawn = (px + ox) * d[0] + (py + oy) * d[1]
            if (np.diff(drawn) < -1).any():
                return False
    return True


def css_ends(css):
    """CSS 底图按 sRGB 普通叠在白色、纯黑上,和线性光的结果比,商品区内的平均差和 99% 分位(0–255)。
    一个透明度管不了三个通道各自的差,背后颜色带色的边缘、发丝处会差得多一些,只占极少数像素"""
    a, rgb = css[..., 3:], css[..., :3]
    inside = union[..., None].repeat(3, axis=2)
    diffs = [np.abs(over - target)[inside] * 255 for over, target in ((1 - a + rgb * a, on_white), (rgb * a, on_black))]
    return [(float(d.mean()), float(np.percentile(d, 99))) for d in diffs]


def within_stretch(surface, fold, printable):
    """Warping 0–100% 时可印区内(设计稿只画在这里)各方向每像素前进量超出范围的面积不到 STRETCH_OUTLIER"""
    inside = printable > 0.5
    for warp in np.linspace(0, 1, 5):
        offset = offsets(surface, fold, warp)
        smin, smax = stretch_range(*jacobian(offset[..., 0], offset[..., 1], 1))
        if ((smin[inside] < STRETCH_RANGE[0]) | (smax[inside] > STRETCH_RANGE[1])).mean() > STRETCH_OUTLIER:
            return False
    return True


def joined_share(printable):
    """可印区最大一块占的比例。遮挡物(如交握的手)横穿可印区时,设计稿在它下面照样铺着,隔开的两边算一块"""
    visible = printable > 0.5
    labels, count = ndimage.label(visible | occluders)
    if not count:
        return 0.0
    return float(ndimage.sum(visible, labels, range(1, count + 1)).max() / visible.sum())


checks = {
    'sizes equal': all(x.shape[:2] == (H, W) for x in [base_out, *p, *q]),
    'flow maps 1/4 size, B unused': all(x.shape == (H // DISP_SCALE, W // DISP_SCALE, 3)
                                        and bool((x[..., 2] == 128 / 255).all()) for x in s + f),
    'aspect 4:5 or 5:4': W * 5 == H * 4 or W * 4 == H * 5,
    'products overlap < 0.1%': all(overlap(a, b) < 0.001 for i, a in enumerate(p) for b in p[i + 1:]),
    'print outside product < 0.1%': all(float(((qq > 0.5) & (pp < 0.5)).sum() / max(1, (qq > 0.5).sum())) < 0.001 for pp, qq in zip(p, q)),
    'base opaque outside products': bool((base_out[..., 3][~union] == 1).all()),
    'base matches formula (≤ 2/255)': bool(np.abs(base_out[..., 3] - alpha).max() <= 2 / 255 + 1e-6),
    # 防换算写错;webp 有损压缩(质量 85)和边缘本身就会差几个色阶,明暗倍数大的商品平均差到 1 以上,公式错了会差几十
    'base-css matches on white / black (mean ≤ 2, 99% ≤ 12 of 255)': all(mean <= 2 and p99 <= 12 for mean, p99 in css_ends(css_out)),
    'base-css sizes equal': css_out.shape == base_out.shape,
    'design centre dragged outward never runs back': all(drag_never_back(ss, ff, qq) for ss, ff, qq in zip(s, f, q)),
    f'stretch {STRETCH_RANGE[0]}–{STRETCH_RANGE[1]} at Warping 0–100% (outside ≤ {STRETCH_OUTLIER:.1%})': all(within_stretch(ss, ff, qq) for ss, ff, qq in zip(s, f, q)),
    f'products ≥ {PRODUCT_MIN:.1%} of frame': all((pp > 0.5).mean() >= PRODUCT_MIN for pp in p),
    'prints not empty': all((qq > 0.5).any() for qq in q),
    'print largest component ≥ 80%': all(joined_share(qq) >= 0.8 for qq in q),
    'logo center inside print': all(q[i][int(r['logo']['cy'] * H), int(r['logo']['cx'] * W)] > 0.5 for i, r in enumerate(regions)),
}
(OUT / 'checks.json').write_text(json.dumps(checks, indent=2))
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
# 和配置里的参考蒙版(如竞品模板自带的)对比,只作参考:transparent 取半透明的部分,否则取白色 × 透明度
for index, item in enumerate(spec['products']):
    for kind, ours in (('product', p[index]), ('print', q[index])):
        if kind not in item.get('reference', {}):
            continue
        ref = item['reference'][kind]
        rgba = np.asarray(Image.open(config.ROOT / ref['file']).convert('RGBA')).astype(np.float32) / 255
        theirs = rgba[..., 3] < 1 if ref.get('transparent') else luma(rgba[..., :3]) * rgba[..., 3] > 0.5
        print(f"IoU vs reference {kind}-{index + 1}: {((ours > 0.5) & theirs).sum() / ((ours > 0.5) | theirs).sum():.3f}")
print('base-css vs linear on white / black (mean, 99%, of 255): ' + ' / '.join(f'{m:.2f}, {q:.1f}' for m, q in css_ends(css_out)))
for f in sorted(OUT.iterdir()):
    print(f'{f.stat().st_size / 1024:8.1f} KB  {f.name}')
