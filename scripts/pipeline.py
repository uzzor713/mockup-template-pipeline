"""
按模板配置生成模板文件到 output/<id>/,并跑自动校验。

模型的结果由前面的脚本存在 work/<id>/:
- 分割(第 2 步):SAM 3(文字提示)或 SAM 2.1(手点)分出的商品、不可印部件和遮挡物,ViTMatte 求的边缘透明度(run_segment.py);
  可印区域按配置里的规则从商品区里取。
- 底图(第 3 步):Marigold 本征分解(intrinsic.py 产出 albedo.png、shading.png、residual.png)。
- 流向图(第 4 步):大曲面和透视取 MoGe-2 的 3D 坐标(geometry.py 产出 points.z、geometry.json),
  褶皱取 Marigold 法线(normals.py 产出 normals.png)。
其余步骤都是固定代码。用法:.venv/bin/python -I scripts/pipeline.py batches/<批次>/specs/<id>.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pymatting import estimate_foreground_ml
from scipy import ndimage, sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config
import logo
from segment import drop_specks, grow

spec = config.load(sys.argv[1])
WORK, OUT = config.folder(spec, 'work'), config.folder(spec, 'output')

WHITE_ALBEDO = 0.87  # 白布料在本征分解里的固有色亮度(白 T 恤实测 0.86–0.88):选白色时还原照片的亮度
DARK_PHOTO = 0.1  # 商品实心部分照片亮度(线性)的中位数低于它算深色商品(sRGB 约 0.35)
DARK_BRIGHT = 0.9  # 深色商品抬亮时对齐的亮部百分位
DARK_TEXTURE = 0.5  # 深色商品的纹理(对数)保留这么多
DARK_CONTRAST = 0.25  # 深色商品的明暗起伏(对数)保留这么多:压到四分之一后选白色的亮度和浅色商品相当
WHITE_CAP = 0.95  # 选白色时每件商品实心部分 WHITE_CAP_AT 分位的亮度(线性)至多这么亮,更亮的整件压暗:截成纯白的约 5–7%,和正常照片相当
WHITE_CAP_AT = 90
# 浅色照片做的布料商品照样保留照片的明暗,强光下亮处接近纯白、暗处很深,选白色显得银灰发亮、不像棉布;
# 深色照片的明暗已按 DARK_CONTRAST 压过,选白色更柔和。所以浅色布料的大块明暗收到深色布料的水平:
# 跨度 = 选白色时实心部分在 3 倍纹理半径内平均后亮度(对数)的 p90 − p10,本机深色布料 942 个的 80% 分位约 0.92
FABRIC = {'t-shirt', 'hoodie', 'cap', 'jersey', 'sweatpants', 'sweatshirt', 'jacket', 'socks', 'tracksuit', 'silk-scarf', 'jeans', 'blanket',
          'beanie', 'apron', 'crop-top', 'pillow', 'bucket-hat', 'tote-bag', 'polo-shirt', 'bandana', 'shorts', 'tank-top', 'cloth-napkin'}
FABRIC_SPREAD = 0.917  # 浅色布料选白色时大块明暗的跨度至多这么大,超出的把光照起伏(对数)按中位数收,纹理、光泽不动
FABRIC_CONTRAST_MIN = 0.5  # 光照起伏至少保留这么多:照片里明暗特别大的(侧光短裤、T 恤)收到跨度上限会整件压成平的中灰
WHITE_FLOOR = 0.75  # 布料、画框 / 画布选白色时实心部分 WHITE_CAP_AT 分位的亮度(线性)至少这么亮,暗的整件提亮:照片偏暗又不走深色那条路时整件发灰
WHITE_FLOOR_KINDS = FABRIC | {'frame', 'canvas'}  # 纸品普遍在 0.72 上下、看着不灰,不提
GLOSS_BASE = 0.03  # 每件商品叠加光泽(乘 GLOSS_KEEP 之后)的中位数至多这么多:光泽整片有底子时按加法叠在每种颜色上,黑色发灰、白色发雾,多出的底子并进明暗
GLOSS_KEEP = 0.6  # 残差里只有这一份当光泽叠在颜色上,其余并入明暗:白色亮度不变,深色、纯色的暗通道和竞品对齐(白 T 恤上约为白色亮度的 3%)
SHADE_HEADROOM = 99.9  # 明暗倍数让商品区这个百分位以内的明暗都放得下,更亮的极少数截掉:倍数越大,8 位 alpha 里明暗的精度越低
TEXTURE_RATIO = 1 / 256  # 纹理半径 / 长边:照片里比它细的起伏算纹理,粗的算明暗
TEXTURE_CLIP = (-0.65, 0.44)  # 纹理(对数)的截断,个别像素不过亮过暗
SIGMA_RATIO = 0.006  # 起伏图模糊半径 / 宽度
DISP_SCALE = 4  # 起伏图缩小倍数:按块求平均,抹掉 8 位台阶和纹理噪点;使用时双线性放大
SLOPE_BLUR = 2.0  # 求坡度前在缩小后的起伏上再模糊(格):坡度再求一次导就是形变,对噪点更敏感
STRETCH_RANGE = (0.5, 2.5)  # Warping 0–100% 时各方向每像素前进量的范围:拉伸至多 2 倍、压缩至多 2.5 倍
STRETCH_OUTLIER = 0.005  # 校验时可印区里至多这么多面积可以超出上面的范围:人穿着的衣服侧面、袖口这些陡边上,大曲面层自己就会超出一点
DRAG_BACK = 3  # 校验时设计稿中心在可印区里往外拖,画上去的位置比可印区里走过的最远处往回退超过这么多像素才算往回跑:8 位量化有一两个像素的抖动,拖动时看不出
PRODUCT_MIN = 0.005  # 每件商品至少占画面这么多;源图里商品被去印花一起抹掉、只分出一点碎块时由校验标出来(胸牌约 1%)
FACE_ANGLE = 35  # print.face:法线和主朝向夹角在这么多度以内的算同一面;盒子相邻两面差 90°
FACE_MIN = 0.3  # print.face:正对的那一面至少要占原来可印区这么多才收窄;更小的不收,由校验标出来
CAP_MARGIN = (0.1, 0.2)  # 减弱时比上面的范围再收一点,留给 8 位量化和双线性放大
SLACK_SHARE = 0.5  # 大曲面层 8 位量化的余量按相邻格差一级的这么多份留:扣满一级时位移幅度大的商品褶皱几乎全被减掉,不留时可印区超出范围的面积过 0.5%
CAP_AREA = 0.2  # 每件商品的坡度系数按它定:Warping 100% 时可印区域里这么多面积刚好顶到上限
SLOPE_GAIN_MAX = 1.4e-3  # 坡度系数(位移 / 宽度 ÷ 起伏对(坐标 / 宽度)的坡度)的上限:几乎没褶的商品不把噪点放大成假褶皱
SURFACE_RATIO = 0.02  # 大曲面:度量的平滑半径 / 宽度
SURFACE_RANGE = (0.7, 2.0)  # 大曲面每像素前进量的范围:近处至多放大到 1 / 0.7,侧面至多压缩 2 倍
SURFACE_INSET = 4  # 商品边上几个像素的 3D 坐标混着背景,往里收这么多再算
SURFACE_KEEP_MIN = 0.3  # 大曲面层存成 8 位后超出校验范围时,位移至多收到这么多份
CSS_MEAN, CSS_P99 = 2, 14  # base-css 叠在白色、纯黑上和线性光结果的差(0–255),商品区内平均、99% 分位的上限
PLANE_INSET = 7  # surface: plane:可印区往里收这么多像素再拟合平面
PLANE_BEND = 0.05  # surface: plane:可印区 3D 点离拟合平面的距离 90% 分位不到可印区边长(面积开方)的这么多才按平面贴。平的约 1–2%;贴在罐子、杯子上的圆贴纸约 6–12%,三折页、拿在手里的报纸约 5–15%
PLANE_QUAD_ERROR = 0.02  # 轮廓和四边形的面积差不到这么多时,用实际轮廓校准 MoGe 估出的透视
PLANE_BOW = 0.005  # 校验时按平面贴的可印区里,屏幕上的直线段映到设计稿后离两端连线最远处占长度的中位数至多这么多:本机 321 个按平面贴的区域中位 0.09%、最大 0.36%(8 位量化的抖动),不记方向的 surface_flow 贴同样的平面 0.7–2.4%
BOW_LINES = 200  # 上面这项量多少条线段
CYL_WIDE = 0.85  # surface: cylinder:先拿宽度到最宽(90% 分位)这么多的最长一段拟合左右轮廓,瓶颈、瓶肩、圆底不算
CYL_ROWS = 0.4  # surface: cylinder:左右轮廓挨着背景的行各至少占这么多、落在直线上的一段至少占可印区高度的这么多才按圆柱
CYL_TILT = 25  # surface: cylinder:轴偏离竖直超过这么多度不按圆柱
CYL_ARC = 0.6  # surface: cylinder:量上下沿的弧时只用中间这么宽(× 那一圈的半宽),两边陡,锯齿影响大
CYL_RING = 0.7  # surface: cylinder:量到弧的那一圈半宽不到身子的这么多(瓶颈、瓶肩)不算;罐口收窄的一圈约 0.85
CYL_DK = 0.15  # surface: cylinder:下沿减上沿的扁度至少是按焦距算的这么多份(实测 0.35–1.0),对不上的那头量到的是别的东西(杯座、手)
PERSPECTIVE = 0.7  # surface: cylinder:只量到一头的弧或量不出弧时,纬线扁度随高度的变化按焦距算的视线角留这么多份
SMOOTH = 2.0  # 二值轮廓的平滑半径(px),台阶越长要越大
GAP_RATIO = 0.004  # 商品和遮挡物之间的缝,两边都在这么远(× 宽度)以内的算遮挡物
FLOW_BLEND = 4  # 流向图商品外铺成边上的值,离守住范围的格子这么多格以外完全换成模糊过的值(格)
BRIDGE_RATIO = 0.08  # 可印区几块之间的缝(敞开的拉链外套两片前襟之间露出的内搭)宽不到 2 倍这么多(× 宽度)的,拖过去往回退时位移在缝里按不变形接过去


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


def surface_flow(points, focal, product, small_size, sigma, gap):
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
    # 可印区几块之间的缝(gap)也算,目标同样是不变形:两片前襟直接接上,不再只经过帽子、下摆绕一圈连着
    region = (cv2.dilate((cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) | gap
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


def stored_limit(flow, guard):
    """save_flows 存这张时 0 / 255 对应 ∓多少像素:先减去守住范围的格子里各方向最大、最小位移的中点,再取最大位移"""
    flow = flow - (flow[guard].max(axis=0) + flow[guard].min(axis=0)) / 2
    return max(float(np.ceil(np.abs(flow[guard]).max() * 10)) / 10, 0.1)


def stored_beyond(flow, guard, inside, limit=None):
    """位移(每格 DISP_SCALE 像素)按 save_flows 存成 8 位、按编辑页双线性放大回模板尺寸后,
    inside 里各方向每像素前进量低于、高于 STRETCH_RANGE 的面积占比。
    limit:存图时的范围,几件商品存在一张图里时按位移最大的那件定;不给就按这张自己的"""
    limit = limit or stored_limit(flow, guard)
    flow = flow - (flow[guard].max(axis=0) + flow[guard].min(axis=0)) / 2
    stored = np.round((0.5 + np.clip(flow, -limit, limit) / (2 * limit)) * 255) / 255
    full = cv2.resize(((stored - 0.5) * 2 * limit).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    smin, smax = stretch_range(*jacobian(full[..., 0], full[..., 1], 1))
    return float((smin[inside] < STRETCH_RANGE[0]).mean()), float((smax[inside] > STRETCH_RANGE[1]).mean())


def plane_outline(printable, center):
    """轮廓清晰的四边形(海报、卡片、盒子面,斜放、斜着拍的也算):从照片里的四角取得中心处的拉伸和透视梯度。"""
    contours, _ = cv2.findContours((printable > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    # 盒子棱角的小倒角、盖沿会多出一两个角:近似的容差从 0.5% 周长逐步放宽到 2%,取第一个四边形,面积差照样卡 PLANE_QUAD_ERROR
    for tolerance in (0.005, 0.01, 0.015, 0.02):
        quad = cv2.approxPolyDP(contour, tolerance * cv2.arcLength(contour, True), True)[:, 0].astype(np.float64)
        if len(quad) <= 4:
            break
    if len(quad) != 4 or not cv2.isContourConvex(quad.astype(np.float32)):
        return None
    area = cv2.contourArea(contour)
    if area == 0 or abs(area - cv2.contourArea(quad.astype(np.float32))) / area > PLANE_QUAD_ERROR:
        return None
    # 四角按绕中心的方向排(屏幕上顺时针),上边取中点在中心以上、最接近水平的那条:斜放、斜着拍的盒子面也认
    centroid = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0]))]
    def flatness(k):
        a, b = quad[k], quad[(k + 1) % 4]
        d = b - a
        return (0 if (a[1] + b[1]) / 2 < centroid[1] else 1, abs(d[1]) / (np.hypot(*d) + 1e-9))
    k = min(range(4), key=flatness)
    tl, tr, br, bl = (quad[(k + i) % 4] for i in range(4))
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width < 100 or height < 100:
        return None
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    homography = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), target)
    cx, cy = center
    at_center = homography @ np.array([cx, cy, 1.0])
    if at_center[2] <= 0:
        return None
    tilt = homography[2, :2] / at_center[2]

    def map_point(x, y):
        mapped = homography @ np.array([x, y, 1.0])
        return mapped[:2] / mapped[2]

    jac = np.stack([map_point(cx + 0.5, cy) - map_point(cx - 0.5, cy),
                    map_point(cx, cy + 0.5) - map_point(cx, cy - 0.5)], -1)
    return jac, tilt


def plane_flow(points, intrinsics, product, printable, small_size, gap, outline_mask=None):
    """平的商品(配置写 surface: plane):可印区的 3D 坐标拟合一个平面,屏幕到平面是单应(透视变换),设计稿按它贴上去,直线还是直的。
    surface_flow 只记每格拉伸多少、不记方向,积分时斜着拍的平面也会被弯成弧。
    长度单位和 surface_flow 相同;可印区中心处只拉伸、不转,拉伸截到 SURFACE_RANGE。
    返回 (位移, 说明, 框转几度);可印区离平面远(贴在罐子上的贴纸、三折页、拿在手里的报纸)或透视太陡的,位移为 None,照旧按曲面算。
    按四角校准的,中心处丢掉的那一下转动就是设计稿该跟着面转的角度(屏幕上顺时针为正),交给 Logo 框;没校准的为 None"""
    inside = printable > 0.5
    if not inside.any():
        return None, '没有可印区', None
    fit = cv2.erode(inside.astype(np.uint8), np.ones((2 * PLANE_INSET + 1,) * 2, np.uint8)) > 0
    pts = points[::DISP_SCALE, ::DISP_SCALE][(fit if fit.sum() >= 500 else inside)[::DISP_SCALE, ::DISP_SCALE]].astype(np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 50:
        return None, '可印区太小', None
    keep = np.ones(len(pts), bool)
    for _ in range(3):  # 剔除离面 3 倍中位距离以外的点(没分出去的手指、夹子)再拟合
        center = pts[keep].mean(0)
        axes = np.linalg.svd(pts[keep] - center, full_matrices=False)[2]
        distance = np.abs((pts - center) @ axes[2])
        keep = distance < 3 * np.median(distance[keep]) + 1e-12
    solid = cv2.erode((product > 0.5).astype(np.uint8), np.ones((2 * SURFACE_INSET + 1,) * 2, np.uint8)) > 0
    unit = np.median(points[..., 2][solid if solid.any() else product > 0.5]) / intrinsics[0, 0]
    bend = np.percentile(distance, 90) / unit / np.sqrt(inside.sum())
    if bend > PLANE_BEND:
        return None, f'可印区离平面 {bend:.1%}(边长的),按曲面', None
    fx, fy, ox, oy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    normal, offset = axes[2], center @ axes[2]

    def on_plane(x, y):
        """模板像素 (x, y) 的视线和平面的交点在平面内的坐标(÷ unit)"""
        ray = np.stack([(x + 0.5 - ox) / fx, (y + 0.5 - oy) / fy, np.ones_like(x)], -1)
        hit = (offset / (ray @ normal))[..., None] * ray - center
        return np.stack([hit @ axes[0], hit @ axes[1]], -1) / unit

    # 屏幕到平面是单应:相对可印区中心 c,平面坐标的差 = J Δ / (1 + h·Δ),Δ = 像素 − c,
    # J 是中心处的雅可比,h 是视线和平面交点深度的倒数沿屏幕的变化率(近大远小都在 h 里)
    ys, xs = np.nonzero(inside)
    cx, cy = float(xs.mean()), float(ys.mean())
    facing = np.array([(cx + 0.5 - ox) / fx, (cy + 0.5 - oy) / fy, 1]) @ normal
    tilt = np.array([normal[0] / fx, normal[1] / fy]) / facing
    jac = np.stack([on_plane(np.float64(cx + 0.5), np.float64(cy)) - on_plane(np.float64(cx - 0.5), np.float64(cy)),
                    on_plane(np.float64(cx), np.float64(cy + 0.5)) - on_plane(np.float64(cx), np.float64(cy - 0.5))], -1)
    outline = plane_outline(printable if outline_mask is None else outline_mask, (cx, cy))
    turn = None
    if outline is not None:
        jac, tilt = outline
    # 设计稿坐标的差 = M Δ / (1 + α h·Δ),仍是单应,直线照样直。J = U S Vᵀ,M = V S' Vᵀ:中心处只拉伸不转,
    # S' 是 S 截到 SURFACE_RANGE;α 是透视留多少,1 时和照片里的平面一样近大远小
    u, s, vt = np.linalg.svd(jac)
    if outline is not None:
        rotation = u @ vt  # J = (U Vᵀ)(V S Vᵀ),位移只用后一半,设计稿比面多转了 (U Vᵀ)ᵀ,框转回来
        turn = round(-float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))), 1) + 0.0
    gy, gx = (np.mgrid[0:small_size[1], 0:small_size[0]] * DISP_SCALE + (DISP_SCALE - 1) / 2).astype(np.float64)
    delta = np.stack([gx - cx, gy - cy], -1)  # 缩小后每格的中心相对 c 的模板像素坐标,和双线性放大对齐
    product_cells = cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0
    region = (cv2.dilate(product_cells.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) | gap  # 算位移的格子(含可印区几块之间的缝),再往外铺最近值
    guard = (cv2.dilate(product_cells.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) | gap  # 和存图时一样,按这些格子定 8 位的范围
    if (1 + delta[region] @ tilt).min() < 0.2:
        return None, '平面在商品范围里转到镜头后面,按曲面', None
    cells = cv2.resize(printable, small_size, interpolation=cv2.INTER_AREA) > 0.5
    _, (iy, ix) = ndimage.distance_transform_edt(~region, return_indices=True)

    def design(scale, perspective):
        """每格的位移(设计稿坐标 − 像素坐标),和可印区各格每像素前进量的最小、最大值"""
        stretch = vt.T @ np.diag(scale) @ vt
        across = 1 + perspective * delta @ tilt
        flow = (delta @ stretch.T / across[..., None] - delta).astype(np.float32)
        flow -= flow[cells].mean(axis=0)
        flow[~region] = 0
        # M Δ / t 的雅可比 = (M t − M Δ (α h)ᵀ) / t²
        t = across[cells][:, None, None]
        local = (stretch * t - (delta[cells] @ stretch.T)[:, :, None] * (perspective * tilt)[None, None, :]) / t ** 2
        return flow[iy, ix], np.linalg.svd(local, compute_uv=False)

    # 大曲面层不减弱,可印区里各方向每像素前进量要自己落在校验范围里:按 save_flows 存成 8 位、按编辑页放大回来再量,
    # 超出的面积至多 STRETCH_OUTLIER 的一半(另一半留给褶皱层)。斜着拍的,近处放大、远处压扁,位移幅度大了 8 位量化的抖动也大:
    # 先少留透视(α 从 1 往 0 降,近处远处的差别变小),还不够再把压得最扁的方向整体少压一点。直线都照样直
    clipped = np.clip(s, *SURFACE_RANGE)
    scale = clipped.copy()
    while True:
        for perspective in np.linspace(1, 0, 6):
            flow, spread = design(scale, perspective)
            under, over = stored_beyond(flow, guard, inside)
            if under + over <= STRETCH_OUTLIER / 2:
                break
        if under + over <= STRETCH_OUTLIER / 2 or scale[0] <= 1:
            break
        scale[0] = max(1.0, scale[0] * 0.9)
    if under + over > STRETCH_OUTLIER / 2:
        return None, f'透视太陡,可印区 {under:.1%} 前进量不到 {STRETCH_RANGE[0]}、{over:.1%} 超过 {STRETCH_RANGE[1]},按曲面', None
    notes = [f'可印区离平面 {bend:.1%}', f'前进量 {spread.min():.2f}–{spread.max():.2f}']
    if outline is not None:
        notes.append('透视按商品轮廓校准' if outline_mask is not None else '透视按可印区轮廓校准')
    if perspective < 0.999:
        notes.append(f'透视留 {perspective:.0%}')
    if scale[0] < clipped[0] * 0.995:
        notes.append(f'压得最扁的方向少压到 {scale[0] / clipped[0]:.0%}')
    return flow, f"按平面({','.join(notes)})", turn


def robust_line(y, x):
    """x = p[0]·y + p[1] 的稳健拟合(离线远的点降权),返回系数和残差的稳健标准差"""
    p = np.polyfit(y, x, 1)
    for _ in range(4):
        r = x - np.polyval(p, y)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 0.5
        p = np.polyfit(y, x, 1, w=np.sqrt(1 / np.maximum(1, np.abs(r) / (2 * s))))
    return p, 1.4826 * np.median(np.abs(x - np.polyval(p, y)))


def longest_run(mask, gap, near=None):
    """一维布尔序列里最长的一段(中间断开不超过 gap 的算连着)的下标;给了 near 时取包含它的那段"""
    labels, count = ndimage.label(ndimage.binary_closing(mask, gap) | mask)
    if not count:
        return np.arange(len(mask))
    pick = labels[near] if near is not None and labels[near] > 0 else 1 + int(np.argmax(np.bincount(labels)[1:]))
    return np.nonzero(labels == pick)[0]


def cylinder_flow(product, printable, parts, blocked, normals, intrinsics, small_size, gap, front_rim=None):
    """罐、杯、瓶、蜡烛这类竖着的回转体(配置写 surface: cylinder):按照片里左右轮廓的两条直线和上下沿的弧算出每一圈纬线,
    设计稿按 (转角, 沿母线的长度) 贴上去,俯视、仰视时横线跟着纬线弯成弧,侧面压缩。
    surface_flow 只记每格拉伸多少、不记方向,横线弯不成弧。
    parts 是不印的部件(盖子、瓶盖),blocked 是遮挡物和上层商品:挨着它们的轮廓不是商品自己的边。
    长度单位和 surface_flow 相同(可印区中心那一圈的像素),可印区中心处只压缩、不转。
    返回 (位移, 说明);轮廓不像竖着的圆柱、圆台的,位移为 None,照旧按曲面算"""
    body = (product > 0.5) & ~parts
    inside = printable > 0.5
    if inside.sum() < 500:
        return None, '可印区太小,按曲面'
    wall = blocked.copy()
    wall[:2], wall[-2:], wall[:, :2], wall[:, -2:] = True, True, True, True
    wall = grow(wall, 3)
    part_near = grow(parts, 3)
    ys, xs = np.nonzero(inside)
    center = np.array([xs.mean(), ys.mean()])
    def measure(angle, CW, CH, shift):
        """从 angle 度起转到轴竖直的坐标系 (x', y') 量轮廓,按量出的轴再转一次。画布 CW × CH,转完再平移 shift。
        返回 (说明, 量到的东西);量不出的说明里写原因,量到的东西为 None"""
        for _ in range(2):
            M = cv2.getRotationMatrix2D(tuple(center), -angle, 1.0)
            M[:, 2] += shift
            turn = lambda mask, border=0: cv2.warpAffine(mask.astype(np.uint8), M, (CW, CH), flags=cv2.INTER_NEAREST, borderValue=border) > 0
            rb, rw, rpn, ri = largest(turn(body)), turn(wall, 1), turn(part_near), turn(inside)
            iy = np.nonzero(ri.any(1))[0]
            y0, y1 = iy[0], iy[-1]
            rows = np.arange(y0, y1 + 1)
            rows = rows[rb[rows].any(1)]
            left = np.array([np.argmax(rb[y]) for y in rows])
            right = np.array([CW - 1 - np.argmax(rb[y][::-1]) for y in rows])
            # 挨着背景的轮廓才是商品自己的边
            free_l = np.array([not rw[y, max(x - 4, 0):x].any() and not rpn[y, max(x - 4, 0):x + 1].any() for y, x in zip(rows, left)])
            free_r = np.array([not rw[y, x + 1:x + 5].any() and not rpn[y, x:x + 5].any() for y, x in zip(rows, right)])
            # 先拿够宽的最长一段拟合左右两条线,再接上落在线上的行(被挡住的行也算):瓶颈、瓶肩、圆底不算,圆台窄的那头算
            span = np.ones(max(3, len(rows) // 12), bool)
            run = longest_run(right - left >= CYL_WIDE * np.percentile(right - left, 90), span)
            for extend in (True, False):
                trim = max(1, int(0.05 * len(run)))
                fit = run[trim:len(run) - trim] if len(run) > 2 * trim + 10 else run
                ok_l, ok_r = free_l[fit], free_r[fit]
                if ok_l.mean() < CYL_ROWS or ok_r.mean() < CYL_ROWS:
                    return f'左右轮廓挨着背景的行不够(左 {ok_l.mean():.0%}、右 {ok_r.mean():.0%})', None
                pl, sl = robust_line(rows[fit][ok_l].astype(float), left[fit][ok_l].astype(float))
                pr, sr = robust_line(rows[fit][ok_r].astype(float), right[fit][ok_r].astype(float))
                if extend:
                    tol = max(3.0, 0.03 * np.polyval((pr - pl) / 2, rows[run].mean()))
                    on = ((np.abs(left - np.polyval(pl, rows)) < tol) | ~free_l) & ((np.abs(right - np.polyval(pr, rows)) < tol) | ~free_r)
                    run = longest_run(on, span, near=run[len(run) // 2])
            if rows[run[-1]] - rows[run[0]] < CYL_ROWS * (y1 - y0):
                return f'直的一段只占可印区高度的 {(rows[run[-1]] - rows[run[0]]) / max(y1 - y0, 1):.0%}', None
            tilt = np.degrees(np.arctan((pl[0] + pr[0]) / 2))
            angle += tilt
            if abs(tilt) < 0.3:
                break
        if max(sl, sr) > max(2.0, 0.03 * np.polyval((pr - pl) / 2, (y0 + y1) / 2)):
            return f'左右轮廓不直(残差 {max(sl, sr):.1f}px)', None
        return '', (angle, M, CW, CH, rb, rw, rpn, y0, y1, pl, pr)

    # 先从竖直起量;量不出或轴歪了 CYL_TILT 以上(横放、斜放的瓶罐:左右轮廓量到的是两头)时,按商品身子的主轴转正再量一次,
    # 画布放大成对角线长的正方形、中心挪到可印区中心,转过去不出界。从主轴起量出来的轴不限角度
    ys, xs = np.nonzero(inside)
    center = np.array([xs.mean(), ys.mean()])
    note, measured = measure(0.0, W, H, np.zeros(2))
    if measured is not None and abs(measured[0]) > CYL_TILT:
        note, measured = f'轴歪了 {measured[0]:.0f}°', None
    if measured is None:
        m = cv2.moments(largest(body).astype(np.uint8), True)
        theta = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])  # 主轴和 x 轴的夹角,屏幕上顺时针为正
        dx, dy = np.cos(theta), np.sin(theta)
        axis = float(np.degrees(np.arctan2(dx, dy) if dy >= 0 else np.arctan2(-dx, -dy)))  # 同 tilt:轴往下偏右为正
        side = int(np.ceil(np.hypot(W, H)))
        again, measured = measure(axis, side, side, side / 2 - center)
        if measured is None:
            return None, f'{note};按主轴 {axis:.0f}° 转正后{again},按曲面'
        note = f'按主轴转正(从竖直起量:{note})'
    angle, M, CW, CH, rb, rw, rpn, y0, y1, pl, pr = measured
    c_line, a_line = (pl + pr) / 2, (pr - pl) / 2  # 轴的 x' 和半宽,都是 y' 的一次函数

    def half_width(y):
        """第 y' 行商品最左到最右的一半(罐口里面没分进商品的也算)"""
        span = np.flatnonzero(rb[int(np.clip(round(y), 0, CH - 1))])
        return (span[-1] - span[0]) / 2 if len(span) else 0.0

    def arc(side):
        """上沿 / 下沿中间一段量弧 y' = yc + s·√(1 − u²),u = (x' − 轴) / r,返回 (yc, 扁度 s/r, 挨着部件的占比);量不准的返回 None。
        r 从那头的半宽起,弧往里几行的半宽比它窄就换成那个半宽再量:罐口收窄的一圈量的是它自己;
        瓶颈、瓶肩量出来的那一圈不到身子的 CYL_RING,不算"""
        profile = np.where(rb.any(0), np.argmax(rb, 0) if side == 'top' else CH - 1 - np.argmax(rb[::-1], 0), -1)
        out = -1 if side == 'top' else 1
        yc = float(np.median(profile[profile >= 0]))
        r = float(np.polyval(a_line, yc))
        for _ in range(4):
            cx = np.polyval(c_line, yc)
            cols = np.arange(int(cx - CYL_ARC * r), int(cx + CYL_ARC * r) + 1)
            cols = cols[(cols >= 0) & (cols < CW)]
            yy = profile[cols]
            good = (yy >= 0) & np.array([0 <= y + 4 * out < CH and not rw[min(y, y + 4 * out):max(y, y + 4 * out) + 1, x].any() for x, y in zip(cols, yy)])
            if good.mean() < 0.7:
                return None
            cols, yy = cols[good], yy[good].astype(float)

            def residual(p, r=r):
                u = np.clip((cols - np.polyval(c_line, p[0])) / r, -1, 1)
                return p[0] + p[1] * np.sqrt(1 - u * u) - yy
            fit = least_squares(residual, [yy.mean(), 0.0], bounds=([-np.inf, -r], [np.inf, r]), loss='soft_l1', f_scale=2)
            yc, s = float(fit.x[0]), float(fit.x[1])
            # 弧最里面那点再往里一小段的半宽,轮廓有锯齿,取最宽的一行
            inner = (max if side == 'top' else min)(yc, yc + s)
            w = max(half_width(inner - d * out) for d in range(3, 4 + int(0.05 * r)))
            if w >= 0.93 * r:
                break
            r = w
            if r < CYL_RING * np.polyval(a_line, yc):
                break
        if r < CYL_RING * np.polyval(a_line, yc) or abs(w - r) > 0.15 * r:
            return None
        if 1.4826 * np.median(np.abs(residual(fit.x))) > max(2.0, 0.015 * r):
            return None
        touching = np.mean([rpn[int(y) + 2 * out, x] for x, y in zip(cols, yy) if 0 <= int(y) + 2 * out < CH])
        return yc, s / r, float(touching)

    top, bottom = arc('top'), arc('bottom')
    if front_rim is not None:
        # 盖住瓶口的泵头附近,商品外轮廓可能是后侧顶沿;正面可见的环线才是印花横线的参照。
        y, k = front_rim['y'], front_rim['flattening']
        if not (0 <= y <= 1 and -0.8 <= k <= 0.8):
            raise ValueError('cylinderFrontRim 的 y 必须在 0–1、flattening 必须在 -0.8–0.8')
        rim_y = float((M @ np.array([center[0], y * H, 1]))[1])
        top = (rim_y, float(k), 1.0)
    f = intrinsics[1, 1]
    p_y = (M @ np.array([intrinsics[0, 2], intrinsics[1, 2], 1]))[1]  # 主点的 y'

    def k_model(phi, y):
        """镜头俯仰 φ 时纬线椭圆的扁度(短轴 / 长轴,俯视为正)。视线角按 PERSPECTIVE 份算:
        MoGe 估的焦距偏短,两头都量到弧的,上下扁度的差约为按焦距算的 0.4–0.8"""
        return np.sin(phi + PERSPECTIVE * np.arctan((y - p_y) / f))

    def phi_of(k, y):
        return np.arcsin(np.clip(k, -0.99, 0.99)) - PERSPECTIVE * np.arctan((y - p_y) / f)

    # 法线给出的镜头俯仰:可印区中间一片侧面的法线绕 x' 轴的倾角,用来挑弧的方向、量不出弧时用
    band = [(x, y) for y in np.linspace(y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0), 9)
            for x in np.polyval(c_line, y) + np.polyval(a_line, y) * np.linspace(-0.3, 0.3, 9)]
    src = (cv2.invertAffineTransform(M) @ np.array([[x, y, 1] for x, y in band]).T).round().astype(int)
    n = normals[np.clip(src[1], 0, H - 1), np.clip(src[0], 0, W - 1)]
    phi_n = float(np.median(np.arctan2(-(M[:, :2] @ n[:, :2].T)[1], n[:, 2])))

    def candidates(a, side):
        """弧的扁度可能是哪几个,可能的排前面。商品自己的轮廓上沿只会往上鼓、下沿只会往下鼓,俯视(上沿是后半圈)、
        仰视(上沿是前半圈)都一样,只量得出大小,按法线的俯仰挑;挨着部件的边(标签边、罐盖接缝)和反着弯的是前半圈,扁度就是量到的"""
        yc, k, touching = a
        if touching > 0.5 or (k > 0 if side == 'top' else k < 0):
            return [k]
        return sorted([abs(k), -abs(k)], key=lambda c: abs(c - k_model(phi_n, yc)) + 0.03 * (c < 0))

    def dk_full(kt):
        """上沿扁度为 kt 时,按焦距算(视线角全算)的下沿减上沿"""
        return float(np.sin(np.arcsin(np.clip(kt, -0.99, 0.99)) - np.arctan((top[0] - p_y) / f) + np.arctan((bottom[0] - p_y) / f)) - kt)

    if top and bottom:
        # 越往下纬线越扁(俯视越多):下沿减上沿至少是按焦距算的 CYL_DK 份,对不上的那头多半量到的是别的东西(杯座、手),
        # 只留挨着部件的那头;都不挨着的按曲面
        pairs = [(kt, kb) for kt in candidates(top, 'top') for kb in candidates(bottom, 'bottom') if kb - kt >= CYL_DK * dk_full(kt) - 0.03]
        if not pairs:
            keep = [a for a in (top, bottom) if a[2] > 0.5]
            if not keep:
                return None, '上下沿的弧对不上,按曲面'
            top, bottom = (keep[0], None) if keep[0] is top else (None, keep[0])
    notes = [note] if note else []
    if front_rim is not None:
        notes.append('上沿用正面环线')
    if top and bottom:
        kt, kb = pairs[0]
        slope = (kb - kt) / max(bottom[0] - top[0], 1)
        k_at = lambda y: np.clip(kt + slope * (y - top[0]), -0.8, 0.8)
        notes.append(f'上沿 {kt:+.2f}、下沿 {kb:+.2f}')
    elif top or bottom:
        a, side = (top, 'top') if top else (bottom, 'bottom')
        k0 = candidates(a, side)[0]
        phi = phi_of(k0, a[0])
        k_at = lambda y: np.clip(k_model(phi, y), -0.8, 0.8)
        notes.append(f"只量到{'上沿' if top else '下沿'} {k0:+.2f},另一头按焦距")
    else:
        k_at = lambda y: np.clip(k_model(phi_n, y), -0.8, 0.8)
        notes.append(f'量不出弧,按法线俯仰 {np.degrees(phi_n):+.0f}°')

    # 设计稿坐标:横向 = 参考半宽 × 转角,纵向 = 沿前面那条母线的长度(按扁度还原俯视造成的缩短、按半宽还原近大远小),
    # 都换算成可印区中心那一圈的像素
    yc_ref = solve_latitude(np.array([center[0]]), np.array([center[1]]), c_line, a_line, k_at)[0]
    a_ref = np.polyval(a_line, yc_ref)
    grid_y = np.linspace(y0 - 0.6 * (y1 - y0), y1 + 0.6 * (y1 - y0), 2001)
    front = grid_y + k_at(grid_y) * np.polyval(a_line, grid_y)
    v_table = np.cumsum(np.gradient(front) * a_ref / np.maximum(np.polyval(a_line, grid_y), 1) / np.sqrt(1 - k_at(grid_y) ** 2))
    v_table -= np.interp(yc_ref, grid_y, v_table)
    gy, gx = (np.mgrid[0:small_size[1], 0:small_size[0]] * DISP_SCALE + (DISP_SCALE - 1) / 2).astype(np.float64)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0)  # 缩小后每格中心的模板像素坐标,和双线性放大对齐
    rx, ry = M @ pts
    product_cells = cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0
    region = (cv2.dilate(product_cells.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) | gap  # 算位移的格子(含可印区几块之间的缝),再往外铺最近值
    guard = (cv2.dilate(product_cells.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) | gap  # 和存图时一样,按这些格子定 8 位的范围
    sel = region.ravel()
    back = cv2.invertAffineTransform(M)[:, :2]
    cells = cv2.resize(printable, small_size, interpolation=cv2.INTER_AREA) > 0.5
    _, (iy, ix) = ndimage.distance_transform_edt(~region, return_indices=True)

    def design(cap):
        """每格的位移(设计稿坐标 − 像素坐标)。横向压缩到 cap 倍处(sinθ = s_c)以外,
        转角按 x' 线性延长、纬线沿切线延长:侧面压缩、纵向挤压都不再变大"""
        s_cap = lambda yc: np.sqrt(np.maximum(1 - (a_ref / (np.maximum(np.polyval(a_line, yc), 1) * cap)) ** 2, 0))
        yc = solve_latitude(rx[sel], ry[sel], c_line, a_line, k_at, s_cap)
        u = (rx[sel] - np.polyval(c_line, yc)) / np.maximum(np.polyval(a_line, yc), 1)
        s_c = s_cap(yc)
        U = a_ref * (np.arcsin(np.clip(u, -s_c, s_c)) + np.sign(u) * np.maximum(np.abs(u) - s_c, 0) / np.maximum(np.sqrt(1 - s_c ** 2), 1e-6))
        V = np.interp(yc, grid_y, v_table)
        flow = np.zeros(region.shape + (2,), np.float32)
        flow[region] = (back @ np.stack([U, V]) + center[:, None] - pts[:2, sel]).T
        flow -= flow[cells].mean(axis=0)
        flow[~region] = 0
        return flow[iy, ix]

    # 和 plane_flow 一样,可印区里存成 8 位后超出校验范围的面积至多 STRETCH_OUTLIER 的一半;超了先少压侧面
    for cap in (SURFACE_RANGE[1], 1.8, 1.6, 1.4):
        flow = design(cap)
        under, over = stored_beyond(flow, guard, inside)
        if under + over <= STRETCH_OUTLIER / 2:
            break
    else:
        return None, f'可印区 {under:.1%} 前进量不到 {STRETCH_RANGE[0]}、{over:.1%} 超过 {STRETCH_RANGE[1]},按曲面'
    notes.insert(0, f'轴 {angle:+.1f}°、半宽 {a_ref:.0f}px')
    if cap < SURFACE_RANGE[1]:
        notes.append(f'侧面压缩截到 {cap}')
    return flow, f"按圆柱({','.join(notes)})"


def solve_latitude(x, y, c_line, a_line, k_at, s_cap=None, iters=40):
    """旋转后的 (x', y') 落在哪一圈纬线上:y' = yc + k(yc)·a(yc)·h(u),u = (x' − c(yc)) / a(yc),二分求 yc。
    s_cap(yc) 给出时 |u| 超过它的部分沿切线延长,侧面的纬线不竖起来"""
    lo, hi = y - 2000.0, y + 2000.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        a = np.maximum(np.polyval(a_line, mid), 1)
        u = (x - np.polyval(c_line, mid)) / a
        above = mid + k_at(mid) * a * front_h(u, None if s_cap is None else s_cap(mid)) > y
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return (lo + hi) / 2


def front_h(u, s_c=None):
    """纬线椭圆前半圈的高度 √(1 − u²)(以短半轴为 1);|u| 超过 s_c 后沿切线延长"""
    if s_c is None:
        return np.sqrt(np.maximum(1 - u * u, 0))
    c_c = np.sqrt(1 - s_c ** 2)
    return np.where(np.abs(u) <= s_c, np.sqrt(np.maximum(1 - u * u, 0)), c_c - s_c / np.maximum(c_c, 1e-6) * (np.abs(u) - s_c))


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


def gaps(printable, small_size):
    """可印区几块之间的缝(缩小后的格子):可印区按半径 BRIDGE_RATIO × 宽度做闭运算多出来的格子。
    敞开的拉链外套两片前襟各自往外积分、只经过帽子或下摆连着,内缘的位移对不上:设计稿中心拖进缝里会往回退,
    两边内缘画的是设计稿上重叠的一段,同一个 Logo 两片前襟上各出现一份。这样的商品缝里也算位移、按不变形接过去,
    设计稿像被内搭挡住一截,拖过去一路往前走"""
    cells = (cv2.resize(printable, small_size, interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    r = max(1, round(BRIDGE_RATIO * W / DISP_SCALE))
    padded = cv2.copyMakeBorder(cells, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))[r:-r, r:-r] > 0
    return closed & ~cells.astype(bool)


def reach(printable, gap):
    """拖动时算数的范围(模板尺寸):可印区,加上 gaps 给的缝"""
    return ((printable > 0.5) | cv2.resize(gap.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)).astype(np.float32)


def never_back(offsets_at, printable, inside):
    """设计稿中心从可印区中心沿各方向往外拖,中心在 inside 里时画上去的位置(中心 + 中心处位移)不比走过的最远处往回退。
    offsets_at(warp) 给 Warping 为 warp(0–1)时模板尺寸的每像素位移。中心出了 inside 时画在哪里不算,
    再进来时和之前走过的最远处比;8 位量化本身有抖动,回退超过 DRAG_BACK 像素才算"""
    ys, xs = np.nonzero(printable > 0.5)
    cx, cy = xs.mean(), ys.mean()
    for warp in np.linspace(0, 1, 5):
        offset = offsets_at(warp)
        for angle in np.radians(np.arange(0, 360, 5)):
            d = np.array([np.cos(angle), np.sin(angle)])
            steps = np.arange(int(np.hypot(W, H)))
            px, py = cx + d[0] * steps, cy + d[1] * steps
            keep = (px >= 0) & (py >= 0) & (px <= W - 1) & (py <= H - 1)
            px, py = px[keep], py[keep]
            on = ndimage.map_coordinates(inside, [py, px], order=0) > 0.5
            px, py = px[on], py[on]
            if not px.size:
                continue
            ox, oy = (ndimage.map_coordinates(offset[..., c], [py, px], order=1) for c in (0, 1))
            drawn = (px + ox) * d[0] + (py + oy) * d[1]
            if (np.maximum.accumulate(drawn) - drawn).max() > DRAG_BACK:
                return False
    return True


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


def free(*names):
    """整幅的中间结果用完就放掉。一幅 20–60MB,三层分解 180MB,都留到最后时一个进程要占 3GB 多,同时能跑几个受内存限制。
    只在一部分模板上才有的名字(深色、布料、不是圆柱的才算)没有就跳过"""
    for name in names:
        globals().pop(name, None)


# ---- 1. 源图
photo, source_alpha = config.photo(spec)
H, W = photo.shape[:2]
Image.fromarray(np.round(photo * 255).astype(np.uint8)).save(OUT / 'cover.webp', quality=85)
# 源图里透明度不到 1 的像素不是原始照片(如从竞品底图还原的商品区),第 3 步换背景色时要用
restored = source_alpha < 1
free('source_alpha')

# ---- 2. 分割(run_segment.py 的结果),商品由上层到下层
read_alpha = lambda name: np.asarray(Image.open(WORK / name)).astype(np.float32) / 65535
read_mask = lambda name: np.asarray(Image.open(WORK / name)) > 127
occluders = read_alpha('occluders.png') > 0.5
# 法线(normals.py)给 print.face 挑正对的那一面、surface: cylinder 定镜头俯仰用,先模糊去掉纹理的起伏
normals = cv2.GaussianBlur(cv2.resize(read_layer('normals.png'), (W, H), interpolation=cv2.INTER_LINEAR) * 2 - 1, (0, 0), SIGMA_RATIO * W)
products, prints, part_masks, blocks = [], [], [], []  # 后两个给 surface: cylinder 认商品自己的轮廓
above, above_bin = np.zeros((H, W), np.float32), np.zeros((H, W), bool)  # 上层商品的覆盖度之和、二值范围
for item in spec['products']:
    segmented = read_alpha(f"{item['id']}-alpha.png")
    # 源图里轮廓是台阶的商品(如从竞品底图还原的),抠图模型救不回来:按二值来源处理,平滑轮廓后重出软边
    alpha = soft(segmented > 0.5) if item.get('binaryEdge') else segmented
    # 下层商品在上层下面接着铺,交界处相加为 1;重叠处归上层
    _, (iy, ix) = ndimage.distance_transform_edt(above_bin, return_indices=True)
    product = quantize(np.minimum(alpha[iy, ix], 1 - above))
    for cx, cy, radius in spec.get('cutouts', {}).get(item['key'], []):
        if not all(0 <= value <= 1 for value in (cx, cy, radius)) or radius <= 0:
            raise ValueError(f"{item['id']}:cutouts 的圆心和半径必须是归一化坐标")
        hole = np.zeros((H, W), np.uint8)
        cv2.circle(hole, (round(cx * W), round(cy * H)), round(radius * W), 1, -1)
        product = quantize(product * (1 - soft(hole > 0)))
    products.append(product)
    part_masks.append(read_mask(f"{item['id']}-parts.png"))
    blocks.append(occluders | above_bin)
    prints.append(quantize(print_area(product, part_masks[-1], blocks[-1], item.get('print', {}))))
    if quad := item.get('print', {}).get('quad'):
        if len(quad) != 4 or any(len(point) != 2 or not all(0 <= value <= 1 for value in point) for point in quad):
            raise ValueError(f"{item['id']}:print.quad 必须是四个归一化坐标")
        face = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(face, np.round(np.array(quad) * [W, H]).astype(np.int32), 1)
        prints[-1] = quantize(np.minimum(prints[-1], soft(face > 0)))
    if item.get('print', {}).get('face') and (prints[-1] > 0.5).any():
        prints[-1] = main_face(prints[-1], normals)
    above, above_bin = np.minimum(above + product, 1), above_bin | (segmented > 0.5)
if spec.get('faces') or spec.get('faceLayouts'):
    expanded = config.region_products(spec)
    divided = ([], [], [], [])
    for item, product, printable, parts, blocked in zip(spec['products'], products, prints, part_masks, blocks):
        faces = config.region_faces(spec, item)
        if not faces:
            for target, value in zip(divided, (product, printable, parts, blocked)):
                target.append(value)
            continue
        owned = np.zeros((H, W), bool)
        bounds = {}
        if 'axes' in faces[0]:
            contours, _ = cv2.findContours((product > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            corners = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea))).astype(np.float32)
            sums, differences = corners.sum(axis=1), corners[:, 1] - corners[:, 0]
            top_left, top_right = corners[np.argmin(sums)], corners[np.argmin(differences)]
            bottom_left = corners[np.argmax(differences)]
            across, downward = top_right - top_left, bottom_left - top_left
            xx = np.arange(W, dtype=np.float32)[None, :] - top_left[0]
            yy = np.arange(H, dtype=np.float32)[:, None] - top_left[1]
            grid_x = (xx * across[0] + yy * across[1]) / max(float(across @ across), 1)
            grid_y = (xx * downward[0] + yy * downward[1]) / max(float(downward @ downward), 1)
        elif 'axis' in faces[0]:
            visible = product > 0.5
            for direction in (faces[0]['axis'],):
                if direction == 'x':
                    bounds['x'] = (visible.any(axis=1), np.argmax(visible, axis=1),
                                   W - 1 - np.argmax(visible[:, ::-1], axis=1))
                else:
                    bounds['y'] = (visible.any(axis=0), np.argmax(visible, axis=0),
                                   H - 1 - np.argmax(visible[::-1, :], axis=0))

        def stripe(direction, interval):
            lower, upper = interval
            occupied, lo, hi = bounds[direction]
            start = lo + (hi - lo) * lower
            end = lo + (hi - lo) * upper
            if direction == 'x':
                coordinate = np.arange(W)[None, :]
                return occupied[:, None] & (coordinate >= start[:, None]) & (
                    coordinate < end[:, None] if upper < 1 else coordinate <= end[:, None])
            coordinate = np.arange(H)[:, None]
            return occupied[None, :] & (coordinate >= start[None, :]) & (
                coordinate < end[None, :] if upper < 1 else coordinate <= end[None, :])
        for face in faces:
            if 'axes' in face:
                x0, x1 = face['axes']['x']
                y0, y1 = face['axes']['y']
                face_mask = (grid_x >= x0) & (grid_x < x1 if x1 < 1 else grid_x <= x1) & (
                    grid_y >= y0) & (grid_y < y1 if y1 < 1 else grid_y <= y1)
            elif 'axis' in face:
                face_mask = stripe(face['axis'], face['interval'])
            elif face.get('remainder'):
                face_mask = ~owned
            else:
                polygon = face['polygon']
                if len(polygon) < 3 or any(len(point) != 2 or not all(0 <= value <= 1 for value in point) for point in polygon):
                    raise ValueError(f"{item['id']}:{face['key']} 的 polygon 必须是归一化坐标")
                mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(mask, [np.round(np.array(polygon) * [W - 1, H - 1]).astype(np.int32)], 1)
                face_mask = mask > 0
            face_mask &= ~owned
            owned |= face_mask
            divided[0].append(quantize(product * face_mask))
            divided[1].append(quantize(printable * face_mask))
            divided[2].append(parts & face_mask)
            divided[3].append(blocked)
    products, prints, part_masks, blocks = divided
    spec['products'] = expanded
empty = [item['id'] for item, product in zip(spec['products'], products) if not (product > 0.5).any()]
if empty:
    raise SystemExit(f"分割没分出商品:{', '.join(empty)}(提示词分不出,或源图里没有这件商品)")
blank = [item['id'] for item, printable in zip(spec['products'], prints) if not (printable > 0.5).any()]
if blank:
    raise SystemExit(f"可印区是空的:{', '.join(blank)}(分出的商品太小,或被部件、内缩扣光了)")
coverage = np.minimum(sum(products), 1)
free('segmented', 'alpha', 'above', 'above_bin', 'ix', 'iy', '_')

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
layered = []
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
    # 深色商品照片里光照和固有色分不开,拟合的光照比例 s 不可信:明暗起伏(对数)压到 DARK_CONTRAST,亮部对齐到白布料在正常光下的亮度。
    # 算出的光照常偏暗;固有色接近 0 时(黑色保温杯)s 又会被拟合到几十倍,明暗偏亮十倍,浅色全截成纯色,所以偏亮偏暗都对齐。
    # 深色照片拆出的光照起伏比浅色的大得多,原样保留时选白色大半发灰(实心部分中位亮度 sRGB 约 0.45–0.65,浅色商品约 0.83)。
    # 反光强的深色商品照片全被残差解释,s 会是 0 或负数,所以按光照层本身的起伏缩放,不乘 s
    if dark:
        lit = blur(luma(layers['shading']))
        lit = (lit / max(float(np.percentile(lit[solid], DARK_BRIGHT * 100)), 1e-4)) ** DARK_CONTRAST
        light = lit * texture
        diffuse = light * WHITE_ALBEDO / max(float(np.percentile(light[solid], DARK_BRIGHT * 100)), 1e-4)
    sheen = luma(max(r, 0) * layers['residual']) * texture
    layered.append((product, solid, diffuse, sheen, texture, dark))
free('layers', 'I', 'terms', 'weight', 'inside', 'lit', 'light')
# 一张图里的几件商品是同一种料子(正背面、上衣和裤子、一对杯子),照片里一件迎光一件背光、深色各自抬亮后亮度也不同,
# 选同一个颜色时会一深一浅:按实心部分 明暗 + 光泽 的中位数整体缩放到最亮的那件,起伏不变。
# 缩放后每件的亮部不超过 WHITE_CAP:浅色、彩色商品的固有色估低时光照偏亮,迎光的那件被提亮时也会,选白色大片截成纯白
level = [float(np.median((diffuse + sheen)[solid])) for _, solid, diffuse, sheen, _, _ in layered]


def look_spread(white, solid):
    """选白色时实心部分大块明暗的跨度:3 倍纹理半径内平均后亮度(对数)的 p90 − p10"""
    weight = solid.astype(np.float32)
    mean = cv2.GaussianBlur(white * weight, (0, 0), 3 * radius) / np.maximum(cv2.GaussianBlur(weight, (0, 0), 3 * radius), 1e-6)
    values = np.log(np.maximum(mean[solid], 1e-4))
    return float(np.percentile(values, 90) - np.percentile(values, 10))


contrasts = []
for index, ((product, solid, diffuse, sheen, texture, dark), own) in enumerate(zip(layered, level)):
    bright = float(np.percentile((diffuse + sheen)[solid], WHITE_CAP_AT))
    k = min(max(level) / max(own, 1e-6), WHITE_CAP / max(bright, 1e-6))
    diffuse, sheen = diffuse * k, sheen * k
    # 浅色布料明暗收到 FABRIC_SPREAD:光照(去掉纹理)按实心部分中位数在对数里乘 C,明暗 + 光泽的中位数对回原值,中间调不变、亮部收、暗部提。
    # 放在亮部封顶之后:放在之前,封顶按收过的亮部整件提亮。光泽、纹理不随 C 变,跨度不和 C 成正比,反复按比例调几次
    contrast = 1.0
    if spec['category'] in FABRIC and not dark and (spread := look_spread(diffuse + sheen, solid)) > FABRIC_SPREAD:
        lit = diffuse / np.maximum(texture, 1e-6)
        middle = float(np.median(lit[solid]))
        before = float(np.median((diffuse + sheen)[solid]))
        start = spread
        for _ in range(6):
            contrast = max(FABRIC_CONTRAST_MIN, contrast * FABRIC_SPREAD / spread)
            softer = middle * (np.maximum(lit, 1e-6) / middle) ** contrast * texture
            spread = look_spread(softer + sheen, solid)
            if contrast == FABRIC_CONTRAST_MIN or abs(spread - FABRIC_SPREAD) < 0.005:
                break
        gain = before / max(float(np.median((softer + sheen)[solid])), 1e-6)
        diffuse, sheen = softer * gain, sheen * gain
        print(f"{spec['products'][index]['id']} 明暗跨度 {start:.2f} 收到 {spread:.2f}(光照起伏保留 {contrast:.2f})")
    contrasts.append(round(contrast, 3))
    # 白色下限放在亮部封顶、收明暗之后:整件按比例提,明暗起伏不变;深色照片已对齐到白布料的亮度,不再提
    if spec['category'] in WHITE_FLOOR_KINDS and not dark:
        bright = float(np.percentile((diffuse + sheen)[solid], WHITE_CAP_AT))
        if bright < WHITE_FLOOR:
            diffuse, sheen = diffuse * (WHITE_FLOOR / bright), sheen * (WHITE_FLOOR / bright)
            print(f"{spec['products'][index]['id']} 白色 p{WHITE_CAP_AT} {bright:.2f} 提到 {WHITE_FLOOR}")
    # 光泽的底子:实心部分中位数以上的才是高光。底子超过 GLOSS_BASE 的部分从光泽挪进明暗,选白色亮度不变,局部高光照旧
    base_sheen = float(np.median(sheen[solid]))
    if base_sheen > GLOSS_BASE / GLOSS_KEEP:
        moved = np.minimum(sheen, base_sheen - GLOSS_BASE / GLOSS_KEEP)
        diffuse, sheen = diffuse + moved, sheen - moved
        print(f"{spec['products'][index]['id']} 光泽底子 {GLOSS_KEEP * base_sheen:.3f} 收到 {GLOSS_BASE}")
    # 边缘像素在照片里混着商品原色:取商品实心部分的值往外延伸,区外那份换成背景色
    for total, layer in ((shade, diffuse), (gloss, sheen)):
        total += product * np.where(solid, layer, local_mean(np.repeat(layer[..., None], 3, -1), solid.astype(np.float32), 2)[..., 0])
free('layered', 'solid', 'diffuse', 'sheen', 'texture', 'lit', 'softer', 'total', 'layer')
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
free('photo', 'photo_linear', 'restored', 'dirty', 'behind', 'background', 'premultiplied', 'rgb', 'base', 'shade', 'gloss', '_')

# 模板墙的 CSS 叠图用的底图:浏览器只能按 sRGB 值普通叠加,由落盘的 base 换算,选白色、纯黑时和线性光结果一致,中间颜色按 sRGB 插值
# 整张不透明时 webp 不存透明度通道,读回来要补上
shipped = np.asarray(Image.open(OUT / 'base.webp').convert('RGBA')).astype(np.float32) / 255
on_white = to_srgb(shade_gain * (1 - shipped[..., 3:]) + to_linear(shipped[..., :3]) * shipped[..., 3:])
on_black = to_srgb(to_linear(shipped[..., :3]) * shipped[..., 3:])
free('shipped')


def css_ends(css, inside):
    """CSS 底图按 sRGB 普通叠在白色、纯黑上,和线性光的结果比,商品区内的平均差和 99% 分位(0–255)。
    一个透明度管不了三个通道各自的差,背后颜色带色的边缘、发丝处会差得多一些,只占极少数像素"""
    a, rgb = css[..., 3:], css[..., :3]
    inside = inside[..., None].repeat(3, axis=2)
    diffs = [np.abs(over - target)[inside] * 255 for over, target in ((1 - a + rgb * a, on_white), (rgb * a, on_black))]
    return [(float(d.mean()), float(np.percentile(d, 99))) for d in diffs]


def save_css(css):
    """存成 webp,返回读回来的样子"""
    Image.fromarray(np.round(css * 255).astype(np.uint8), 'RGBA').save(OUT / 'base-css.webp', quality=85, alpha_quality=100)
    return np.asarray(Image.open(OUT / 'base-css.webp').convert('RGBA')).astype(np.float32) / 255


spread = on_white - on_black   # 每个通道要的 1 − 透明度
css_alpha = np.clip(1 - spread.mean(axis=2, keepdims=True), 1 / 255, 1)
css_split = not all(mean <= CSS_MEAN and p99 <= CSS_P99
                    for mean, p99 in css_ends(save_css(np.dstack([np.clip(on_black / css_alpha, 0, 1), css_alpha])), coverage > 0))
if css_split:
    # 边缘带色多的(挂绳、滑板)三个通道要的透明度差得远,黑底照原样、差全落在白底上会超出校验。
    # 改取三个通道的中点作透明度,白底多出的差一半挪到黑底,两边各差至多三个通道之差的四分之一
    css_alpha = np.clip(1 - (spread.max(axis=2, keepdims=True) + spread.min(axis=2, keepdims=True)) / 2, 1 / 255, 1)
    save_css(np.dstack([np.clip((on_black - (1 - css_alpha - spread) / 2) / css_alpha, 0, 1), css_alpha]))
    print('base-css: 白底、黑底的差各留一半')
free('spread', 'css_alpha', 'coverage')

# ---- 4. 流向图:每件商品两张,R / G = x / y 方向的位移,1/4 尺寸
# - 大曲面(固定):按 MoGe-2 的 3D 坐标,设计稿像贴在曲面上,侧面压缩、近处略放大;
# - 褶皱(Warping 100% 时):起伏取法线的 z 分量(表面正对镜头的程度),位移沿它的坡度,褶皱横着就上下压缩、
#   竖着就左右压缩,和照片的光线方向无关;系数按每件商品自己的褶皱定,Warping 0–100% 时两层合起来超限的地方
#   局部减弱褶皱这层,深褶处不翻折。配置写 surface: cylinder 的(罐、瓶、杯、蜡烛)是硬的,褶皱层不位移:
#   法线 z 分量在它们身上的坡度是圆柱面本身转向侧面,大曲面层已经算过,再加一遍侧面多压、横线的弧被拉平。
# 位移会沿曲面累积,渲染端减去设计稿中心处的位移,中心不动;设计稿中心可以拖出商品,商品外按 extend_outside 铺平。
# 交界两侧的起伏不连续,每件商品单独一张、各自向外延伸,缩小和插值时才不混在一起
facing = cv2.resize(read_layer('normals.png'), (W, H), interpolation=cv2.INTER_LINEAR)[..., 2] * 2 - 1
points = config.load_points(WORK)
intrinsics = np.array(json.loads((WORK / 'geometry.json').read_text())['intrinsics']) * [[W], [H], [1]]  # 存的按宽高归一
focal = intrinsics[0, 0]
sigma = SIGMA_RATIO * W
small_size = (W // DISP_SCALE, H // DISP_SCALE)
to_px = W * W / DISP_SCALE  # 系数 × to_px:缩小后每格的坡度 → 模板像素位移
low, high = STRETCH_RANGE[0] + CAP_MARGIN[0], STRETCH_RANGE[1] - CAP_MARGIN[1]
carriers = [carrier(product, printable) for product, printable in zip(products, prints)]
bridges = [gaps(printable, small_size) for printable in prints]


def build(bridges):
    """各件商品缩小后的大曲面层、褶皱层。bridges 是各件要接上的缝(gaps 的格子,不接的全 False)。
    返回 (大曲面层, 褶皱层, 贴法, 按平面贴的框转几度, 守住范围, 大曲面层收到几成, 褶皱系数, 说明)"""
    # 配置写 surface: plane 的先按平面贴,surface: cylinder 的先按圆柱贴;真弯的、透视太陡的、轮廓不像圆柱的照旧按曲面
    surfaces, fits, turns, notes = [], [], [], []
    for item, product, printable, parts, blocked, gap in zip(spec['products'], carriers, prints, part_masks, blocks, bridges):
        kind = item.get('surface')
        if kind == 'plane':
            reference = item.get('planeOutline', 'print')
            if reference not in ('print', 'product'):
                raise ValueError(f"{item['id']}:planeOutline 必须是 print 或 product")
            shaped, note, turn = plane_flow(points, intrinsics, product, printable, small_size, gap,
                                            product if reference == 'product' else None)
        elif kind == 'cylinder':
            (shaped, note), turn = cylinder_flow(product, printable, parts, blocked, normals, intrinsics, small_size, gap,
                                                  item.get('cylinderFrontRim')), None
        else:
            shaped, note, turn = None, '', None
        if note:
            notes.append(f"{item['id']} 大曲面层:{note}")
        surfaces.append(surface_flow(points, focal, product, small_size, SURFACE_RATIO * W, gap) if shaped is None else shaped)
        fits.append('curved' if shaped is None else kind)
        turns.append(turn)
    # 存图时按这些格子定 8 位的范围:沾到商品的格子再往外两格,双线性放大时商品边上的像素会插到外面一格
    # 可印区几块之间的缝也守住:位移是按不变形接过去的,不能再换成最近值
    guards = [cv2.dilate((cv2.resize(product, small_size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8),
                         np.ones((5, 5), np.uint8)).astype(bool) | gap for product, gap in zip(carriers, bridges)]
    # 和平面、圆柱一样,可印区里存成 8 位后超出校验范围的面积至多 STRETCH_OUTLIER 的一半(另一半留给褶皱层)。
    # 按曲面算的之前没量过:位移幅度 200px 以上的(摊开的书页、毯子、裤腿)一级就有 1.6px 以上,放大回来相邻格的前进量差到 0.4,
    # 压缩到 0.8 上下的地方就出现一圈圈不到 0.5 的细条纹。几件商品存在一张图里时范围按位移最大的那件定,
    # 平放、透视很陡的那张撑大了范围,立着的那张按自己的范围量是过的,存进去也会出条纹。
    # 超了就把大曲面层整体往不变形收,位移依次乘 0.9、0.8…至多收到 SURFACE_KEEP_MIN:按自己的范围也超的收自己,
    # 只是范围被撑大的收位移最大的那件。侧面压缩、近处放大、透视跟着减弱,一级对应的位移也跟着变小
    keeps = [1.0] * len(surfaces)
    measured = [index for index, printable in enumerate(prints) if (printable > 0.5).any()]
    shaped_surfaces = list(surfaces)
    while True:
        limits = [stored_limit(flow, guard) for flow, guard in zip(surfaces, guards)]
        beyond = {index: sum(stored_beyond(surfaces[index], guards[index], prints[index] > 0.5, max(limits))) for index in measured}
        weaken = {index if sum(stored_beyond(surfaces[index], guards[index], prints[index] > 0.5)) > STRETCH_OUTLIER / 2 else int(np.argmax(limits))
                  for index in measured if beyond[index] > STRETCH_OUTLIER / 2}
        weaken = [index for index in weaken if keeps[index] > SURFACE_KEEP_MIN + 1e-6]
        if not weaken:
            break
        for index in weaken:
            keeps[index] = round(keeps[index] - 0.1, 1)
            surfaces[index] = shaped_surfaces[index] * keeps[index]
    for index, keep in enumerate(keeps):
        if keep < 1:
            still = f',可印区还有 {beyond[index]:.1%} 超出' if beyond.get(index, 0) > STRETCH_OUTLIER / 2 else ''
            notes.append(f"{spec['products'][index]['id']} 大曲面层:存成 8 位后超出校验范围,位移收到 {keep:.0%}{still}")
    # 大曲面的 8 位量化让相邻格的位移差至多多出一级,两层叠起来减弱时再扣掉这部分的 SLACK_SHARE:
    # 整级只在两格取整刚好相反时才出现,扣满一级时大曲面层 200px 以上的商品上下限只剩约 1.0–1.9,褶皱几乎全被减掉
    slack = SLACK_SHARE * 2 * max(np.abs(surface).max() for surface in surfaces) / 255 / DISP_SCALE
    folds, gains = [], []
    for index, (item, product, printable) in enumerate(zip(spec['products'], carriers, prints)):
        if item.get('surface') == 'cylinder':
            gains.append(0.0)
            surfaces[index] = extend_outside(surfaces[index], guards[index])
            folds.append(np.zeros((small_size[1], small_size[0], 2), np.float32))
            continue
        inside = product > 0.5
        weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), sigma)
        blurred = cv2.GaussianBlur(facing * inside, (0, 0), sigma) / np.maximum(weight, 1e-6)
        # 区外取最近的区内值再缩小,边上的坡度不会因为区外突变而跳
        _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
        height = cv2.resize(blurred[iy, ix], small_size, interpolation=cv2.INTER_AREA)
        gy, gx = np.gradient(cv2.GaussianBlur(height, (0, 0), SLOPE_BLUR))
        # 沾到商品的格子再往外两格都守住范围:双线性放大时商品边上的像素会插到外面一格
        guarded = guards[index]
        cells = cv2.resize(printable if (printable > 0.5).any() else product, small_size, interpolation=cv2.INTER_AREA) > 0.5
        gain = min(SLOPE_GAIN_MAX, float(np.percentile(max_gain(gx, gy, low, high)[cells], CAP_AREA * 100)) / to_px)
        gains.append(gain)
        ox, oy = cap(gain * to_px * gx, gain * to_px * gy, guarded, low + slack, high - slack,
                     (surfaces[index][..., 0], surfaces[index][..., 1]))
        # 守住范围的格子原样留着(商品边上的像素插值要用到它们),从它们往外铺
        surfaces[index] = extend_outside(surfaces[index], guarded)
        folds.append(extend_outside(np.dstack([ox, oy]).astype(np.float32), guarded))
    return surfaces, folds, fits, turns, guards, keeps, gains, notes


# 先不接缝算一遍;可印区几块之间有缝的,模拟设计稿中心拖过缝(同自动校验),往回退的那件把缝按不变形接上再算一遍。
# 缝两边接得上的不动:可印区边上有凹口的模板很多(挂绳、卡片、裤腿),多数拖过去本来就不往回退
built = build([np.zeros_like(gap) for gap in bridges])
joins = [bool(gap.any()) and not never_back(lambda warp, s=surface, f=fold: cv2.resize(s + warp * f, (W, H), interpolation=cv2.INTER_LINEAR),
                                            printable, reach(printable, gap))
         for surface, fold, printable, gap in zip(built[0], built[1], prints, bridges)]
if any(joins):
    built = build([gap if join else np.zeros_like(gap) for gap, join in zip(bridges, joins)])
surfaces, folds, fits, turns, guards, keeps, gains, notes = built
for note in notes:
    print(note)
for item, join in zip(spec['products'], joins):
    if join:
        print(f"{item['id']} 可印区几块之间的缝两边位移对不上,拖过去会往回退,缝里按不变形接上")
free('points', 'normals', 'built')
free('facing', 'carriers', 'weight', 'blurred', 'inside', 'ix', 'iy', '_')


def save_flows(flows, prefix):
    """通道值 0 / 255 对应 ∓range 像素,按守住范围的格子里的最大位移定,8 位精度用满;更外面的格子用不到,截掉。
    每张先整体减去守住范围的格子里各方向最大、最小位移的中点:渲染端减去设计稿中心处的位移,整体平移不改变结果,
    范围小了每一级对应的位移就小,直线上的台阶跟着变小。每件商品都不位移时(褶皱层,全是圆柱)范围记 0.1"""
    flows = [flow - (flow[g].max(axis=0) + flow[g].min(axis=0)) / 2 for flow, g in zip(flows, guards)]
    range_px = max(0.1, round(float(np.ceil(max(np.abs(f[g]).max() for f, g in zip(flows, guards)) * 10)) / 10, 1))
    for index, flow in enumerate(flows, start=1):
        rgb = np.dstack([0.5 + np.clip(flow, -range_px, range_px) / (2 * range_px), np.full(flow.shape[:2], 0.5, np.float32)])
        Image.fromarray(np.round(rgb * 255).astype(np.uint8), 'RGB').save(OUT / f'{prefix}-{index}.png', optimize=True)
    return range_px


surface_range = save_flows(surfaces, 'surface')
folds_range = save_flows(folds, 'folds')

for index, (product, printable) in enumerate(zip(products, prints), start=1):
    save_mask(product, f'product-{index}.png')
    save_mask(printable, f'print-{index}.png')

# ---- 5. 配置:默认 Logo 框由 logo.py 算(上衣放在胸口,正面带拉链、扣子的放左胸,平面印刷品铺满可印区,跟着可印区摆放的方向转)
regions = []
# 和单独跑 logo.py 时一样读回封面,问 Gemini 的图相同,回答对得上
settings = logo.settle(spec, np.asarray(Image.open(OUT / 'cover.webp').convert('RGB')), [printable > 0.5 for printable in prints], WORK)
for index, (printable, item, logo_settings) in enumerate(zip(prints, spec['products'], settings), start=1):
    regions.append({
        'key': item['key'],
        'name': item['name'],
        'product': f'product-{index}.png',
        'print': f'print-{index}.png',
        'surface': f'surface-{index}.png',
        'folds': f'folds-{index}.png',
        'recolorable': item.get('recolorable', True),
        'logo': logo.box(printable > 0.5, {**logo_settings, 'plane': fits[index - 1] == 'plane', 'turn': turns[index - 1]}),
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
    'pipeline': {'whiteAlbedo': WHITE_ALBEDO, 'darkPhoto': DARK_PHOTO, 'darkBright': DARK_BRIGHT, 'darkTexture': DARK_TEXTURE, 'darkContrast': DARK_CONTRAST,
                 'whiteCap': WHITE_CAP, 'whiteCapAt': WHITE_CAP_AT, 'fabricSpread': FABRIC_SPREAD, 'fabricContrastMin': FABRIC_CONTRAST_MIN, 'fabricContrast': contrasts, 'whiteFloor': WHITE_FLOOR, 'glossBase': GLOSS_BASE,
                 'glossKeep': GLOSS_KEEP, 'textureRadius': round(float(radius), 1), 'textureClip': list(TEXTURE_CLIP),
                 'blurSigma': round(float(sigma), 1),
                 'displacementScale': 1 / DISP_SCALE, 'slopeBlur': SLOPE_BLUR, 'stretchRange': list(STRETCH_RANGE),
                 'capArea': CAP_AREA, 'slackShare': SLACK_SHARE, 'slopeGainMax': SLOPE_GAIN_MAX, 'slopeGain': [float(f'{g:.3g}') for g in gains],
                 'surfaceSigma': round(SURFACE_RATIO * W, 1), 'surfaceRange': list(SURFACE_RANGE),
                 'surfaceFit': fits, **({'planeTurn': turns} if 'plane' in fits else {}), 'surfaceKeep': keeps, 'surfaceKeepMin': SURFACE_KEEP_MIN, 'planeBend': PLANE_BEND, 'cylinderPerspective': PERSPECTIVE,
                 'baseCssSplit': css_split},
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
p = [read(f'product-{i}.png')[..., 3].copy() for i in numbers]  # 只留透明度这一层,不带着整张 RGBA
q = [read(f'print-{i}.png')[..., 3].copy() for i in numbers]
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
    """设计稿中心在可印区里或可印区几块之间的缝里往外拖不往回退(见 never_back)。
    缝两边的位移对不上时,往回退的那一段设计稿在两边各画一份;中心出了这个范围(拖到商品外、经过可印区边上的遮挡物)不算"""
    return never_back(lambda warp: offsets(surface, fold, warp), printable, reach(printable, gaps(printable, small_size)))


def within_stretch(surface, fold, printable):
    """Warping 0–100% 时可印区内(设计稿只画在这里)各方向每像素前进量超出范围的面积不到 STRETCH_OUTLIER"""
    inside = printable > 0.5
    for warp in np.linspace(0, 1, 5):
        offset = offsets(surface, fold, warp)
        smin, smax = stretch_range(*jacobian(offset[..., 0], offset[..., 1], 1))
        if ((smin[inside] < STRETCH_RANGE[0]) | (smax[inside] > STRETCH_RANGE[1])).mean() > STRETCH_OUTLIER:
            return False
    return True


def lines_stay_straight(surface, printable):
    """按平面贴的:可印区里屏幕上的直线段映到设计稿后还是直的。线段两端和整段都在可印区、长过可印区边长(面积开方)的一半,
    取 BOW_LINES 条(随机数固定,每次取的一样),量设计稿上离两端连线最远处占长度多少,中位数至多 PLANE_BOW。
    单应把直线映成直线,落盘的图只多出 8 位量化的抖动;只记拉伸、不记方向的贴法会把斜着拍的平面上的直线弯成弧。
    可印区太细、取不够线段的不量"""
    flow = (surface[..., :2] - 0.5) * 2 * surface_range
    cells = cv2.erode((cv2.resize(printable, small_size, interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.nonzero(cells)
    if not ys.size:
        return True
    along, bows = np.linspace(0, 1, 64), []
    for i, j in np.random.default_rng(0).integers(ys.size, size=(BOW_LINES * 40, 2)):
        px, py = xs[i] + (xs[j] - xs[i]) * along, ys[i] + (ys[j] - ys[i]) * along
        if np.hypot(xs[j] - xs[i], ys[j] - ys[i]) < 0.5 * np.sqrt(ys.size) or not cells[py.round().astype(int), px.round().astype(int)].all():
            continue
        dx, dy = (p * DISP_SCALE + ndimage.map_coordinates(flow[..., c], [py, px], order=1) for c, p in ((0, px), (1, py)))
        chord = np.array([dx[-1] - dx[0], dy[-1] - dy[0]])
        bows.append(np.abs((dy - dy[0]) * chord[0] - (dx - dx[0]) * chord[1]).max() / max(float(chord @ chord), 1e-9))
        if len(bows) == BOW_LINES:
            break
    return len(bows) < BOW_LINES // 4 or float(np.median(bows)) <= PLANE_BOW


def joined_share(printable, product, rule):
    """可印区最大一块占的比例。遮挡物(如交握的手)横穿可印区时,设计稿在它下面照样铺着,隔开的两边算一块。
    print.bridge:敞开的拉链外套,可印区本来就是左右两片前襟,中间的拉链、内搭当遮挡物:
    带可印区的那几块商品合起来的凸包都算连着"""
    visible = printable > 0.5
    joined = visible | occluders
    if rule.get('bridge') and visible.any():
        solid = product > 0.5
        labels, _ = ndimage.label(solid)
        hull = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero((solid & np.isin(labels, np.unique(labels[visible & solid]))).astype(np.uint8))), 1)
        joined |= hull > 0
    labels, count = ndimage.label(joined)
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
    # 防换算写错;webp 有损压缩(质量 85)和边缘本身就会差几个色阶,明暗倍数大的商品平均差到 1 以上,
    # 边缘带色多的(挂绳、酒瓶、滑板)99% 分位到 13–14,公式错了会差几十
    f'base-css matches on white / black (mean ≤ {CSS_MEAN}, 99% ≤ {CSS_P99} of 255)': all(mean <= CSS_MEAN and p99 <= CSS_P99 for mean, p99 in css_ends(css_out, union)),
    'base-css sizes equal': css_out.shape == base_out.shape,
    'design centre dragged outward never runs back inside print or gaps between print pieces': all(drag_never_back(ss, ff, qq) for ss, ff, qq in zip(s, f, q)),
    f'stretch {STRETCH_RANGE[0]}–{STRETCH_RANGE[1]} at Warping 0–100% (outside ≤ {STRETCH_OUTLIER:.1%})': all(within_stretch(ss, ff, qq) for ss, ff, qq in zip(s, f, q)),
    # 下面两项查各模型算出来的东西用得对:按平面贴的保住了 3D 坐标的方向,直线还直;圆柱的转向只由大曲面层算,褶皱层不位移
    f'plane fits keep straight lines straight (median bow ≤ {PLANE_BOW:.1%})': all(lines_stay_straight(ss, qq) for ss, qq, fit in zip(s, q, fits) if fit == 'plane'),
    'cylinders have no folds displacement': all(bool((ff[..., :2] == 128 / 255).all()) for ff, item in zip(f, spec['products']) if item.get('surface') == 'cylinder'),
    f'products ≥ {PRODUCT_MIN:.1%} of frame': all((pp > 0.5).mean() >= PRODUCT_MIN for pp in p),
    'prints not empty': all((qq > 0.5).any() for qq in q),
    'print largest component ≥ 80%': all(joined_share(qq, pp, item.get('print', {})) >= 0.8 for pp, qq, item in zip(p, q, spec['products'])),
    logo.CHECK: logo.centered(q, spec['products'], [r['logo'] for r in regions]),
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
print('base-css vs linear on white / black (mean, 99%, of 255): ' + ' / '.join(f'{m:.2f}, {q:.1f}' for m, q in css_ends(css_out, union)))
for f in sorted(OUT.iterdir()):
    print(f'{f.stat().st_size / 1024:8.1f} KB  {f.name}')
