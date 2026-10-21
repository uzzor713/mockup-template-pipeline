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
GAIN = 1.0  # 起伏图增益
DISP_SCALE = 4  # 起伏图缩小倍数:按块求平均,抹掉 8 位台阶和纹理噪点;使用时双线性放大
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
# 竞品的商品边缘是二值台阶,原图里也一样,只能平滑轮廓重出软边
tee_bin = base_src[..., 3] < 1
tee = quantize(soft(tee_bin))
print_src = np.asarray(Image.open(SRC / 'tee-print.webp').convert('RGBA')).astype(np.float32) / 255
# 源文件是有损 webp,白色只到 246–255
print_tee = np.minimum(np.clip((luma(print_src[..., :3]) - 0.03) / 0.93, 0, 1) * print_src[..., 3], tee)

jeans_sam = np.asarray(Image.open(ROOT / 'work/jeans-mask.png').convert('L')) > 127
jeans_bin = ndimage.binary_fill_holes(largest(jeans_sam)) & ~tee_bin
band_px = max(1, round(BAND_RATIO * W))
jeans = refine(jeans_bin, photo, band_px, max(1, band_px // 3))
# 牛仔裤在 T 恤下面接着铺(取最近的 T 恤外像素),交界处两者相加为 1;重叠处归上层的 T 恤
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
# 边缘像素是商品和背景的混色:亮度取商品自身的颜色,区外那份取背景色,否则改色后边缘留一圈原色
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

# ---- 4. 起伏图:每件商品一个通道(R / G / B = 第 1–3 件),1/4 尺寸
# 各件商品的起伏在交界处不连续,合在一层里缩小或插值会互相串;分层后每层各自向外延伸,在交界处是连续的
sigma = SIGMA_RATIO * W
layers = []
for product in products:
    inside = product > 0.5
    weight = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), sigma)
    blurred = cv2.GaussianBlur(L_norm * inside, (0, 0), sigma) / np.maximum(weight, 1e-6)
    value = np.clip(0.5 + GAIN * (blurred - blurred[inside].mean()), 0, 1)
    # 区外 3σ 以内取最近的区内值,再往外为 0.5
    distance, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    layers.append(np.where(distance <= 3 * sigma, value[iy, ix], 0.5).astype(np.float32))
layers += [np.full((H, W), 0.5, np.float32)] * (3 - len(layers))
small_size = (W // DISP_SCALE, H // DISP_SCALE)
height_map = np.dstack([cv2.resize(layer, small_size, interpolation=cv2.INTER_AREA) for layer in layers])
Image.fromarray(np.round(height_map * 255).astype(np.uint8), 'RGB').save(OUT / 'displacement.png', optimize=True)

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
        'recolorable': True,
        'logo': {'cx': round(float(cx) / W, 4), 'cy': round(float(cy) / H, 4), 'width': round(float(size) / W, 4),
                 'height': round(float(size) / H, 4), 'rotation': 0},
    })
template = {
    'id': 'sample-tee-jeans',
    'version': 1,
    'size': {'width': W, 'height': H},
    'category': 't-shirt',
    'files': {'cover': 'cover.webp', 'base': 'base.webp', 'displacement': 'displacement.png'},
    'regions': regions,
    'pipeline': {'luminancePercentile': PERCENTILE, 'beta': BETA, 'blurSigma': round(float(sigma), 1),
                 'displacementGain': GAIN, 'displacementScale': 1 / DISP_SCALE},
}
(OUT / 'template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2))

# ---- 6. 自动校验(读回落盘的文件)
def read(name):
    return np.asarray(Image.open(OUT / name)).astype(np.float32) / 255

base_out = read('base.webp')
disp_out = read('displacement.png')
p = [read(f'product-{i}.png')[..., 3] for i in (1, 2)]
q = [read(f'print-{i}.png')[..., 3] for i in (1, 2)]
union = (p[0] > 0) | (p[1] > 0)


def neutral_far(channel, product):
    """离这件商品 3σ 以外的格子全是 0.5;没有对应商品的通道整张是 0.5"""
    cells = np.ones(disp_out.shape[:2], bool)
    if product is not None:
        far = ndimage.distance_transform_edt(~(product > 0.5)) > 3 * sigma + DISP_SCALE
        cells = cv2.resize(far.astype(np.float32), small_size, interpolation=cv2.INTER_AREA) == 1
    return bool((np.abs(disp_out[..., channel][cells] - 0.5) <= 1 / 255).all())


checks = {
    'sizes equal': all(x.shape[:2] == (H, W) for x in [base_out, *p, *q]),
    'displacement 1/4 size, 3 channels': disp_out.shape == (H // DISP_SCALE, W // DISP_SCALE, 3),
    'aspect 4:5': W * 5 == H * 4,
    'products overlap < 0.1%': float(((p[0] > 0.5) & (p[1] > 0.5)).sum() / max(1, (p[1] > 0.5).sum())) < 0.001,
    'print outside product < 0.1%': all(float(((qq > 0.5) & (pp < 0.5)).sum() / max(1, (qq > 0.5).sum())) < 0.001 for pp, qq in zip(p, q)),
    'base opaque outside products': bool((base_out[..., 3][~union] == 1).all()),
    'base matches formula (≤ 2/255)': bool(np.abs(base_out[..., 3] - alpha).max() <= 2 / 255 + 1e-6),
    'displacement neutral far outside': all(neutral_far(i, p[i] if i < len(p) else None) for i in range(3)),
    'prints not empty': all((qq > 0.5).any() for qq in q),
    'print largest component ≥ 80%': all(largest(qq > 0.5).sum() / (qq > 0.5).sum() >= 0.8 for qq in q),
    'logo center inside print': all(q[i][int(r['logo']['cy'] * H), int(r['logo']['cx'] * W)] > 0.5 for i, r in enumerate(regions)),
}
(OUT / 'checks.json').write_text(json.dumps(checks, indent=2))
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
for f in sorted(OUT.iterdir()):
    print(f'{f.stat().st_size / 1024:8.1f} KB  {f.name}')
