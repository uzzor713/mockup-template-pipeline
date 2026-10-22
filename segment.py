"""
分割:SAM 2.1 按外接框和提示点分出遮挡物(包带和包)、T 恤、两只袖子和牛仔裤,ViTMatte 在商品和背景、遮挡物的交界求透明度。
存 work/tee-alpha.png、jeans-alpha.png(16 位)、sleeves.png、occluders.png,给 pipeline.py 用。
另存对比图:work/tee-compare.png(T 恤:我们的分割和竞品模板自带的商品区)、work/jeans-compare.png(牛仔裤:颜色加形态学和 SAM)。
用法:.venv/bin/python -I segment.py
"""
import sys, time
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).parent
WORK = ROOT / 'work'
WORK.mkdir(exist_ok=True)
rgba = np.asarray(Image.open(ROOT / 'sources/tee-base.webp').convert('RGBA')).astype(np.float32) / 255
photo = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
photo8 = (photo * 255).round().astype(np.uint8)
reference_tee = rgba[..., 3] < 1  # 竞品模板自带的 T 恤商品区,只用来对比
H, W = reference_tee.shape
disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
grow = lambda mask, r: cv2.dilate(mask.astype(np.uint8), disk(r)) > 0
shrink = lambda mask, r: cv2.erode(mask.astype(np.uint8), disk(r)) > 0


def largest(binary):
    labels, n = ndimage.label(binary)
    return labels == (1 + int(np.argmax(ndimage.sum(binary, labels, range(1, n + 1))))) if n else binary


def drop_specks(binary, ratio=0.01):
    """去掉面积不到整体 ratio 的碎块"""
    labels, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, labels, range(1, n + 1))
    return np.isin(labels, 1 + np.nonzero(sizes >= ratio * binary.sum())[0])


from transformers import Sam2Model, Sam2Processor
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'facebook/sam2.1-hiera-base-plus'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()
processor = Sam2Processor.from_pretrained(MODEL)
model = Sam2Model.from_pretrained(MODEL).to(device).eval()
t1 = time.time()


def sam(box, points):
    """外接框 + 提示点([x, y, 1 正 / 0 负])→ 每像素的前景概率"""
    inputs = processor(images=Image.fromarray(photo8), input_boxes=[[box]],
                       input_points=[[[[x, y] for x, y, _ in points]]], input_labels=[[[l for *_, l in points]]],
                       return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    logits = processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu(), binarize=False)[0]
    return torch.sigmoid(logits.squeeze().float()).numpy()


# 遮挡物:包带上 4 个点、包身上 3 个点
occluders = sam([500, 270, 655, 860], [[625, 300, 1], [615, 400, 1], [607, 500, 1], [600, 575, 1],
                                       [570, 700, 1], [560, 800, 1], [620, 650, 1],
                                       [700, 500, 0], [470, 620, 0], [800, 300, 0]]) > 0.5
# T 恤:包带窄,在它上面放负点会把旁边的 T 恤一起切掉,所以不放,分完再减去遮挡物。
# 包带和包把左袖隔成单独一块,只去碎块,不取最大连通块
tee = sam([395, 185, 1185, 1040], [[820, 500, 1], [750, 800, 1], [900, 650, 1], [470, 600, 1], [1080, 550, 1],
                                   [870, 230, 1], [660, 950, 1], [670, 650, 1], [780, 1000, 0], [950, 960, 0]]) > 0.5
tee = drop_specks(tee & ~occluders)
# 袖子:各自一个框,正点在袖子上,负点在躯干和包带上;左袖的落肩接缝从 (513, 447) 斜到包带,接缝以上是躯干
sleeves = (sam([400, 330, 600, 730], [[470, 600, 1], [520, 580, 1], [440, 660, 1],
                                      [560, 400, 0], [540, 360, 0], [700, 500, 0], [612, 450, 0]]) > 0.5) \
    | (sam([970, 360, 1185, 730], [[1080, 550, 1], [1030, 450, 1], [850, 500, 0]]) > 0.5)
# 袖子边上有伸进躯干的细长毛刺,按宽度的 0.5% 做开运算去掉
sleeves = (cv2.morphologyEx(ndimage.binary_fill_holes(sleeves).astype(np.uint8), cv2.MORPH_OPEN, disk(round(0.005 * W))) > 0) & tee

# 牛仔裤。颜色加形态学:牛仔蓝色相阈值 → 填洞 → 开闭运算 → 最大连通块 → GrabCut 修边,结果给 SAM 当外接框
hsv = cv2.cvtColor(photo8, cv2.COLOR_RGB2HSV).astype(np.float32)
h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
seed = (h > 190) & (h < 240) & (s > 0.12) & (v > 0.18) & ~tee
seed = ndimage.binary_closing(seed, iterations=6)
seed = ndimage.binary_fill_holes(seed)
seed = ndimage.binary_opening(seed, iterations=3)
classic = largest(seed)
gc = np.where(classic, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
gc[ndimage.binary_erosion(classic, iterations=12)] = cv2.GC_FGD
gc[~ndimage.binary_dilation(classic, iterations=12)] = cv2.GC_BGD
gc[tee] = cv2.GC_BGD
cv2.grabCut(cv2.cvtColor(photo8, cv2.COLOR_RGB2BGR), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
classic = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & ~tee
# SAM:正提示点在两条裤腿上,负提示点在鞋面和 T 恤下摆
ys, xs = np.nonzero(classic)
box = [int(xs.min()) - 10, int(ys.min()) - 10, int(xs.max()) + 10, int(ys.max()) + 10]
jeans = sam(box, [[560, 1300, 1], [960, 1300, 1], [700, 900, 1], [500, 1925, 0], [915, 1930, 0], [760, 760, 0]]) > 0.5
jeans = ndimage.binary_fill_holes(largest(jeans & ~tee)) & ~tee
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, jeans box {box}')
print('jeans: classic frac', classic.mean().round(4), 'sam frac', jeans.mean().round(4),
      'IoU', ((classic & jeans).sum() / (classic | jeans).sum()).round(4))

# ViTMatte:待求带只留在商品和背景、遮挡物的交界;牛仔裤和 T 恤的交界不算边缘,T 恤下面由 pipeline.py 接着铺
from transformers import VitMatteForImageMatting, VitMatteImageProcessor
MATTE = 'hustvl/vitmatte-small-composition-1k'
BAND_RATIO = 0.004  # 待求带向外的半宽 / 宽度,要盖住 SAM 的边界误差;向内收 1/3
TILE, OVERLAP = 768, 128  # 整图一次推理要十几 GB 内存,按块推理
band = max(1, round(BAND_RATIO * W))
t2 = time.time()
matte_processor = VitMatteImageProcessor.from_pretrained(MATTE)
matte_model = VitMatteForImageMatting.from_pretrained(MATTE).to(device).eval()
ramp = lambda k: np.minimum(np.arange(k) + 1, k - np.arange(k)).astype(np.float32)


def matte(fg, bg):
    """三分图 fg / bg 之外的像素由 ViTMatte 求透明度"""
    unknown = ~fg & ~bg
    trimap = np.where(fg, 255, np.where(bg, 0, 128)).astype(np.uint8)
    total, weight = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for ty in range(0, H, TILE - OVERLAP):
        for tx in range(0, W, TILE - OVERLAP):
            sl = (slice(ty, min(ty + TILE, H)), slice(tx, min(tx + TILE, W)))
            if not unknown[sl].any():
                continue
            inputs = matte_processor(images=Image.fromarray(photo8[sl]), trimaps=Image.fromarray(trimap[sl]),
                                     return_tensors='pt').to(device)
            with torch.no_grad():
                pred = matte_model(**inputs).alphas[0, 0].float().cpu().numpy()
            h, w = trimap[sl].shape
            # 块边上缺上下文,按离块边的距离加权拼接
            feather = np.clip(np.minimum.outer(ramp(h), ramp(w)) / (OVERLAP / 2), 1e-3, 1)
            total[sl] += pred[:h, :w] * feather
            weight[sl] += feather
    return np.where(unknown, np.clip(total / np.maximum(weight, 1e-6), 0, 1), fg).astype(np.float32), unknown


inner = max(1, band // 3)
tee_alpha, tee_unknown = matte(shrink(tee, inner), ~grow(tee, band) | occluders)
# 牛仔裤按抠好的 T 恤让位,和 pipeline.py 用的同一个范围,交界处不留缝
tee = tee_alpha > 0.5
jeans_alpha, jeans_unknown = matte(shrink(jeans, inner) | (jeans & grow(tee, band)), ~grow(jeans, band) | tee)
print(f'matting {time.time() - t2:.1f}s, unknown band: T-shirt {tee_unknown.mean() * 100:.2f}%, jeans {jeans_unknown.mean() * 100:.2f}%')

for name, alpha in (('tee-alpha', tee_alpha), ('jeans-alpha', jeans_alpha)):
    Image.fromarray(np.round(alpha * 65535).astype(np.uint16)).save(WORK / f'{name}.png')
for name, mask in (('sleeves', sleeves), ('occluders', occluders)):
    Image.fromarray(mask.astype(np.uint8) * 255).save(WORK / f'{name}.png')


def overlay(layers, line):
    """蒙版按颜色半透明叠在原图上,line 的轮廓描黄线"""
    vis = photo8.astype(np.float32)
    for mask, color in layers:
        m = mask.astype(np.float32)[..., None]
        vis = vis * (1 - 0.5 * m) + np.array(color) * 0.5 * m
    vis[cv2.Canny(line.astype(np.uint8) * 255, 50, 150) > 0] = [255, 230, 0]
    return Image.fromarray(vis.clip(0, 255).astype(np.uint8))


# T 恤:绿色为我们的商品区,蓝色为袖子,红色为遮挡物,黄线为竞品模板自带商品区的轮廓
overlay([(tee_alpha * ~sleeves, [0, 200, 80]), (sleeves, [40, 120, 255]), (occluders, [255, 40, 40])], reference_tee) \
    .crop((350, 150, 1250, 1100)).save(WORK / 'tee-compare.png')
# 牛仔裤:左为颜色加形态学,右为 SAM
panels = [overlay([(m, [255, 0, 60])], m).crop((280, 690, 1235, 2000)).resize((474, 650), Image.LANCZOS)
          for m in (classic, jeans)]
sheet = Image.new('RGB', (960, 650), 'white')
sheet.paste(panels[0], (0, 0)); sheet.paste(panels[1], (486, 0))
sheet.save(WORK / 'jeans-compare.png')
