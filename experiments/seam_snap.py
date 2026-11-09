"""
试验:SAM 2.1 只给出袖子的大致范围(sam21_auto.py 的结果),这里沿照片里的接缝重新切分袖子和身体,不用手标。
接缝是布面上的细暗线:按 Hessian 求暗线强度,跨过暗线的代价高;从身体、袖子两侧的种子各求测地距离,
每个像素归近的一边,分界就落在两边之间最强的那条暗线上。结果在 work/<模板 id>/experiments/。
用法:.venv/bin/python -I experiments/seam_snap.py batches/<批次>/specs/<id>.json [SAM 结果的变体名]
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))  # -I 运行时 scripts/ 不在 sys.path 里
import config
from segment import grow

RIDGE_SIGMA = 1.5  # 暗线的尺度(px),接缝线宽两三个像素
CROSS_COST = 60  # 跨过最强暗线时每像素的额外代价(按像素长度计)
TORSO_MARGIN = 0.03  # 身体种子离袖子大致范围至少这么远(× 宽度)
OCCLUDER_MARGIN = 4  # 遮挡物边上几个像素的明暗跳变不算接缝
SEED_QUANTILE = 0.75  # 袖子种子只取离身体最远的 25%(袖口一侧),保证落在接缝以外

spec = config.load(sys.argv[1])
variant = sys.argv[2] if len(sys.argv) > 2 else 'box'
OUT = config.folder(spec, 'work') / 'experiments'
saved = np.load(OUT / 'sam21-auto-plus.npz')
tee, occluders, rough = saved['tee'] & ~saved['occluders'], saved['occluders'], saved[f'sleeves {variant}']
photo, _ = config.photo(spec)
H, W = tee.shape
t0 = time.time()

# 暗线强度:Hessian 较大的特征值(横跨暗线方向的二阶导为正),按尺度归一
luma = photo @ np.array([0.299, 0.587, 0.114], np.float32)
xx, yy, xy = (ndimage.gaussian_filter(luma, RIDGE_SIGMA, order=o) for o in ((0, 2), (2, 0), (1, 1)))
ridge = np.maximum((xx + yy) / 2 + np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2), 0)
ridge = np.clip(ridge / np.percentile(ridge[tee], 99.5), 0, 1)
ridge[~tee | grow(occluders, OCCLUDER_MARGIN)] = 0

# 种子:身体取离所有袖子大致范围足够远的部分;每只袖子取离身体种子最远的一截
sleeves = [s & tee for s in rough]
torso = tee & ~grow(np.logical_or.reduce(sleeves), round(TORSO_MARGIN * W))
far = ndimage.distance_transform_edt(~torso)
seeds = np.zeros_like(tee)
for s in sleeves:
    seeds |= s & (far >= np.quantile(far[s], SEED_QUANTILE))

# 测地距离:域是商品区加遮挡物(左袖和身体被包带隔开,要能从包带上走过去),8 邻接
domain = tee | occluders
index = -np.ones((H, W), int)
index[domain] = np.arange(domain.sum())


def pairs(array, dy, dx):
    """相距 (dy, dx) 的每对像素,两端各一个视图"""
    xa, xb = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    return array[:H - dy, xa], array[dy:, xb]


rows, cols, weights = [], [], []
for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
    pair = np.logical_and(*pairs(domain, dy, dx))
    (ia, ib), (ra, rb) = ((a[pair], b[pair]) for a, b in (pairs(index, dy, dx), pairs(ridge, dy, dx)))
    rows.append(ia), cols.append(ib)
    weights.append(np.hypot(dy, dx) * (1 + CROSS_COST * np.maximum(ra, rb)))
graph = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(domain.sum(),) * 2).tocsr()
to_torso = dijkstra(graph, directed=False, indices=index[torso], min_only=True)
to_sleeve = dijkstra(graph, directed=False, indices=index[seeds], min_only=True)
snapped = np.zeros_like(tee)
snapped[domain] = to_sleeve < to_torso
snapped &= tee
print(f'seam snap {time.time() - t0:.1f}s')

manual = np.asarray(Image.open(OUT.parent / 'tee-parts.png')) > 127
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
print(f'IoU vs manual parts: rough {iou(np.logical_or.reduce(sleeves), manual):.3f}, snapped {iou(snapped, manual):.3f}')
np.save(OUT / f'seam-snap-{variant}.npy', snapped)


def overlay(sleeve, extra=()):
    image = (photo * 255).astype(np.float32)
    for mask, color in [(tee & ~sleeve, (0, 200, 80)), (sleeve, (40, 120, 255)), (occluders, (220, 40, 40)), *extra]:
        image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        image[mask ^ ndimage.binary_erosion(mask, iterations=2)] = color
    return Image.fromarray(image.clip(0, 255).astype(np.uint8)).crop((360, 160, 1220, 800))


ridge_view = Image.fromarray((np.dstack([ridge] * 3) * 255).astype(np.uint8)).crop((360, 160, 1220, 800))
panels = [('SAM 2.1 rough', overlay(np.logical_or.reduce(sleeves), [(seeds, (255, 200, 0)), (torso, (255, 255, 255))])),
          ('seam snapped', overlay(snapped)), ('manual prompts', overlay(manual)), ('ridge', ridge_view)]
w, h = panels[0][1].size
sheet = Image.new('RGB', (w * 2 + 10, h * 2 + 10), 'white')
for i, (name, panel) in enumerate(panels):
    ImageDraw.Draw(panel).text((10, 10), name, fill=(255, 0, 0))
    sheet.paste(panel, ((i % 2) * (w + 10), (i // 2) * (h + 10)))
sheet.save(OUT / f'seam-snap-{variant}.png')
print('wrote', OUT / f'seam-snap-{variant}.png')
