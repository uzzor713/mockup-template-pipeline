"""
分割牛仔裤:颜色加形态学的方法先给出大致范围,作为 SAM 2.1 的外接框;SAM 的结果存 work/jeans-mask.png,给 pipeline.py 用。
两种方法的对比图存 work/jeans-compare.png。用法:.venv/bin/python -I segment.py
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
tee = rgba[..., 3] < 1
H, W = tee.shape

# 颜色加形态学:牛仔蓝色相阈值 → 填洞 → 开闭运算 → 最大连通块 → GrabCut 修边
hsv = cv2.cvtColor(photo8, cv2.COLOR_RGB2HSV).astype(np.float32)
h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
seed = (h > 190) & (h < 240) & (s > 0.12) & (v > 0.18) & ~tee
seed = ndimage.binary_closing(seed, iterations=6)
seed = ndimage.binary_fill_holes(seed)
seed = ndimage.binary_opening(seed, iterations=3)
labels, n = ndimage.label(seed)
sizes = ndimage.sum(seed, labels, range(1, n + 1))
classic = labels == (1 + int(np.argmax(sizes)))
gc = np.where(classic, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
gc[ndimage.binary_erosion(classic, iterations=12)] = cv2.GC_FGD
gc[~ndimage.binary_dilation(classic, iterations=12)] = cv2.GC_BGD
gc[tee] = cv2.GC_BGD
cv2.grabCut(cv2.cvtColor(photo8, cv2.COLOR_RGB2BGR), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
classic = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & ~tee

# SAM 2.1:外接框取上一步的范围,正提示点在两条裤腿上,负提示点在鞋面和 T 恤下摆
from transformers import Sam2Model, Sam2Processor
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'facebook/sam2.1-hiera-base-plus'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()
processor = Sam2Processor.from_pretrained(MODEL)
model = Sam2Model.from_pretrained(MODEL).to(device).eval()
t1 = time.time()
ys, xs = np.nonzero(classic)
box = [int(xs.min()) - 10, int(ys.min()) - 10, int(xs.max()) + 10, int(ys.max()) + 10]
points = [[560, 1300, 1], [960, 1300, 1], [700, 900, 1], [500, 1925, 0], [915, 1930, 0], [760, 760, 0]]
inputs = processor(images=Image.fromarray(photo8), input_boxes=[[box]],
                   input_points=[[[[x, y] for x, y, _ in points]]], input_labels=[[[l for *_, l in points]]],
                   return_tensors='pt').to(device)
with torch.no_grad():
    out = model(**inputs, multimask_output=False)
logits = processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu(), binarize=False)[0]
prob = torch.sigmoid(logits.squeeze().float()).numpy()
sam = prob * (~tee)
Image.fromarray(((sam > 0.5) * 255).astype(np.uint8)).save(WORK / 'jeans-mask.png')
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, box {box}')
print('classic frac', classic.mean().round(4), 'sam frac', (sam > 0.5).mean().round(4),
      'IoU', ((classic & (sam > 0.5)).sum() / (classic | (sam > 0.5)).sum()).round(4),
      'sam soft-edge px', int(((sam > 0.02) & (sam < 0.98)).sum()))

def overlay(mask, name):
    vis = photo8.astype(np.float32).copy()
    m = mask[..., None] if mask.dtype != bool else mask[..., None].astype(np.float32)
    vis = vis * (1 - 0.55 * m) + np.array([255, 0, 60]) * 0.55 * m
    edge = cv2.Canny((m[..., 0] > 0.5).astype(np.uint8) * 255, 50, 150) > 0
    vis[edge] = [255, 230, 0]
    Image.fromarray(vis.clip(0, 255).astype(np.uint8)).save(WORK / f'{name}.png')
overlay(classic, 'jeans-classic'); overlay(sam, 'jeans-sam')

# 对比图:左为颜色加形态学,右为 SAM
panels = [Image.open(WORK / f'{n}.png').crop((280, 690, 1235, 2000)).resize((474, 650), Image.LANCZOS)
          for n in ('jeans-classic', 'jeans-sam')]
sheet = Image.new('RGB', (960, 650), 'white')
sheet.paste(panels[0], (0, 0)); sheet.paste(panels[1], (486, 0))
sheet.save(WORK / 'jeans-compare.png')
