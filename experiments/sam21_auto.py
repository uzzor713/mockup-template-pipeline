"""
试验:detect_boxes.py 检测出的框直接当 SAM 2.1 的提示,不用手点,和手点提示的现有结果(work/<模板 id>/ 下)对比。
袖子比三种提示:只给框 / 框 + 框中心正点 + T 恤中心负点 / 同样的点,三个候选里挑最贴框的。
结果在 work/<模板 id>/experiments/。
用法:.venv/bin/python -I experiments/sam21_auto.py specs/<模板>.json [SAM 模型] [检测模型]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))  # -I 运行时仓库根目录不在 sys.path 里
import config
import torch
from transformers import Sam2Model, Sam2Processor

spec = config.load(sys.argv[1])
SAM = sys.argv[2] if len(sys.argv) > 2 else 'facebook/sam2.1-hiera-base-plus'
DETECTOR = sys.argv[3] if len(sys.argv) > 3 else 'IDEA-Research/grounding-dino-base'
WORK = config.folder(spec, 'work')
OUT = WORK / 'experiments'
detections = json.loads((OUT / f"detections-{DETECTOR.split('/')[-1]}.json").read_text())
tag = SAM.split('-')[-1]

photo, _ = config.photo(spec)
photo = np.round(photo * 255).astype(np.uint8)
H, W = photo.shape[:2]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
processor = Sam2Processor.from_pretrained(SAM)
model = Sam2Model.from_pretrained(SAM).to(device).eval()


def sam(box, points=(), multimask=False):
    """返回 [(蒙版, 模型自评分)];multimask 时三个候选"""
    kwargs = {'input_boxes': [[box]]}
    if points:
        kwargs['input_points'] = [[[[x, y] for x, y, _ in points]]]
        kwargs['input_labels'] = [[[label for *_, label in points]]]
    inputs = processor(images=Image.fromarray(photo), return_tensors='pt', **kwargs).to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=multimask)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu(), binarize=False)[0]
    masks = torch.sigmoid(masks.float()).numpy()[0] > 0.5
    return list(zip(masks, out.iou_scores.float().cpu().numpy()[0, 0]))


def best(label):
    return max((d for d in detections if d['label'] == label), key=lambda d: d['score'])


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def overlap(a, b):
    """框 a 有多少比例落在框 b 里"""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / ((a[2] - a[0]) * (a[3] - a[1]))


def union(masks):
    return np.logical_or.reduce(masks) if masks else np.zeros((H, W), bool)


t0 = time.time()
tee_box = best('t-shirt')['box']
tee = sam(tee_box)[0][0]
jeans = sam(best('jeans')['box'])[0][0]
occluders = union([sam(d['box'])[0][0] for d in detections if d['label'] in ('bag', 'bag strap')])

# 同一只袖子可能检出几个框,按分数去重
kept = []
for d in sorted((d for d in detections if d['label'] == 'sleeve'), key=lambda d: -d['score']):
    if all(overlap(d['box'], k['box']) < 0.5 for k in kept):
        kept.append(d)
print('sleeve boxes', [(k['box'], k['score']) for k in kept])

torso = center(tee_box)
variants = {'box': [], 'box+points': [], 'box+points best-fit': []}
for d in kept:
    box = d['box']
    variants['box'].append(sam(box)[0][0])
    points = [[*center(box), 1], [*torso, 0]]
    variants['box+points'].append(sam(box, points)[0][0])
    filled = np.zeros((H, W), bool)
    filled[box[1]:box[3], box[0]:box[2]] = True
    variants['box+points best-fit'].append(max(sam(box, points, multimask=True), key=lambda c: iou(c[0], filled & tee))[0])
print(f'sam {time.time() - t0:.1f}s on {device}')
np.savez_compressed(OUT / f'sam21-auto-{tag}.npz', tee=tee, jeans=jeans, occluders=occluders,
                    **{f"sleeves {name}": np.array(masks) for name, masks in variants.items()})

read = lambda name: np.asarray(Image.open(WORK / name))
manual_tee, manual_jeans = read('tee-alpha.png') > 32767, read('jeans-alpha.png') > 32767
manual_parts, manual_occ = read('tee-parts.png') > 127, read('occluders.png') > 32767
print(f'IoU vs manual: tee {iou(tee & ~occluders, manual_tee):.3f}  jeans {iou(jeans & ~tee, manual_jeans):.3f}  '
      f'occluders {iou(occluders, manual_occ):.3f}')
for name, masks in variants.items():
    print(f'sleeves [{name}] IoU vs manual parts {iou(union(masks) & tee, manual_parts):.3f}')


def overlay(layers, crop):
    image = photo.astype(np.float32)
    for mask, color in layers:
        image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        image[mask ^ ndimage.binary_erosion(mask, iterations=2)] = color
    return Image.fromarray(image.clip(0, 255).astype(np.uint8)).crop(crop)


tee_crop = (360, 160, 1220, 800)
panels = [('manual', overlay([(manual_tee & ~manual_parts, (0, 200, 80)), (manual_parts, (40, 120, 255)),
                              (manual_occ, (220, 40, 40))], tee_crop))]
for name, masks in variants.items():
    sleeves = union(masks) & tee
    panels.append((name, overlay([(tee & ~occluders & ~sleeves, (0, 200, 80)), (sleeves, (40, 120, 255)),
                                  (occluders, (220, 40, 40))], tee_crop)))
w, h = panels[0][1].size
sheet = Image.new('RGB', (w * 2 + 10, h * 2 + 10), 'white')
for index, (name, panel) in enumerate(panels):
    ImageDraw.Draw(panel).text((10, 10), name, fill=(0, 0, 0))
    sheet.paste(panel, ((index % 2) * (w + 10), (index // 2) * (h + 10)))
sheet.save(OUT / f'sam21-auto-{tag}.png')
full = overlay([(tee & ~occluders, (0, 200, 80)), (jeans & ~tee & ~occluders, (255, 150, 0)), (occluders, (220, 40, 40))],
               (0, 0, W, H))
full.resize((W // 3, H // 3)).save(OUT / f'sam21-auto-full-{tag}.png')
print('wrote', OUT / f'sam21-auto-{tag}.png')
