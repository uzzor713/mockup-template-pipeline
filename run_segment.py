"""
按模板的分割配置跑 segment.py,结果存到 work/:每件商品 <id>-alpha.png(16 位透明度)、<id>-parts.png(不可印部件),
遮挡物 occluders.png,以及人工检查用的叠图 segment-check.png。
配置见 specs/,源图路径相对仓库根目录;带透明度的源图叠在白色上。
用法:.venv/bin/python -I run_segment.py specs/sample-tee-jeans.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
# -I 运行时脚本所在目录不在 sys.path 里
sys.path.insert(0, str(ROOT))
from segment import Models, segment

WORK = ROOT / 'work'
WORK.mkdir(exist_ok=True)
spec = json.loads(Path(sys.argv[1]).read_text())
rgba = np.asarray(Image.open(ROOT / spec['photo']).convert('RGBA')).astype(np.float32) / 255
photo = np.round((rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]) * 255).astype(np.uint8)

t0 = time.time()
models = Models()
t1 = time.time()
occluders, products = segment(photo, spec, models)
print(f'model load {t1 - t0:.1f}s, segment {time.time() - t1:.1f}s, device {models.device}')

for item, result in zip(spec['products'], products):
    Image.fromarray(np.round(result['alpha'] * 65535).astype(np.uint16)).save(WORK / f"{item['id']}-alpha.png")
    Image.fromarray(result['parts'].astype(np.uint8) * 255).save(WORK / f"{item['id']}-parts.png")
Image.fromarray(occluders.astype(np.uint8) * 255).save(WORK / 'occluders.png')

# 检查叠图:各件商品按顺序着绿、橙、紫,不可印部件蓝色,遮挡物红色
layers = [(r['alpha'] * ~r['parts'], c) for r, c in zip(products, [[0, 200, 80], [255, 150, 0], [170, 60, 220]])]
layers += [(r['parts'], [40, 120, 255]) for r in products] + [(occluders, [255, 40, 40])]
check = photo.astype(np.float32)
for mask, color in layers:
    m = mask.astype(np.float32)[..., None]
    check = check * (1 - 0.5 * m) + np.array(color) * 0.5 * m
Image.fromarray(check.clip(0, 255).astype(np.uint8)).save(WORK / 'segment-check.png')
