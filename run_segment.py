"""
按模板配置跑 segment.py,结果存到 work/<模板 id>/:每件商品 <商品 id>-alpha.png(16 位透明度)、<商品 id>-parts.png
(不可印部件),遮挡物 occluders.png(16 位透明度),以及人工检查用的叠图 segment-check.png。
用到文字提示或 MODEL_BACKEND=replicate(抠图也走 Replicate)时要 Replicate 的 token:环境变量或仓库根目录 .env.local 里的 REPLICATE_API_TOKEN。
用法:.venv/bin/python -I run_segment.py specs/<模板>.json
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config
from segment import Models, Sam3Replicate, segment, uses_text

spec = config.load(sys.argv[1])
WORK = config.folder(spec, 'work')
photo, _ = config.photo(spec)
photo = np.round(photo * 255).astype(np.uint8)

t0 = time.time()
on_replicate = config.backend() == 'replicate'
client = Sam3Replicate(**config.replicate()) if uses_text(spec) or on_replicate else None
models = Models(sam3=client if uses_text(spec) else None, remote=client if on_replicate else None)
t1 = time.time()
occluders, products = segment(photo, spec, models)
print(f'model load {t1 - t0:.1f}s, segment {time.time() - t1:.1f}s, device {models.device or "replicate"}')

for item, result in zip(spec['products'], products):
    Image.fromarray(np.round(result['alpha'] * 65535).astype(np.uint16)).save(WORK / f"{item['id']}-alpha.png")
    Image.fromarray(result['parts'].astype(np.uint8) * 255).save(WORK / f"{item['id']}-parts.png")
Image.fromarray(np.round(occluders * 65535).astype(np.uint16)).save(WORK / 'occluders.png')

# 检查叠图:各件商品按顺序着绿、橙、紫,不可印部件蓝色,遮挡物红色
layers = [(r['alpha'] * ~r['parts'], c) for r, c in zip(products, [[0, 200, 80], [255, 150, 0], [170, 60, 220]])]
layers += [(r['parts'], [40, 120, 255]) for r in products] + [(occluders, [255, 40, 40])]
check = photo.astype(np.float32)
for mask, color in layers:
    m = mask.astype(np.float32)[..., None]
    check = check * (1 - 0.5 * m) + np.array(color) * 0.5 * m
Image.fromarray(check.clip(0, 255).astype(np.uint8)).save(WORK / 'segment-check.png')
