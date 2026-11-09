"""
试验:SAM 3 按文字提示分割,每个提示的各个实例分别存下蒙版,再出一张叠图,用来给新模板挑提示词。
走 segment.py 的 Replicate 后端,会把源图上传到 Replicate;token 是环境变量或 .env.local 里的 REPLICATE_API_TOKEN。
结果在 work/<模板 id>/experiments/sam3/。
用法:.venv/bin/python -I experiments/sam3_replicate.py batches/<批次>/specs/<id>.json [提示 ...]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))  # -I 运行时 scripts/ 不在 sys.path 里
import config
from segment import Sam3Replicate

PROMPTS = sys.argv[2:] or ['t-shirt', 'sleeve', 'neckline', 'bag', 'bag strap', 'jeans']
COLORS = [(0, 200, 80), (255, 150, 0), (40, 120, 255), (220, 40, 40), (170, 60, 220), (0, 190, 200)]

spec = config.load(sys.argv[1])
OUT = config.folder(spec, 'work') / 'experiments' / 'sam3'
OUT.mkdir(parents=True, exist_ok=True)
photo, _ = config.photo(spec)
photo = np.round(photo * 255).astype(np.uint8)
sam3 = Sam3Replicate(**config.replicate())

panels = []
for prompt in PROMPTS:
    masks = sam3.instances(photo, prompt)
    print(f'{prompt}: {len(masks)} instances, areas {[int(m.sum()) for m in masks]}')
    overlay = photo.astype(np.float32)
    for index, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8) * 255).save(OUT / f"{prompt.replace(' ', '-')}-{index}.png")
        overlay[mask] = overlay[mask] * 0.45 + np.array(COLORS[index % len(COLORS)]) * 0.55
    panel = Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).resize((400, 500), Image.LANCZOS)
    ImageDraw.Draw(panel).text((8, 8), f'{prompt} ({len(masks)})', fill=(0, 0, 0))
    panels.append(panel)

sheet = Image.new('RGB', (400 * len(panels), 500), 'white')
for index, panel in enumerate(panels):
    sheet.paste(panel, (400 * index, 0))
sheet.save(OUT / 'sheet.png')
print('wrote', OUT / 'sheet.png')
