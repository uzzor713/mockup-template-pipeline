"""
试验:开放词汇检测模型按文字找出各部分的外接框,给 sam21_auto.py 当 SAM 2.1 的提示,省掉手点。
结果在 work/<模板 id>/experiments/:detections-<模型>.json 和画了框的图。
用法:.venv/bin/python -I experiments/detect_boxes.py batches/<批次>/specs/<id>.json [检测模型]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))  # -I 运行时 scripts/ 不在 sys.path 里
import config
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

LABELS = ['t-shirt', 'sleeve', 'neckline', 'bag', 'bag strap', 'jeans']
COLORS = ['lime', 'orange', 'cyan', 'red', 'magenta', 'blue']
THRESHOLD = 0.2

spec = config.load(sys.argv[1])
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'IDEA-Research/grounding-dino-base'
OUT = config.folder(spec, 'work') / 'experiments'
OUT.mkdir(exist_ok=True)
photo, _ = config.photo(spec)
image = Image.fromarray(np.round(photo * 255).astype(np.uint8))

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).eval()
t1 = time.time()
inputs = processor(images=image, text='. '.join(LABELS) + '.', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
result = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids, threshold=THRESHOLD, text_threshold=THRESHOLD, target_sizes=[image.size[::-1]])[0]
print(f'load {t1 - t0:.1f}s, detect {time.time() - t1:.1f}s')

detections = [{'label': label.replace(' - ', '-'), 'score': round(score, 3), 'box': [round(v) for v in box]}
              for box, score, label in zip(result['boxes'].tolist(), result['scores'].tolist(), result['text_labels'])]
for d in detections:
    print(f"{d['label']:12s} {d['score']:.2f} {d['box']}")
name = MODEL.split('/')[-1]
(OUT / f'detections-{name}.json').write_text(json.dumps(detections, indent=1))

draw = ImageDraw.Draw(image)
for d in detections:
    color = dict(zip(LABELS, COLORS)).get(d['label'], 'yellow')
    draw.rectangle(d['box'], outline=color, width=4)
    draw.text((d['box'][0] + 6, d['box'][1] + 4), f"{d['label']} {d['score']}", fill=color)
image.resize((image.width // 2, image.height // 2)).save(OUT / f'detections-{name}.png')
