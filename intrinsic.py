"""
本征分解:Marigold(光照版)把照片拆成固有色、漫反射光照和残差(高光等),存 work/<id>/albedo.png、shading.png、residual.png
(线性值,16 位,模型处理分辨率)。pipeline.py 用光照和残差出明暗,花纹和染料留在固有色里。
用法:.venv/bin/python -I intrinsic.py specs/<模板>.json;首次运行会从 Hugging Face 下载模型(约 2.6GB)
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import MarigoldIntrinsicsPipeline
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config

MODEL = 'prs-eth/marigold-iid-lighting-v1-1'
RESOLUTION = 768  # 长边;不放大回原图,pipeline.py 按这个分辨率决定从照片补回多细的纹理

spec = config.load(sys.argv[1])
WORK = config.folder(spec, 'work')
photo, _ = config.photo(spec)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()
pipe = MarigoldIntrinsicsPipeline.from_pretrained(
    MODEL, variant='fp16', dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
t1 = time.time()
result = pipe(Image.fromarray(np.round(photo * 255).astype(np.uint8)), processing_resolution=RESOLUTION,
              match_input_resolution=False, generator=torch.Generator().manual_seed(0))
for name, layer in zip(pipe.target_properties['target_names'], result.prediction):
    cv2.imwrite(str(WORK / f'{name}.png'), np.round(np.clip(layer[..., ::-1], 0, 1) * 65535).astype(np.uint16))
height, width = result.prediction.shape[1:3]
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, size {width}×{height}')
