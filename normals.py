"""
法线:Marigold(法线版)估计照片里每个像素的表面朝向,存 work/normals.png(16 位,(n + 1) / 2,模型处理分辨率)。
pipeline.py 用它出起伏图。用法:.venv/bin/python -I normals.py;首次运行会从 Hugging Face 下载模型(约 2.6GB)
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import MarigoldNormalsPipeline
from PIL import Image

ROOT = Path(__file__).parent
WORK = ROOT / 'work'
WORK.mkdir(exist_ok=True)
MODEL = 'prs-eth/marigold-normals-v1-1'
RESOLUTION = 768

rgba = np.asarray(Image.open(ROOT / 'sources/tee-base.webp').convert('RGBA')).astype(np.float32) / 255
photo = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()
pipe = MarigoldNormalsPipeline.from_pretrained(
    MODEL, variant='fp16', dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
t1 = time.time()
result = pipe(Image.fromarray(np.round(photo * 255).astype(np.uint8)), processing_resolution=RESOLUTION,
              match_input_resolution=False, generator=torch.Generator().manual_seed(0))
normals = result.prediction[0]
cv2.imwrite(str(WORK / 'normals.png'), np.round(np.clip((normals[..., ::-1] + 1) / 2, 0, 1) * 65535).astype(np.uint16))
height, width = normals.shape[:2]
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, size {width}×{height}')
