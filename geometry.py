"""
几何:MoGe-2 估计照片里每个像素的 3D 坐标和相机内参,存 work/points.npy(相机坐标,x 右、y 下、z 朝前,原图尺寸)
和 work/geometry.json(归一化内参)。pipeline.py 用它出起伏图里的大曲面和透视。
用法:.venv/bin/python -I geometry.py;首次运行会从 Hugging Face 下载模型(约 1.3GB)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from moge.model.v2 import MoGeModel
from PIL import Image

ROOT = Path(__file__).parent
WORK = ROOT / 'work'
WORK.mkdir(exist_ok=True)
MODEL = 'Ruicheng/moge-2-vitl-normal'

rgba = np.asarray(Image.open(ROOT / 'sources/tee-base.webp').convert('RGBA')).astype(np.float32) / 255
photo = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()
model = MoGeModel.from_pretrained(MODEL).to(device).eval()
t1 = time.time()
with torch.no_grad():
    result = model.infer(torch.from_numpy(photo).permute(2, 0, 1).to(device), use_fp16=False)
points = result['points'].float().cpu().numpy().astype(np.float32)
np.save(WORK / 'points.npy', points)
(WORK / 'geometry.json').write_text(json.dumps({'intrinsics': result['intrinsics'].float().cpu().numpy().round(6).tolist()}))
height, width = points.shape[:2]
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, size {width}×{height}')
