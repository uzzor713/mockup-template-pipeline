"""
法线:Marigold(法线版)估计照片里每个像素的表面朝向,存 work/<id>/normals.png(16 位,(n + 1) / 2,模型处理分辨率)。
pipeline.py 用它出流向图里的褶皱。
MODEL_BACKEND=local 本机跑,首次运行会从 Hugging Face 下载模型(约 2.6GB);replicate 调 Replicate 上的封装(见 remote.py)。
用法:.venv/bin/python -I scripts/normals.py batches/<批次>/specs/<id>.json
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config

MODEL = 'prs-eth/marigold-normals-v1-1'
RESOLUTION = 768

spec = config.load(sys.argv[1])
WORK = config.folder(spec, 'work')
photo = config.photo8(spec)
t0 = time.time()
if config.backend() == 'replicate':
    import remote
    device = 'replicate'
    t1 = time.time()
    normals = remote.normals(remote.Replicate(**config.replicate()), photo, RESOLUTION)
else:
    import torch
    from diffusers import MarigoldNormalsPipeline
    from PIL import Image
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pipe = MarigoldNormalsPipeline.from_pretrained(
        MODEL, variant='fp16', dtype=torch.float16 if device == 'mps' else torch.float32).to(device)
    t1 = time.time()
    result = pipe(Image.fromarray(photo), processing_resolution=RESOLUTION,
                  match_input_resolution=False, generator=torch.Generator().manual_seed(0))
    normals = result.prediction[0]
cv2.imwrite(str(WORK / 'normals.png'), np.round(np.clip((normals[..., ::-1] + 1) / 2, 0, 1) * 65535).astype(np.uint16))
height, width = normals.shape[:2]
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, size {width}×{height}')
