"""
几何:MoGe-2 估计照片里每个像素的 3D 坐标和相机内参,存 work/<id>/points.npy(相机坐标,x 右、y 下、z 朝前,原图尺寸)
和 work/<id>/geometry.json(归一化内参)。pipeline.py 用它出流向图里的大曲面和透视。
MODEL_BACKEND=local 本机跑,首次运行会从 Hugging Face 下载模型(约 1.3GB);replicate 调 Replicate 上的封装(见 remote.py)。
用法:.venv/bin/python -I geometry.py specs/<模板>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config

MODEL = 'Ruicheng/moge-2-vitl-normal'

spec = config.load(sys.argv[1])
WORK = config.folder(spec, 'work')
photo, _ = config.photo(spec)
t0 = time.time()
if config.backend() == 'replicate':
    import remote
    device = 'replicate'
    t1 = time.time()
    client = remote.Replicate(**config.replicate())
    points, intrinsics = remote.geometry(client, np.round(photo * 255).astype(np.uint8))
else:
    import torch
    from moge.model.v2 import MoGeModel
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = MoGeModel.from_pretrained(MODEL).to(device).eval()
    t1 = time.time()
    with torch.no_grad():
        result = model.infer(torch.from_numpy(photo).permute(2, 0, 1).to(device), use_fp16=False)
    points = result['points'].float().cpu().numpy().astype(np.float32)
    intrinsics = result['intrinsics'].float().cpu().numpy().round(6).tolist()
np.save(WORK / 'points.npy', points)
(WORK / 'geometry.json').write_text(json.dumps({'intrinsics': np.round(intrinsics, 6).tolist()}))
height, width = points.shape[:2]
print(f'model load {t1 - t0:.1f}s, inference {time.time() - t1:.1f}s, device {device}, size {width}×{height}')
