"""
模板配置(specs/*.json)和密钥的读取,各脚本共用。配置里的路径相对仓库根目录;
每个模板的中间结果放 work/<id>/,模板文件放 output/<id>/,预览图放 preview/<id>/。
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent


def load(path):
    return json.loads(Path(path).read_text())


def folder(spec, kind):
    """work / output / preview 下这个模板的目录,没有就建"""
    path = ROOT / kind / spec['id']
    path.mkdir(parents=True, exist_ok=True)
    return path


def secret(name):
    """先看环境变量,没有就读仓库根目录 .env.local(不进仓库)里的同名一行"""
    if os.environ.get(name):
        return os.environ[name]
    path = ROOT / '.env.local'
    lines = path.read_text().splitlines() if path.exists() else []
    values = [line.split('=', 1)[1].strip().strip('"\'') for line in lines if line.startswith(f'{name}=')]
    if not values or not values[-1]:
        raise SystemExit(f'缺少 {name}:设环境变量,或写进 {path}')
    return values[-1]


def photo(spec):
    """源图:0–1 的 RGB,带透明度的叠在白色上;另返回透明度,不到 1 的像素不是原始照片(如从竞品底图还原的商品区)。
    配置里有 crop 时先裁"""
    image = Image.open(ROOT / spec['photo']).convert('RGBA')
    if 'crop' in spec:
        image = image.crop(spec['crop'])
    rgba = np.asarray(image).astype(np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]), rgba[..., 3]
