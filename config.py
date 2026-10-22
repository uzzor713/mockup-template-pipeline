"""
模板配置(specs/*.json)的读取,各脚本共用。配置里的路径相对仓库根目录;
每个模板的中间结果放 work/<id>/,模板文件放 output/<id>/,预览图放 preview/<id>/。
"""
import json
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


def photo(spec):
    """源图:0–1 的 RGB,带透明度的叠在白色上;另返回透明度,不到 1 的像素不是原始照片(如从竞品底图还原的商品区)"""
    rgba = np.asarray(Image.open(ROOT / spec['photo']).convert('RGBA')).astype(np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]), rgba[..., 3]
