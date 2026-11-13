"""
模板配置(batch.py 生成在 batches/<批次>/specs/<id>.json)和密钥的读取,各脚本共用。配置里的路径相对仓库根目录;
每个模板的中间结果放 work/<id>/,模板文件放 output/<id>/,预览图放 preview/<id>/。
"""
import json
import os
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent  # 本文件在 scripts/ 下


def load(path):
    return json.loads(Path(path).read_text())


def region_faces(spec, item):
    """逐张标注的多边形,或按纸张轮廓比例切开的可见印刷面。"""
    key = item['key']
    faces = spec.get('faces', {}).get(key, [])
    layout = spec.get('faceLayouts', {}).get(key)
    if faces and layout:
        raise ValueError(f"{spec['id']}:{key} 不能同时设置 faces 和 faceLayouts")
    if not layout:
        return faces
    axis, breaks = layout['axis'], layout['breaks']
    if axis == 'xy':
        if layout.get('fit') != 'oriented':
            raise ValueError(f"{spec['id']}:{key} 的 xy faceLayouts 必须按商品方向定位")
        if len(breaks) != 2 or any(not 0 < value < 1 for value in breaks):
            raise ValueError(f"{spec['id']}:{key} 的 xy faceLayouts 必须有 x、y 两条折线")
        omit = layout.get('omit')
        names = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        if omit not in names:
            raise ValueError(f"{spec['id']}:{key} 的 xy faceLayouts 必须指定不选的最小面")
        xb, yb = breaks
        intervals = {
            'top-left': ([0, xb], [0, yb]),
            'top-right': ([xb, 1], [0, yb]),
            'bottom-left': ([0, xb], [yb, 1]),
            'bottom-right': ([xb, 1], [yb, 1]),
        }
        return [{'key': name, 'name': f'{name.replace("-", " ").title()} {layout.get("name", "Panel")}',
                 'axes': {'x': intervals[name][0], 'y': intervals[name][1]}}
                for name in names if name != omit]
    if axis not in ('x', 'y') or len(breaks) not in (1, 2) or any(
            not 0 < value < 1 for value in breaks) or breaks != sorted(set(breaks)):
        raise ValueError(f"{spec['id']}:{key} 的 faceLayouts 必须是 1–2 条有序的 x/y 折线")
    names = (['Left', 'Right'] if axis == 'x' else ['Top', 'Bottom']) if len(breaks) == 1 else (
        ['Left', 'Center', 'Right'] if axis == 'x' else ['Top', 'Middle', 'Bottom'])
    return [{'key': name.lower(), 'name': f'{name} {layout.get("name", "Panel")}',
             'axis': axis, 'interval': [lower, upper]}
            for name, lower, upper in zip(names, [0, *breaks], [*breaks, 1])]


def region_products(spec):
    """把已分割的商品按可见印刷面展开,最多三个独立区域。"""
    regions = []
    for item in spec['products']:
        faces = region_faces(spec, item)
        if not faces:
            regions.append(item)
            continue
        if len(faces) > 3 or len({face['key'] for face in faces}) != len(faces):
            raise ValueError(f"{spec['id']}:{item['key']} 的面数必须在 1–3 之间,且 key 不重复")
        for face in faces:
            regions.append({**item, 'id': f"{item['id']}-{face['key']}",
                            'key': face['key'], 'name': face['name'],
                            **{key: face[key] for key in ('surface', 'logo') if key in face}})
    if len(regions) > 3:
        raise ValueError(f"{spec['id']}:模板最多三个区域")
    return regions


def folder(spec, kind):
    """work / output / preview 下这个模板的目录,没有就建"""
    path = ROOT / kind / spec['id']
    path.mkdir(parents=True, exist_ok=True)
    return path


def setting(name, default=None):
    """先看环境变量,再看仓库根目录 .env.local(不进仓库,放密钥和本机的改动),最后看 .env(进仓库,共用的默认值)"""
    if os.environ.get(name):
        return os.environ[name]
    for file in ('.env.local', '.env'):
        path = ROOT / file
        lines = path.read_text().splitlines() if path.exists() else []
        values = [line.split('=', 1)[1].strip().strip('"\'') for line in lines if line.startswith(f'{name}=')]
        if values and values[-1]:
            return values[-1]
    return default


def secret(name):
    value = setting(name)
    if not value:
        raise SystemExit(f'缺少 {name}:设环境变量,或写进 {ROOT / ".env.local"}')
    return value


def replicate():
    """建 Replicate 客户端的参数:token 和每个进程两次预测之间的最短间隔(秒)"""
    return {'token': secret('REPLICATE_API_TOKEN'), 'min_interval': float(setting('REPLICATE_MIN_INTERVAL', '0'))}


def backend():
    """本征分解、法线、几何和抠图在哪跑:local(本机)或 replicate"""
    value = setting('MODEL_BACKEND', 'local')
    if value not in ('local', 'replicate'):
        raise SystemExit(f'MODEL_BACKEND 只能是 local 或 replicate,现在是 {value}')
    return value


def source_rgba(spec):
    """源图的 RGBA(uint8),配置里有 crop 时先裁,有 size 时再缩放到这个尺寸"""
    image = Image.open(ROOT / spec['photo']).convert('RGBA')
    if 'crop' in spec:
        image = image.crop(spec['crop'])
    if 'size' in spec:
        image = image.resize(tuple(spec['size']), Image.LANCZOS)
    return np.asarray(image)


def photo(spec):
    """源图:0–1 的 RGB,带透明度的叠在白色上;另返回透明度,不到 1 的像素不是原始照片(如从竞品底图还原的商品区)"""
    rgba = source_rgba(spec).astype(np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]), rgba[..., 3]


def photo8(spec, rows=128):
    """photo() 的 8 位版,和 np.round(photo * 255) 逐位相同,给只要 8 位图的分割和模型脚本。
    按条带换算:整张一次换算的浮点中间结果有几百 MB,macOS 释放后不还给系统,
    进程等 Replicate 的几分钟里一直占着,批量并发高时内存大半耗在这里"""
    rgba = source_rgba(spec)
    out = np.empty(rgba.shape[:2] + (3,), np.uint8)
    for y in range(0, len(rgba), rows):
        strip = rgba[y:y + rows].astype(np.float32) / 255
        out[y:y + rows] = np.round((strip[..., :3] * strip[..., 3:] + (1 - strip[..., 3:])) * 255)
    return out


def save_points(work, points):
    """3D 坐标(H×W×3 float32)无损压缩存 work/points.z:x、y、z 分开,每个数的四个字节按位置归到一起再 zlib,
    约为原来的一半(work/ 里最占地方的就是它)。第一行是 JSON 写的形状"""
    planes = np.ascontiguousarray(points.astype(np.float32).transpose(2, 0, 1)).view(np.uint8).reshape(-1, 4).T
    header = json.dumps({'shape': list(points.shape)}).encode() + b'\n'
    (Path(work) / 'points.z').write_bytes(header + zlib.compress(planes.tobytes(), 9))


def load_points(work):
    """save_points 存的 3D 坐标;更早存成 points.npy 的照样读"""
    path = Path(work) / 'points.z'
    if not path.exists():
        return np.load(Path(work) / 'points.npy')
    header, body = path.read_bytes().split(b'\n', 1)
    height, width, channels = json.loads(header)['shape']
    planes = np.frombuffer(zlib.decompress(body), np.uint8).reshape(4, -1).T.copy().view(np.float32)
    return np.ascontiguousarray(planes.reshape(channels, height, width).transpose(1, 2, 0))
