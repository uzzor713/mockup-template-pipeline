"""
Replicate 的调用:上传图片、按固定版本跑预测、取回结果,以及 MODEL_BACKEND=replicate 时四个模型的封装和输出换算。
SAM 3(segment.py)也用这里的 Replicate。不读写本地文件;token 由调用方给。

四个封装都是社区上传的,版本固定写死。输出和本机版对过(T 恤、咖啡袋两张):
- MoGe-2:关掉 fp16 后 3D 坐标、内参和本机逐位相同;
- Marigold 本征分解、法线:不能设随机种子和处理分辨率,只出 8 位 PNG。albedo、shading 存的是线性值的 1/2.2 次方,
  residual 另按最大值缩放过(pipeline.py 每件商品自己拟合比例,缩放不影响);法线存的是 (n + 1) / 2。
  换算后和本机的相关系数:albedo、shading 0.995 以上,法线 0.96–0.98,residual 约 0.94(数值小,8 位损失大);
- ViTMatte:三分图照常给,长边超过 2048 时在缩小的图上抠,再放大回原图。
"""
import hashlib
import http.client
import io
import json
import os
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image

API = 'https://api.replicate.com/v1'
RETRY_WAIT = 11  # 限流(429)又没给 Retry-After 时等这么久
RETRIES = 6  # 网络偶发 TLS 握手失败、连接中断:隔 3、6、12、24、48 秒再试,一共等一分半
USER_AGENT = 'curl/8.7.1'  # Python 默认的 User-Agent 会被 Cloudflare 拦下(错误码 1010)
INTRINSICS = 'jasonod888/marigold-intrinsics:0d39a1c5b555598f1797ec783d79e3d2b5d3e5b594625625ab59351c5be50467'
NORMALS = 'jasonod888/marigold-normalsv2:3129bd0f2d415e73d6cb6bccca11eba9c6dc9105129f0f3637b5dec5baa6cb4d'
GEOMETRY = 'jasonod888/moge2:daa7a9329b3d6bb513f3a20def9451b6e3cf234afe2bb60ba38f96ecd33c81f7'
MATTE = 'superhighfives/vitmatte-trimap:193b4b013b262e9b64e57755d38ff82e497dd2f14537e2e274bc553a28c03237'
MATTE_MAX = 2048  # 抠图封装能处理的最大长边


class Replicate:
    """同一张图只上传一次;各次预测之间至少隔 min_interval 秒(本进程内,余额低、每分钟限 6 次的账号设 11)"""

    def __init__(self, token, min_interval=0.0):
        self.token = token
        self.min_interval = min_interval
        self.uploaded = {}
        self.next_start = 0.0

    def fetch(self, url, data=None, method=None, headers=None, auth=False):
        """取回响应内容;连接失败和限流(429)重试,其他 HTTP 错误照常抛出。结果在 replicate.delivery,不带 token"""
        headers = {'User-Agent': USER_AGENT, **({'Authorization': f'Bearer {self.token}'} if auth else {}), **(headers or {})}
        for attempt in range(RETRIES):
            try:
                request = urllib.request.Request(url, data=data, method=method, headers=headers)
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == RETRIES - 1:
                    raise
                time.sleep(float(error.headers.get('Retry-After') or RETRY_WAIT))
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(3 * 2 ** attempt)

    def call(self, method, url, body=None, headers=None):
        return json.loads(self.fetch(url, body, method, headers, auth=True))

    def upload(self, image):
        """uint8 的 H×W×3 或 H×W → 文件地址"""
        key = hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()
        if key not in self.uploaded:
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, 'PNG')
            boundary = 'replicate-upload-boundary'
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="content"; filename="image.png"\r\n'
                    'Content-Type: image/png\r\n\r\n').encode() + buffer.getvalue() + f'\r\n--{boundary}--\r\n'.encode()
            self.uploaded[key] = self.call('POST', f'{API}/files', body,
                                           {'Content-Type': f'multipart/form-data; boundary={boundary}'})['urls']['get']
        return self.uploaded[key]

    def predict(self, model, inputs):
        """model 写成 所有者/名称:版本;返回预测的 output"""
        time.sleep(max(0.0, self.next_start - time.monotonic()))
        self.next_start = time.monotonic() + self.min_interval
        body = json.dumps({'version': model.split(':')[1], 'input': inputs}).encode()
        prediction = self.call('POST', f'{API}/predictions', body, {'Content-Type': 'application/json', 'Prefer': 'wait=60'})
        while prediction['status'] not in ('succeeded', 'failed', 'canceled'):
            time.sleep(2)
            prediction = self.call('GET', prediction['urls']['get'])
        if prediction['status'] != 'succeeded':
            raise RuntimeError(f"{model.split(':')[0]}: {prediction['status']} {prediction.get('error')}")
        return prediction['output']

    def image(self, url):
        return np.asarray(Image.open(io.BytesIO(self.fetch(url))))


def fit(image, longest):
    """缩到长边 longest(不放大),和本机版的处理分辨率一致"""
    h, w = image.shape[:2]
    k = longest / max(h, w)
    return image if k >= 1 else cv2.resize(image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def intrinsics(client, photo, resolution):
    """photo:uint8 RGB。返回 {albedo, shading, residual},线性值,长边 resolution"""
    urls = client.predict(INTRINSICS, {'image': client.upload(photo)})
    layers = {url.rsplit('/', 1)[1].split('.')[0]: client.image(url) for url in urls}
    return {name: fit((layers[name].astype(np.float32) / 255) ** 2.2, resolution) for name in ('albedo', 'shading', 'residual')}


def normals(client, photo, resolution):
    """返回法线,−1–1,长边 resolution"""
    encoded = client.image(client.predict(NORMALS, {'image': client.upload(photo)})).astype(np.float32) / 255
    return fit(encoded[..., :3] * 2 - 1, resolution)


def geometry(client, photo):
    """返回 (原图尺寸的 3D 坐标, 归一化内参)"""
    output = client.predict(GEOMETRY, {'image': client.upload(photo), 'fp16': False, 'debug_write_local': False})
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    points = cv2.imdecode(np.frombuffer(client.fetch(output['points_exr']), np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
    return np.ascontiguousarray(points, dtype=np.float32), json.loads(client.fetch(output['intrinsics_json']))['intrinsics']


def matte(client, photo, trimap):
    """trimap:uint8,0 / 128 / 255。返回原图尺寸的透明度,0–1"""
    h, w = trimap.shape
    alpha = client.image(client.predict(MATTE, {'image': client.upload(photo), 'trimap': client.upload(trimap),
                                                'max_size': min(MATTE_MAX, max(h, w))}))
    alpha = (alpha if alpha.ndim == 2 else alpha[..., 0]).astype(np.float32) / np.iinfo(alpha.dtype).max
    return alpha if alpha.shape == (h, w) else cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
