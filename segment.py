"""
分割:SAM 3 按文字提示(或 SAM 2.1 按外接框和提示点)分出遮挡物、商品和商品的不可印部件,
ViTMatte 在商品和背景、遮挡物的交界求透明度。
不读写本地文件,也不含具体模板的内容;源图、提示和密钥由调用方给,见 run_segment.py。
SAM 3 现在走 Replicate,会把源图上传过去。

提示的格式(spec):
- 一个提示是 {text: 名词短语}(SAM 3,图里所有符合的实例合在一起),
  或 {box: [x0, y0, x1, y1], points: [[x, y, 1 正 / 0 负], ...]}(SAM 2.1,文字分不准时手点兜底);
- occluders:遮挡物的提示列表,合成一张蒙版;
- products:商品列表,由上层到下层,每项本身是一个提示,加可选的 parts(不可印部件的提示列表,如袖子、领口)。
"""
import hashlib, io, json, time, zipfile, urllib.error, urllib.request
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

SAM_MODEL = 'facebook/sam2.1-hiera-base-plus'
SAM3_VERSION = 'd73db077226443ba4fafd34e233b3626b552eac2a433f90c7c32a9ac89bd9e72'  # Replicate 上的 mattsays/sam3-image
REPLICATE_API = 'https://api.replicate.com/v1'
RATE_LIMIT_WAIT = 11  # Replicate 余额低的账号每分钟限 6 次
RETRIES = 4  # 网络偶发 TLS 握手失败、连接中断
MATTE_MODEL = 'hustvl/vitmatte-small-composition-1k'
BAND_RATIO = 0.004  # 待求带向外的半宽 / 宽度,要盖住 SAM 的边界误差;向内收 1/3
SPECK_RATIO = 0.01  # 面积不到整件这么多的碎块去掉
SPUR_RATIO = 0.005  # 部件边上伸进相邻部件的细长毛刺,按这个半径 / 宽度做开运算去掉
TILE, OVERLAP = 768, 128  # 整图一次推理要十几 GB 内存,按块推理


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)


def grow(mask, r):
    return cv2.dilate(mask.astype(np.uint8), disk(r)) > 0


def shrink(mask, r):
    return cv2.erode(mask.astype(np.uint8), disk(r)) > 0


def drop_specks(binary, ratio=SPECK_RATIO):
    """去掉面积不到整体 ratio 的碎块"""
    labels, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, labels, range(1, n + 1))
    return np.isin(labels, 1 + np.nonzero(sizes >= ratio * binary.sum())[0])


class Sam3Replicate:
    """SAM 3 按文字提示分割,走 Replicate;同一张源图只上传一次"""

    def __init__(self, token):
        self.token = token
        self.uploaded = {}
        self.next_start = 0.0

    def fetch(self, request):
        """取回响应内容;连接失败和限流(429)重试,其他 HTTP 错误照常抛出"""
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == RETRIES - 1:
                    raise
                time.sleep(float(error.headers.get('Retry-After') or RATE_LIMIT_WAIT))
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(3 * (attempt + 1))

    def call(self, method, url, body=None, headers=None):
        request = urllib.request.Request(url, data=body, method=method,
                                         headers={'Authorization': f'Bearer {self.token}', **(headers or {})})
        return json.loads(self.fetch(request))

    def upload(self, photo):
        key = hashlib.sha1(photo.tobytes()).hexdigest()
        if key not in self.uploaded:
            buffer = io.BytesIO()
            Image.fromarray(photo).save(buffer, 'PNG')
            boundary = 'segment-upload-boundary'
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="content"; filename="photo.png"\r\n'
                    'Content-Type: image/png\r\n\r\n').encode() + buffer.getvalue() + f'\r\n--{boundary}--\r\n'.encode()
            self.uploaded[key] = self.call('POST', f'{REPLICATE_API}/files', body,
                                           {'Content-Type': f'multipart/form-data; boundary={boundary}'})['urls']['get']
        return self.uploaded[key]

    def instances(self, photo, text):
        """文字提示 → 各实例的蒙版"""
        image = self.upload(photo)
        time.sleep(max(0.0, self.next_start - time.monotonic()))
        self.next_start = time.monotonic() + RATE_LIMIT_WAIT
        body = json.dumps({'version': SAM3_VERSION, 'input': {
            'image': image, 'prompt': text, 'mask_only': True, 'return_zip': True, 'threshold': 0.5}}).encode()
        prediction = self.call('POST', f'{REPLICATE_API}/predictions', body,
                               {'Content-Type': 'application/json', 'Prefer': 'wait=60'})
        while prediction['status'] not in ('succeeded', 'failed', 'canceled'):
            time.sleep(2)
            prediction = self.call('GET', prediction['urls']['get'])
        if prediction['status'] != 'succeeded':
            raise RuntimeError(f"SAM 3「{text}」: {prediction['status']} {prediction.get('error')}")
        output = prediction['output']
        if not output:
            return []
        data = self.fetch(output if isinstance(output, str) else output[0])  # 结果在 replicate.delivery,不带 token
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            return [np.asarray(Image.open(package.open(name)).convert('L')) > 127
                    for name in sorted(package.namelist()) if name.lower().endswith('.png')]


class Models:
    """ViTMatte 加载一次反复用;SAM 2.1 用到手点提示时才加载;给了 sam3 才能用文字提示"""

    def __init__(self, device=None, sam3=None):
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        self.torch = torch
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.sam3 = sam3
        self.sam2 = None
        self.matte_processor = VitMatteImageProcessor.from_pretrained(MATTE_MODEL)
        self.matte_model = VitMatteForImageMatting.from_pretrained(MATTE_MODEL).to(self.device).eval()

    def sam(self, photo, box, points):
        """SAM 2.1:外接框 + 提示点 → 每像素的前景概率"""
        if self.sam2 is None:
            from transformers import Sam2Model, Sam2Processor
            self.sam2 = Sam2Processor.from_pretrained(SAM_MODEL), Sam2Model.from_pretrained(SAM_MODEL).to(self.device).eval()
        processor, model = self.sam2
        inputs = processor(images=Image.fromarray(photo), input_boxes=[[box]],
                           input_points=[[[[x, y] for x, y, _ in points]]],
                           input_labels=[[[label for *_, label in points]]], return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            out = model(**inputs, multimask_output=False)
        logits = processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu(), binarize=False)[0]
        return self.torch.sigmoid(logits.squeeze().float()).numpy()

    def concept(self, photo, text):
        """SAM 3:文字提示 → 所有实例合成的蒙版"""
        if self.sam3 is None:
            raise ValueError(f'文字提示「{text}」要用 SAM 3')
        masks = self.sam3.instances(photo, text)
        return np.logical_or.reduce(masks) if masks else np.zeros(photo.shape[:2], bool)

    def matte(self, photo, fg, bg):
        """三分图 fg / bg 之外的像素由 ViTMatte 求透明度"""
        H, W = fg.shape
        unknown = ~fg & ~bg
        trimap = np.where(fg, 255, np.where(bg, 0, 128)).astype(np.uint8)
        total, weight = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        ramp = lambda k: np.minimum(np.arange(k) + 1, k - np.arange(k)).astype(np.float32)
        for ty in range(0, H, TILE - OVERLAP):
            for tx in range(0, W, TILE - OVERLAP):
                sl = (slice(ty, min(ty + TILE, H)), slice(tx, min(tx + TILE, W)))
                if not unknown[sl].any():
                    continue
                inputs = self.matte_processor(images=Image.fromarray(photo[sl]), trimaps=Image.fromarray(trimap[sl]),
                                              return_tensors='pt').to(self.device)
                with self.torch.no_grad():
                    pred = self.matte_model(**inputs).alphas[0, 0].float().cpu().numpy()
                h, w = trimap[sl].shape
                # 块边上缺上下文,按离块边的距离加权拼接
                feather = np.clip(np.minimum.outer(ramp(h), ramp(w)) / (OVERLAP / 2), 1e-3, 1)
                total[sl] += pred[:h, :w] * feather
                weight[sl] += feather
        return np.where(unknown, np.clip(total / np.maximum(weight, 1e-6), 0, 1), fg).astype(np.float32)


def prompted(models, photo, prompts):
    """每个提示各出一张蒙版,取并集"""
    union = np.zeros(photo.shape[:2], bool)
    for prompt in prompts:
        if 'text' in prompt:
            union |= models.concept(photo, prompt['text'])
        else:
            union |= models.sam(photo, prompt['box'], prompt['points']) > 0.5
    return union


def uses_text(spec):
    """配置里有没有文字提示,有就要给 Models 传 SAM 3"""
    prompts = list(spec.get('occluders', []))
    for item in spec['products']:
        prompts += [item, *item.get('parts', [])]
    return any('text' in prompt for prompt in prompts)


def segment(photo, spec, models):
    """photo:H×W×3 uint8。返回遮挡物蒙版,和每件商品的 {alpha: 透明度, parts: 不可印部件蒙版}"""
    W = photo.shape[1]
    band = max(1, round(BAND_RATIO * W))
    occluders = prompted(models, photo, spec.get('occluders', []))
    above = np.zeros(photo.shape[:2], bool)  # 已经分好的上层商品
    products = []
    for item in spec['products']:
        # 遮挡物和上层商品分完再减,不在提示里排除:手点时在包带这类窄的遮挡物上放负点,会把旁边的商品一起切掉。
        # 遮挡物可能把商品切成几块,只去碎块,不取最大连通块
        mask = drop_specks(ndimage.binary_fill_holes(prompted(models, photo, [item])) & ~occluders & ~above)
        parts = ndimage.binary_fill_holes(prompted(models, photo, item.get('parts', [])))
        parts = (cv2.morphologyEx(parts.astype(np.uint8), cv2.MORPH_OPEN, disk(round(SPUR_RATIO * W))) > 0) & mask
        # 待求带只留在商品和背景、遮挡物的交界;和上层商品的交界不算边缘,下层商品在上层下面接着铺
        alpha = models.matte(photo, shrink(mask, max(1, band // 3)) | (mask & grow(above, band)),
                             ~grow(mask, band) | occluders | above)
        products.append({'alpha': alpha, 'parts': parts})
        above |= alpha > 0.5
    return occluders, products
