"""
分割:SAM 3 按文字提示(或 SAM 2.1 按外接框和提示点)分出遮挡物、商品和商品的不可印部件,
ViTMatte 在商品和背景、遮挡物的交界求透明度。
不读写本地文件,也不含具体模板的内容;源图、提示和密钥由调用方给,见 run_segment.py。
SAM 3 走 Replicate,会把源图上传过去;给了 remote 时抠图也走 Replicate(见 remote.py)。

提示的格式(spec):
- 一个提示是 {text: 名词短语}(SAM 3,图里所有符合的实例合在一起),
  或 {box: [x0, y0, x1, y1], points: [[x, y, 1 正 / 0 负], ...]}(SAM 2.1,文字分不准时手点兜底);
- occluders:遮挡物的提示列表,合在一起求透明度;头发这类边上有细丝的,提示里加 matteBand(和商品交界的待求带半宽 / 宽度);
- products:商品列表,由上层到下层,每项本身是一个提示,加可选的 parts(不可印部件的提示列表,如袖子、领口)。
"""
import io, zipfile
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import remote
from remote import Replicate

SAM_MODEL = 'facebook/sam2.1-hiera-base-plus'
SAM3 = 'mattsays/sam3-image:d73db077226443ba4fafd34e233b3626b552eac2a433f90c7c32a9ac89bd9e72'
MATTE_MODEL = 'hustvl/vitmatte-small-composition-1k'
BAND_RATIO = 0.004  # 待求带向外的半宽 / 宽度,要盖住 SAM 的边界误差;向内收 1/3
MATTE_FLOOR = 0.1  # 遮挡物待求带宽时,抠图在纯色衣服上也会给出这么多以内的透明度:低于它算 0、高于 1 − 它算 1,中间线性拉开
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


class Sam3Replicate(Replicate):
    """SAM 3 按文字提示分割,走 Replicate"""

    def instances(self, photo, text):
        """文字提示 → 各实例的蒙版"""
        output = self.predict(SAM3, {'image': self.upload(photo), 'prompt': text, 'mask_only': True, 'return_zip': True,
                                     'threshold': 0.5})
        if not output:
            return []
        with zipfile.ZipFile(io.BytesIO(self.fetch(output if isinstance(output, str) else output[0]))) as package:
            return [np.asarray(Image.open(package.open(name)).convert('L')) > 127
                    for name in sorted(package.namelist()) if name.lower().endswith('.png')]


class Models:
    """ViTMatte 加载一次反复用,给了 remote(Replicate)就走 Replicate;SAM 2.1 用到手点提示时才加载;给了 sam3 才能用文字提示"""

    def __init__(self, device=None, sam3=None, remote=None):
        self.device = device
        self.sam3 = sam3
        self.remote = remote
        self.sam2 = None
        if remote is None:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
            self.load_torch()
            self.matte_processor = VitMatteImageProcessor.from_pretrained(MATTE_MODEL)
            self.matte_model = VitMatteForImageMatting.from_pretrained(MATTE_MODEL).to(self.device).eval()

    def load_torch(self):
        """本机推理才 import torch,并定下设备"""
        import torch
        self.torch = torch
        self.device = self.device or ('mps' if torch.backends.mps.is_available() else 'cpu')

    def sam(self, photo, box, points):
        """SAM 2.1:外接框 + 提示点 → 每像素的前景概率"""
        if self.sam2 is None:
            from transformers import Sam2Model, Sam2Processor
            self.load_torch()
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
        if self.remote is not None:
            return np.where(unknown, np.clip(remote.matte(self.remote, photo, trimap), 0, 1), fg).astype(np.float32)
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
    """photo:H×W×3 uint8。返回遮挡物的透明度,和每件商品的 {alpha: 透明度, parts: 不可印部件蒙版}"""
    W = photo.shape[1]
    band = max(1, round(BAND_RATIO * W))
    prompts = spec.get('occluders', [])
    masks = [prompted(models, photo, [prompt]) for prompt in prompts]
    occluders = np.logical_or.reduce(masks) if masks else np.zeros(photo.shape[:2], bool)
    raws = [ndimage.binary_fill_holes(prompted(models, photo, [item])) for item in spec['products']]
    # 遮挡物和商品的交界两边都待求,遮挡物当前景抠(抠图模型最擅长头发这类前景),商品在这里的覆盖度 = 1 − 遮挡物透明度。
    # 发丝伸出去多远按各自的 matteBand 给;离商品远的边不影响模板,不求
    near = np.logical_or.reduce(raws)
    unknown = np.zeros_like(occluders)
    for prompt, mask in zip(prompts, masks):
        r = max(1, round(prompt.get('matteBand', BAND_RATIO) * W))
        unknown |= grow(mask, r) & ~shrink(mask, r) & grow(near, r)
    hidden = models.matte(photo, occluders & ~unknown, ~occluders & ~unknown) if unknown.any() else occluders.astype(np.float32)
    hidden = np.clip((hidden - MATTE_FLOOR) / (1 - 2 * MATTE_FLOOR), 0, 1)
    covered = occluders | (hidden > 0.5)
    _, (iy, ix) = ndimage.distance_transform_edt(covered, return_indices=True)  # 被盖住的像素离它最近的未遮挡像素
    above = np.zeros(photo.shape[:2], bool)  # 已经分好的上层商品
    products = []
    for item, raw in zip(spec['products'], raws):
        # 遮挡物和上层商品分完再减,不在提示里排除:手点时在包带这类窄的遮挡物上放负点,会把旁边的商品一起切掉。
        # 遮挡物可能把商品切成几块,只去碎块,不取最大连通块
        mask = drop_specks(raw & ~covered & ~above)
        parts = ndimage.binary_fill_holes(prompted(models, photo, item.get('parts', [])))
        parts = (cv2.morphologyEx(parts.astype(np.uint8), cv2.MORPH_OPEN, disk(round(SPUR_RATIO * W))) > 0) & mask
        # 遮挡物下面按最近的未遮挡像素归属,商品在下面接着铺;商品和背景的交界照常抠,再乘上没被遮挡的那份。
        # 和上层商品的交界不算边缘,下层商品在上层下面接着铺
        spread = mask | (covered & mask[iy, ix])
        alpha = models.matte(photo, shrink(spread, max(1, band // 3)) | (spread & grow(above, band)),
                             ~grow(spread, band) | above) * (1 - hidden)
        products.append({'alpha': alpha, 'parts': parts})
        above |= alpha > 0.5
    return hidden, products
