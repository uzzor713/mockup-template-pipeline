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
  文字提示分出几个实例时(如书架上的书、背景里路人的包),交给 Gemini 看编号叠图挑出要展示的那几个;
  可选的 alternatives 是备选说法,text 一个实例都分不出时按顺序换着试;
  可选的 instance 只取挑剩的实例里从左往右第几个(从 0 数),同一张图里的两件同类商品(正反两面、两只杯子)各成一件。
"""
import io, json, re, zipfile
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import remote
from remote import Replicate

SAM_MODEL = 'facebook/sam2.1-hiera-base-plus'
SAM3 = 'mattsays/sam3-image:d73db077226443ba4fafd34e233b3626b552eac2a433f90c7c32a9ac89bd9e72'
GEMINI = 'google/gemini-3-flash:61ebd56fbf92e15093c9abae6e4437e5c06260bd64f839e784f68d9d7f3e95b0'
PICK_SIZE = 1024  # 交给 Gemini 挑实例的图,长边缩到这么大
PICK_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]
PICK_PROMPT = '''Both images are the same photo, used to make a mockup template for: {what}.
In the second image, candidate regions are tinted and labeled with numbers.
Which numbers belong to the {what} being showcased, i.e. the item a customer would put their own design on?
It is usually the most prominent, blank (unprinted) item. Leave out props, background objects and other items,
even of the same kind, that are only in the background. If several items are showcased together, keep all of them;
if a candidate covers only part of a showcased item, keep it too.
Answer with JSON only, like {{"keep": [1, 3]}}.'''
MATTE_MODEL = 'hustvl/vitmatte-small-composition-1k'
BAND_RATIO = 0.004  # 待求带向外的半宽 / 宽度,要盖住 SAM 的边界误差;向内收 1/3
MATTE_FLOOR = 0.1  # 遮挡物待求带宽时,抠图在纯色衣服上也会给出这么多以内的透明度:低于它算 0、高于 1 − 它算 1,中间线性拉开
SPECK_RATIO = 0.01  # 面积不到整件这么多的碎块去掉
HOLE_RATIO = 0.002  # 商品蒙版里面积不到整件这么多的洞补上;更大的是包带、提手围出来的背景,留着
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


def fill_small_holes(binary, ratio=HOLE_RATIO):
    """补上面积不到整体 ratio 的洞"""
    holes = ndimage.binary_fill_holes(binary) & ~binary
    labels, n = ndimage.label(holes)
    sizes = ndimage.sum(holes, labels, range(1, n + 1))
    return binary | np.isin(labels, 1 + np.nonzero(sizes < ratio * binary.sum())[0])


def touching(binary, mask):
    """binary 里碰到 mask 的连通块"""
    labels, _ = ndimage.label(binary)
    return np.isin(labels, np.unique(labels[mask & binary])) & binary


def despur(parts, r):
    """开运算去掉毛刺;开运算后剩不到一半的块本身就细(如绳子提手),原样留着"""
    opened = cv2.morphologyEx(parts.astype(np.uint8), cv2.MORPH_OPEN, disk(r)) > 0
    labels, n = ndimage.label(parts)
    kept = ndimage.sum(opened, labels, range(1, n + 1)) / ndimage.sum(parts, labels, range(1, n + 1))
    return opened | np.isin(labels, 1 + np.nonzero(kept < 0.5)[0])


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

    def pick(self, photo, masks, what):
        """SAM 3 分出几个实例时,把它们编号画在图上,和原图一起交给 Gemini,返回要展示的商品是哪几个(下标);
        Gemini 说一个都不是时返回空列表,回答看不懂时全留"""
        h, w = photo.shape[:2]
        k = min(1, PICK_SIZE / max(h, w))
        size = (round(w * k), round(h * k))
        small = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
        tinted = small.astype(np.float32)
        labels = []
        for index, mask in enumerate(masks):
            m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
            color = np.array(PICK_COLORS[index % len(PICK_COLORS)], np.float32)
            tinted[m] = tinted[m] * 0.5 + color * 0.5
            if m.any():
                # 编号写在实例最里面的点(离边最远),细长的实例也落在它自己身上
                dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                y, x = np.unravel_index(np.argmax(dist), dist.shape)
                labels.append((str(index + 1), int(x), int(y), color))
        tinted = tinted.astype(np.uint8)
        for text, x, y, color in labels:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(tinted, (x - tw // 2 - 6, y - th // 2 - 8), (x + tw // 2 + 6, y + th // 2 + 8), (255, 255, 255), -1)
            cv2.putText(tinted, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple(int(c) for c in color), 3)
        output = self.predict(GEMINI, {'prompt': PICK_PROMPT.format(what=what), 'images': [self.upload(small), self.upload(tinted)],
                                       'temperature': 0, 'max_output_tokens': 2000})
        answer = ''.join(output) if isinstance(output, list) else str(output)
        try:
            keep = sorted({int(n) - 1 for n in json.loads(re.search(r'\{.*\}', answer, re.S).group())['keep']} & set(range(len(masks))))
        except (AttributeError, ValueError, KeyError, TypeError):
            keep = None
        print(f'{what}:SAM 3 分出 {len(masks)} 个实例,Gemini 留下 {"看不懂回答,全留" if keep is None else [i + 1 for i in keep] or "一个都不是"}'
              f'({answer.strip()})')
        return list(range(len(masks))) if keep is None else keep


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


def product_mask(models, photo, item, what, found):
    """一件商品的蒙版。text 分出几个实例时由 Gemini 挑出要展示的那几个;一个实例都分不出、或 Gemini 说一个都不是时,
    按顺序换 alternatives 里的说法,用第一个挑得出的;写了 instance 的只取挑剩的里从左往右第几个;
    found 存每个提示词挑好的实例,同一张图里两件同类商品只分、只挑一次"""
    if 'text' not in item:
        return prompted(models, photo, [item])
    if models.sam3 is None:
        raise ValueError(f"文字提示「{item['text']}」要用 SAM 3")
    if item['text'] not in found:
        for text in [item['text'], *item.get('alternatives', [])]:
            masks = [m for m in models.sam3.instances(photo, text) if m.any()]
            if len(masks) > 1:
                masks = [masks[i] for i in models.sam3.pick(photo, masks, what)]
            if masks:
                if text != item['text']:
                    print(f"「{item['text']}」分不出,换成「{text}」")
                break
        found[item['text']] = sorted(masks, key=lambda m: np.nonzero(m)[1].mean())
    masks = found[item['text']]
    if 'instance' in item:
        if item['instance'] >= len(masks):
            raise ValueError(f"「{item['text']}」只挑出 {len(masks)} 个实例,没有从左数第 {item['instance'] + 1} 个")
        masks = [masks[item['instance']]]
    return np.logical_or.reduce(masks) if masks else np.zeros(photo.shape[:2], bool)


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
    # 部件(如提手)可能伸出商品的蒙版,碰到商品的部件块并进商品。只补小洞:包带、提手和包身围出来的是背景
    raws, part_raws = [], []
    found = {}
    for item in spec['products']:
        raw = product_mask(models, photo, item, f"{item.get('text', item.get('name'))} (a {spec.get('category', 'product').replace('-', ' ')} mockup)", found)
        parts = ndimage.binary_fill_holes(prompted(models, photo, item.get('parts', [])))
        raws.append(fill_small_holes(raw | touching(parts, grow(raw, band))))
        part_raws.append(parts)
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
    for raw, parts in zip(raws, part_raws):
        # 遮挡物和上层商品分完再减,不在提示里排除:手点时在包带这类窄的遮挡物上放负点,会把旁边的商品一起切掉。
        # 遮挡物可能把商品切成几块,只去碎块,不取最大连通块
        mask = drop_specks(raw & ~covered & ~above)
        parts = despur(parts, round(SPUR_RATIO * W)) & mask
        # 遮挡物下面按最近的未遮挡像素归属,商品在下面接着铺;商品和背景的交界照常抠,再乘上没被遮挡的那份。
        # 和上层商品的交界不算边缘,下层商品在上层下面接着铺
        spread = mask | (covered & mask[iy, ix])
        alpha = models.matte(photo, shrink(spread, max(1, band // 3)) | (spread & grow(above, band)),
                             ~grow(spread, band) | above) * (1 - hidden)
        products.append({'alpha': alpha, 'parts': parts})
        above |= alpha > 0.5
    return hidden, products
