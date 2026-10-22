"""
分割:SAM 2.1 按外接框和提示点分出遮挡物、商品和商品的不可印部件,ViTMatte 在商品和背景、遮挡物的交界求透明度。
只有函数,不读写文件,也不含具体模板的内容;源图和提示由调用方给,见 run_segment.py。

提示的格式(spec):
- occluders:遮挡物的提示列表,每项 {box: [x0, y0, x1, y1], points: [[x, y, 1 正 / 0 负], ...]},合成一张蒙版;
- products:商品列表,由上层到下层,每项是一组提示加可选的 parts(不可印部件的提示列表,如袖子)。
"""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

SAM_MODEL = 'facebook/sam2.1-hiera-base-plus'
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


class Models:
    """SAM 2.1 和 ViTMatte,加载一次反复用"""

    def __init__(self, device=None):
        import torch
        from transformers import Sam2Model, Sam2Processor, VitMatteForImageMatting, VitMatteImageProcessor
        self.torch = torch
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.sam_processor = Sam2Processor.from_pretrained(SAM_MODEL)
        self.sam_model = Sam2Model.from_pretrained(SAM_MODEL).to(self.device).eval()
        self.matte_processor = VitMatteImageProcessor.from_pretrained(MATTE_MODEL)
        self.matte_model = VitMatteForImageMatting.from_pretrained(MATTE_MODEL).to(self.device).eval()

    def sam(self, photo, box, points):
        """外接框 + 提示点 → 每像素的前景概率"""
        inputs = self.sam_processor(images=Image.fromarray(photo), input_boxes=[[box]],
                                    input_points=[[[[x, y] for x, y, _ in points]]],
                                    input_labels=[[[label for *_, label in points]]], return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            out = self.sam_model(**inputs, multimask_output=False)
        logits = self.sam_processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu(), binarize=False)[0]
        return self.torch.sigmoid(logits.squeeze().float()).numpy()

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
    """每组提示各出一张蒙版,取并集"""
    union = np.zeros(photo.shape[:2], bool)
    for prompt in prompts:
        union |= models.sam(photo, prompt['box'], prompt['points']) > 0.5
    return union


def segment(photo, spec, models):
    """photo:H×W×3 uint8。返回遮挡物蒙版,和每件商品的 {alpha: 透明度, parts: 不可印部件蒙版}"""
    W = photo.shape[1]
    band = max(1, round(BAND_RATIO * W))
    occluders = prompted(models, photo, spec.get('occluders', []))
    above = np.zeros(photo.shape[:2], bool)  # 已经分好的上层商品
    products = []
    for item in spec['products']:
        # 遮挡物和上层商品分完再减:窄的遮挡物(如包带)上放负点,会把旁边的商品一起切掉。
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
