"""Measure planar artwork perspective against a visible quadrilateral.

Usage: .venv/bin/python -I scripts/plane_perspective_scan.py [--only ids.txt] [--out report.csv]
The score is the difference in projective denominator across the logo span.
An inset print mask is only a candidate reference and needs photo review.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
QUAD_ERROR = 0.02
PERSPECTIVE_ERROR = 0.05
FIT_ERROR_P90 = 4.0


def alpha(path):
    return np.asarray(Image.open(path).convert('RGBA'))[..., 3] > 127


def outline(mask, center):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)[:, 0].astype(np.float64)
    area = cv2.contourArea(contour)
    if len(quad) != 4 or not cv2.isContourConvex(quad.astype(np.float32)) or area <= 0:
        return None
    if abs(area - cv2.contourArea(quad.astype(np.float32))) / area > QUAD_ERROR:
        return None
    ordered = quad[np.argsort(quad[:, 1])]
    top, bottom = (pair[np.argsort(pair[:, 0])] for pair in (ordered[:2], ordered[2:]))
    tl, tr, bl, br = *top, *bottom
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width < 100 or height < 100 or abs(tr[1] - tl[1]) > width / 4 or abs(br[1] - bl[1]) > width / 4 \
            or abs(bl[0] - tl[0]) > height / 4 or abs(br[0] - tr[0]) > height / 4:
        return None
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    homography = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), target)
    return gradient(homography, center), (width, height)


def gradient(homography, center):
    denominator = homography[2] @ np.array([*center, 1])
    return homography[2, :2] / denominator


def used_perspective(path, mask, center, range_px):
    rgb = np.asarray(Image.open(path).convert('RGB'), np.float64)
    sh, sw = rgb.shape[:2]
    cell = mask.shape[1] / sw
    if abs(cell - mask.shape[0] / sh) > 0.01:
        return None
    inside = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    yy, xx = np.nonzero(inside)
    if len(xx) < 50:
        return None
    pick = np.arange(0, len(xx), max(1, len(xx) // 5000))
    xx, yy = xx[pick], yy[pick]
    camera = np.stack([(xx + 0.5) * cell - 0.5, (yy + 0.5) * cell - 0.5], -1)
    flow = (rgb[yy, xx, :2] / 255 - 0.5) * (2 * range_px)
    design = camera + flow
    homography, _ = cv2.findHomography(camera.astype(np.float64), design.astype(np.float64), 0)
    if homography is None:
        return None
    predicted = cv2.perspectiveTransform(camera[None].astype(np.float32), homography)[0]
    error = np.percentile(np.linalg.norm(predicted - design, axis=1), 90)
    return gradient(homography, center), float(error)


def scan(path):
    template = json.loads(path.read_text())
    root = path.parent
    size = template['size']
    width, height = size['width'], size['height']
    fits = template.get('pipeline', {}).get('surfaceFit', [])
    rows = []
    for index, region in enumerate(template['regions']):
        if index >= len(fits) or fits[index] != 'plane':
            continue
        printable = alpha(root / region['print'])
        ys, xs = np.nonzero(printable)
        if not len(xs):
            continue
        center = (float(xs.mean()), float(ys.mean()))
        product = alpha(root / region['product'])
        print_shape = outline(printable, center)
        product_shape = outline(product, center)
        row = {'id': template['id'], 'region': index + 1, 'category': template['category'],
               'printQuad': bool(print_shape), 'productQuad': bool(product_shape),
               'printAreaOfProduct': round(float(printable.sum() / max(product.sum(), 1)), 3),
               'fitP90Px': '', 'scoreX': '', 'scoreY': '', 'score': '', 'status': 'no_print_quad'}
        if print_shape:
            actual = used_perspective(root / region['surface'], printable, center,
                                      template['displacement']['surfaceRangePx'])
            if actual:
                actual_h, error = actual
                span = np.minimum([region['logo']['width'] * width, region['logo']['height'] * height],
                                  [xs.max() - xs.min(), ys.max() - ys.min()])
                scores = np.abs(actual_h - print_shape[0]) * span
                row.update(fitP90Px=round(error, 2), scoreX=round(float(scores[0]), 4),
                           scoreY=round(float(scores[1]), 4), score=round(float(scores.max()), 4))
                row['status'] = 'unreliable_flow' if error > FIT_ERROR_P90 else \
                    'review_reference' if row['printAreaOfProduct'] < 0.85 else \
                    'mismatch' if scores.max() >= PERSPECTIVE_ERROR else 'aligned'
            else:
                row['status'] = 'unreliable_flow'
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=Path, help='one template id per line')
    parser.add_argument('--ids', help='comma-separated template ids')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    paths = sorted((ROOT / 'output').glob('*/template.json'))
    if args.only:
        ids = {line.strip() for line in args.only.read_text().splitlines() if line.strip() and not line.startswith('#')}
        paths = [path for path in paths if path.parent.name in ids]
    if args.ids:
        ids = set(args.ids.split(','))
        paths = [path for path in paths if path.parent.name in ids]
    rows = [row for path in paths for row in scan(path)]
    from collections import Counter
    print(f'{len(paths)} templates, {len(rows)} plane regions: {dict(Counter(row["status"] for row in rows))}')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', newline='') as file:
            writer = csv.DictWriter(file, rows[0].keys() if rows else ['id'])
            writer.writeheader()
            writer.writerows(rows)
        print(args.out)


if __name__ == '__main__':
    main()
