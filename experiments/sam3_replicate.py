"""
试验:Replicate 上的 SAM 3(mattsays/sam3-image)按文字提示分割,看袖子、领口这类没有明显边界的部件能不能分准。
每个提示存下各个实例的蒙版和一张叠图,结果在 work/<模板 id>/experiments/sam3/。会把源图上传到 Replicate。
token 取环境变量 REPLICATE_API_TOKEN,没有就读仓库根目录 .env.local 里的同名一行(不进仓库)。
用法:.venv/bin/python -I experiments/sam3_replicate.py specs/<模板>.json [提示 ...]
"""
import io, json, os, sys, time, zipfile, urllib.error, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))  # -I 运行时仓库根目录不在 sys.path 里
import config

VERSION = 'd73db077226443ba4fafd34e233b3626b552eac2a433f90c7c32a9ac89bd9e72'
API = 'https://api.replicate.com/v1'


def env_local(name):
    path = config.ROOT / '.env.local'
    lines = path.read_text().splitlines() if path.exists() else []
    values = [line.split('=', 1)[1].strip().strip('"\'') for line in lines if line.startswith(f'{name}=')]
    if not values or not values[-1]:
        raise SystemExit(f'缺少 {name}:设环境变量,或写进 {path}')
    return values[-1]


TOKEN = os.environ.get('REPLICATE_API_TOKEN') or env_local('REPLICATE_API_TOKEN')
PROMPTS = sys.argv[2:] or ['t-shirt', 'sleeve', 'neckline', 'bag', 'bag strap', 'jeans']
COLORS = [(0, 200, 80), (255, 150, 0), (40, 120, 255), (220, 40, 40), (170, 60, 220), (0, 190, 200)]
RATE_LIMIT_WAIT = 11  # 余额低的账号每分钟限 6 次
RETRIES = 4  # 网络偶发 TLS 握手失败、连接中断


def fetch(request):
    """取回响应内容;连接层面的失败重试,HTTP 错误照常抛出"""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(3 * (attempt + 1))


def call(method, url, body=None, headers=None):
    request = urllib.request.Request(url, data=body, method=method,
                                     headers={'Authorization': f'Bearer {TOKEN}', **(headers or {})})
    return json.loads(fetch(request))


def upload(path):
    boundary = 'sam3-experiment-boundary'
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="content"; filename="{path.name}"\r\n'
            'Content-Type: image/png\r\n\r\n').encode()
    body = head + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    return call('POST', f'{API}/files', body, {'Content-Type': f'multipart/form-data; boundary={boundary}'})['urls']['get']


def predict(image, prompt):
    """返回各实例的蒙版和用时"""
    body = json.dumps({'version': VERSION, 'input': {
        'image': image, 'prompt': prompt, 'mask_only': True, 'return_zip': True, 'threshold': 0.5}}).encode()
    prediction = call('POST', f'{API}/predictions', body, {'Content-Type': 'application/json', 'Prefer': 'wait=60'})
    while prediction['status'] not in ('succeeded', 'failed', 'canceled'):
        time.sleep(2)
        prediction = call('GET', prediction['urls']['get'])
    if prediction['status'] != 'succeeded':
        raise RuntimeError(f"{prompt}: {prediction['status']} {prediction.get('error')}")
    output = prediction['output']
    url = output if isinstance(output, str) else output[0]
    data = fetch(url)  # 结果在 replicate.delivery,不带 token
    masks = []
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        for name in sorted(package.namelist()):
            if name.lower().endswith('.png'):
                masks.append(np.asarray(Image.open(package.open(name)).convert('L')) > 127)
    return masks, prediction.get('metrics', {}).get('predict_time', 0)


spec = config.load(sys.argv[1])
OUT = config.folder(spec, 'work') / 'experiments' / 'sam3'
OUT.mkdir(parents=True, exist_ok=True)
photo, _ = config.photo(spec)
photo = np.round(photo * 255).astype(np.uint8)
Image.fromarray(photo).save(OUT / 'photo.png')

schema = call('GET', f'{API}/models/mattsays/sam3-image/versions/{VERSION}')['openapi_schema']['components']['schemas']
print('input:', json.dumps(schema['Input']['properties'], ensure_ascii=False)[:1500])

image = upload(OUT / 'photo.png')
panels = []
for prompt in PROMPTS:
    masks, seconds = predict(image, prompt)
    print(f'{prompt}: {len(masks)} instances, areas {[int(m.sum()) for m in masks]}, {seconds:.1f}s')
    overlay = photo.astype(np.float32)
    for index, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8) * 255).save(OUT / f"{prompt.replace(' ', '-')}-{index}.png")
        overlay[mask] = overlay[mask] * 0.45 + np.array(COLORS[index % len(COLORS)]) * 0.55
    panel = Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).resize((400, 500), Image.LANCZOS)
    ImageDraw.Draw(panel).text((8, 8), f'{prompt} ({len(masks)})', fill=(0, 0, 0))
    panels.append(panel)
    time.sleep(RATE_LIMIT_WAIT)

sheet = Image.new('RGB', (400 * len(panels), 500), 'white')
for index, panel in enumerate(panels):
    sheet.paste(panel, (400 * index, 0))
sheet.save(OUT / 'sheet.png')
print('wrote', OUT / 'sheet.png')
