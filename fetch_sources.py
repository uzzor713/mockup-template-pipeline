"""
把样本用的竞品素材下载到 sources/。竞品素材只作内部示例,不进仓库、不用于产品。
用法:.venv/bin/python -I fetch_sources.py
"""
import hashlib
import urllib.request
from pathlib import Path

SOURCES = Path(__file__).parent / 'sources'
# 竞品 Mockey 免费模板 04674:底图(商品区半透明)和可印区域蒙版。地址后缀是 .png,内容是 webp
FILES = {
    'tee-base.webp': (
        'https://assets.mockey.ai/mockups/tshirt/1790268599918-model-4674.png',
        'b9ff73138fe9d03cf3b5f54948396d2821c206fe70ad4e34b6f217978e44c33b',
    ),
    'tee-print.webp': (
        'https://assets.mockey.ai/mockups/tshirt/1790268600969-model-4674-mask-design.png',
        '5fb44b990a0454fb0f565d33a344675d134d933a55665000aa00b0bb4292981e',
    ),
}

SOURCES.mkdir(exist_ok=True)
for name, (url, expected) in FILES.items():
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    (SOURCES / name).write_bytes(data)
    same = hashlib.sha256(data).hexdigest() == expected
    print(f'{len(data) / 1024:8.1f} KB  {name}' + ('' if same else '  (与样本生成时的文件不同,产出可能对不上)'))
