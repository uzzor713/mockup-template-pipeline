"""
按模板配置的 fetch 把源图等素材下载到配置里的路径,并核对是否与生成样本时的文件一致。
样本用的竞品素材只作内部示例,不进仓库、不用于产品。
用法:.venv/bin/python -I fetch_sources.py specs/<模板>.json
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # -I 运行时脚本所在目录不在 sys.path 里
import config

spec = config.load(sys.argv[1])
for name, item in spec.get('fetch', {}).items():
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    path = config.ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    same = hashlib.sha256(data).hexdigest() == item['sha256']
    print(f'{len(data) / 1024:8.1f} KB  {name}' + ('' if same else '  (与样本生成时的文件不同,产出可能对不上)'))
