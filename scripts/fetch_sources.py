"""
按模板配置的 fetch 把源图等素材下载到配置里的路径,并按 sha256 核对是否与生成配置时的文件一致。
用法:.venv/bin/python -I scripts/fetch_sources.py batches/<批次>/specs/<id>.json
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
    print(f'{len(data) / 1024:8.1f} KB  {name}' + ('' if same else '  (与生成配置时的文件不同,产出可能对不上)'))
