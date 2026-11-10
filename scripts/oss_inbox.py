"""
OSS 收件箱(vorla-assets/mockup-template-inbox/)里有哪些模板目录。inbox.py --update 按这个找模板原来的目录,
fixes.py sync 按这个给同步正式库的命令分前缀。要用 ossutil(PATH 里或 ~/bin/ossutil)列整个收件箱,约 1 分钟。
"""
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath

INBOX = 'oss://vorla-assets/mockup-template-inbox/'


def inbox_dirs():
    """OSS 收件箱里有 template.json 的目录:{id: [(相对收件箱根目录的目录, 目录里的文件名)]}"""
    ossutil = shutil.which('ossutil') or str(Path.home() / 'bin' / 'ossutil')
    listing = subprocess.run([ossutil, 'ls', INBOX], capture_output=True, text=True, check=True).stdout
    files = defaultdict(set)
    for line in listing.splitlines():
        name = line.split()[-1] if line.strip() else ''
        if name.startswith(INBOX) and not name.endswith('/'):
            path = PurePosixPath(name.removeprefix(INBOX))
            files[str(path.parent)].add(path.name)
    found = defaultdict(list)
    for folder, names in sorted(files.items()):
        if 'template.json' in names:
            found[PurePosixPath(folder).name].append((folder, names))
    return found


def complete(places):
    """同一个 id 有几个目录时,有展示图和白模图的那份(发布脚本见同名目录几份都挡掉,另几份是不全的旧目录)"""
    return max(places, key=lambda place: {'cover.webp', 'white.png'} <= place[1])[0]


def prefix(places):
    """发布脚本 --batch 用的前缀(相对收件箱根目录):只有一个目录时是整个收件箱(空字符串);
    有几个时取 complete() 那份往上最浅、又不含其它几份的一层,发布脚本只扫到这一份和它的 manifest.csv"""
    if len(places) == 1:
        return ''
    folder = complete(places)
    others = [place for place, _ in places if place != folder]
    parts = PurePosixPath(folder).parts
    for depth in range(1, len(parts) + 1):
        candidate = '/'.join(parts[:depth]) + '/'
        if not any(other.startswith(candidate) for other in others):
            return candidate
    return folder + '/'
