"""
模板墙(mockup.listTemplates,每页至多 48 个):每个模板的版本号、显示名、小品类、标签、各区域的 Logo 框和最后发布的时间,
就是库里 template 表那一行。测试站读测试库,正式站读正式库;两边的文件在 OSS 同一个 bucket。
inbox.py 按测试站给已经发布过的模板升版本,fixes.py 按两边看修好的模板进了测试库、同步到正式库没有。
用法:.venv/bin/python -I scripts/wall.py [--prod] [存成.csv]   # 打出总数和各版本的个数,给了路径时每个模板一行存下来
"""
import csv
import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

SITES = {'test': 'https://vorla.drfans.com', 'prod': 'https://www.vorla.ai'}
API = '/api/trpc/mockup.listTemplates'
PAGE = 48  # 接口每页的上限


def page(number, site='test'):
    query = urllib.parse.urlencode({'input': json.dumps({'json': {'page': number, 'pageSize': PAGE}})})
    request = urllib.request.Request(f'{SITES[site]}{API}?{query}', headers={'User-Agent': 'mockup-template-pipeline'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)['result']['data']['json']
        except (OSError, ValueError, KeyError, http.client.HTTPException):  # HTTPException:响应读到一半断了(IncompleteRead)
            if attempt == 2:
                raise
            time.sleep(5)


def fetch(site='test'):
    """{id: {version, name, category, tags, logos, updated}};site 是 test 或 prod,
    logos 是各区域的 Logo 框,updated 是最后发布的时间(UTC,ISO 格式)"""
    first = page(1, site)
    with ThreadPoolExecutor(12) as pool:
        pages = [first, *pool.map(lambda n: page(n, site), range(2, first['totalPages'] + 1))]
    wall = {}
    for item in (item for p in pages for item in p['items']):
        wall[item['slug']] = {
            'version': int(re.search(r'/v(\d+)/', item['coverImage']).group(1)),
            'name': item['displayName'],
            'category': item['mockupSubcategory'],
            'tags': item['mockupTags'],
            'logos': [region['logo'] for region in item['materials']['regions']],
            'updated': item['updatedAt'],
        }
    if len(wall) != first['total']:
        # 翻页的同时有模板发布或下架,前后页会错开
        raise SystemExit(f"{site} 模板墙共 {first['total']} 个,只拉到 {len(wall)} 个不重复的,过一会儿再拉")
    return wall


if __name__ == '__main__':
    args = sys.argv[1:]
    site = 'prod' if '--prod' in args else 'test'
    args = [a for a in args if a != '--prod']
    wall = fetch(site)
    versions = Counter(item['version'] for item in wall.values())
    print(f"{'正式站' if site == 'prod' else '测试站'}模板墙 {len(wall)} 个:" + ','.join(f'v{v} {n} 个' for v, n in sorted(versions.items())))
    if args:
        with open(args[0], 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'version', 'name', 'category', 'tags', 'updated'])
            for key, item in sorted(wall.items()):
                writer.writerow([key, item['version'], item['name'], item['category'] or '', ';'.join(item['tags']), item['updated']])
        print(f'存到 {args[0]}')
