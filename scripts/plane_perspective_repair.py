"""Rebuild photo-verified planar perspective candidates and keep only measured improvements.

Usage: .venv/bin/python -I scripts/plane_perspective_repair.py
Reads fixes/perspective-plane-confirmed-2026-11-12.txt and the scan CSV. No model steps run.
"""

import csv
import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plane_perspective_scan import ROOT, scan

PLAN = ROOT / 'fixes' / 'plan'
BASELINE = PLAN / 'perspective-scan-2026-11-12.csv'
CONFIRMED = ROOT / 'fixes' / 'perspective-plane-confirmed-2026-11-12.txt'
BACKUP = PLAN / 'perspective-2026-11-12-before'
LOGS = PLAN / 'perspective-2026-11-12-logs'
RESULT = PLAN / 'perspective-2026-11-12-repair.csv'


def copy_files(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def restore_files(source, target):
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def run_one(template_id, before):
    steps = json.loads((ROOT / 'work' / template_id / 'steps.json').read_text())
    spec = ROOT / 'batches' / steps['batch'] / 'specs' / f'{template_id}.json'
    if not spec.exists():
        return {'id': template_id, 'status': 'missing_spec', 'oldScore': '', 'newScore': '', 'reason': str(spec)}
    old_score = max(float(row['score']) for row in before if row['score'])
    output = ROOT / 'output' / template_id
    preview = ROOT / 'preview' / template_id
    copy_files(output, BACKUP / template_id / 'output')
    if preview.exists():
        copy_files(preview, BACKUP / template_id / 'preview')
    log = LOGS / f'{template_id}.log'
    try:
        with log.open('w') as stream:
            for script in ('pipeline.py', 'preview.py'):
                command = [sys.executable, '-I', str(ROOT / 'scripts' / script), str(spec)]
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{script} exited {result.returncode}')
        checks = json.loads((output / 'checks.json').read_text())
        if not checks or not all(checks.values()):
            raise RuntimeError('material checks failed')
        after = {row['region']: row for row in scan(output / 'template.json')}
        bad = [row for row in before if row['status'] == 'mismatch' and
               (row['region'] not in after or after[row['region']]['status'] != 'aligned')]
        if bad:
            raise RuntimeError('perspective score remains above 0.05')
        new_score = max(float(after[row['region']]['score']) for row in before if row['score'])
        return {'id': template_id, 'status': 'repaired', 'oldScore': old_score,
                'newScore': round(new_score, 4), 'reason': ''}
    except Exception as error:
        restore_files(BACKUP / template_id / 'output', output)
        if (BACKUP / template_id / 'preview').exists():
            restore_files(BACKUP / template_id / 'preview', preview)
        return {'id': template_id, 'status': 'needs_review', 'oldScore': old_score,
                'newScore': '', 'reason': str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', help='comma-separated template ids for a follow-up pass')
    args = parser.parse_args()
    result_path = RESULT if not args.ids else PLAN / 'perspective-2026-11-12-repair-followup.csv'
    confirmed = {line.strip() for line in CONFIRMED.read_text().splitlines()
                 if line.strip() and not line.startswith('#')}
    if args.ids:
        confirmed &= set(args.ids.split(','))
    baseline = defaultdict(list)
    for row in csv.DictReader(BASELINE.open()):
        if row['id'] in confirmed:
            row['region'] = int(row['region'])
            baseline[row['id']].append(row)
    candidates = {template_id: rows for template_id, rows in baseline.items()
                  if any(row['status'] == 'mismatch' for row in rows)
                  and all(row['status'] in ('mismatch', 'aligned') for row in rows)}
    BACKUP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f'{len(candidates)} photo-verified, measurable templates queued', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_one, template_id, rows): template_id
                   for template_id, rows in sorted(candidates.items())}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(candidates)} {result['id']} {result['status']} "
                  f"{result['oldScore']} -> {result['newScore']} {result['reason']}", flush=True)
            with result_path.open('w', newline='') as file:
                writer = csv.DictWriter(file, ['id', 'status', 'oldScore', 'newScore', 'reason'])
                writer.writeheader()
                writer.writerows(sorted(results, key=lambda row: row['id']))
    print(result_path)


if __name__ == '__main__':
    main()
