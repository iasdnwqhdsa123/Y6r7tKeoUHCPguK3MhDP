from __future__ import annotations
import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from airspace import VERSION as AIRSPACE_VERSION
from aviation import VERSION as AVIATION_VERSION
from sources import OPEN_AHEAD_H, OPEN_SLACK_H, PAUSED, REQUIRED, SOURCES, candidate_runs, latest_complete_run, needed_times
AIRSPACE_MAX_AGE = timedelta(days=7)
WAIT_MAX = timedelta(minutes=30)
WAIT_STEP = timedelta(minutes=2)
DUE = {'hrrr': timedelta(hours=1, minutes=45), 'gfs': timedelta(hours=5), 'icon': timedelta(hours=4, minutes=45), 'icond2': timedelta(hours=1, minutes=45), 'ecmwf': timedelta(hours=9, minutes=45)}
LATE_FOR = timedelta(hours=2)

def late_run(src, now=None):
    now = now or datetime.now(timezone.utc)
    run, _ = latest_complete_run(src, needed_times(src))
    due = DUE.get(src.id)
    landing = None
    if due is not None:
        for cand in candidate_runs(src, now):
            if run is not None and cand <= run:
                break
            if due <= now - cand <= due + LATE_FOR:
                landing = cand
                break
    return (run, landing)

def newest_run(src, wait=True):
    start = datetime.now(timezone.utc)
    run, landing = late_run(src)
    while wait and landing is not None and (datetime.now(timezone.utc) - start + WAIT_STEP <= WAIT_MAX):
        print(f'{src.id}: waiting for {landing:%Y-%m-%d %HZ} to land')
        time.sleep(WAIT_STEP.total_seconds())
        run, landing = late_run(src)
    return run

def published(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wx-pipeline/1.0', 'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None

def map_data_due(ages: list, todo: list[str]) -> bool:
    if any((a is None for a in ages)):
        return True
    oldest = max(ages)
    if oldest > AIRSPACE_MAX_AGE + timedelta(days=1):
        return True
    return oldest > AIRSPACE_MAX_AGE and bool(todo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-url', required=True)
    ap.add_argument('--open-url', help='where the open data (free, airspace) is published, if not under --data-url')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no-wait', action='store_true', help="don't wait for a run that's landing late")
    ap.add_argument('--missing-only', action='store_true', help="only build what isn't published at all")
    args = ap.parse_args()
    base = args.data_url.rstrip('/')
    open_base = (args.open_url or base).rstrip('/')
    todo = []
    for mid, src in SOURCES.items():
        if mid in PAUSED:
            continue
        cur = published(f'{base}/{mid}/latest.json')
        run = newest_run(src, wait=not args.no_wait)
        have = cur.get('runId') if cur and cur.get('complete') else None
        want = f'{run:%Y%m%d%H}' if run else None
        print(f'{mid}: published {have or '-'}, newest complete {want or '-'}')
        if want and (args.force or want != have) and (not args.missing_only or have is None):
            todo.append(mid)
    free_due = False
    for mid, ahead in OPEN_AHEAD_H.items():
        if mid in PAUSED:
            continue
        fm = published(f'{open_base}/free/{mid}/latest.json')
        until = fm and fm.get('freeUntil')
        span = timedelta(hours=ahead + OPEN_SLACK_H - 1)
        behind = not until or (not args.missing_only and datetime.now(timezone.utc) + span >= datetime.fromisoformat(until))
        print(f'free {mid}: until {until or '-'}{(' → refresh' if behind else '')}')
        free_due = free_due or behind
    now = datetime.now(timezone.utc)
    ages = []
    for name, url, version in (('airspace', f'{open_base}/airspace/meta.json', AIRSPACE_VERSION), ('aviation map data', f'{open_base}/airspace/aviation/meta.json', AVIATION_VERSION)):
        meta = published(url)
        updated = meta and meta.get('updated')
        age = now - datetime.fromisoformat(updated) if updated and meta.get('version') == version else None
        print(f'{name}: updated {updated or '-'}{('' if age is not None else ' (missing or new version)')}')
        ages.append(age)
    airspace = args.force or (any((a is None for a in ages)) if args.missing_only else map_data_due(ages, todo))
    aviation = airspace
    print(f'map data: {('refresh' if airspace else 'keep')}')
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a') as f:
            f.write(f'models={' '.join(todo)}\n')
            f.write(f'matrix={json.dumps(todo)}\n')
            f.write(f'active={json.dumps([m for m in SOURCES if m not in PAUSED])}\n')
            f.write(f'required={json.dumps([m for m in REQUIRED if m not in PAUSED])}\n')
            f.write(f'airspace={('true' if airspace else 'false')}\n')
            f.write(f'aviation={('true' if aviation else 'false')}\n')
            f.write(f'free={('true' if free_due or todo else 'false')}\n')
            f.write(f'any={('true' if todo or airspace or aviation or free_due else 'false')}\n')
    print('build:', ' '.join(todo) or 'nothing')
if __name__ == '__main__':
    main()
