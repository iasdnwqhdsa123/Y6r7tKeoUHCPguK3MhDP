from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
ARCHIVE_DAYS = 31
RUN_BY = {'hrrr': 12, 'gfs': 6, 'ecmwf': 0, 'icon': 6, 'icond2': 6}
SKIP = {'ens'}
DAY = re.compile('^\\d{8}$')
SND = re.compile('^(\\d{8})\\.pack$')

def run_time(run_id: str) -> datetime:
    return datetime.strptime(run_id, '%Y%m%d%H').replace(tzinfo=timezone.utc)

def day_for_run(model: str, run: datetime) -> str:
    t = run - timedelta(hours=RUN_BY.get(model, 6))
    d = t.date() if (t.hour, t.minute, t.second) == (0, 0, 0) else t.date() + timedelta(days=1)
    return f'{d:%Y%m%d}'

def trim_manifest(m: dict, day: str) -> dict:
    out = dict(m)
    stamps = list(m['days'][day])
    out['days'] = {day: stamps}
    out['quarters'] = {day: m['quarters'][day]} if day in (m.get('quarters') or {}) else {}
    snd = m.get('sounding')
    if snd and snd.get('days'):
        utc = {s[:8] for s in stamps}
        out['sounding'] = {**snd, 'days': {k: v for k, v in snd['days'].items() if k in utc}}
    out['older'] = []
    out.pop('hours', None)
    out['archive'] = {'day': day}
    return out

def wanted(rel: str, day: str, snd_days: set[str]) -> bool:
    parts = rel.split(os.sep)
    if parts[0] == 'h':
        return False
    if parts[0] == 'snd':
        m = SND.match(parts[-1]) if len(parts) == 2 else None
        if m:
            return m.group(1) in snd_days
        return len(parts) > 2 and parts[1] in snd_days
    if any((p.endswith('.building') or p.endswith('.tmp.npy') for p in parts)):
        return False
    return all((not DAY.match(p) or p == day for p in parts[:-1]))

def stage(dist: str, model: str, out: str) -> tuple[str, str] | None:
    if model in SKIP:
        return None
    base = os.path.join(dist, model)
    try:
        with open(os.path.join(base, 'latest.json')) as f:
            m = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    run_id = m.get('runId') or ''
    if not re.fullmatch('\\d{10}', run_id) or not m.get('complete', True):
        return None
    day = day_for_run(model, run_time(run_id))
    if not (m.get('days') or {}).get(day):
        return None
    rdir = os.path.join(base, run_id)
    if not os.path.isdir(rdir):
        return None
    am = trim_manifest(m, day)
    snd_days = set(((am.get('sounding') or {}).get('days') or {}).keys())
    dest = os.path.join(out, day, model)
    shutil.rmtree(dest, ignore_errors=True)
    for root, dirs, files in os.walk(rdir):
        dirs[:] = [d for d in dirs if not d.endswith('.building')]
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, rdir)
            if not wanted(rel, day, snd_days):
                continue
            to = os.path.join(dest, run_id, rel)
            os.makedirs(os.path.dirname(to), exist_ok=True)
            try:
                os.link(src, to)
            except OSError:
                shutil.copy2(src, to)
    with open(os.path.join(dest, 'latest.json'), 'w') as f:
        json.dump(am, f, separators=(',', ':'))
    return (day, run_id)

def _today(s: str | None) -> datetime:
    if s:
        return datetime.strptime(s, '%Y%m%d').replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

def _days_models(keys) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for k in keys:
        parts = k.strip().split('/')
        if len(parts) >= 3 and parts[-1] == 'latest.json' and DAY.match(parts[-3]) and (parts[-2] in RUN_BY):
            out.setdefault(parts[-3], set()).add(parts[-2])
    return out

def first_kept(today: datetime) -> str:
    return f'{today - timedelta(days=ARCHIVE_DAYS - 1):%Y%m%d}'

def build_index(keys, today: datetime) -> dict:
    lo, hi = (first_kept(today), f'{today + timedelta(days=1):%Y%m%d}')
    dm = _days_models(keys)
    days = [{'day': d, 'models': sorted(dm[d])} for d in sorted(dm, reverse=True) if lo <= d <= hi]
    return {'days': days, 'keep': ARCHIVE_DAYS, 'updated': datetime.now(timezone.utc).isoformat(timespec='seconds')}

def old_days(keys, today: datetime) -> list[str]:
    lo = first_kept(today)
    return sorted((d for d in _days_models(keys) if d < lo))

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--dist', default='dist-data')
    s.add_argument('--out', required=True)
    s.add_argument('models', nargs='+')
    for name in ('index', 'old'):
        p = sub.add_parser(name)
        p.add_argument('--keys', required=True)
        p.add_argument('--today')
    a = ap.parse_args(argv)
    if a.cmd == 'stage':
        for model in a.models:
            r = stage(a.dist, model, a.out)
            if r:
                print(f'{r[0]} {model} {r[1]}')
        return 0
    with open(a.keys) as f:
        keys = f.read().split()
    today = _today(a.today)
    if a.cmd == 'index':
        json.dump(build_index(keys, today), sys.stdout, separators=(',', ':'))
        sys.stdout.write('\n')
    else:
        print('\n'.join(old_days(keys, today)))
    return 0
if __name__ == '__main__':
    sys.exit(main())
