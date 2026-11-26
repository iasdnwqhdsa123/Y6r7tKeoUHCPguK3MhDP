from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blend
import build
import opslake
import verify
UTC = timezone.utc
REGIONS = {'conus': {'base': 'hrrr', 'model': 'blend', 'lead_max': 48, 'run_hours': (0, 6, 12, 18), 'box': None}, 'sh': {'base': 'ecmwf', 'model': 'blendsh', 'lead_max': 48, 'run_hours': (0, 12), 'box': (112.0, -49.0, 179.75, -18.5)}, 'alps': {'base': 'icond2', 'model': 'blendalps', 'lead_max': 48, 'run_hours': (0, 6, 12, 18), 'box': (5.5, 43.5, 16.0, 48.5)}}
CLASS_CODES = ('cuclass',)
KEEP_OUT = ('overdev', 'elevation')
KEY = re.compile('^cache/blendin/v=(\\d+)/([a-z0-9]+)/(\\d{10})/([a-z]+)\\.npz$')

def log(*a):
    print('blend leg:', *a, flush=True)

def parse_run(s: str) -> datetime:
    return datetime.strptime(s, '%Y%m%d%H').replace(tzinfo=UTC)

def utc(s: str | None) -> datetime:
    if not s:
        return datetime.now(UTC)
    t = datetime.fromisoformat(s.replace('Z', '+00:00'))
    return t if t.tzinfo else t.replace(tzinfo=UTC)

def region_grid(region: str) -> tuple:
    cfg = REGIONS[region]
    full = tuple(build.REGIONS[cfg['base']][:5])
    if cfg['box'] is None:
        return (full, None)
    win = opslake.conus_window(full, cfg['box'])
    if win is None:
        raise ValueError(f"blend leg: {region}'s box is off {cfg['base']}'s grid")
    y0, y1, x0, x1 = win
    w, s, e, n, step = full
    cut = (round(w + x0 * step, 6), round(n - (y1 - 1) * step, 6), round(w + (x1 - 1) * step, 6), round(n - y0 * step, 6), step)
    return (cut, win)

def runs_whole(hours_dir: str, region: str) -> list:
    cfg = REGIONS[region]
    by = {}
    for f in os.listdir(hours_dir):
        if not f.endswith('.run') or len(f) != 14:
            continue
        run = build.stamp_of(os.path.join(hours_dir, f))[0]
        if not re.fullmatch('\\d{10}', run or '') or not os.path.exists(os.path.join(hours_dir, f[:-4] + '.npz')):
            continue
        lead = int((parse_run(f[:-4]) - parse_run(run)).total_seconds() // 3600)
        by.setdefault(run, set()).add(lead)
    want = set(range(cfg['lead_max'] + 1))
    return sorted((r for r, leads in by.items() if parse_run(r).hour in cfg['run_hours'] and want <= leads), reverse=True)

def plan(bundle: dict, listing: list, region: str, run: str, issued: datetime) -> dict:
    r = bundle['regions'][region]
    base, ages = (r['base'], r.get('max_age_h') or {})
    best = {}
    for o in listing:
        m = KEY.match(o.get('Key') or '')
        if not m or int(m.group(1)) != opslake.BLEND_V or m.group(4) != region:
            continue
        model, mrun = (m.group(2), m.group(3))
        if model == base or model not in r['members']:
            continue
        if utc(o.get('LastModified')) > issued:
            continue
        age = (issued - parse_run(mrun)).total_seconds() / 3600
        if age < 0 or (model in ages and age > ages[model]):
            continue
        if model not in best or mrun > best[model][0]:
            best[model] = (mrun, o['Key'])
    return {m: k for m, (_, k) in sorted(best.items())}

class Member:

    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as z:
            self.model = str(z['model'])
            self.run = parse_run(str(z['run']))
            self.fields = [str(x) for x in z['fields']]
            self.codes = z['codes']
            self.stamps = [parse_run(str(s)) for s in z['stamps']]
            self.elev = z['elev']
            g = json.loads(str(z['grid']))
            self.aircorr = bool(z['aircorr']) if 'aircorr' in z.files else True
        self.lons = np.round(g['west'] + g['step'] * np.arange(g['nx']), 6)
        self.lats = np.round(g['north'] - g['step'] * np.arange(g['ny']), 6)

    def at(self, valid: datetime) -> dict | None:
        if valid in self.stamps:
            c = self.codes[self.stamps.index(valid)]
            out = {f: c[j] for j, f in enumerate(self.fields)}
        else:
            before = [i for i, s in enumerate(self.stamps) if s < valid]
            after = [i for i, s in enumerate(self.stamps) if s > valid]
            if not before or not after:
                return None
            i0, i1 = (max(before, key=lambda i: self.stamps[i]), min(after, key=lambda i: self.stamps[i]))
            t = (valid - self.stamps[i0]) / (self.stamps[i1] - self.stamps[i0])
            out = {}
            for j, f in enumerate(self.fields):
                a, b = (self.codes[i0, j], self.codes[i1, j])
                if f in CLASS_CODES:
                    out[f] = a if t < 0.5 else b
                else:
                    mix = np.rint(a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)
                    out[f] = np.where((a == 255) | (b == 255), 255, mix).astype(np.uint8)
        out['elevation'] = self.elev
        return out

    def member(self, valid: datetime, base_region, wregion) -> dict | None:
        codes = self.at(valid)
        if codes is None:
            return None
        return blend.member(self.model, self.run, codes, self.lats, self.lons, base_region, wregion, self.aircorr)
_CTX: dict = {}

def _blend_hour(valid: datetime) -> tuple:
    c = _CTX
    with np.load(os.path.join(c['hours_dir'], f'{valid:%Y%m%d%H}.npz')) as h:
        win = c['win']
        codes = {k: h[k] if win is None else h[k][win[0]:win[1], win[2]:win[3]] for k in h.files if h[k].ndim == 2}
    base_hour = {'model': c['base'], 'run': c['run'], 'valid': valid, 'region': c['breg'], 'codes': codes, 'aircorr': c['aircorr']}
    members = {}
    for m in c['members']:
        mm = m.member(valid, c['breg'], c['wreg'])
        if mm is not None:
            members[m.model] = mm
    try:
        res = blend.blend(base_hour, members, c['bundle'], c['region'], issued=c['issued'])
    except blend.Refused as e:
        return (valid, None, f'refused: {e}')
    keep = set(opslake.STATION_FIELDS) | set(KEEP_OUT)
    out = {k: v for k, v in res['codes'].items() if k in keep}
    info = {'blended': res['fields_blended'], 'from_base': sorted(res['fields_from_base']), 'members': [x['model'] for x in res['members']], 'refused': res['refused']}
    return (valid, out, info)

def blend_run(hours_dir: str, run: str, bundle: dict, member_paths: list, region: str, out: str, issued: datetime, workers: int=1, build_id: str | None=None, stations=None) -> dict | None:
    cfg = REGIONS[region]
    r = bundle['regions'][region]
    if r['base'] != cfg['base']:
        raise ValueError(f"blend leg: the bundle's {region} base is {r['base']}, the leg's {cfg['base']}")
    breg, win = region_grid(region)
    wreg = blend.weight_region(breg, float(r['grid_step']))
    r0 = parse_run(run)
    hours = [v for v, ld in opslake.own_hours(hours_dir, run) if ld <= cfg['lead_max']]
    members = [Member(p) for p in member_paths]
    tmp = os.path.join(out, '_blended', region)
    os.makedirs(tmp, exist_ok=True)
    _CTX.update(hours_dir=hours_dir, base=cfg['base'], run=r0, breg=breg, wreg=wreg, win=win, members=members, bundle=bundle, region=region, issued=issued, aircorr=os.environ.get('AIR_CORRECTIONS', '0') == '1')
    took, t0 = ({}, time.time())
    workers = max(1, min(workers, len(hours) or 1))
    pool = mp.get_context('fork').Pool(workers) if workers > 1 else None
    try:
        for valid, codes, info in pool.imap(_blend_hour, hours) if pool else map(_blend_hour, hours):
            took[f'{valid:%Y%m%d%H}'] = info
            if codes is None:
                continue
            np.savez(os.path.join(tmp, f'{valid:%Y%m%d%H}.npz'), **codes)
            with open(os.path.join(tmp, f'{valid:%Y%m%d%H}.run'), 'w') as f:
                f.write(f'{run}\n')
    finally:
        if pool:
            pool.close()
            pool.join()
        _CTX.clear()
    n = sum((1 for v in took.values() if isinstance(v, dict)))
    log(f'{region} {run}: {n} of {len(hours)} hours blended in {time.time() - t0:.0f} s')
    if not n:
        return None
    stations = stations if stations is not None else verify.load_stations()
    cells = verify.station_cells(cfg['base'], stations, breg)
    done = opslake.write(tmp, run, out, model=cfg['model'], build_id=build_id, region=breg, stations=stations, blend_in=False, workers=1, complete=True, day_has_grids=True, cells=cells)
    import shutil
    shutil.rmtree(os.path.join(out, '_blended'), ignore_errors=True)
    if done is not None:
        done['blend'] = {'bundle': bundle['id'], 'region': region, 'issued': issued.strftime('%Y-%m-%dT%H:%M:%SZ'), 'members': sorted({m for v in took.values() if isinstance(v, dict) for m in v['members']}), 'hours': {k: v for k, v in took.items()}}
        p = os.path.join(out, 'ops', cfg['model'], f'{r0:%Y%m%d}', f'{r0:%H}', f"b{done['build']}", 'done.json')
        with open(p, 'w') as f:
            json.dump(done, f, separators=(',', ':'))
    return done

def turned_off(text: str) -> list[str]:
    return [f for f in (t.strip() for t in (text or '').split(',')) if f in blend.FIELDS]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('base')
    a.add_argument('--hours', required=True)
    a.add_argument('--region', required=True, choices=sorted(REGIONS))
    p = sub.add_parser('plan')
    p.add_argument('--bundle', required=True)
    p.add_argument('--sha256')
    p.add_argument('--listing', required=True, help="aws s3api list-objects-v2's JSON of cache/blendin/")
    p.add_argument('--region', required=True, choices=sorted(REGIONS))
    p.add_argument('--run', required=True)
    p.add_argument('--issued')
    r = sub.add_parser('run')
    r.add_argument('--hours', required=True)
    r.add_argument('--run', required=True)
    r.add_argument('--bundle', required=True)
    r.add_argument('--sha256')
    r.add_argument('--members', required=True, help="a folder of the members' inputs (*.npz)")
    r.add_argument('--region', required=True, choices=sorted(REGIONS))
    r.add_argument('--out', required=True)
    r.add_argument('--issued')
    r.add_argument('--workers', type=int, default=3)
    r.add_argument('--build')
    r.add_argument('--off', default='', help='the fields the pointer turns off for this region (its fields_off), comma-separated')
    x = ap.parse_args(argv)
    if x.cmd == 'base':
        runs = runs_whole(x.hours, x.region) if os.path.isdir(x.hours) else []
        print(runs[0] if runs else '')
        return 0
    with open(x.bundle, 'rb') as f:
        bundle = blend.load(f.read(), x.sha256)
    if x.region not in bundle['regions']:
        log(f'the bundle has no {x.region} region: nothing to do')
        return 0
    issued = utc(x.issued)
    if x.cmd == 'plan':
        with open(x.listing) as f:
            listing = (json.load(f) or {}).get('Contents') or []
        for k in plan(bundle, listing, x.region, x.run, issued).values():
            print(k)
        return 0
    for f in turned_off(x.off):
        fd = bundle['regions'][x.region]['fields'].get(f)
        if fd is not None:
            fd['off'] = True
            log(f"{x.region}: {f} off (its source code moved since its training): the base's own")
    paths = sorted((os.path.join(x.members, f) for f in os.listdir(x.members) if f.endswith('.npz'))) if os.path.isdir(x.members) else []
    done = blend_run(x.hours, x.run, bundle, paths, x.region, x.out, issued, x.workers, x.build)
    return 0 if done is not None else 1
if __name__ == '__main__':
    sys.exit(main())
