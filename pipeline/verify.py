from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = os.path.join(HERE, 'verify_stations.json')
UTC = timezone.utc
VALID_H = (0, 9, 12, 15, 18, 21)
LEAD_MAX = {'hrrr': 48, 'icond2': 48, 'icon': 180, 'gfs': 240, 'ecmwf': 240}
BACK_DAYS = {'hrrr': 30, 'gfs': 30, 'ecmwf': 30, 'icon': 7, 'icond2': 7}
RUN_HOURS = (0, 12)
READY_H = {'hrrr': 3, 'icond2': 4, 'icon': 6, 'gfs': 6, 'ecmwf': 10}
HOUR_S = {'hrrr': 90, 'icond2': 60, 'icon': 60, 'gfs': 45, 'ecmwf': 150}
WORKERS = {'hrrr': 2, 'ecmwf': 3}
FIELDS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'gust', 'cloud', 'cloudlow', 'ceil', 'vis', 'rain', 'cape', 'wstar', 'zi', 'depth', 'cubase', 'cudepth', 'cuclass', 'cucover', 'elevation')

def log(*a):
    print(*a, flush=True)

def load_stations():
    with open(STATIONS) as f:
        return json.load(f)['stations']

def make_stations(history, out):
    rows = list(csv.DictReader(open(history, encoding='utf-8', errors='replace')))
    best = {}
    for r in rows:
        icao = r['ICAO'].strip()
        if len(icao) != 4 or not icao.isalpha() or r['END'] < '20250801' or (r['BEGIN'] > '20200101'):
            continue
        try:
            lat, lon, elev = (float(r['LAT']), float(r['LON']), float(r['ELEV(M)']))
        except ValueError:
            continue
        if not -56 < lat < 72 or not -170 < lon < 179.5 or elev < -400 or (elev > 5000) or (lat == 0 and lon == 0):
            continue
        fine = 24 < lat < 50 and -125 < lon < -66 or (35 < lat < 60 and -10 < lon < 30)
        k = (int(math.floor(lat)), int(math.floor(lon))) if fine else (int(lat // 3), int(lon // 3), 'c')
        if k not in best or r['BEGIN'] < best[k][0]:
            best[k] = (r['BEGIN'], [icao, round(lat, 3), round(lon, 3), round(elev)])
    st = sorted((v[1] for v in best.values()))
    with open(out, 'w') as f:
        json.dump({'source': 'NOAA ISD station history: an airport a 1° box (US, Europe), 3° elsewhere', 'stations': st}, f, separators=(',', ':'))
    log(f'{len(st)} stations')

def iem_id(icao):
    return icao[1:] if icao.startswith('K') else icao

def runs_to_do(model, done, now):
    out = []
    t = now.replace(minute=0, second=0, microsecond=0)
    last = t - timedelta(hours=READY_H[model])
    first = t - timedelta(days=BACK_DAYS[model])
    r = last.replace(hour=0)
    while r >= first:
        for h in sorted(RUN_HOURS, reverse=True):
            run = r.replace(hour=h)
            if first <= run <= last and f'{run:%Y%m%d%H}' not in done:
                out.append(run)
        r -= timedelta(days=1)
    return out

def valid_hours(src, run, model):
    out = []
    for fh in range(1, min(LEAD_MAX[model], src.last_fh(run)) + 1):
        v = run + timedelta(hours=fh)
        if v.hour in VALID_H and src.fstep(fh) and src.builds(fh):
            out.append(v)
    return out

def _build(model, run_id, out, workers, valid):
    sys.path.insert(0, HERE)
    import build
    hours = [datetime.strptime(v, '%Y%m%d%H').replace(tzinfo=UTC) for v in valid]
    build.needed_times = lambda s, days=None, now=None: hours
    sys.argv = ['build.py', '--model', model, '--out', out, '--shard', '0/1', '--run', run_id, '--workers', str(workers)]
    return build.main()

def station_cells(model, stations):
    import build
    w, s, e, n, step = build.REGIONS[model][:5]
    lons, lats = build.target_grid(build.REGIONS[model])
    idx, iy, ix = ([], [], [])
    for i, (_, lat, lon, _) in enumerate(stations):
        x, y = (round((lon - w) / step), round((n - lat) / step))
        if 0 <= x < len(lons) and 0 <= y < len(lats):
            (idx.append(i), iy.append(y), ix.append(x))
    return (np.array(idx, np.int32), np.array(iy), np.array(ix))

def sample_hours(model, hours_dir, stations):
    idx, iy, ix = station_cells(model, stations)
    stamps = sorted((f[:-4] for f in os.listdir(hours_dir) if f.endswith('.npz') and (not f.endswith('.tmp.npz'))))
    if not stamps:
        return None
    codes = np.full((len(stamps), len(idx), len(FIELDS)), 255, np.uint8)
    for k, st in enumerate(stamps):
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            for j, name in enumerate(FIELDS):
                if name in h.files:
                    codes[k, :, j] = h[name][iy, ix]
    has = (codes[:, :, FIELDS.index('t2')] != 255).any(axis=0)
    return (stamps, idx[has], codes[:, has, :])

def cmd_sample(a):
    sys.path.insert(0, HERE)
    import build
    import sources
    model, state = (a.model, a.state)
    deadline = datetime.fromisoformat(a.deadline.replace('Z', '+00:00'))
    os.makedirs(state, exist_ok=True)
    book_path = os.path.join(state, 'done.json')
    book = json.load(open(book_path)) if os.path.exists(book_path) else {}
    done, failed = (book.setdefault('done', {}), book.setdefault('failed', {}))
    per_hour = book.get('hour_s', HOUR_S[model])
    stations = load_stations()
    src = sources.SOURCES[model]
    now = datetime.now(UTC)
    gave_up = {r for r, f in failed.items() if f.get('n', 0) >= 2}
    todo = runs_to_do(model, {**done, **dict.fromkeys(gave_up)}, now)
    if a.part:
        i, n = (int(x) for x in a.part.split('/'))
        todo = [r for r in todo if int(r.timestamp() // 43200) % n == i]
    log(f'{model}: {len(todo)} runs to do, {len(done)} done; until {deadline:%H:%M}Z')
    enc = {k: list(build.ENC[k]) for k in FIELDS if k in build.ENC}
    scratch = os.path.join(a.scratch, model)
    for run in todo:
        rid = f'{run:%Y%m%d%H}'
        valid = valid_hours(src, run, model)
        if not valid:
            continue
        workers = WORKERS.get(model, 3)
        need = per_hour * len(valid) / workers * 1.2 + 120
        left = (deadline - datetime.now(UTC)).total_seconds()
        if left < need:
            log(f'{model}: {rid} would take ~{need / 60:.0f} min, {left / 60:.0f} left: stopping here')
            break
        shutil.rmtree(scratch, ignore_errors=True)
        t0 = time.time()
        log(f'{model}: {rid}: {len(valid)} hours')
        cmd = [sys.executable, os.path.abspath(__file__), '_build', model, rid, scratch, str(workers), *[f'{v:%Y%m%d%H}' for v in valid]]
        try:
            subprocess.run(cmd, check=True, timeout=max(300, left - 60), cwd=HERE)
            got = sample_hours(model, os.path.join(scratch, model, '_hours'), stations)
        except subprocess.TimeoutExpired:
            log(f'{model}: {rid} not done by the deadline: next time')
            break
        except Exception as e:
            log(f'{model}: {rid} failed: {e}')
            failed[rid] = {'n': failed.get(rid, {}).get('n', 0) + 1, 'e': str(e)[:200]}
            got = None
        if got:
            stamps, idx, codes = got
            np.savez_compressed(os.path.join(state, f'{rid}.npz'), stamps=np.array(stamps), idx=idx, codes=codes, fields=np.array(FIELDS), enc=json.dumps(enc), model=model, run=rid)
            took = time.time() - t0
            per_hour = round(0.5 * per_hour + 0.5 * took * workers / len(stamps), 1)
            done[rid] = {'hours': len(stamps), 'stations': len(idx), 's': round(took)}
            log(f'{model}: {rid}: {len(stamps)} hours at {len(idx)} stations in {took / 60:.1f} min')
            failed.pop(rid, None)
        elif rid not in failed:
            failed[rid] = {'n': 2, 'e': 'no hours'}
        book['hour_s'] = per_hour
        with open(book_path, 'w') as f:
            json.dump(book, f)
    shutil.rmtree(scratch, ignore_errors=True)
    oldest = f'{now - timedelta(days=BACK_DAYS[model] + 11):%Y%m%d%H}'
    for f in os.listdir(state):
        if f.endswith('.npz') and f[:10] < oldest:
            os.remove(os.path.join(state, f))
    return 0
OBS = ('t2', 'td2', 'ws', 'gust', 'vis', 'ceil', 'cu', 'cubase', 'cover')
CU_BASE_FT = (1000, 9000)

def parse_reports(text, ids):
    out = defaultdict(dict)
    best = {}
    for r in csv.DictReader(io.StringIO(text)):
        i = ids.get(r.get('station', ''))
        if i is None:
            continue
        try:
            t = datetime.strptime(r['valid'], '%Y-%m-%d %H:%M').replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        hr = (t + timedelta(minutes=30)).replace(minute=0)
        off = abs((t - hr).total_seconds())
        if off > 1200 or best.get((i, hr), 1000000000.0) <= off:
            continue
        best[i, hr] = off
        f = lambda k: float(r[k]) if r.get(k) not in (None, '', 'M') else math.nan
        layers = []
        for n in (1, 2, 3, 4):
            c, b = (r.get(f'skyc{n}', ''), f(f'skyl{n}'))
            if c in ('FEW', 'SCT', 'BKN', 'OVC', 'VV'):
                layers.append((c, b))
        clear = any((r.get(f'skyc{n}') in ('CLR', 'SKC', 'NSC', 'NCD', 'CAVOK') for n in (1, 2, 3, 4))) or not layers
        ceil = min((b for c, b in layers if c in ('BKN', 'OVC', 'VV') and math.isfinite(b)), default=math.inf)
        low = [(c, b) for c, b in layers if math.isfinite(b) and b < CU_BASE_FT[0]]
        cu = [b for c, b in layers if c in ('FEW', 'SCT') and math.isfinite(b) and (CU_BASE_FT[0] <= b <= CU_BASE_FT[1])]
        cover = {'FEW': 0.19, 'SCT': 0.44, 'BKN': 0.75, 'OVC': 1.0, 'VV': 1.0}
        tot = 1 - np.prod([1 - cover[c] for c, _ in layers]) if layers else 0.0
        known = bool(layers) or clear
        t2 = (f('tmpf') - 32) / 1.8
        td2 = (f('dwpf') - 32) / 1.8
        out[i][hr] = [t2, td2, f('sknt') * 0.5144, f('gust') * 0.5144, f('vsby') * 1609.34, ceil * 0.3048 if known else math.nan, (1.0 if cu and (not low) else 0.0) if known else math.nan, min(cu) * 0.3048 if cu else math.nan, tot if known else math.nan]
    return out

def fetch_day(day, stations, batch=150):
    ids = {iem_id(s[0]): i for i, s in enumerate(stations)}
    arr = np.full((len(stations), 24, len(OBS)), np.nan, np.float32)
    names = list(ids)
    data = ['tmpf', 'dwpf', 'sknt', 'gust', 'vsby'] + [f'sky{c}{n}' for n in (1, 2, 3, 4) for c in 'cl']
    for k in range(0, len(names), batch):
        q = [('station', s) for s in names[k:k + batch]] + [('data', d) for d in data] + [('tz', 'Etc/UTC'), ('format', 'onlycomma'), ('missing', 'empty'), ('trace', 'empty'), ('report_type', '3'), ('sts', f'{day - timedelta(minutes=20):%Y-%m-%dT%H:%MZ}'), ('ets', f'{day + timedelta(hours=23, minutes=40):%Y-%m-%dT%H:%MZ}')]
        url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urllib.parse.urlencode(q)
        for attempt in range(4):
            try:
                text = urllib.request.urlopen(url, timeout=180).read().decode('utf-8', 'replace')
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log(f'reports {day:%Y-%m-%d}: {e}; again')
                time.sleep(10 * (attempt + 1))
        for i, hours in parse_reports(text, ids).items():
            for hr, vals in hours.items():
                if hr.date() == day.date():
                    arr[i, hr.hour] = vals
        time.sleep(1)
    return arr

def cmd_obs(a):
    stations = load_stations()
    os.makedirs(a.state, exist_ok=True)
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    deadline = datetime.fromisoformat(a.deadline.replace('Z', '+00:00')) if a.deadline else None
    for d in range(a.days, -1, -1):
        day = today - timedelta(days=d)
        path = os.path.join(a.state, f'{day:%Y%m%d}.npy')
        if os.path.exists(path) and d > 1:
            continue
        if deadline and datetime.now(UTC) > deadline:
            break
        arr = fetch_day(day, stations)
        np.save(path, arr)
        log(f'reports {day:%Y-%m-%d}: {int(np.isfinite(arr[:, :, 0]).sum())} station-hours')
    for f in os.listdir(a.state):
        if f.endswith('.npy') and f[:8] < f'{today - timedelta(days=a.days + 5):%Y%m%d}':
            os.remove(os.path.join(a.state, f))
    return 0
LEADS = ((1, 12), (13, 24), (25, 48), (49, 72), (73, 120), (121, 240))
DAY_H = {'us': (15, 18, 21), 'europe': (9, 12, 15)}

def region_of(lat, lon):
    if 24 < lat < 50 and -125 < lon < -66:
        return 'us'
    if 35 < lat < 60 and -10 < lon < 30:
        return 'europe'
    return 'elsewhere'

def lead_of(h):
    for lo, hi in LEADS:
        if lo <= h <= hi:
            return f'{lo}-{hi}h'
    return None

def decode(codes, lo, hi, transform):
    c = codes.astype(np.float64)
    if transform == 'sqrt':
        a, b = (math.sqrt(max(lo, 0)), math.sqrt(hi))
        v = (a + c / 254 * (b - a)) ** 2
    else:
        v = lo + c / 254 * (hi - lo)
    return np.where(codes == 255, np.nan, v)

def cu_class(code):
    return np.where(code == 255, -1, np.where(code < 225, np.round(code / 50), code - 225 + 5))

def pairs(samples_dirs, obs_dir, stations):
    obs = {}
    for f in os.listdir(obs_dir):
        if f.endswith('.npy'):
            obs[f[:8]] = np.load(os.path.join(obs_dir, f), mmap_mode='r')
    lat = np.array([s[1] for s in stations])
    lon = np.array([s[2] for s in stations])
    elev = np.array([s[3] for s in stations], float)
    regions = np.array([region_of(a, b) for a, b in zip(lat, lon)])
    for d in samples_dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith('.npz'):
                continue
            with np.load(os.path.join(d, f)) as z:
                model, run = (str(z['model']), str(z['run']))
                fields = [str(x) for x in z['fields']]
                enc = json.loads(str(z['enc']))
                idx, codes, stamps = (z['idx'], z['codes'], [str(s) for s in z['stamps']])
            r0 = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC)
            val = {k: decode(codes[:, :, j], *enc[k]) for j, k in enumerate(fields) if k in enc}
            if 'cuclass' in fields:
                val['cuclass'] = cu_class(codes[:, :, fields.index('cuclass')])
            for k, st in enumerate(stamps):
                v = datetime.strptime(st, '%Y%m%d%H').replace(tzinfo=UTC)
                ob = obs.get(st[:8])
                lead = lead_of(int((v - r0).total_seconds() // 3600))
                if ob is None or lead is None:
                    continue
                o = ob[idx, v.hour]
                yield (model, lead, v, idx, regions[idx], elev[idx], {n: a[k] for n, a in val.items()}, o)

def cmd_score(a):
    stations = load_stations()
    obi = {k: i for i, k in enumerate(OBS)}
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    cont = defaultdict(lambda: np.zeros(4))
    keys = defaultdict(dict)
    n_rows = 0
    for model, lead, v, idx, reg, elev, fc, ob in pairs(a.samples, a.obs, stations):
        n_rows += len(idx)
        lapse = 0.0065 * (fc['elevation'] - elev) if 'elevation' in fc else 0.0
        meas = {'t2': fc.get('t2') + lapse - ob[:, obi['t2']] if 't2' in fc else None, 'td2': fc.get('td2') - ob[:, obi['td2']] if 'td2' in fc else None, 'wind': np.hypot(fc['u_sfc'], fc['v_sfc']) - ob[:, obi['ws']] if 'u_sfc' in fc else None}
        day = np.array([v.hour in DAY_H.get(r, ()) for r in reg])
        if 'cubase' in fc and 'cuclass' in fc:
            both = day & np.isin(fc['cuclass'], (2, 3, 4)) & (ob[:, obi['cu']] == 1)
            meas['cubase'] = np.where(both, fc['cubase'] - ob[:, obi['cubase']], np.nan)
        for name, err in meas.items():
            if err is None:
                continue
            for r in np.unique(reg):
                e = err[(reg == r) & np.isfinite(err)]
                if len(e):
                    acc[model, r, lead][name] += [len(e), e.sum(), np.abs(e).sum(), (e * e).sum()]
        if 'cuclass' in fc:
            f_cu = np.isin(fc['cuclass'], (2, 3, 4, 5))
            o_cu = ob[:, obi['cu']]
            ok = day & np.isfinite(o_cu) & (fc['cuclass'] >= 0)
            for r in np.unique(reg[ok]):
                s = ok & (reg == r)
                fo, oo = (f_cu[s], o_cu[s] == 1)
                cont[model, r, lead] += [(fo & oo).sum(), (fo & ~oo).sum(), (~fo & oo).sum(), (~fo & ~oo).sum()]
        if meas['t2'] is not None:
            for j in np.nonzero(np.isfinite(meas['t2']))[0]:
                keys[reg[j], lead].setdefault(model, {})[int(idx[j]), v] = abs(float(meas['t2'][j]))
    rows = []
    for (model, r, lead), m in sorted(acc.items()):
        row = {'model': model, 'region': r, 'lead': lead}
        for name, (n, s, sa, s2) in m.items():
            row[name] = {'n': int(n), 'bias': round(s / n, 2), 'mae': round(sa / n, 2), 'rmse': round(math.sqrt(s2 / n), 2)}
        h, fa, mi, cn = cont.get((model, r, lead), np.zeros(4))
        nn = h + fa + mi + cn
        if nn:
            exp = ((h + fa) * (h + mi) + (mi + cn) * (fa + cn)) / nn
            row['cumulus'] = {'n': int(nn), 'pod': round(h / max(1, h + mi), 2), 'far': round(fa / max(1, h + fa), 2), 'bias': round((h + fa) / max(1, h + mi), 2), 'heidke': round((h + cn - exp) / (nn - exp), 3) if nn != exp else None}
        rows.append(row)
    side = []
    for (r, lead), by in sorted(keys.items()):
        if len(by) < 2:
            continue
        common = set.intersection(*(set(d) for d in by.values()))
        if len(common) < 50:
            continue
        side.append({'region': r, 'lead': lead, 'n': len(common), 't2_mae': {m: round(sum((d[k] for k in common)) / len(common), 2) for m, d in sorted(by.items())}})
    out = {'generated': f'{datetime.now(UTC):%Y-%m-%dT%H:%MZ}', 'pairs': n_rows, 'scores': rows, 'side_by_side': side}
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    log(f'{n_rows} forecasts paired with reports')
    log('\nT2 mean error (°C; same stations and hours for every model):')
    for s in side:
        log(f'  {s['region']:10} {s['lead']:9} n={s['n']:6}  ' + '  '.join((f'{m} {e:.2f}' for m, e in s['t2_mae'].items())))
    log('\nCumulus or not, daytime (Heidke skill, 1 = perfect, 0 = no skill; POD, false alarms):')
    for row in rows:
        c = row.get('cumulus')
        if c and c['n'] >= 100:
            log(f'  {row['region']:10} {row['lead']:9} {row['model']:7} n={c['n']:6} heidke {c['heidke']}  pod {c['pod']}  far {c['far']}')
    log('\nCloudbase where both have cumulus (m: bias, mean error):')
    for row in rows:
        c = row.get('cubase')
        if c and c['n'] >= 50:
            log(f'  {row['region']:10} {row['lead']:9} {row['model']:7} n={c['n']:6} bias {c['bias']:+.0f}  mae {c['mae']:.0f}')
    return 0

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_build':
        _, _, model, run_id, out, workers, *valid = sys.argv
        return _build(model, run_id, out, int(workers), valid)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('stations')
    p.add_argument('history')
    p.add_argument('out', nargs='?', default=STATIONS)
    p = sub.add_parser('sample')
    p.add_argument('--model', required=True, choices=sorted(LEAD_MAX))
    p.add_argument('--state', required=True)
    p.add_argument('--deadline', required=True, help='UTC, e.g. 2026-10-24T07:55Z: no run started that would end after it')
    p.add_argument('--part', help='i/n: only every n-th run (several runners a model)')
    p.add_argument('--scratch', default=os.path.join(HERE, '..', 'verify-scratch'))
    p = sub.add_parser('obs')
    p.add_argument('--state', required=True)
    p.add_argument('--days', type=int, default=32)
    p.add_argument('--deadline')
    p = sub.add_parser('score')
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--obs', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'stations':
        return make_stations(a.history, a.out)
    return {'sample': cmd_sample, 'obs': cmd_obs, 'score': cmd_score}[a.cmd](a)
if __name__ == '__main__':
    sys.exit(main())
