from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from airspace import west_pacific
VERSION = 9
UA = {'User-Agent': 'wx-pipeline/1.0'}
OURAIRPORTS = ('https://davidmegginson.github.io/ourairports-data', 'https://ourairports.com/data')
REGIONS = ((-180.0, 5.0, -50.0, 75.0), (-32.0, 26.0, 60.0, 72.0), (-82.0, -56.0, -32.0, 5.0), (9.0, -36.0, 38.0, -14.0), (165.0, -48.0, 179.9, -33.0))
REGION_NAMES = ('North America', 'Europe', 'South America', 'southern Africa', 'New Zealand')
IDENTS_FIRST = 2
REGION = (-180.0, -56.0, 180.0, 75.0)
BORDER_REGION = REGIONS[0]
CELL = 5.0
TYPES = {'large_airport': 'L', 'medium_airport': 'M', 'small_airport': 'S', 'heliport': 'H', 'seaplane_base': 'W'}
NAVAID_TYPES = {'VOR', 'VORTAC', 'VOR-DME', 'TACAN', 'NDB', 'NDB-DME', 'DME'}
FAA_SERVICES = 'https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services'
CHART_BASE = 'https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/services'
CHARTS = {'vfr': ('VFR_Sectional', 'VFR Sectional'), 'tac': ('VFR_Terminal', 'VFR Terminal Area'), 'ifrlo': ('IFR_AreaLow', 'IFR Low Enroute'), 'ifrhi': ('IFR_High', 'IFR High Enroute')}

def get(url: str, tries: int=4, timeout: int=120) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404) or attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError('unreachable')

def status(url: str) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0

def ourairports(name: str) -> list[dict]:
    last = None
    for base in OURAIRPORTS:
        try:
            text = get(f'{base}/{name}').decode('utf-8', 'replace')
            rows = list(csv.DictReader(io.StringIO(text)))
            print(f'{name}: {len(rows)} rows from {base}', flush=True)
            return rows
        except Exception as e:
            last = e
            print(f'{name}: {base} failed ({e})', flush=True)
    raise RuntimeError(f'{name} unavailable: {last}')

def num(x, nd=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v

def region_of(lat, lon) -> int | None:
    if lat is None or lon is None:
        return None
    return next((i for i, (w, s, e, n) in enumerate(REGIONS) if s <= lat <= n and w <= lon <= e), None)

def cell_of(lat: float, lon: float) -> str:
    return f'{math.floor(lon / CELL)}_{math.floor(lat / CELL)}'

def write(path: str, obj) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(path + '.tmp', path)
    return os.path.getsize(path)

def merge_later(idents, later):
    for key, v in later.items():
        if key in idents or (len(key) == 3 and f'K{key}' in idents) or (len(key) == 4 and key[0] == 'K' and (key[1:] in idents)):
            continue
        idents[key] = v

def build_airports(d: str) -> dict:
    airports = ourairports('airports.csv')
    runways = ourairports('runways.csv')
    freqs = ourairports('airport-frequencies.csv')
    navaids = ourairports('navaids.csv')
    rw_by = defaultdict(list)
    for r in runways:
        if r.get('closed') == '1':
            continue
        ids = '/'.join((x for x in (r.get('le_ident'), r.get('he_ident')) if x))
        rw_by[r['airport_ref']].append([ids, num(r.get('length_ft'), 0), num(r.get('width_ft'), 0), (r.get('surface') or '').strip()[:12], 1 if r.get('lighted') == '1' else 0, num(r.get('le_heading_degT'), 0)])
    fq_by = defaultdict(list)
    for f in freqs:
        mhz = num(f.get('frequency_mhz'), 3)
        if mhz:
            fq_by[f['airport_ref']].append([(f.get('type') or '').strip()[:8], (f.get('description') or '').strip()[:40], mhz])
    cells = defaultdict(lambda: {'a': [], 'n': []})
    overview = {'a': [], 'n': []}
    idents, later = ({}, {})
    search = []
    kinds, by_region = (Counter(), Counter())
    for a in airports:
        t = TYPES.get(a.get('type'))
        lat, lon = (num(a.get('latitude_deg'), 5), num(a.get('longitude_deg'), 5))
        reg = region_of(lat, lon)
        if not t or reg is None:
            continue
        code = (a.get('icao_code') or a.get('gps_code') or a.get('local_code') or a.get('ident') or '').strip().upper()
        if not code:
            continue
        lid = (a.get('local_code') or '').strip().upper()
        fq = sorted(fq_by.get(a['id'], []), key=lambda x: x[0])
        towered = 1 if any((x[0] in ('TWR', 'TOWER') for x in fq)) else 0
        rec = [code, (a.get('name') or '')[:60], t, lat, lon, num(a.get('elevation_ft'), 0), towered, lid if lid != code else '', (a.get('iata_code') or '').strip(), (a.get('municipality') or '')[:40], (a.get('iso_region') or '')[:6], sorted(rw_by.get(a['id'], []), key=lambda r: -(r[1] or 0)), fq]
        cells[cell_of(lat, lon)]['a'].append(rec)
        kinds[t] += 1
        by_region[REGION_NAMES[reg]] += 1
        if t in ('L', 'M'):
            overview['a'].append(rec[:7])
        if t != 'H':
            for key in {code, lid} - {''}:
                (idents if reg < IDENTS_FIRST else later).setdefault(key, [lat, lon, 'A', rec[1][:30]])
            search.append([code, rec[1], t, round(lat, 3), round(lon, 3), rec[9], rec[10], ' '.join(sorted({lid, rec[8]} - {'', code}))])
    vars_ = []
    for n in navaids:
        t = (n.get('type') or '').upper()
        lat, lon = (num(n.get('latitude_deg'), 5), num(n.get('longitude_deg'), 5))
        reg = region_of(lat, lon)
        if t not in NAVAID_TYPES or reg is None:
            continue
        khz = num(n.get('frequency_khz'))
        freq = None if khz is None else round(khz / 1000, 2) if khz >= 10000 else round(khz)
        mv = num(n.get('magnetic_variation_deg'), 1)
        rec = [(n.get('ident') or '').upper(), (n.get('name') or '')[:40], t, lat, lon, freq, num(n.get('elevation_ft'), 0), (n.get('dme_channel') or '').strip()]
        cells[cell_of(lat, lon)]['n'].append(rec)
        kinds[t] += 1
        if t in ('VOR', 'VORTAC', 'VOR-DME'):
            overview['n'].append(rec[:6])
        if mv is not None:
            vars_.append((lat, lon, mv))
        if rec[0]:
            (idents if reg < IDENTS_FIRST else later).setdefault(rec[0], [lat, lon, 'N', rec[1][:30], t])
    merge_later(idents, later)
    print('kept:', dict(kinds), flush=True)
    print('airports by region:', {k: by_region[k] for k in REGION_NAMES}, flush=True)
    if sum((kinds[k] for k in 'LMS')) < 1000:
        raise SystemExit('too few airports; not overwriting the published files')
    sizes = 0
    for key, c in cells.items():
        sizes += write(os.path.join(d, 'cells', f'{key}.json'), c)
    write(os.path.join(d, 'cells.json'), {'size': CELL, 'cells': sorted(cells)})
    sizes += write(os.path.join(d, 'overview.json'), overview)
    sizes += write(os.path.join(d, 'idents.json'), idents)
    sizes += write(os.path.join(d, 'airports.json'), {'v': 1, 'a': search})
    print(f'airports: {len(cells)} cells, {len(idents)} idents, {len(search)} searchable, {sizes / 1000000.0:.1f} MB', flush=True)
    write(os.path.join(d, 'magvar.json'), magvar_grid(vars_))
    return dict(kinds)

def magvar_grid(samples, step: float=2.0) -> dict:
    west = [v for lat, lon, v in samples if 30 < lat < 50 and -125 < lon < -110]
    sign = -1.0 if west and sorted(west)[len(west) // 2] < 0 else 1.0
    print(f'magvar: {len(samples)} samples, west-US median {(sorted(west)[len(west) // 2] if west else '-')} → sign {sign:+.0f}')
    w, s, e, n = REGION
    nx, ny = (int((e - w) / step) + 1, int((n - s) / step) + 1)
    buckets = defaultdict(list)
    for lat, lon, v in samples:
        buckets[int((lon - w) / 10), int((lat - s) / 10)].append((lat, lon, sign * v))
    vals = []
    for j in range(ny):
        lat = s + j * step
        row = []
        for i in range(nx):
            lon = w + i * step
            cx, cy = (int((lon - w) / 10), int((lat - s) / 10))
            near = [p for dx in (-1, 0, 1) for dy in (-1, 0, 1) for p in buckets.get((cx + dx, cy + dy), [])]
            if not near:
                row.append(None)
                continue
            acc = wsum = 0.0
            for plat, plon, v in near:
                dd = (plat - lat) ** 2 + ((plon - lon) * math.cos(math.radians(lat))) ** 2
                wt = 1 / (dd + 0.25) ** 2
                acc += wt * v
                wsum += wt
            row.append(round(acc / wsum, 1))
        vals.append(row)
    return {'west': w, 'south': s, 'step': step, 'nx': nx, 'ny': ny, 'values': vals, 'eastPositive': True}

def arcgis_json(url: str, params: dict | None=None, tries: int=10) -> dict:
    full = f'{url}?{urllib.parse.urlencode(params)}' if params else url
    for attempt in range(tries):
        data = json.loads(get(full, tries=3, timeout=300))
        err = data.get('error') if isinstance(data, dict) else None
        if not err:
            return data
        if err.get('code') == 429 and attempt < tries - 1:
            print('  FAA quota exceeded, waiting 65 s', flush=True)
            time.sleep(65)
            continue
        raise RuntimeError(f'FAA service error: {err}')
    raise RuntimeError('unreachable')

def limit(val, uom, code) -> str:
    v = num(val)
    code = str(code or '').upper()
    if code == 'SFC' or v == 0:
        return 'SFC'
    if v is None or v < 0:
        return str(val or '?')
    if str(uom or '').upper() == 'FL':
        return f'FL{round(v)}'
    return f'{round(v):,} ft{(' AGL' if code == 'AGL' else '')}'

def pick(p: dict, *names):
    for n in names:
        for k in (n, n.upper(), n.lower()):
            if p.get(k) not in (None, ''):
                return p[k]
    return None

def build_sua(d: str) -> int:
    services = arcgis_json(FAA_SERVICES, {'f': 'json'}).get('services', [])
    names = [s['name'] for s in services]
    print('FAA services:', ', '.join(names[:80]), flush=True)
    match = sorted((s for s in services if 'special_use_airspace' in s['name'].lower() and 'pending' not in s['name'].lower()), key=lambda s: len(s['name']))
    if not match:
        raise RuntimeError('no Special Use Airspace service found')
    url = f'{FAA_SERVICES}/{match[0]['name']}/{match[0].get('type', 'FeatureServer')}/0'
    info = arcgis_json(url, {'f': 'json'})
    print('SUA:', url, '| fields:', [f['name'] for f in info.get('fields', [])], flush=True)
    feats, offset = ([], 0)
    page = max(100, min(int(info.get('maxRecordCount') or 1000), 2000))
    while True:
        data = arcgis_json(f'{url}/query', {'where': '1=1', 'outFields': '*', 'outSR': '4326', 'f': 'geojson', 'geometryPrecision': '4', 'maxAllowableOffset': '0.001', 'resultOffset': str(offset), 'resultRecordCount': str(page)})
        batch = data.get('features') or []
        feats += batch
        if not batch or not (data.get('exceededTransferLimit') or (data.get('properties') or {}).get('exceededTransferLimit')):
            break
        offset += len(batch)
    if feats:
        print('SUA sample properties:', json.dumps(feats[0].get('properties'))[:600], flush=True)
    out = []
    for f in feats:
        p = f.get('properties') or {}
        g = f.get('geometry')
        if not g or west_pacific(g):
            continue
        kind = str(pick(p, 'TYPE_CODE', 'TYPE', 'LOCAL_TYPE', 'SUA_TYPE') or '').upper()
        out.append({'type': 'Feature', 'geometry': g, 'properties': {'t': kind[:12], 'n': str(pick(p, 'NAME', 'IDENT') or '')[:60], 'lo': limit(pick(p, 'LOWER_VAL'), pick(p, 'LOWER_UOM'), pick(p, 'LOWER_CODE')), 'hi': limit(pick(p, 'UPPER_VAL'), pick(p, 'UPPER_UOM'), pick(p, 'UPPER_CODE')), 'time': str(pick(p, 'TIMESOFUSE', 'TIMES_OF_USE', 'WORKHR_CODE', 'SCHEDULE') or '')[:80], 'ctl': str(pick(p, 'CONT_AGENT', 'CONTROLLING_AGENCY', 'CONTAGENT') or '')[:60]}})
    print('SUA kept:', Counter((f['properties']['t'] for f in out)).most_common(12), flush=True)
    if len(out) < 50:
        raise RuntimeError(f'only {len(out)} SUA features')
    size = write(os.path.join(d, 'sua.json'), {'type': 'FeatureCollection', 'features': out})
    print(f'SUA: {len(out)} features, {size / 1000000.0:.1f} MB', flush=True)
    return len(out)

def tile_xy(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return (x, y)

def build_charts(d: str) -> dict:
    out = {}
    probes = [(39.1, -94.6), (40.7, -74.0), (34.0, -118.2)]
    for cid, (svc, name) in CHARTS.items():
        base = f'{CHART_BASE}/{svc}/MapServer'
        rec = {'name': name, 'url': f'{base}/tile/{{z}}/{{y}}/{{x}}', 'ok': False}
        try:
            info = json.loads(get(f'{base}?f=json', tries=2, timeout=60))
            lods = [l.get('level') for l in (info.get('tileInfo') or {}).get('lods', [])]
            rec.update({'lods': [min(lods), max(lods)] if lods else None, 'minLOD': info.get('minLOD'), 'maxLOD': info.get('maxLOD'), 'copyright': str(info.get('copyrightText') or '')[:120]})
        except Exception as e:
            rec['error'] = str(e)[:120]
        zooms = []
        for z in range(3, 15):
            ok = False
            for lat, lon in probes:
                x, y = tile_xy(lat, lon, z)
                if status(f'{base}/tile/{z}/{y}/{x}') == 200:
                    ok = True
                    break
            if ok:
                zooms.append(z)
        if zooms:
            rec.update({'ok': True, 'minZoom': min(zooms), 'maxZoom': max(zooms)})
        print(f'chart {cid}: {rec}', flush=True)
        out[cid] = rec
    write(os.path.join(d, 'charts.json'), out)
    return out

def probe_tfr():
    for url in ('https://tfr.faa.gov/tfrapi/exportTfrList', 'https://tfr.faa.gov/geoserver/TFR/ows?service=WFS&version=1.1.0&request=GetFeature&typeName=TFR:V_TFR_LOC&maxFeatures=3&outputFormat=application/json&srsname=EPSG:4326'):
        try:
            body = get(url, tries=1, timeout=60).decode('utf-8', 'replace')
            print(f'TFR probe {url[:60]}…: {len(body)} bytes: {body[:700]}', flush=True)
        except Exception as e:
            print(f'TFR probe {url[:60]}…: {e}', flush=True)
NATURAL_EARTH = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson'

def simplify_line(pts, tol):
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = (pts[a], pts[b])
        dx, dy = (x2 - x1, y2 - y1)
        norm = (dx * dx + dy * dy) ** 0.5
        far, idx = (-1.0, None)
        for i in range(a + 1, b):
            x, y = pts[i]
            dd = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm if norm else ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
            if dd > far:
                far, idx = (dd, i)
        if idx is not None and far > tol:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(pts, keep) if k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dist-data')
    args = ap.parse_args()
    d = os.path.join(args.out, 'airspace', 'aviation')
    meta = {'updated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'version': VERSION}
    for part, fn in (('airports', build_airports), ('sua', build_sua), ('charts', build_charts)):
        try:
            meta[part] = fn(d)
        except BaseException as e:
            print(f'{part} failed: {e}', flush=True)
            meta[part] = None
    probe_tfr()
    if meta['airports'] is None and meta['sua'] is None:
        raise SystemExit('aviation data unavailable')
    write(os.path.join(d, 'meta.json'), meta)
    open(os.path.join(args.out, '.changed'), 'w').close()
if __name__ == '__main__':
    main()
