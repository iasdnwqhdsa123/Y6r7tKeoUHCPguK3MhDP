from __future__ import annotations
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

def west_pacific(geometry) -> bool:

    def lons(c):
        if c and isinstance(c[0], (int, float)):
            yield c[0]
        else:
            for x in c or ():
                yield from lons(x)
    xs = list(lons((geometry or {}).get('coordinates')))
    return bool(xs) and all((100 <= x <= 172 for x in xs))
SERVICE = 'https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/Class_Airspace/FeatureServer/0'
ITEM = 'c6a62360338e408cb1512366ad61559e'
EXPORT = f'https://hub.arcgis.com/api/download/v1/items/{ITEM}/geojson?redirect=true&layers=0'
UA = {'User-Agent': 'wx-pipeline/1.0'}
VERSION = 8

def get_json(url: str, params: dict | None=None, tries: int=12) -> dict:
    full = f'{url}?{urllib.parse.urlencode(params)}' if params else url
    for attempt in range(tries):
        req = urllib.request.Request(full, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
        err = data.get('error') if isinstance(data, dict) else None
        if not err:
            return data
        if err.get('code') == 429 and attempt < tries - 1:
            print(f'  FAA quota exceeded, waiting 65 s ({attempt + 1}/{tries})', flush=True)
            time.sleep(65)
            continue
        raise RuntimeError(f'FAA service error: {err}')
    raise RuntimeError('unreachable')

def simplify(pts: list, tol: float) -> list:
    if len(pts) < 5:
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
            d = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm if norm else ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
            if d > far:
                far, idx = (d, i)
        if idx is not None and far > tol:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    out = [p for p, k in zip(pts, keep) if k]
    return out if len(out) >= 4 else pts[::max(1, len(pts) // 4)] + [pts[-1]]

def rounded(geom: dict, nd: int=4, tol: float=0.0005) -> dict:

    def ring(r):
        pts = simplify([[x, y] for x, y, *_ in r], tol)
        out = []
        for x, y in pts:
            p = [round(x, nd), round(y, nd)]
            if not out or out[-1] != p:
                out.append(p)
        return out
    if geom['type'] == 'Polygon':
        return {'type': 'Polygon', 'coordinates': [ring(r) for r in geom['coordinates']]}
    if geom['type'] == 'MultiPolygon':
        return {'type': 'MultiPolygon', 'coordinates': [[ring(r) for r in poly] for poly in geom['coordinates']]}
    return geom

def from_export() -> list:
    print('downloading export:', EXPORT, flush=True)
    data = get_json(EXPORT, tries=1)
    return data.get('features') or []

def from_service() -> list:
    info = get_json(SERVICE, {'f': 'json'})
    print('service:', info.get('name'), '| maxRecordCount', info.get('maxRecordCount'))
    print('fields:', [f['name'] for f in info.get('fields', [])])
    feats, offset = ([], 0)
    page = max(100, min(int(info.get('maxRecordCount') or 1000), 2000))
    while True:
        data = get_json(f'{SERVICE}/query', {'where': '1=1', 'outFields': '*', 'outSR': '4326', 'f': 'geojson', 'geometryPrecision': '4', 'maxAllowableOffset': '0.0008', 'orderByFields': 'OBJECTID', 'resultOffset': str(offset), 'resultRecordCount': str(page)})
        batch = data.get('features') or []
        feats += batch
        more = data.get('exceededTransferLimit') or (data.get('properties') or {}).get('exceededTransferLimit')
        print(f'  fetched {len(batch)} (total {len(feats)})', flush=True)
        if not batch or not more:
            return feats
        offset += len(batch)

def limit(val, uom, code) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '?'
    if code == 'SFC' or v == 0:
        return 'SFC'
    if v < 0:
        return '?'
    if uom == 'FL':
        return f'FL{round(v)}'
    return f'{round(v):,} ft{(' AGL' if code == 'AGL' else '')}'

def airspace_class(p: dict) -> str | None:
    c = str(p.get('CLASS') or '').strip().upper()
    if c in ('B', 'C', 'D'):
        return c
    lt = str(p.get('LOCAL_TYPE') or p.get('TYPE_CODE') or '').upper()
    for k in ('B', 'C', 'D'):
        if lt in (f'CLASS_{k}', f'CLASS {k}', k):
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dist-data')
    args = ap.parse_args()
    keyed = bool(os.environ.get('OPENAIP_API_KEY', '').strip())
    retry = keyed
    try:
        import europe_airspace
        print('open flightmaps and openAIP:', europe_airspace.build(args.out), flush=True)
        retry = keyed and any((not os.path.exists(os.path.join(args.out, 'airspace', f'{a}.json')) for a in europe_airspace.OPENAIP_AREAS))
    except Exception as e:
        print('open flightmaps and openAIP airspace failed:', e, flush=True)
    if retry:
        print('openAIP: an area has no file yet; the next job tries again', flush=True)
    try:
        feats = from_export()
    except Exception as e:
        print('export unavailable:', e)
        feats = []
    if not feats:
        feats = from_service()
    print('CLASS values:', Counter((str(f['properties'].get('CLASS')) for f in feats)).most_common(12))
    print('LOCAL_TYPE values:', Counter((str(f['properties'].get('LOCAL_TYPE')) for f in feats)).most_common(12))
    out = []
    for f in feats:
        p = f.get('properties') or {}
        c = airspace_class(p)
        if not c or not f.get('geometry') or west_pacific(f['geometry']):
            continue
        out.append({'type': 'Feature', 'geometry': rounded(f['geometry']), 'properties': {'c': c, 'n': p.get('NAME') or '', 'lo': limit(p.get('LOWER_VAL'), p.get('LOWER_UOM'), p.get('LOWER_CODE')), 'hi': limit(p.get('UPPER_VAL'), p.get('UPPER_UOM'), p.get('UPPER_CODE'))}})
    print('kept:', Counter((f['properties']['c'] for f in out)))
    if not out:
        raise SystemExit('no Class B/C/D features found; not overwriting the published file')
    d = os.path.join(args.out, 'airspace')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'us.json'), 'w') as fh:
        json.dump({'type': 'FeatureCollection', 'features': out}, fh, separators=(',', ':'))
    with open(os.path.join(d, 'meta.json'), 'w') as fh:
        stamp = datetime.now(timezone.utc) - (timedelta(days=7, hours=1) if retry else timedelta(0))
        json.dump({'updated': stamp.isoformat(timespec='seconds'), 'count': len(out), 'version': VERSION, 'source': 'FAA Class_Airspace'}, fh)
    open(os.path.join(args.out, '.changed'), 'w').close()
    print(f'wrote {len(out)} features, {os.path.getsize(os.path.join(d, 'us.json')) / 1000000.0:.1f} MB')
if __name__ == '__main__':
    main()
