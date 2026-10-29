from __future__ import annotations
import argparse
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import date, timedelta
from airspace import simplify
UA = {'User-Agent': 'wx-pipeline/1.0'}
SNAPSHOTS = 'https://snapshots.openflightmaps.org/live'
REGIONS = [(('ed', 'edww', 'edgg', 'edmm'), 'ed'), (('lovv',), 'lo'), (('lsas',), 'ls'), (('lkaa',), 'lk'), (('lzbb',), 'lz'), (('ebbu',), 'eb'), (('ehaa',), 'eh'), (('ekdk',), 'ek'), (('esaa',), 'es'), (('epww',), 'ep'), (('lhcc',), 'lh'), (('ljla',), 'lj'), (('ldzo',), 'ld'), (('lbsr',), 'lb'), (('lrbb',), 'lr'), (('efin',), 'ef'), (('lggg',), 'lg'), (('limm', 'li'), 'li')]
CONTROLLED = {'CTR', 'TMA', 'CTA', 'CLASS', 'AWY'}
ZONES = ('TMZ', 'RMZ')
OPENAIP = 'https://api.core.openaip.net/api/airspaces'
OPENAIP_AREAS = {'eu': ('GB', 'IE', 'ES', 'PT', 'FR', 'MA', 'DZ', 'TN', 'NO', 'EE', 'LV', 'LT', 'LU', 'IS', 'RS', 'BA', 'ME', 'MK', 'AL', 'XK', 'MD', 'CY', 'MT', 'TR', 'IL'), 'sa': ('BR', 'AR', 'CL', 'UY', 'PY', 'BO', 'PE', 'EC', 'CO', 'VE'), 'af': ('ZA', 'NA', 'BW', 'ZW', 'MZ', 'LS', 'SZ', 'ZM'), 'nz': ('NZ',), 'ca': ('CA',), 'mx': ('MX', 'GT', 'BZ', 'SV', 'HN', 'NI', 'CR', 'PA'), 'au': ('AU',), 'jp': ('JP',)}
SOURCES = {'eu': 'open flightmaps + openAIP'}
OPENAIP_BUDGET = 600
EAST_OF_180 = {'NZ'}
OA_CONTROLLED = {4: 'CTR', 7: 'TMA', 15: 'AWY', 26: 'CTA'}
OA_ZONES = {5: 'TMZ', 6: 'RMZ'}
OA_SUA = {1: 'R', 2: 'D', 3: 'P', 8: 'TRA', 9: 'TSA', 13: 'ATZ', 14: 'MATZ', 21: 'GLD'}
OA_CLASS = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
OA_UNIT = {0: 'M', 1: 'FT', 6: 'FL'}
OA_DATUM = {0: 'HEI', 1: 'ALT', 2: 'STD'}
SUA = {'P': 'P', 'R': 'R', 'D': 'D', 'TRA': 'TRA', 'TSA': 'TSA'}
NM = 1852.0

def airac(day: date) -> str:
    eff = date(2020, 1, 2) + timedelta(days=28 * ((day - date(2020, 1, 2)).days // 28))
    first = eff
    while (first - timedelta(days=28)).year == eff.year:
        first -= timedelta(days=28)
    return f'{eff.year % 100:02d}{(eff - first).days // 28 + 1:02d}'

def cycles(day: date) -> list[str]:
    return [airac(day), airac(day + timedelta(days=28)), airac(day - timedelta(days=28))]

def get(url: str, timeout: int=120) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise

def coord(text: str | None) -> float | None:
    m = re.fullmatch('\\s*([\\d.]+)\\s*([NSEW])\\s*', text or '')
    if not m:
        return None
    v, h = (m.group(1), m.group(2))
    whole = v.split('.')[0]
    lon = h in 'EW'
    if len(whole) >= (7 if lon else 6):
        d, mi, s = (float(whole[:-4]), float(whole[-4:-2]), float(whole[-2:] + ('.' + v.split('.')[1] if '.' in v else '')))
        x = d + mi / 60 + s / 3600
    else:
        x = float(v)
    return -x if h in 'SW' else x

def limit(code: str | None, val: str | None, uom: str | None) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '?'
    u = (uom or '').upper()
    if u == 'FL' or (code or '').upper() == 'STD':
        return f'FL{round(v)}'
    ft = v / 0.3048 if u == 'M' else v
    if ft <= 0 and (code or '').upper() in ('HEI', ''):
        return 'SFC'
    return f'{round(ft):,} ft{(' AGL' if (code or '').upper() == 'HEI' else '')}'

def bearing_dist(lat0, lon0, lat, lon):
    k = math.cos(math.radians(lat0))
    dx, dy = ((lon - lon0) * k * 111320.0, (lat - lat0) * 111320.0)
    return (math.atan2(dx, dy), math.hypot(dx, dy))

def offset(lat0, lon0, brg, dist):
    k = math.cos(math.radians(lat0))
    return (lat0 + dist * math.cos(brg) / 111320.0, lon0 + dist * math.sin(brg) / (111320.0 * k))

def arc(c_lat, c_lon, start, end, clockwise, step=math.radians(5)):
    b0, r0 = bearing_dist(c_lat, c_lon, *start)
    b1, r1 = bearing_dist(c_lat, c_lon, *end)
    sweep = (b1 - b0) % (2 * math.pi) if clockwise else -((b0 - b1) % (2 * math.pi))
    n = max(1, int(abs(sweep) / step))
    return [offset(c_lat, c_lon, b0 + sweep * i / n, r0 + (r1 - r0) * i / n) for i in range(n)]

def circle(c_lat, c_lon, radius_m, n=72):
    return [offset(c_lat, c_lon, 2 * math.pi * i / n, radius_m) for i in range(n)]

def text(el, path):
    x = el.find(path)
    return x.text.strip() if x is not None and x.text else None

def uid(el) -> tuple | None:
    if el is None:
        return None
    return (el.get('region') or '', text(el, 'codeType') or '', text(el, 'codeId') or '')

def ring_of(abd) -> list | None:
    circ = abd.find('Circle')
    if circ is not None:
        la, lo = (coord(text(circ, 'geoLatCen')), coord(text(circ, 'geoLongCen')))
        r = float(text(circ, 'valRadius') or 0)
        u = (text(circ, 'uomRadius') or 'NM').upper()
        r *= NM if u == 'NM' else 1000.0 if u == 'KM' else 1.0
        return circle(la, lo, r) if la is not None and lo is not None and (r > 0) else None
    vx = [(text(v, 'codeType') or 'GRC', coord(text(v, 'geoLat')), coord(text(v, 'geoLong')), coord(text(v, 'geoLatArc')), coord(text(v, 'geoLongArc'))) for v in abd.findall('Avx')]
    vx = [v for v in vx if v[1] is not None and v[2] is not None]
    if not vx:
        return None
    if len(vx) == 1 and vx[0][0] in ('CWA', 'CCA') and (vx[0][3] is not None):
        _, la, lo, cla, clo = vx[0]
        return circle(cla, clo, bearing_dist(cla, clo, la, lo)[1])
    pts = []
    for i, (kind, la, lo, cla, clo) in enumerate(vx):
        nxt = vx[(i + 1) % len(vx)]
        if kind in ('CWA', 'CCA') and cla is not None and (clo is not None):
            pts += arc(cla, clo, (la, lo), (nxt[1], nxt[2]), kind == 'CWA')
        else:
            pts.append((la, lo))
    return pts if len(pts) >= 3 else None

def polygon(ring) -> dict:
    pts = [[round(lo, 5), round(la, 5)] for la, lo in ring]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    pts = simplify(pts, 0.0005)
    return {'type': 'Polygon', 'coordinates': [[[round(x, 4), round(y, 4)] for x, y in pts]]}

def zone_of(kind: str, name: str, cls: str='') -> str | None:
    if kind in ZONES:
        return kind
    if kind in SUA or kind == 'CTR' or (kind in CONTROLLED and cls in ('A', 'B', 'C', 'D')):
        return None
    words = set(re.findall('[A-Z]+', name.upper()))
    return next((z for z in ZONES if z in words), None)

def parse(xml: bytes, kinds: Counter | None=None):
    root = ET.fromstring(xml)
    ases = {uid(a.find('AseUid')): a for a in root.iter('Ase')}
    rings = {}
    for abd in root.iter('Abd'):
        u = uid(abd.find('AbdUid/AseUid'))
        if u and u not in rings:
            r = ring_of(abd)
            if r:
                rings[u] = r
    for adg in root.iter('Adg'):
        u, same = (uid(adg.find('AdgUid/AseUid')), uid(adg.find('AseUidSameExtent')))
        if u and same in rings and (u not in rings):
            rings[u] = rings[same]
    controlled, sua, seen = ([], [], set())
    for u, a in ases.items():
        if u not in rings:
            continue
        kind = u[1].upper()
        cls = (text(a, 'codeClass') or '').upper()
        lo = limit(text(a, 'codeDistVerLower'), text(a, 'valDistVerLower'), text(a, 'uomDistVerLower'))
        hi = limit(text(a, 'codeDistVerUpper'), text(a, 'valDistVerUpper'), text(a, 'uomDistVerUpper'))
        name = (text(a, 'txtName') or u[2])[:60]
        if kinds is not None:
            kinds[kind] += 1
        zone = zone_of(kind, name, cls)
        if zone:
            g = polygon(rings[u])
            key = (zone, lo, hi, json.dumps(g['coordinates'][0][:4]))
            if key not in seen:
                seen.add(key)
                controlled.append({'type': 'Feature', 'geometry': g, 'properties': {'c': zone, 'n': name, 'lo': lo, 'hi': hi}})
        elif kind in CONTROLLED:
            c = cls if cls in ('A', 'B', 'C', 'D') else 'D' if kind == 'CTR' and (not cls) else None
            if not c:
                continue
            g = polygon(rings[u])
            key = (c, lo, hi, json.dumps(g['coordinates'][0][:4]))
            if key in seen:
                continue
            seen.add(key)
            controlled.append({'type': 'Feature', 'geometry': g, 'properties': {'c': c, 'n': name, 'lo': lo, 'hi': hi}})
        elif kind in SUA:
            sua.append({'type': 'Feature', 'geometry': polygon(rings[u]), 'properties': {'t': SUA[kind], 'n': name, 'lo': lo, 'hi': hi, 'time': (text(a, 'Att/codeWorkHr') or '')[:80], 'ctl': ''}})
    return (controlled, sua)

def oa_limit(lim) -> str:
    lim = lim or {}
    return limit(OA_DATUM.get(lim.get('referenceDatum')), lim.get('value'), OA_UNIT.get(lim.get('unit')))

def oa_features(items, cc: str='') -> tuple[list, list]:
    controlled, sua = ([], [])
    east = cc in EAST_OF_180
    for a in items:
        g = a.get('geometry') or {}
        if g.get('type') != 'Polygon' or not g.get('coordinates'):
            continue
        ring = [(y, x % 360 if east else x) for x, y, *_ in g['coordinates'][0]]
        if len(ring) < 3:
            continue
        t, name = (a.get('type'), str(a.get('name') or '')[:60])
        lo, hi = (oa_limit(a.get('lowerLimit')), oa_limit(a.get('upperLimit')))
        if t in OA_ZONES:
            controlled.append({'type': 'Feature', 'geometry': polygon(ring), 'properties': {'c': OA_ZONES[t], 'n': name, 'lo': lo, 'hi': hi, 'cc': cc}})
        elif t in OA_CONTROLLED or (t == 0 and a.get('icaoClass') in (1, 2, 3)):
            c = OA_CLASS.get(a.get('icaoClass')) or ('D' if t == 4 else None)
            if c:
                controlled.append({'type': 'Feature', 'geometry': polygon(ring), 'properties': {'c': c, 'n': name, 'lo': lo, 'hi': hi, 'cc': cc}})
        elif t in OA_SUA:
            sua.append({'type': 'Feature', 'geometry': polygon(ring), 'properties': {'t': OA_SUA[t], 'n': name, 'lo': lo, 'hi': hi, 'time': '', 'ctl': '', 'cc': cc}})
    return (controlled, sua)

def openaip_page(url: str, key: str, tries: int=4) -> dict:
    req = urllib.request.Request(url, headers={**UA, 'x-openaip-api-key': key, 'Accept': 'application/json'})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == tries - 1:
                raise
            wait = e.headers.get('Retry-After')
            time.sleep(min(120, float(wait)) if wait and wait.isdigit() else 10 * 2 ** attempt)
    raise RuntimeError('unreachable')

def openaip(country: str, key: str, deadline: float | None=None) -> list:
    items, page = ([], 1)
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f'out of time after {page - 1} page(s)')
        data = openaip_page(f'{OPENAIP}?country={country}&limit=1000&page={page}', key)
        time.sleep(1)
        items += data.get('items') or []
        if page >= int(data.get('totalPages') or 1):
            return items
        page += 1

def snapshot(folder: str, suffix: str, cyc: str) -> bytes | None:
    data = get(f'{SNAPSHOTS}/{cyc}/ofmx/{folder}/latest/ofmx_{suffix}.zip')
    if not data:
        return None
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.endswith('.ofmx') or (n.endswith('.xml') and 'Shape' not in n)]
        names.sort(key=lambda n: ('isolated' not in n, len(n)))
        return z.read(names[0]) if names else None

def area_paths(out: str, area: str) -> tuple[str, str]:
    d = os.path.join(out, 'airspace')
    return (os.path.join(d, f'{area}.json'), os.path.join(d, 'aviation', f'sua-{area}.json'))

def last_features(out: str, area: str) -> tuple[list, list]:
    feats = []
    for path in area_paths(out, area):
        try:
            with open(path) as f:
                feats.append(json.load(f).get('features') or [])
        except (OSError, ValueError, AttributeError):
            feats.append([])
    return (feats[0], feats[1])

def build(out: str, today: date | None=None) -> dict:
    today = today or date.today()
    areas = {area: ([], []) for area in OPENAIP_AREAS}
    controlled, sua = areas['eu']
    got = []
    for folders, suffix in REGIONS:
        done = False
        for cyc in cycles(today):
            for folder in folders:
                try:
                    xml = snapshot(folder, suffix, cyc)
                    if not xml:
                        continue
                    kinds = Counter()
                    c, s = parse(xml, kinds)
                    print(f'{folder}@{cyc} types: {kinds.most_common()}', flush=True)
                except Exception as e:
                    print(f'europe airspace {folder} {cyc}: {e}', flush=True)
                    continue
                controlled += c
                sua += s
                got.append(f'{folder}@{cyc}: {len(c)}+{len(s)}')
                done = True
                break
            if done:
                break
        if not done:
            print(f'europe airspace: no snapshot for {suffix} ({', '.join(folders)})', flush=True)
    key = os.environ.get('OPENAIP_API_KEY', '').strip()
    deadline = time.monotonic() + OPENAIP_BUDGET
    for area, countries in OPENAIP_AREAS.items() if key else ():
        last = None
        for cc in countries:
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"skipped, openAIP's {OPENAIP_BUDGET // 60} min are up")
                items = openaip(cc, key, deadline)
                if not items:
                    raise RuntimeError('no airspace records')
                print(f'openaip {cc}: types {Counter((a.get('type') for a in items)).most_common(12)}, classes {Counter((a.get('icaoClass') for a in items)).most_common(8)}', flush=True)
                c, s = oa_features(items, cc)
                got.append(f'openaip {cc}: {len(c)}+{len(s)}')
            except Exception as e:
                if last is None:
                    last = last_features(out, area)
                c, s = ([f for f in feats if (f.get('properties') or {}).get('cc') == cc] for feats in last)
                print(f'openaip {cc}: {e}; kept {len(c)}+{len(s)} from the last {area} files', flush=True)
            areas[area][0].extend(c)
            areas[area][1].extend(s)
    if not key:
        print("airspace: no OPENAIP_API_KEY, so only open flightmaps' Europe (no UK, Ireland, Iberia, France, Norway, the Baltics, the Balkans, Turkey, North Africa, South America, southern Africa, New Zealand, Canada, Mexico, Central America, Australia or Japan)", flush=True)
    print('airspace:', ', '.join(got) or 'none', flush=True)
    written = {}
    os.makedirs(os.path.join(out, 'airspace', 'aviation'), exist_ok=True)
    for area, (controlled, sua) in areas.items():
        print(f'{area} classes:', Counter((f['properties']['c'] for f in controlled)), '| areas:', Counter((f['properties']['t'] for f in sua)), flush=True)
        if not controlled and (not sua):
            print(f'{area}: no airspace; its files stay as they were', flush=True)
            continue
        for path, feats in zip(area_paths(out, area), (controlled, sua)):
            with open(path + '.tmp', 'w') as f:
                json.dump({'type': 'FeatureCollection', 'features': feats, 'source': SOURCES.get(area, 'openAIP')}, f, separators=(',', ':'))
            os.replace(path + '.tmp', path)
            print(f'{os.path.relpath(path, out)}: {len(feats)} features, {os.path.getsize(path) / 1000.0:,.0f} kB', flush=True)
        written[area] = (len(controlled), len(sua))
    if not written:
        raise RuntimeError('no airspace from open flightmaps or openAIP')
    return written
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dist-data')
    print(build(ap.parse_args().out))
