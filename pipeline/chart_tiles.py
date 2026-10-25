from __future__ import annotations
import argparse
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from aviation import CHART_BASE, UA, write
CYCLE_0 = datetime(2025, 1, 23, tzinfo=timezone.utc)
CYCLE_DAYS = 56
VERSION = 1
AREAS = ((-126.0, 23.5, -65.0, 52.0), (-180.0, 50.5, -129.0, 72.0), (-161.0, 18.5, -154.0, 22.6), (-68.2, 17.5, -64.2, 18.9))
CHARTS = {'vfr': {'svc': 'VFR_Sectional', 'native': 8, 'low': (4, 7), 'fmt': 'jpeg'}, 'tac': {'svc': 'VFR_Terminal', 'native': 10, 'low': (6, 9), 'fmt': 'png', 'stride': 2}}
QUALITY = 72
WORKERS = 12

def log(*a):
    print(*a, flush=True)

def cycle_of(t: datetime) -> int:
    return (t - timedelta(days=1) - CYCLE_0).days // CYCLE_DAYS

def tile_range(area, z):
    w, s, e, n = area
    k = 2 ** z
    x = lambda lon: int((lon + 180) / 360 * k)
    y = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * k)
    return (range(x(w), min(k - 1, x(e)) + 1), range(y(n), min(k - 1, y(s)) + 1))

def fetch(svc, z, y, x):
    from PIL import Image
    url = f'{CHART_BASE}/{svc}/MapServer/tile/{z}/{y}/{x}'
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404):
                return None
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        return None
    try:
        im = Image.open(io.BytesIO(data)).convert('RGBA')
    except Exception:
        return None
    return im if im.getextrema()[3][1] > 0 else None

def native_tiles(c):
    z, svc, got = (c['native'], c['svc'], {})
    todo = set()
    for area in AREAS:
        xs, ys = tile_range(area, z)
        step = c.get('stride', 1)
        todo |= {(x, y) for x in xs[::step] for y in ys[::step]}
    seen = set()
    with ThreadPoolExecutor(WORKERS) as pool:
        while todo:
            batch = sorted(todo - seen)
            seen |= set(batch)
            todo = set()
            for (x, y), im in zip(batch, pool.map(lambda t: fetch(svc, z, t[1], t[0]), batch)):
                if im is None:
                    continue
                got[x, y] = im
                if c.get('stride'):
                    todo |= {(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - seen
            log(f'  {svc} zoom {z}: {len(seen)} asked, {len(got)} with chart')
    return got

def save(im, path, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == 'jpeg':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel('A'))
        im = bg
    im.save(path, 'WEBP', quality=QUALITY, method=4)

def build(cid, c, out):
    from PIL import Image
    tiles = native_tiles(c)
    if not tiles:
        raise RuntimeError(f'{cid}: no tiles')
    lo, hi = c['low']
    d = os.path.join(out, cid)
    n = 0
    for z in range(hi, lo - 1, -1):
        parents = {}
        for (x, y), im in tiles.items():
            parents.setdefault((x // 2, y // 2), []).append((x % 2 * 256, y % 2 * 256, im))
        tiles = {}
        for (x, y), kids in parents.items():
            big = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
            for ox, oy, im in kids:
                big.paste(im, (ox, oy))
            tiles[x, y] = small = big.resize((256, 256), Image.LANCZOS)
            save(small, os.path.join(d, str(z), str(y), f'{x}.webp'), c['fmt'])
        n += len(tiles)
        log(f'  {cid} zoom {z}: {len(tiles)} tiles')
    return {'minZoom': lo, 'maxZoom': hi, 'tiles': n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--charts', nargs='*', default=list(CHARTS))
    a = ap.parse_args()
    root = os.path.join(a.out, 'airspace', 'aviation')
    meta_path = os.path.join(root, 'charts-low.json')
    try:
        meta = json.load(open(meta_path))
    except (OSError, ValueError):
        meta = {}
    cycle = cycle_of(datetime.now(timezone.utc))
    if not a.force and meta.get('cycle') == cycle and (meta.get('version') == VERSION) and all((c in meta.get('charts', {}) for c in a.charts)):
        log(f'chart overviews: cycle {cycle} on hand')
        return 0
    tmp = os.path.join(root, 'charts.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    charts = {}
    for cid in a.charts:
        t0 = time.time()
        log(f'chart overviews: {cid}')
        charts[cid] = build(cid, CHARTS[cid], tmp)
        log(f'  {cid}: {charts[cid]['tiles']} tiles in {(time.time() - t0) / 60:.1f} min')
    final = os.path.join(root, 'charts')
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    write(meta_path, {'cycle': cycle, 'version': VERSION, 'updated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'charts': charts})
    open(os.path.join(a.out, '.changed'), 'w').close()
    size = sum((os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(final) for f in fs))
    log(f'chart overviews: cycle {cycle}, {size / 1000000.0:.1f} MB')
    return 0
if __name__ == '__main__':
    sys.exit(main())
