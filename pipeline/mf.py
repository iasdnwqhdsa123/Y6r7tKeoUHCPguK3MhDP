from __future__ import annotations
import fcntl
import os
import tempfile
import threading
import time
import grib
BASE = 'https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt'
CONST = 'https://www.data.gouv.fr/fr/datasets/r/2ea15b4c-36e1-47ba-98d3-43c09c9aef65'
LAST_FH = 51
KEEP_S = 12 * 3600
PART = 8 << 20

def group(fh: int) -> tuple[int, int]:
    if fh <= 6:
        return (0, 6)
    lo = (fh - 7) // 6 * 6 + 7
    return (lo, min(lo + 5, LAST_FH))

def url(run, pkg: str, fh: int) -> str:
    lo, hi = group(fh)
    r = f'{run:%Y-%m-%dT%H:%M:%SZ}'
    return f'{BASE}/{r}/arome/0025/{pkg}/arome__0025__{pkg}__{lo:02d}H{hi:02d}H__{r}.grib2'

def folder() -> str:
    d = os.environ.get('MF_CACHE') or os.path.join(tempfile.gettempdir(), 'arome-mf')
    os.makedirs(d, exist_ok=True)
    return d

def _tidy(d: str, keep: str):
    now = time.time()
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if p != keep and (not f.endswith('.lock')) and (now - os.path.getmtime(p) > KEEP_S):
            for q in (p, p + '.lock'):
                try:
                    os.remove(q)
                except OSError:
                    pass

def local(u: str, workers: int=8) -> str:
    import om
    d = folder()
    path = os.path.join(d, u.rsplit('/', 1)[-1] if '/pnt/' in u else u.rsplit('/', 1)[-1] + '.grib2')
    if os.path.exists(path):
        return path
    with open(path + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not os.path.exists(path):
                if '/pnt/' in u:
                    tmp = om.download(u, workers=workers, part=PART, folder=d)
                else:
                    data = grib.http_get(u, timeout=120)
                    fd, tmp = tempfile.mkstemp(dir=d)
                    with os.fdopen(fd, 'wb') as f:
                        f.write(data)
                os.replace(tmp, path)
                _tidy(d, path)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return path
_INDEX: dict = {}
_INDEX_LOCK = threading.Lock()

def index(path: str) -> dict:
    with _INDEX_LOCK:
        if path in _INDEX:
            return _INDEX[path]
    import eccodes
    out = {}
    with open(path, 'rb') as fh:
        while True:
            g = eccodes.codes_grib_new_from_file(fh)
            if g is None:
                break
            try:
                key = (eccodes.codes_get(g, 'shortName'), str(eccodes.codes_get(g, 'stepRange')))
                out.setdefault(key, (int(eccodes.codes_get(g, 'offset')), int(eccodes.codes_get(g, 'totalLength'))))
            finally:
                eccodes.codes_release(g)
    with _INDEX_LOCK:
        _INDEX[path] = out
    return out

def message(path: str, key) -> bytes | None:
    at = index(path).get(key)
    if at is None:
        return None
    with open(path, 'rb') as f:
        f.seek(at[0])
        return f.read(at[1])

class Field:

    def __init__(self, grid, parts):
        self.grid, self.parts = (grid, parts)

    def read(self, region, boxes=None):
        import numpy as np
        i_n, j_w, ny, nx = self.grid.window(region)
        top = self.grid.ny - 1 - i_n
        v = None
        for c, m in self.parts:
            a = grib.decode(m)[0]
            if a.shape != (self.grid.ny, self.grid.nx):
                raise ValueError(f'AROME message on a {a.shape} grid, not {(self.grid.ny, self.grid.nx)}')
            v = c * a if v is None else v + c * a
        v = v[top:top + ny, j_w:j_w + nx]
        if not boxes:
            return v.astype(np.float32)
        out = np.full((ny, nx), np.nan, np.float32)
        for y0, y1, x0, x1 in boxes:
            out[y0:y1, x0:x1] = v[y0:y1, x0:x1]
        return out

def instant(run, pkg, name, fh):
    m = message(local(url(run, pkg, fh)), (name, str(fh)))
    return None if m is None else [(1.0, m)]

def hour_mean(run, pkg, name, fh, sign=1.0):
    now = message(local(url(run, pkg, fh)), (name, f'0-{fh}'))
    if now is None:
        return None
    parts = [(sign / 3600, now)]
    if fh > 1:
        prev = message(local(url(run, pkg, fh - 1)), (name, f'0-{fh - 1}'))
        if prev is None:
            return None
        parts.append((-sign / 3600, prev))
    return parts

def constant(name):
    m = message(local(CONST), (name, '0'))
    return None if m is None else [(1.0, m)]
