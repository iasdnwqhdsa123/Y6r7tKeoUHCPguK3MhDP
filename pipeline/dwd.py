from __future__ import annotations
import bz2
from concurrent.futures import ThreadPoolExecutor
import grib
BASE = 'https://opendata.dwd.de/weather/nwp/icon-d2/grib'

def url(run, fh: int, name: str, p: int) -> str:
    return f'{BASE}/{run:%H}/{name}/icon-d2_germany_regular-lat-lon_pressure-level_{run:%Y%m%d%H}_{fh:03d}_{p}_{name}.grib2.bz2'

def message(u: str) -> bytes:
    return bz2.decompress(grib.http_get(u, tries=4, timeout=60))

class Field:

    def __init__(self, grid, msg: bytes):
        self.grid, self.msg = (grid, msg)

    def read(self, region, boxes=None):
        import numpy as np
        i_n, j_w, ny, nx = self.grid.window(region)
        a, _ = grib.decode(self.msg)
        if a.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(f'ICON-D2 message on a {a.shape} grid, not {(self.grid.ny, self.grid.nx)}')
        v = a[i_n - ny + 1:i_n + 1, j_w:j_w + nx][::-1]
        if not boxes:
            return np.ascontiguousarray(v, np.float32)
        out = np.full((ny, nx), np.nan, np.float32)
        for y0, y1, x0, x1 in boxes:
            out[y0:y1, x0:x1] = v[y0:y1, x0:x1]
        return out

def submit(pool: ThreadPoolExecutor, run, fh: int, keys, names: dict) -> dict:
    return {k: pool.submit(message, url(run, fh, names[k[0]], int(k[1][:-3]))) for k in keys}
