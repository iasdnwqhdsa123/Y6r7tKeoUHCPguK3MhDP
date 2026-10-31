from __future__ import annotations
import os
import numpy as np
import leelattice
import runwave
import shade as shade_mod
import sky
FIELDS = ('rrun', 'rrunf', 'wrun', 'wrunf', 'wrunz', 'wtop', 'crest', 'cshade', 'cthin', 'csun')
RUN_FIELDS = ('rrun', 'rrunf', 'wrun', 'wrunf', 'wrunz', 'wtop', 'crest')
FINE = {'rrunf': 'rrun', 'wrunf': 'wrun'}
SHADE_FIELDS = ('cshade', 'cthin', 'csun')
HERE = os.path.dirname(os.path.abspath(__file__))
_DIRS = {k: [os.path.join(HERE, 'geo', k), os.path.join(HERE, '..', 'public', 'geo', k)] for k in ('lee', 'ridge')}
_EXPOSURES: dict = {}

def exposure(kind):
    if kind not in _EXPOSURES:
        d = next((p for p in _DIRS[kind] if os.path.exists(os.path.join(p, 'index.json'))), None)
        _EXPOSURES[kind] = runwave.Exposure(d) if d else None
    return _EXPOSURES[kind]

def _steps(v, lo, hi, transform):
    x = np.clip(np.asarray(v, np.float64), lo, hi)
    if transform == 'sqrt':
        a, b = (np.sqrt(max(lo, 0)), np.sqrt(hi))
        return (np.sqrt(x) - a) / (b - a) * 254
    return (x - lo) / (hi - lo) * 254

def _encode(v, lo, hi, transform):
    s = _steps(v, lo, hi, transform)
    return np.where(np.isfinite(s), np.rint(np.nan_to_num(s)), 255).astype(np.uint8)

def publish(q, enc, grid, when, mid_by_ceil=False, agl_heights=True):
    if 'wstar' not in q or 'elevation' not in q:
        return {}
    dec = {n: sky.decode(c, *enc[n]) for n, c in q.items() if n in enc and getattr(c, 'ndim', 0) == 2}
    out = {}
    lee, ridge = (exposure('lee'), exposure('ridge'))
    bands = None
    if lee is not None and 'wtrap' in q:
        lat = leelattice.lattice(dec, grid, lee.edges)
        bands = leelattice.lattice_cells(lat, grid)
    r = runwave.run_fields(dec, grid, lee, ridge, bands)
    out['rrun'] = r['ridge']
    out['wrun'] = r['wave']
    for f, n in FINE.items():
        s = _steps(out[n], *enc[n])
        out[f] = s - np.rint(s)
    out['wrunz'] = np.where(r['wave'] > 0, r['wave_agl'], np.nan)
    out['wtop'] = r['wave_top']
    out['crest'] = r['crest']
    s = shade_mod.shade(dec, grid, when, mid_by_ceil, agl_heights, rays=True)
    out['cshade'], out['cthin'], out['csun'] = (s['shade'], s['thin'], s['sun'])
    nodata = ~np.isfinite(dec['wstar'])
    return {n: np.where(nodata, 255, _encode(v, *enc[n])).astype(np.uint8) for n, v in out.items()}
