from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.spatial import cKDTree
import grib
import physics
import physics as P
import pack
import tiles
from sources import SOURCES, candidate_runs, latest_complete_run, needed_times
from tiles import downsample
REGIONS = {'hrrr': (-125.0, 24.0, -66.5, 50.0, 0.03), 'gfs': (-170.0, -54.75, 179.75, 70.0, 0.25), 'ecmwf': (-130.0, 20.0, -60.0, 55.0, 0.25), 'icon': (-170.0, -54.875, 179.75, 70.0, 0.125)}
KEEP = {'gfs': ((-130.0, 20.25, -52.0, 56.0), (-168.0, 59.0, -130.25, 70.0), (-163.0, 54.5, -148.0, 59.0), (-137.0, 54.5, -130.25, 59.0), (-25.0, 34.0, 44.75, 70.0), (-13.0, 27.0, 12.0, 34.0), (-76.0, -54.75, -53.0, -16.0), (-53.0, -34.0, -34.0, -2.0), (-82.0, -16.0, -66.0, 12.5), (11.0, -35.0, 36.0, -16.0), (166.0, -48.0, 179.0, -34.0), (114.0, -35.5, 120.5, -28.5), (131.5, -26.0, 136.5, -21.5), (144.0, -27.0, 154.0, -18.5), (136.0, -39.5, 154.0, -27.0), (143.5, -44.0, 148.75, -39.5))}
KEEP['icon'] = KEEP['gfs']

def keep_mask(src_id, lat2d, lon2d):
    boxes = KEEP.get(src_id)
    if not boxes:
        return None
    m = np.zeros(lat2d.shape, bool)
    for w, s, e, n in boxes:
        m |= (lon2d >= w) & (lon2d <= e) & (lat2d >= s) & (lat2d <= n)
    return m

def keep_windows(src_id, region, halo=1):
    boxes = KEEP.get(src_id)
    if not boxes:
        return None
    lons, lats = target_grid(region)
    out = []
    for w, s, e, n in boxes:
        xs = np.nonzero((lons >= w - 1e-06) & (lons <= e + 1e-06))[0]
        ys = np.nonzero((lats >= s - 1e-06) & (lats <= n + 1e-06))[0]
        if not len(xs) or not len(ys):
            continue
        y0, y1 = (max(0, ys[0] - halo), min(len(lats), ys[-1] + 1 + halo))
        x0, x1 = (max(0, xs[0] - halo), min(len(lons), xs[-1] + 1 + halo))
        out.append(((y0, y1, x0, x1), (ys[0] - y0, ys[-1] + 1 - y0, xs[0] - x0, xs[-1] + 1 - x0)))
    return out

def _crop(a, y0, y1, x0, x1, shape):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and (a.shape[-2:] == shape):
        return np.ascontiguousarray(a[..., y0:y1, x0:x1])
    if isinstance(a, tuple):
        return tuple((_crop(x, y0, y1, x0, x1, shape) for x in a))
    if isinstance(a, list):
        return [_crop(x, y0, y1, x0, x1, shape) for x in a]
    if isinstance(a, dict):
        return {k: _crop(x, y0, y1, x0, x1, shape) for k, x in a.items()}
    return a

def paste(windows, grids, shape):
    if windows is None:
        return grids[0]
    out = None
    for ((y0, y1, x0, x1), (iy0, iy1, ix0, ix1)), v in zip(windows, grids):
        if v is None:
            continue
        if out is None:
            out = np.full(shape, np.nan, np.float32)
        out[y0 + iy0:y0 + iy1, x0 + ix0:x0 + ix1] = v[iy0:iy1, ix0:ix1]
    return out

def analyse_parts(parts, coords, grid_step, valid, windows, shape):
    done = [analyse(s, lv, levels, grid_step, lat2d, lon2d, valid) for (s, lv, levels), (lat2d, lon2d) in zip(parts, coords)]
    names = dict.fromkeys((k for part in done for k in part))
    return {k: paste(windows, [part.get(k) for part in done], shape) for k in names}

def part_coords(regrids):
    return [np.meshgrid(rg.lats, rg.lons, indexing='ij') for rg in regrids]

def analyse_kept(s, lv, levels, grid_step, lat2d, lon2d, valid, windows):
    if windows is None:
        return analyse(s, lv, levels, grid_step, lat2d, lon2d, valid)
    shape = lat2d.shape
    cut = [lambda a, w=w: _crop(a, *w, shape) for w, _ in windows]
    return analyse_parts([(c(s), c(lv), levels) for c in cut], [(c(lat2d), c(lon2d)) for c in cut], grid_step, valid, windows, shape)
TILE = {'hrrr': 100, 'gfs': 100, 'ecmwf': 100, 'icon': 200}
OVERVIEW = {'hrrr': (2, 4, 8)}
LEVEL_TILE = 200
CLASS_FIELDS = {'ptype', 'fltcat', 'sinkby'}
WIND_ALTS_FT = [2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 24000, 30000]
WAVE_ALTS_FT = [8000, 10000, 12000, 14000, 18000, 24000]
SAMPLED_FIELDS = {f'wv{ft}' for ft in WAVE_ALTS_FT} | {'wlow'}
ENC = {'wstar': (0, 6, None), 'zi': (0, 6000, None), 'depth': (0, 6000, None), 'cubase': (0, 7000, None), 'cudepth': (0, 12000, None), 'blapse': (0, 15, None), 'twidth': (0, 1000, None), 'overdev': (0, 100, None), 'bs': (0, 40, None), 'cape': (0, 6000, 'sqrt'), 'rain': (0, 50, 'sqrt'), 'cloud': (0, 100, None), 'cloudlow': (0, 100, None), 'cloudmid': (0, 100, None), 'cloudhigh': (0, 100, None), 't2': (-40, 50, None), 'td2': (-45, 35, None), 'sst': (-2, 36, None), 'snowd': (0, 3, 'sqrt'), 'mslp': (950, 1060, None), 'heat': (0, 800, None), 'densalt': (-1000, 7000, None), 'gust': (0, 50, None), 'wavemax': (0, 8, None), 'wavemaxz': (0, 12000, None), 'wavepot': (0, 100, None), 'wavelen': (0, 60000, None), 'nlow': (0, 0.04, None), 'wlow': (-4, 4, None), 'wtrap': (0, 100, None), 'wlam': (0, 60000, None), 'conv': (-2, 2, None), 'sink': (0, 6, None), 'sinkth': (0, 3, None), 'sinkby': (0, 254, None), 'ridge': (0, 4, None), 'bumpy': (0, 1, None), 'pm25': (0, 300, 'sqrt'), 'elevation': (-100, 6000, None), 'ceil': (0, P.CEIL_NONE, 'sqrt'), 'vis': (0, 16100, 'sqrt'), 'fltcat': (0, 254, None), 'ptype': (0, 254, None), 'slr': (P.SLR_RANGE[0], P.SLR_RANGE[1], None), 'frzl': (-100, 6500, None), 'icing': (0, 100, None), 'icebase': (0, 8000, None), 'icetop': (0, 8000, None), 'turb': (0, 25, None), 'turbz': (0, 11000, None), 'turbbase': (0, 11000, None), 'turbtop': (0, 11000, None), 'llws': (0, 40, None), 'ctop': (0, 16000, None), 'refc': (0, 75, None)}
for ft in WAVE_ALTS_FT:
    ENC[f'wv{ft}'] = (-8, 8, None)
    ENC[f'dpd{ft}'] = (0, 40, None)
    ENC[f't{ft}'] = (-70, 40, None)
ML_TOP_DZ = (200, 500, 1000, 1800, 3000)
ENC['thml'] = (250, 350, None)
ENC['thx'] = (0, 5, None)
for dz in ML_TOP_DZ:
    ENC[f'dth_m{dz}'] = (-3, 22, None)
    ENC[f'q_m{dz}'] = (0, 20, None)
for p in P.CC_LEVELS:
    ENC[f'cc{p}'] = (0, 100, None)
WIND_AGL_M = [100, 300]
WIND_ENC, WIND_ENC_ALOFT, WIND_ALOFT_FT = ((-60, 60, None), (-100, 100, None), 12000)
for key in ['sfc', 'blavg', 'bltop'] + [f'a{ft}' for ft in WIND_ALTS_FT] + [f'g{m}' for m in WIND_AGL_M]:
    enc = WIND_ENC_ALOFT if key.startswith('a') and int(key[1:]) >= WIND_ALOFT_FT else WIND_ENC
    ENC[f'u_{key}'] = enc
    ENC[f'v_{key}'] = enc
LEVEL_NONE = {'ceil': 254}
LEVEL_SKIP = {f'cc{p}' for p in P.CC_LEVELS} | {'thml', 'thx'} | {f'{k}_m{dz}' for dz in ML_TOP_DZ for k in ('dth', 'q')}

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def _fmt_dur(sec):
    sec = int(round(sec))
    return f'{sec // 60}m{sec % 60:02d}s' if sec >= 60 else f'{sec}s'

class Progress:

    def __init__(self, label, total):
        self.label, self.total, self.done, self.t0 = (label, total, 0, time.time())
        log(f'{label}: {total} hours to compute')

    def tick(self):
        self.done += 1
        el = time.time() - self.t0
        left = el / self.done * (self.total - self.done)
        eta = time.strftime('%H:%M', time.localtime(time.time() + left))
        log(f'{self.label}: [{self.done}/{self.total}] {_fmt_dur(el)} elapsed' + (f', ~{_fmt_dur(left)} left (done ≈ {eta})' if self.done < self.total else ', all hours done'))

def target_grid(region):
    w, s, e, n, step = region[:5]
    return (np.arange(w, e + step / 2, step), np.arange(n, s - step / 2, -step))

class Regridder:

    def __init__(self, src_lat, src_lon, region):
        w, s, e, n, step = region
        self.lons, self.lats = target_grid(region)
        glon, glat = np.meshgrid(self.lons, self.lats)
        lat0 = math.radians((s + n) / 2)
        src = np.column_stack([src_lat.ravel(), src_lon.ravel() * math.cos(lat0)])
        dst = np.column_stack([glat.ravel(), glon.ravel() * math.cos(lat0)])
        dist, self.index = cKDTree(src).query(dst, workers=-1)
        src_step = np.median(np.abs(np.diff(src_lat[:, src_lat.shape[1] // 2]))) if src_lat.ndim == 2 else step
        self.outside = dist > max(step, float(src_step)) * 1.5
        self.shape = glat.shape
        ix = np.round((src_lon - w) / step).astype(np.int64)
        iy = np.round((n - src_lat) / step).astype(np.int64)
        ny, nx = self.shape
        self.cell = np.where((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny), iy * nx + ix, -1).ravel()
        self._plan = None

    def __call__(self, field):
        out = field.ravel()[self.index].reshape(self.shape).astype(np.float32)
        out[self.outside.reshape(self.shape)] = np.nan
        return out

    def peak(self, field):
        out = self(field)
        if self._plan is None:
            ok = np.nonzero(self.cell >= 0)[0]
            order = ok[np.argsort(self.cell[ok], kind='stable')]
            tgt = self.cell[order]
            starts = np.r_[0, np.nonzero(np.diff(tgt))[0] + 1] if len(tgt) else np.zeros(0, np.int64)
            self._plan = (order, starts, tgt[starts])
        order, starts, tgt = self._plan
        if len(order):
            flat = out.reshape(-1)
            vals = np.fmax.reduceat(np.asarray(field, np.float32).ravel()[order], starts)
            keep = np.isfinite(vals) & ~self.outside.reshape(-1)[tgt]
            flat[tgt[keep]] = vals[keep]
        return out

    @classmethod
    def identity(cls, region):
        r = object.__new__(cls)
        r.lons, r.lats = target_grid(region)
        r.shape = (len(r.lats), len(r.lons))
        n = r.shape[0] * r.shape[1]
        r.index = np.arange(n)
        r.outside = np.zeros(n, bool)
        r.cell = np.arange(n)
        r._plan = None
        return r

    def part(self, y0, y1, x0, x1):
        p = object.__new__(Regridder)
        p.lats, p.lons = (self.lats[y0:y1], self.lons[x0:x1])
        p.shape = (y1 - y0, x1 - x0)
        p.index = self.index.reshape(self.shape)[y0:y1, x0:x1].ravel()
        p.outside = self.outside.reshape(self.shape)[y0:y1, x0:x1].ravel()
        nx = self.shape[1]
        iy, ix = (self.cell // nx, self.cell % nx)
        inside = (self.cell >= 0) & (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
        p.cell = np.where(inside, (iy - y0) * (x1 - x0) + (ix - x0), -1)
        p._plan = None
        return p

def quantize(arr, lo, hi, transform):
    a = np.asarray(arr, dtype=np.float32)
    if transform == 'sqrt':
        a, lo, hi = (np.sqrt(np.maximum(a, 0)), math.sqrt(max(lo, 0)), math.sqrt(hi))
    q = np.round((np.clip(a, lo, hi) - lo) / (hi - lo) * 254)
    return np.where(np.isfinite(a), q, 255).astype(np.uint8)

def solar_estimate(valid, lat, lon, cloud_pct):
    doy = valid.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    hour = valid.hour + valid.minute / 60 + lon / 15
    ha = np.radians(15 * (hour - 12))
    la = np.radians(lat)
    cosz = np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)
    ghi = 1098 * np.maximum(cosz, 0) * np.exp(-0.057 / np.maximum(cosz, 0.05))
    return ghi * (1 - 0.75 * (np.clip(cloud_pct, 0, 100) / 100) ** 3.4)

def instant_factor(valid, hours, lat, lon):
    now = solar_estimate(valid, lat, lon, 0)
    n = 12
    mean = sum((solar_estimate(valid - timedelta(hours=hours * k / n), lat, lon, 0) * (0.5 if k in (0, n) else 1) for k in range(n + 1))) / n
    return np.where(mean > 20, np.clip(now / np.maximum(mean, 0.001), 0, 2.5), np.where(now > 1, 1.0, 0.0))

def heat_from_sunshine(ghi, t2, td2, land=None):
    frac = np.clip(0.09 + 0.0203 * np.minimum(np.maximum(t2 - td2, 0), 16), 0.05, 0.42)
    heat = frac * np.maximum(ghi, 0)
    return heat if land is None else heat * (land >= 0.5)

def water_temp(s):
    t, land = (s.get('tsfc'), s.get('lsm'))
    if t is None or land is None:
        return None
    water = land < 0.5
    if s.get('ice') is not None:
        water &= ~(np.nan_to_num(s['ice'], nan=0) >= 0.5)
    return np.where(water, t, np.nan)

def sun_elevation(valid, lat, lon):
    doy = valid.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    hour = valid.hour + valid.minute / 60 + lon / 15
    ha = np.radians(15 * (hour - 12))
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.clip(np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha), -1, 1)))
SUN_GATE = (1.0, 8.0)

def sun_gate(valid, lat, lon):
    lo, hi = SUN_GATE
    return np.clip((sun_elevation(valid, lat, lon) - lo) / (hi - lo), 0, 1)
CU_MIN_LOW_CLOUD = 5.0
CU_MARGIN_NO_CLOUD = 200.0
DALR = 9.8
BLOCK_LAYER = 500.0
WAVE_BAND_M = (300.0, 600.0, 900.0)
WAVE_BAND_MAX_STEP = 0.1

def thermal_width(zi, blapse, bs):
    d0 = 2 * np.maximum(10, 0.102 * 0.6 ** (1 / 3) * (1 - 0.25 * 0.6) * zi)
    stab = np.clip(1.25 - 0.15 * (DALR - np.nan_to_num(blapse, nan=DALR - 5 / 3)), 0.7, 1.15)
    shear = np.clip(0.55 + 0.09 * np.nan_to_num(bs, nan=5), 0.6, 1.0)
    return np.where(zi > 200, d0 * stab * shear, np.nan)
STORM_DBZ = 50.0

def cumulus_forms(ws, zi, lcl, low_cloud=None):
    reach = (ws > 0.5) & (lcl < zi)
    if low_cloud is None or np.all(np.isnan(low_cloud)):
        return reach & (zi - lcl > CU_MARGIN_NO_CLOUD)
    return reach & (np.nan_to_num(low_cloud, nan=0) >= CU_MIN_LOW_CLOUD)
DRIFT_KM = (2.5, 5.0, 7.5, 10.0, 15.0)
DRIFT_THERMAL_S = 900.0
DRIFT_WIND_M = 500.0
DRIFT_MIN_WS = 0.25

def thermal_drift(zs, hth, zi, ws, u, v, t2, ps, lat2d, grid_step):
    U = np.hypot(np.nan_to_num(u, nan=0), np.nan_to_num(v, nan=0))
    windy = U > 0.5
    ex = np.where(windy, np.nan_to_num(u, nan=0) / np.maximum(U, 1e-06), 0)
    ey = np.where(windy, np.nan_to_num(v, nan=0) / np.maximum(U, 1e-06), 0)
    ny, nx = zi.shape
    km_x = grid_step * 111.32 * np.maximum(np.cos(np.radians(lat2d)), 0.05)
    km_y = grid_step * 111.32
    rows, cols = np.indices(zi.shape)
    ws_in = np.zeros_like(ws)
    top_in = np.zeros_like(zi)
    lam = np.maximum(U * DRIFT_THERMAL_S / 1000, 1.0)
    for d in DRIFT_KM:
        c = np.clip(np.rint(cols - ex * d / km_x), 0, nx - 1).astype(np.int64)
        r = np.clip(np.rint(rows + ey * d / km_y), 0, ny - 1).astype(np.int64)
        cand = np.nan_to_num(ws[r, c], nan=0) * np.exp(-d / lam)
        better = cand > ws_in
        ws_in = np.where(better, cand, ws_in)
        top_in = np.where(better, np.nan_to_num(zs[r, c] + zi[r, c], nan=0), top_in)
    drifted = windy & (ws_in >= DRIFT_MIN_WS) & (ws_in > np.nan_to_num(ws, nan=0)) & np.isfinite(zs)
    zi2 = np.where(drifted, np.fmax(zi, top_in - zs), zi)
    ws2 = np.where(drifted, ws_in, ws)
    return (zi2.astype(zi.dtype, copy=False), ws2.astype(ws.dtype, copy=False))

def analyse(s, lv, levels, grid_step, lat2d, lon2d=None, valid=None):
    zs = s['zs']
    prof = P.Profile(zs, s['ps'], s['t2'], s['td2'], s['u10'], s['v10'], lv['z'], levels, lv['t'], lv['td'], lv['u'], lv['v'], s.get('uh'), s.get('vh'), s.get('hub_m', 100.0))
    if s.get('shf') is not None:
        heat = np.maximum(np.nan_to_num(s['shf'], nan=0), 0)
    else:
        heat = heat_from_sunshine(s['sw'], s['t2'], s['td2'], s.get('lsm'))
    hth = heat * sun_gate(valid, lat2d, lon2d) if valid is not None and lon2d is not None else heat
    zi = s.get('hpbl')
    if zi is None or np.all(np.isnan(zi)):
        zi = P.parcel_ml_depth(prof, np.where(hth > 20, 1.0, 0.2))
    zi = np.maximum(np.nan_to_num(zi, nan=0), 0)
    ws = np.where(hth > 10, P.wstar(hth, zi, s['t2'], s['ps']), 0)
    ud, vd = prof.at(zs + DRIFT_WIND_M, 'u', 'v')
    zi, ws = thermal_drift(zs, hth, zi, ws, ud, vd, s['t2'], s['ps'], lat2d, grid_step)
    up15, vp15 = prof.at(zs + 1500, 'u', 'v')
    cool = P.night_cooling(s['shf']) if s.get('shf') is not None else None
    pump = P.mountain_plain(zs, hth, up15, vp15, grid_step, cool=cool, coslat=np.cos(np.radians(lat2d)))
    pump_sink, pump_lift = (pump['sink'], pump['lift'])
    held = np.where(hth > 0, np.minimum(pump_sink * P.pump_unresolved(grid_step) * P.PUMP_TOP_S, P.PUMP_TOP_CAP * zi), 0)
    ws = np.where((held > 0) & (zi > 0), ws * np.cbrt(np.maximum(zi - held, 1) / np.maximum(zi, 1)), ws)
    zi = np.maximum(zi - held, 0)
    wml = np.zeros_like(zs)
    n = 8
    for i in range(n):
        td, p = prof.at(zs + 2 + zi * i / (n - 1), 'td', 'lnp')
        wml += P.mixing_ratio(td, p)
    wml /= n
    lcl = np.maximum(0, 125 * (s['t2'] - P.dewpoint_from_w(wml, s['ps'])))
    has_cu = cumulus_forms(ws, zi, lcl, s.get('lcc'))
    depth = np.where(has_cu, lcl, np.where(ws > 0, zi, 0))
    d = np.maximum(np.where(has_cu, lcl, zi), 100)
    ub, vb = P.layer_mean(prof, zs + 2, zs + 2 + d, ['u', 'v'])
    ut, vt = prof.at(zs + 2 + d, 'u', 'v')
    ul, vl = prof.at(zs + 2 + 0.1 * d, 'u', 'v')
    bs = P.bs_ratio(ws, np.hypot(ut - ul, vt - vl))
    z1, z2 = (zs + 2 + 0.3 * zi, zs + 2 + 0.9 * zi)
    blapse = np.where(zi >= 300, (prof.at(z1, 't') - prof.at(z2, 't')) / np.maximum(z2 - z1, 1) * 1000, np.nan)
    twidth = np.where(ws > 0.5, thermal_width(zi, blapse, bs), np.nan)
    base, tbase = (zs + 2 + lcl, s['t2'] - P.DRY_LAPSE * lcl)
    ctop = P.cumulus_top(prof, base, tbase, has_cu, mix=P.cumulus_mix(twidth))
    cudepth = np.where(has_cu, np.maximum(ctop - base, zi - lcl), 0)
    cape = np.nan_to_num(s.get('cape', np.zeros_like(zs)), nan=0)
    rain = np.nan_to_num(s.get('prate', np.zeros_like(zs)), nan=0)
    refc = s.get('refc')
    refc = None if refc is None else np.nan_to_num(refc, nan=0)
    convecting = (rain > 0.5) & (cape > 300)
    if refc is not None:
        convecting |= refc >= STORM_DBZ - 10
    if convecting.any():
        deep = P.cumulus_top(prof, base, tbase, has_cu & convecting)
        cudeep = np.where(has_cu & convecting, np.maximum(deep - base, zi - lcl), cudepth)
    else:
        cudeep = cudepth
    overdev = np.clip((cudeep - 2000) / 4000, 0, 1) * np.clip(0.4 + cape / 1500, 0, 1) * 100
    overdev = np.where((rain > 0.5) & (cape > 300), np.maximum(overdev, 60), overdev)
    if refc is not None:
        storm = (refc >= STORM_DBZ) | (refc >= STORM_DBZ - 10) & (cape >= 100)
        overdev = np.where(storm, np.maximum(overdev, 70), overdev)
    thermals_here = ws > 0
    zt = zs + 2 + zi
    thml = P.theta(*prof.at(zs + 2 + 0.5 * zi, 't', 'lnp'))
    thx = np.clip(P.theta(s['t2'], s['ps']) - thml, 0, 5)
    sounding = {'thml': thml, 'thx': thx}
    for dz in ML_TOP_DZ:
        t, p = prof.at(zt + dz, 't', 'lnp')
        td = prof.at(zt + dz, 'td')
        sounding[f'dth_m{dz}'] = P.theta(t, p) - thml
        sounding[f'q_m{dz}'] = P.mixing_ratio(np.minimum(td, t), p) * 1000
    sounding = {k: np.where(thermals_here, v, np.nan) for k, v in sounding.items()}
    out = {'wstar': ws, 'zi': zi, 'depth': depth, 'cubase': np.where(has_cu, lcl, np.nan), 'cudepth': cudepth, 'overdev': overdev, 'bs': bs, 'blapse': np.clip(blapse, 0, 15), 'twidth': twidth, 'cape': cape, 'rain': rain, 't2': s['t2'], 'td2': s['td2'], 'mslp': s.get('mslp'), 'sst': water_temp(s), 'heat': heat, 'densalt': P.density_altitude(s['ps'], s['t2']), 'gust': s.get('gust'), 'cloud': s.get('tcc'), 'cloudlow': s.get('lcc'), 'cloudmid': s.get('mcc'), 'cloudhigh': s.get('hcc'), 'pm25': s.get('smoke'), 'snowd': None if s.get('snod') is None else np.maximum(s['snod'], 0), 'u_sfc': s['u10'], 'v_sfc': s['v10'], 'u_blavg': ub, 'v_blavg': vb, 'u_bltop': ut, 'v_bltop': vt}
    out.update(cloud_levels(s, lv, levels))
    out.update(sounding)
    for m in WIND_AGL_M:
        out[f'u_g{m}'], out[f'v_g{m}'] = prof.at(zs + m, 'u', 'v')
    for key, z in (('sfc', 10.0), *((f'g{m}', float(m)) for m in WIND_AGL_M)):
        w = np.clip(1 - z / pump['depth'], 0, 1)
        out[f'u_{key}'] = out[f'u_{key}'] + w * pump['u']
        out[f'v_{key}'] = out[f'v_{key}'] + w * pump['v']
    for ft in WIND_ALTS_FT:
        z = np.full_like(zs, ft * P.FT)
        u, v = prof.at(z, 'u', 'v')
        below = z < zs + 30
        out[f'u_a{ft}'] = np.where(below, np.nan, u)
        out[f'v_a{ft}'] = np.where(below, np.nan, v)
    dy = grid_step * 111320.0
    dx = dy * np.cos(np.radians(lat2d))

    def ddx(a):
        return np.gradient(a, axis=1) / dx

    def ddy(a):
        return -np.gradient(a, axis=0) / dy
    out['conv'] = -(ddx(ub) + ddy(vb)) * np.maximum(zi, 300) + pump_lift
    relief = P.local_relief(zs, max(1, round(0.1 / grid_step)))
    out['bumpy'] = np.where(np.isfinite(s['t2']), P.bumpiness(prof, ws, zi, s.get('gust'), s['u10'], s['v10'], s.get('ustar'), relief), np.nan)
    hilly = relief >= 150
    ur, vr = prof.at(zs + 300, 'u', 'v')
    upslope = ur * ddx(zs) + vr * ddy(zs)
    out['ridge'] = np.where(hilly, np.maximum(0, upslope), np.nan)
    th0 = P.theta(*prof.at(zs + 10, 't', 'lnp'))
    th1 = P.theta(*prof.at(zs + 10 + BLOCK_LAYER, 't', 'lnp'))
    n2 = P.G / (0.5 * (th0 + th1)) * (th1 - th0) / BLOCK_LAYER
    out['nlow'] = np.where(hilly, np.sqrt(np.maximum(n2, 0)), np.nan)
    if lv.get('w') is not None:
        wz = lv['z']
        w = lv['w']
        order = np.argsort(-np.asarray(levels))
        wz, w = (wz[order], w[order])
        above = wz > zs[None] + 500
        wmask = np.where(above, w, -np.inf)
        k = np.argmax(wmask, axis=0)
        wmax = P.take_along_levels(wmask, k)
        out['wavemax'] = np.where(np.isfinite(wmax), wmax, np.nan)
        out['wavemaxz'] = np.where(np.isfinite(wmax) & (wmax > 0), P.take_along_levels(wz, k), np.nan)
        for ft in WAVE_ALTS_FT:
            zq = ft * P.FT
            hi = P.level_positions(np.clip(P.levels_below(wz, zq), 1, wz.shape[0] - 1))
            lo = hi - zs.size
            z0, z1 = (P.take_level(wz, lo, zs.shape), P.take_level(wz, hi, zs.shape))
            w0, w1 = (P.take_level(w, lo, zs.shape), P.take_level(w, hi, zs.shape))
            f = np.clip((zq - z0) / np.maximum(z1 - z0, 1), 0, 1)
            out[f'wv{ft}'] = np.where(zq > zs + 100, w0 + f * (w1 - w0), np.nan)
    for ft in WAVE_ALTS_FT:
        zq = np.full_like(zs, ft * P.FT)
        tq, tdq = prof.at(zq, 't', 'td')
        out[f'dpd{ft}'] = np.where(zq > zs + 100, np.maximum(0, tq - tdq), np.nan)
        out[f't{ft}'] = np.where(zq > zs + 100, tq, np.nan)
    wz_up = w_up = None
    if lv.get('w') is not None:
        up = np.argsort(-np.asarray(levels))
        wz_up, w_up = (lv['z'][up], lv['w'][up])
    if w_up is not None and grid_step <= WAVE_BAND_MAX_STEP:
        under = wz_up < (zs + 10)[None]
        k = np.arange(wz_up.shape[0], dtype=np.float32)[:, None, None]
        zc = np.concatenate([zs[None], np.where(under, zs[None] + 0.01 * (k + 1), wz_up)])
        wc = np.concatenate([np.zeros_like(zs)[None], np.where(under, 0, np.nan_to_num(w_up, nan=0))])
        wl = sum((P.column_at(zc, wc, zs + h) for h in WAVE_BAND_M)) / len(WAVE_BAND_M)
        out['wlow'] = np.where(hilly & np.isfinite(s['t2']), wl, np.nan)
    lee = np.maximum(0, -upslope) * P.SINK_LEE * np.clip((relief - 100) / 300, 0, 1)
    falling = P.precip_type(rain, s.get('crain'), s.get('csnow'), s.get('cfrzr'), s.get('cicep'), s.get('ptype'), s['t2'])
    snow = P.snow_share(falling, prof, depth)
    sink, between, why = P.sink_rate(prof, wz_up, w_up, ws, zi, depth, rain, s.get('refd1'), lee, parts=True, snow=snow)
    if s.get('snod') is not None and valid is not None and (lon2d is not None):
        gl = P.glacier_sink(s['snod'], s['t2'], np.hypot(ddx(zs), ddy(zs)), sun_gate(valid, lat2d, lon2d)) - between
        why = np.where((gl > sink) & (gl >= 0.05), np.float32(4), why)
        sink = np.maximum(sink, gl)
    why = np.where((pump_sink > sink) & (pump_sink >= 0.05), np.float32(5), why)
    sink = np.maximum(sink, pump_sink)
    inside = np.isfinite(s['t2'])
    out['sink'], out['sinkth'], out['sinkby'] = (np.where(inside, a, np.nan) for a in (sink, between, why))
    score, wavelen = P.wave_environment(prof)
    forcing = np.clip((relief - 100) / 600, 0, 1)
    out['wavepot'] = score * forcing * 100
    out['wavelen'] = np.where(out['wavepot'] >= 20, wavelen, np.nan)
    trapping = np.nan_to_num(score, nan=0) > 0
    out['wtrap'] = np.where(trapping, score * 100, np.nan)
    out['wlam'] = np.where(trapping, wavelen, np.nan)
    out.update(general_aviation(s, lv, levels, prof, rain, grid_step, ddx, ddy, [out[f'cc{p}'] for p in P.CC_LEVELS], zi=zi))
    return out

def cloud_levels(s, lv, levels):
    ps = np.where(np.isfinite(s['t2']), s['ps'], np.nan)
    if lv.get('cloud') is not None:
        levels, rh, clw, own = lv['cloud']
    else:
        rh = lv['rh'] if lv.get('rh') is not None else P.rh_cloud(lv['t'], lv['td'])
        clw, own = (lv.get('clw'), lv.get('cc'))
    have = [float(p) for p in levels]
    if own is not None and all((p in have for p in P.CC_LEVELS)):
        own = np.clip(own[[have.index(p) for p in P.CC_LEVELS]], 0, 100)
        with np.errstate(invalid='ignore'):
            cc = np.where(np.array(P.CC_LEVELS, np.float32)[:, None, None] <= ps[None], own, np.nan)
    else:
        cc = P.cloud_cover_levels(levels, rh, ps, P.rh_cloud(s['t2'], s['td2']), clw)
    return {f'cc{p}': a for p, a in zip(P.CC_LEVELS, cc)}
RADAR_BLUR_KM = 4.0

def blur_dbz(dbz, sigma):
    from scipy.ndimage import gaussian_filter
    z = np.power(10, np.asarray(dbz, np.float32) / 10)
    return (10 * np.log10(np.maximum(gaussian_filter(z, sigma, mode='nearest'), 1))).astype(np.float32)

def radar(s, rain=None, step=None):
    if s.get('radar') is not None:
        return np.maximum(np.nan_to_num(s['radar'], nan=0), 0)
    if s.get('refd1') is not None:
        d = np.maximum(np.nan_to_num(s['refd1'], nan=0), 0)
        sigma = RADAR_BLUR_KM / (step * 111.32) if step else 0
        return blur_dbz(d, sigma) if sigma >= 0.3 else d
    if s.get('refc') is not None:
        return np.maximum(np.nan_to_num(s['refc'], nan=0), 0)
    return None if rain is None else P.reflectivity_from_rain(rain)

def general_aviation(s, lv, levels, prof, rain, grid_step, ddx, ddy, cc=None, zi=None):
    zs = s['zs']
    inside = np.isfinite(s['t2'])
    tcc = s.get('tcc')
    if s.get('ceil') is not None:
        ceil = P.ceiling_agl(s['ceil'], zs, inside)
    elif cc is not None:
        zc = P.level_heights(levels, lv['z'])
        ceil = np.where(inside, P.ceiling_from_cover(zc, np.stack(cc), zs, P.rh_from_td(s['t2'], s['td2']), tcc), np.nan)
    else:
        ceil = np.where(inside, P.ceiling_from_humidity(prof, tcc), np.nan)
    ptype = P.precip_type(rain, s.get('crain'), s.get('csnow'), s.get('cfrzr'), s.get('cicep'), s.get('ptype'), s['t2'])
    vis = np.clip(s['vis'], 0, 16100) if s.get('vis') is not None else P.visibility_estimate(s['t2'], s['td2'], rain, ptype == 2)
    vis = np.where(inside, vis, np.nan)
    derived_frz = P.freezing_level(prof)
    frz = s.get('frz')
    frzl = derived_frz if frz is None else np.where(np.isfinite(frz), np.maximum(frz, zs), derived_frz)
    ctop = s['ctop'] if s.get('ctop') is not None else P.cloud_top_from_humidity(prof, tcc)
    refc = radar(s, rain, grid_step)
    order = np.argsort(-np.asarray(levels))
    z_l, t_l = (lv['z'][order], lv['t'][order])
    rh_l = lv['rh'][order] if lv.get('rh') is not None else P.rh_from_td(t_l, lv['td'][order])
    clw_l = lv['clw'][order] if lv.get('clw') is not None else None
    icing, icebase, icetop = P.icing_potential(z_l, t_l, rh_l, clw_l, zs)
    from scipy.ndimage import uniform_filter
    k = max(1, round(0.25 / grid_step))
    smooth = (lambda a: uniform_filter(a, k, mode='nearest')) if k > 1 else None
    turb, turbz, turbbase, turbtop = P.ellrod_turbulence(prof, ddx, ddy, smooth, zi, band=True)
    u2k, v2k = prof.at(zs + 2 + 610, 'u', 'v')
    llws = np.hypot(u2k - s['u10'], v2k - s['v10'])
    nan = lambda a: np.where(inside, a, np.nan)
    snowing = inside & np.isin(ptype, (2, 3))
    slr = np.full(zs.shape, np.nan, np.float32)
    if snowing.any():
        slr = np.where(snowing, P.kuchera_ratio(prof), np.nan)
    return {'ceil': ceil, 'vis': vis, 'fltcat': P.flight_category(ceil, vis), 'ptype': nan(ptype), 'slr': slr, 'refc': nan(refc), 'frzl': nan(frzl), 'ctop': nan(ctop), 'icing': nan(icing), 'icebase': icebase, 'icetop': icetop, 'turb': nan(turb), 'turbz': np.where(turb >= 2, turbz, np.nan), 'turbbase': turbbase, 'turbtop': turbtop, 'llws': nan(llws)}

def _decode(m, region=None, boxes=None):
    if hasattr(m, 'read'):
        return m.read(region, boxes)
    return grib.decode(m)[0]

def _fetch(src, run, fh, keys):
    raw = src.fetch(run, fh, src.inventory(run, fh, keys))
    return {k: grib.decode(m)[0] for k, m in raw.items()}

def prev_step(src, fh):
    p = fh - 1
    while p > 0 and (not src.fstep(p)):
        p -= 1
    return p

def _window_mean(src, run, fh, canon, now_vals, earlier):
    kind = src.windowed[canon]
    key = src.surface[canon]
    if fh <= 0:
        return (None, 0)
    if kind == 'avg6':
        start = (fh - 1) // 6 * 6
        prev = prev_step(src, fh)
        if prev <= start:
            return (now_vals, fh - start)
        a_prev = earlier(prev, key)
        if a_prev is None:
            return (now_vals, fh - start)
        return (((fh - start) * now_vals - (prev - start) * a_prev) / (fh - prev), fh - prev)
    if kind == 'acc1h':
        return (now_vals / 3600, 1)
    if kind in ('mean', 'sum'):
        step = fh - prev_step(src, fh)
        return (now_vals if kind == 'mean' else now_vals / (step * 3600), step)
    if kind in ('acc', 'accw'):
        prev = prev_step(src, fh)
        a_prev = earlier(prev, key)
        if a_prev is None:
            return (None, 0)
        step = fh - prev
        return ((now_vals - a_prev) / (step * 3600) * (1000 if kind == 'accw' else 1), step)
    raise ValueError(kind)
SUNLIT = {'sw', 'shf'}

def resolve_windows(src, run, fh, native, earlier):
    for canon in src.windowed:
        if native.get(canon) is None:
            native[canon] = native.get(f'{canon}_now')
            continue
        mean, hours = _window_mean(src, run, fh, canon, native[canon], earlier)
        native[canon] = ('win', mean, hours) if mean is not None else native.get(f'{canon}_now')
    if src.gust_window and fh == 0:
        native['gust'] = None
    return native

def fetch_hour(src, run, fh):
    return src.fetch(run, fh, src.inventory(run, fh))

def load_hour(src, run, fh, regrid_holder, region, valid, raw=None):
    raw = fetch_hour(src, run, fh) if raw is None else raw
    if regrid_holder.get('r') is None and hasattr(src, 'static_url'):
        regrid_holder['r'], regrid_holder['rot'] = (Regridder.identity(region), None)
    if regrid_holder.get('r') is None:
        any_msg = next(iter(raw.values()))
        lat, lon = grib.latlons(any_msg)
        if lat.ndim == 1 or src.grid == 'latlon':
            lat = lat.reshape(lat.shape[0], -1)
        regrid_holder['r'] = Regridder(lat, lon, region)
        regrid_holder['rot'] = grib.earth_rotation(raw.get(src.surface['u10']) or any_msg)
    rg = regrid_holder['r']
    if 'parts' not in regrid_holder:
        windows = regrid_holder['windows'] = keep_windows(src.id, region)
        regrid_holder['parts'] = [rg] if windows is None else [rg.part(*w) for w, _ in windows]
        regrid_holder['boxes'] = None if windows is None else [w for w, _ in windows]
    windows, parts = (regrid_holder['windows'], regrid_holder['parts'])
    boxes = regrid_holder['boxes']
    coords = [np.meshgrid(p.lats, p.lons, indexing='ij') for p in parts]
    native = {name: None if raw.get(k) is None else _decode(raw[k], region, boxes) for name, k in src.surface.items()}
    cache = {}

    def earlier(pfh, key):
        if pfh not in cache:
            keys = [src.surface[c] for c in src.windowed if native.get(c) is not None]
            try:
                cache[pfh] = _fetch(src, run, pfh, keys)
            except Exception as e:
                log(f'  f{fh:03d}: earlier step f{pfh:03d} unavailable ({e})')
                cache[pfh] = {}
        return cache[pfh].get(key)
    resolve_windows(src, run, fh, native, earlier)
    if src.zs_step0_only:
        if regrid_holder.get('zs') is None:
            key = src.surface['zs']
            z0 = native.get('zs') if fh == 0 else _fetch(src, run, 0, [key]).get(key)
            regrid_holder['zs'] = [None if z0 is None else p(z0 / (P.G if src.zs_geopotential else 1)) for p in parts]
        native['zs'] = ('done', regrid_holder['zs'])

    def surface(v, name, i):
        p, (lat2d, lon2d) = (parts[i], coords[i])
        if v is None:
            return None
        if isinstance(v, tuple) and v[0] == 'win':
            a = p(v[1])
            if name not in SUNLIT:
                return a
            f = instant_factor(valid, v[2], lat2d, lon2d)
            return np.where(a > 0, a * f, a) if name == 'shf' else a * f
        if isinstance(v, tuple) and v[0] == 'done':
            return v[1][i]
        return p(v)
    as_fraction = {}
    for k in ('tcc', 'lcc', 'mcc', 'hcc') if windows is not None else ():
        v = native.get(k)
        if v is not None:
            as_fraction[k] = np.nanmax(rg(v[1] if isinstance(v, tuple) else v)) <= 1.01

    def get(key):
        m = raw.get(key)
        if m is None:
            return [None] * len(parts)
        f = _decode(m, region, boxes)
        return [p(f) for p in parts]
    lvp = {canon: [get((gname, f'{l} mb')) for l in src.levels] for canon, gname in src.level_vars.items()}
    hip = {c: [get((src.level_vars[c], f'{l} mb')) for l in src.cloud_levels] for c in src.cloud_vars()}
    rot = regrid_holder.get('rot')
    levels = np.array(src.levels, dtype=np.float32)
    out = []
    for i in range(len(parts)):
        lat2d, lon2d = coords[i]
        s = {name: surface(v, name, i) for name, v in native.items()}
        if isinstance(native.get('refd1'), np.ndarray):
            s['radar'] = parts[i].peak(native['refd1'])
        if src.zs_geopotential and (not src.zs_step0_only) and (s.get('zs') is not None):
            s['zs'] = s['zs'] / P.G
        lv = {}
        for canon, stack in lvp.items():
            stack = [x[i] for x in stack]
            lv[canon] = None if any((x is None for x in stack)) else np.stack(stack)
        hi = {c: [x[i] for x in v] for c, v in hip.items()}
        hi_ok = [j for j in range(len(src.cloud_levels)) if all((hi[c][j] is not None for c in hi if c != 'cc'))]
        for a, b in (('u10', 'v10'), ('uh', 'vh')):
            s[a], s[b] = grib.earth_winds(s.get(a), s.get(b), lon2d, rot)
        lv['u'], lv['v'] = grib.earth_winds(lv.get('u'), lv.get('v'), lon2d[None], rot)
        for k in ('t2', 'td2', 'tsfc'):
            if s.get(k) is not None:
                s[k] = s[k] - 273.15
        lv['t'] = lv['t'] - 273.15
        if src.moisture == 'td' and lv.get('td') is not None:
            lv['td'] = lv['td'] - 273.15
        s['ps'] = s['ps'] / 100
        if s.get('mslp') is not None:
            s['mslp'] = s['mslp'] / 100
        if s.get('prate') is not None:
            s['prate'] = np.maximum(s['prate'], 0) * 3600
        if s.get('smoke') is not None:
            s['smoke'] = s['smoke'] * 1000000000.0
        for k in ('tcc', 'lcc', 'mcc', 'hcc'):
            if s.get(k) is not None and (np.nanmax(s[k]) <= 1.01 if windows is None else as_fraction[k]):
                s[k] = s[k] * 100
        if src.moisture == 'rh':
            lv['td'] = P.dewpoint_from_rh(lv['t'], lv['rh'])
        rh_c = lv['rh'] if lv.get('rh') is not None else P.rh_cloud(lv['t'], lv['td'])
        clw_c, cc_c = (lv.get('clw'), lv.get('cc'))
        if hi_ok:
            pick = lambda c: np.stack([hi[c][j] for j in hi_ok])
            rh_hi = pick('rh') if 'rh' in hi else P.rh_cloud(pick('t') - 273.15, pick('td') - 273.15)
            rh_c = np.concatenate([rh_c, rh_hi])
            clw_c = None if clw_c is None else np.concatenate([clw_c, pick('clw')])
            cc_c = None if cc_c is None or any((hi['cc'][j] is None for j in hi_ok)) else np.concatenate([cc_c, pick('cc')])
        lv['cloud'] = (np.array(list(src.levels) + [src.cloud_levels[j] for j in hi_ok], np.float32), rh_c, clw_c, cc_c)
        if lv.get('w') is not None and src.omega:
            p = levels[:, None, None] * 100
            lv['w'] = -lv['w'] * P.RD * (lv['t'] + 273.15) / (p * P.G)
        if s.get('zs') is None:
            tv = s['t2'] + 273.15
            s['zs'] = np.maximum(0, P.RD * tv / P.G * np.log(s['mslp'] / s['ps']))
        if s.get('sw') is None:
            s['sw'] = solar_estimate(valid, lat2d, lon2d, s.get('tcc', np.zeros_like(s['t2'])))
        s['hub_m'] = src.hub_m
        out.append((s, lv, levels))
    return out
_CTX: dict = {}
HOURS_VERSION = 14

def _job_hour(job):
    run, fh = job if isinstance(job, tuple) else (_CTX['run'], job)
    return (run, fh, run + timedelta(hours=fh))

def _kept(run, valid):
    hours_dir = _CTX['hours']
    npz, stamp = (os.path.join(hours_dir, f'{valid:%Y%m%d%H}{ext}') for ext in ('.npz', '.run'))
    if _CTX.get('recompute') or not (os.path.exists(npz) and os.path.exists(stamp)):
        return False
    with open(stamp) as f:
        return f.read().strip() == f'{run:%Y%m%d%H}'

def _prefetch(job):
    run, fh, valid = _job_hour(job)
    if _kept(run, valid):
        return None
    try:
        return fetch_hour(_CTX['src'], run, fh)
    except Exception as e:
        return e

def _process_hour(job, raw=None):
    src, region, hours_dir, holder = (_CTX[k] for k in ('src', 'region', 'hours', 'holder'))
    run, fh, valid = _job_hour(job)
    t0 = time.time()
    npz, stamp = (os.path.join(hours_dir, f'{valid:%Y%m%d%H}{ext}') for ext in ('.npz', '.run'))
    if _kept(run, valid):
        log(f'  f{fh:03d} {valid:%a %HZ} kept (already computed from this run)')
        return valid
    try:
        if isinstance(raw, Exception):
            raise raw
        parts = load_hour(src, run, fh, holder, region, valid, raw=raw)
    except Exception as e:
        log(f'  f{fh:03d} skipped: {e}')
        return None
    rg, windows = (holder['r'], holder['windows'])
    lat2d = np.repeat(rg.lats[:, None], len(rg.lons), axis=1)
    lon2d = np.repeat(rg.lons[None, :], len(rg.lats), axis=0)
    fields = analyse_parts(parts, part_coords(holder['parts']), region[4], valid, windows, lat2d.shape)
    fields['elevation'] = paste(windows, [s['zs'] for s, _, _ in parts], lat2d.shape)
    q = {k: quantize(v, *ENC[k]) for k, v in fields.items() if v is not None and k in ENC}
    keep = keep_mask(src.id, lat2d, lon2d)
    if keep is not None:
        for v in q.values():
            v[~keep] = 255
    if _CTX.pop('pick_physics', False):
        pick_physics(q, lambda: analyse_parts(parts, part_coords(holder['parts']), region[4], valid, windows, lat2d.shape), fh, keep)
    part = os.path.join(hours_dir, f'{valid:%Y%m%d%H}.tmp.npz')
    tiles.savez(part, q)
    os.replace(part, npz)
    with open(stamp, 'w') as f:
        f.write(f'{run:%Y%m%d%H}')
    log(f'  f{fh:03d} {valid:%a %HZ} done in {time.time() - t0:.1f}s')
    return valid
NATIVE_TOLERANCE = 0.0001

def pick_physics(q, compute, fh, keep=None):
    global P
    import physics_native
    if not physics_native.available():
        log(f'physics: physics.py (C++ not available: {physics_native.error})')
        return
    t0 = time.time()
    P = physics_native
    try:
        fields = compute()
        qc = {k: quantize(v, *ENC[k]) for k, v in fields.items() if v is not None and k in ENC}
        qc['elevation'] = q['elevation']
        if keep is not None:
            for v in qc.values():
                v[~keep] = 255
    except Exception as e:
        P = physics
        log(f'::warning::physics: the C++ failed on f{fh:03d} ({e!r}), physics.py for this run')
        return
    bad, near = ([], [])
    for k in sorted(set(q) | set(qc)):
        a, b = (q.get(k), qc.get(k))
        if a is None or b is None or a.shape != b.shape:
            bad.append(f'{k} missing')
            continue
        diff = a != b
        if not diff.any():
            continue
        share = float(diff.mean())
        worst = int(np.abs(a.astype(np.int16) - b.astype(np.int16))[diff].max())
        (near if share <= NATIVE_TOLERANCE and worst <= 1 else bad).append(f'{k} {diff.sum()} cells, {worst} code(s)')
    if bad:
        P = physics
        log(f'::warning::physics: the C++ differs from physics.py on f{fh:03d} ({'; '.join(bad[:8])}), physics.py for this run')
        return
    log(f'physics: C++ (matches physics.py on f{fh:03d}: {len(q) - len(near)} of {len(q)} fields identical' + (f'; {', '.join(near)}' if near else '') + f'; checked in {time.time() - t0:.1f}s)')
PREFETCH_AHEAD = 2

def _lane(jobs, results):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(PREFETCH_AHEAD) as ex:
        ahead = deque((ex.submit(_prefetch, job) for _, job in jobs[:PREFETCH_AHEAD]))
        for n, (i, job) in enumerate(jobs):
            try:
                raw = ahead.popleft().result()
            except Exception as e:
                raw = e
            if n + PREFETCH_AHEAD < len(jobs):
                ahead.append(ex.submit(_prefetch, jobs[n + PREFETCH_AHEAD][1]))
            try:
                r = _process_hour(job, raw)
            except Exception as e:
                log(f'  {job}: failed ({e!r})')
                r = None
            del raw
            results.put((i, r))

def process_lanes(jobs, workers, progress=None):
    import multiprocessing as mp
    import queue as _queue
    ctx = mp.get_context('fork')
    results = ctx.Queue()
    indexed = list(enumerate(jobs))
    procs = [ctx.Process(target=_lane, args=(indexed[w::workers], results), daemon=True) for w in range(workers)]
    for p in procs:
        p.start()
    out, got = ([None] * len(jobs), 0)
    while got < len(jobs):
        try:
            i, r = results.get(timeout=30)
        except _queue.Empty:
            if not any((p.is_alive() for p in procs)):
                log(f'::warning::hour workers stopped with {len(jobs) - got} hours left')
                break
            continue
        out[i] = r
        got += 1
        if progress:
            progress.tick()
    for p in procs:
        p.join(timeout=10)
    return out
QUARTER_MAX_FH = 18
QUARTER_FIELDS = ('ceil', 'vis', 'refc', 'rain', 'ptype', 'u_sfc', 'v_sfc', 'gust', 't2', 'td2', 'ctop')
SUBH_VARS = {'zs': ('HGT', 'surface'), 't2': ('TMP', '2 m above ground'), 'td2': ('DPT', '2 m above ground'), 'u10': ('UGRD', '10 m above ground'), 'v10': ('VGRD', '10 m above ground'), 'gust': ('GUST', 'surface'), 'vis': ('VIS', 'surface'), 'refc': ('REFC', 'entire atmosphere'), 'refd1': ('REFD', '1000 m above ground'), 'prate': ('PRATE', 'surface'), 'ceil': ('HGT', 'cloud ceiling'), 'ctop': ('HGT', 'cloud top'), 'crain': ('CRAIN', 'surface'), 'csnow': ('CSNOW', 'surface'), 'cfrzr': ('CFRZR', 'surface'), 'cicep': ('CICEP', 'surface')}

def subh_url(run, fh):
    return f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrfsubhf{fh:02d}.grib2'

def quarter_fields(s, step=None):
    t2 = s['t2'] - 273.15
    inside = np.isfinite(t2)
    nan = lambda a: None if a is None else np.where(inside, a, np.nan)
    rain = np.maximum(np.nan_to_num(s['prate'], nan=0), 0) * 3600 if s.get('prate') is not None else None
    out = {'t2': t2, 'td2': None if s.get('td2') is None else s['td2'] - 273.15, 'u_sfc': s.get('u10'), 'v_sfc': s.get('v10'), 'gust': nan(s.get('gust')), 'vis': None if s.get('vis') is None else nan(np.clip(s['vis'], 0, 16100)), 'refc': nan(radar({'radar': s.get('refd1'), 'refc': s.get('refc')}, step=step)), 'rain': nan(rain), 'ctop': nan(s.get('ctop'))}
    if s.get('ceil') is not None and s.get('zs') is not None:
        out['ceil'] = P.ceiling_agl(s['ceil'], s['zs'], inside)
    if rain is not None:
        out['ptype'] = nan(P.precip_type(rain, s.get('crain'), s.get('csnow'), s.get('cfrzr'), s.get('cicep'), None, t2))
    return {k: v for k, v in out.items() if v is not None}

def _process_quarters(fh):
    run, region, qdir, holder = (_CTX[k] for k in ('run', 'region', 'quarters', 'holder'))
    t0 = time.time()
    url = subh_url(run, fh)
    done = []
    try:
        rows = grib.parse_noaa_idx(grib.http_get(url + '.idx', tries=4).decode())
    except Exception as e:
        log(f'  q{fh:03d} skipped: {e}')
        return done
    for m in (fh * 60 - 45, fh * 60 - 30, fh * 60 - 15):
        valid = run + timedelta(minutes=m)
        npz, stamp = (os.path.join(qdir, f'{valid:%Y%m%d%H%M}{ext}') for ext in ('.npz', '.run'))
        if not _CTX.get('recompute') and os.path.exists(npz) and os.path.exists(stamp):
            with open(stamp) as f:
                if f.read().strip() == f'{run:%Y%m%d%H}':
                    done.append(valid)
                    continue
        want = f'{m} min fcst'
        sel = {}
        for canon, (var, lev) in SUBH_VARS.items():
            r = next((r for r in rows if r['var'] == var and r['level'] == lev and (r['fcst'] == want)), None)
            if r:
                sel[canon] = r
        if 't2' not in sel:
            continue
        try:
            raw = grib.fetch_messages(url, sel, workers=8, gap=1 << 20, tries=4)
            if holder.get('r') is None:
                lat, lon = grib.latlons(next(iter(raw.values())))
                holder['r'] = Regridder(lat, lon, region)
            if 'rot' not in holder:
                holder['rot'] = grib.earth_rotation(raw.get('u10') or next(iter(raw.values())))
            rg = holder['r']
            s = {k: (rg.peak if k == 'refd1' else rg)(grib.decode(v)[0]) for k, v in raw.items()}
            s['u10'], s['v10'] = grib.earth_winds(s.get('u10'), s.get('v10'), rg.lons[None, :], holder['rot'])
            fields = quarter_fields(s, region[4])
        except Exception as e:
            log(f'  q{fh:03d} +{m % 60:02d} skipped: {e}')
            continue
        q = {k: quantize(v, *ENC[k]) for k, v in fields.items() if k in ENC}
        part = os.path.join(qdir, f'{valid:%Y%m%d%H%M}.tmp.npz')
        tiles.savez(part, q)
        os.replace(part, npz)
        with open(stamp, 'w') as f:
            f.write(f'{run:%Y%m%d%H}')
        done.append(valid)
    log(f'  q{fh:03d} quarter hours {', '.join((f'{t:%H%M}' for t in done)) or 'none'} in {time.time() - t0:.1f}s')
    return done

def _quarter_job(job):
    rdir, qdir, day, stamps, name, tile = job
    frames = [np.load(os.path.join(qdir, f'{s}.npz')) for s in stamps]
    shape = frames[0]['t2'].shape
    stack = np.stack([f[name] if name in f.files else np.full(shape, 255, np.uint8) for f in frames])
    for f in frames:
        f.close()
    return (f'q_{name}', _write_strips(stack, tile, os.path.join(rdir, day, f'q_{name}')))

def write_quarters(rdir, qdir, times, tile, only=None, workers=1):
    days: dict[str, list] = {}
    for t in sorted(times):
        days.setdefault(day_key(t), []).append(t)
    jobs = []
    for day, dts in days.items():
        stamps = [f'{t:%Y%m%d%H%M}' for t in dts]
        files = set()
        for st in stamps:
            with np.load(os.path.join(qdir, f'{st}.npz')) as f:
                files |= set(f.files)
        jobs += [(rdir, qdir, day, stamps, n, tile) for n in QUARTER_FIELDS if (only is None or n in only) and n in files]
    present: dict[str, set] = {}
    for name, keys in run_jobs(_quarter_job, jobs, workers):
        if keys:
            present.setdefault(name, set()).update(keys)
    return ({d: [t.strftime('%Y%m%d%H%M') for t in ts] for d, ts in days.items()}, present)

def quarter_manifest(qdays, qpresent):
    return {k: {'min': ENC[k[2:]][0], 'max': ENC[k[2:]][1], 'transform': ENC[k[2:]][2], 'tiles': sorted(v)} for k, v in sorted(qpresent.items())}

def stored_quarters(qdir, lo, hi, prune=False):
    out = []
    if not os.path.isdir(qdir):
        return out
    for f in sorted(os.listdir(qdir)):
        if not f.endswith('.npz') or f.endswith('.tmp.npz'):
            continue
        t = datetime.strptime(f[:-4], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
        if lo <= t <= hi:
            out.append(t)
        elif prune:
            for ext in ('.npz', '.run'):
                if os.path.exists(os.path.join(qdir, f[:-4] + ext)):
                    os.remove(os.path.join(qdir, f[:-4] + ext))
    return out

def day_key(valid):
    return f'{valid - timedelta(hours=10):%Y%m%d}'

def _sqrt_from_zero(name):
    e = ENC.get(name) or (ENC.get(name[2:]) if name.startswith('q_') else None)
    return bool(e) and e[2] == 'sqrt' and (e[0] == 0)

def level_grid(grid, k, tile=LEVEL_TILE):
    step = grid['step']
    nx, ny = (-(-grid['nx'] // k), -(-grid['ny'] // k))
    return {'west': round(grid['west'] + (k - 1) / 2 * step, 9), 'north': round(grid['north'] - (k - 1) / 2 * step, 9), 'step': round(k * step, 9), 'nx': nx, 'ny': ny, 'tile': tile, 'ntx': -(-nx // tile), 'nty': -(-ny // tile)}

def overview_manifest(grid, levels, lpresent):
    return [{'k': k, 'path': f'o{k}', 'grid': level_grid(grid, k, tile), 'tiles': {n: sorted(v) for n, v in sorted(lpresent.get(k, {}).items())}} for k, tile in levels]
COMPRESS_LEVEL = 4
PNG_TOO_UNTIL = datetime(2026, 10, 11, 19, tzinfo=timezone.utc)

def png_too():
    return datetime.now(timezone.utc) < PNG_TOO_UNTIL

def _write_strips(stack, tile, folder):
    stack = stack if stack.ndim == 3 else stack[None]
    n, ny, nx = stack.shape
    ntx, nty = (-(-nx // tile), -(-ny // tile))
    keys, blobs = (set(), [b''] * (ntx * nty))
    for ty in range(0, ny, tile):
        for tx in range(0, nx, tile):
            block = stack[:, ty:ty + tile, tx:tx + tile]
            if (block == 255).all():
                continue
            pad = np.full((n, tile, tile), 255, np.uint8)
            pad[:, :block.shape[1], :block.shape[2]] = block
            blobs[ty // tile * ntx + tx // tile] = tiles.png_encode(pad.reshape(-1, tile), COMPRESS_LEVEL)
            keys.add(f'{tx // tile}_{ty // tile}')
    if keys:
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        pack.write_pack(folder + '.pack', blobs, ntx, nty)
        if png_too():
            os.makedirs(folder, exist_ok=True)
            for k in keys:
                tx, ty = (int(v) for v in k.split('_'))
                with open(os.path.join(folder, f'{k}.png'), 'wb') as f:
                    f.write(blobs[ty * ntx + tx])
    return keys

def run_jobs(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(min(workers, len(jobs))) as pool:
        return list(pool.imap_unordered(fn, jobs))

def _link_strips(src, dst):
    if os.path.exists(src + '.pack'):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.link(src + '.pack', dst + '.pack')
        names = sorted((f for f in os.listdir(src) if f.endswith('.png'))) if png_too() and os.path.isdir(src) else []
        if names:
            os.makedirs(dst, exist_ok=True)
        for f in names:
            os.link(os.path.join(src, f), os.path.join(dst, f))
        return pack.keys_in(dst + '.pack')
    return None if os.path.isdir(src) else set()

def _tile_job(job):
    rdir, hours_dir, day, stamps, name, tile, levels, reuse = job
    if reuse is not None:
        keys = _link_strips(os.path.join(reuse, day, name), os.path.join(rdir, day, name))
        lkeys = {}
        for k, _ in levels if name not in LEVEL_SKIP else ():
            lk = _link_strips(os.path.join(reuse, f'o{k}', day, name), os.path.join(rdir, f'o{k}', day, name))
            if lk is None:
                keys = None
            elif lk:
                lkeys[k] = lk
        if keys is not None:
            return (name, keys, lkeys)
        for p in [os.path.join(rdir, day, name)] + [os.path.join(rdir, f'o{k}', day, name) for k, _ in levels]:
            if os.path.exists(p + '.pack'):
                os.remove(p + '.pack')
            shutil.rmtree(p, ignore_errors=True)
    hours = [np.load(os.path.join(hours_dir, f'{s}.npz')) for s in stamps]
    if day is None:
        grid = hours[0]['elevation']
    else:
        shape = hours[0]['wstar'].shape
        grid = np.stack([h[name] if name in h.files else np.full(shape, 255, np.uint8) for h in hours])
    for h in hours:
        h.close()
    path = () if day is None else (day,)
    keys = _write_strips(grid, tile, os.path.join(rdir, *path, name))
    lkeys = {}
    for k, ltile in levels if name not in LEVEL_SKIP else ():
        lk = _write_strips(downsample(grid, k, nearest=name in CLASS_FIELDS or name in SAMPLED_FIELDS, none=LEVEL_NONE.get(name), rms=_sqrt_from_zero(name)), ltile, os.path.join(rdir, f'o{k}', *path, name))
        if lk:
            lkeys[k] = lk
    return (name, keys, lkeys)

def write_tiles(rdir, hours_dir, times, tile, elev_time, only=None, levels=(), workers=1, reuse=None, day_of=None, elevation=True):
    days: dict[str, list] = {}
    for t in times:
        days.setdefault((day_of or day_key)(t), []).append(t)
    present: dict[str, set] = {}
    lpresent: dict[int, dict[str, set]] = {k: {} for k, _ in levels}
    jobs = []
    for day, dts in days.items():
        stamps = [f'{t:%Y%m%d%H}' for t in dts]
        files = set()
        for st in stamps:
            with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
                files |= set(h.files)
        names = sorted(files - {'elevation'})
        if only is not None:
            names = [n for n in names if n in only]
        same = reuse is not None and reuse[1].get(day) == stamps and os.path.isdir(os.path.join(reuse[0], day))
        jobs += [(rdir, hours_dir, day, stamps, n, tile, levels, reuse[0] if same else None) for n in names]
    if days and elevation:
        jobs.append((rdir, hours_dir, None, [f'{elev_time:%Y%m%d%H}'], 'elevation', tile, levels, None))
    for name, keys, lkeys in run_jobs(_tile_job, jobs, workers):
        if keys:
            present.setdefault(name, set()).update(keys)
        for k, lk in lkeys.items():
            lpresent[k].setdefault(name, set()).update(lk)
    return ({d: [t.strftime('%Y%m%d%H') for t in ts] for d, ts in days.items()}, present, lpresent)
POINT_FIELDS = ('t2', 'td2', 'rain', 'ptype', 'slr', 'u_sfc', 'v_sfc', 'gust', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'ctop', 'vis', 'mslp', 'cape', 'frzl', 'wstar', 'depth', 'zi', 'cubase', 'cudepth', 'overdev', 'bumpy', 'densalt', 'refc', *(f'cc{p}' for p in P.CC_LEVELS), 'elevation')
POINT_BLOCK = 16

def _pack_job(job):
    path, start, keys, folder = job
    mm = np.load(path, mmap_mode='r')
    for i, key in zip(range(start, start + len(keys)), keys):
        tiles.write_png(os.path.join(folder, f'{key}.png'), np.ascontiguousarray(mm[i]), COMPRESS_LEVEL)
    return len(keys)

def write_points(rdir, hours_dir, times, elev_time, only=None, workers=1, block=POINT_BLOCK):
    if not times:
        return None
    stamps = [f'{t:%Y%m%d%H}' for t in sorted(times)]
    files = set()
    for st in stamps:
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            files |= set(h.files)
    names = [n for n in POINT_FIELDS if n in files and (only is None or n in only or n == 'elevation')]
    with np.load(os.path.join(hours_dir, f'{elev_time:%Y%m%d%H}.npz')) as h:
        elev = h['elevation'] if 'elevation' in h.files else None
        ny, nx = h['wstar'].shape
    if elev is None and 'elevation' in names:
        names.remove('elevation')
    B, W, nf = (block, block + 1, len(names))
    nby, nbx = (max(1, -(-(ny - 1) // B)), max(1, -(-(nx - 1) // B)))
    NY, NX = (nby * B + 1, nbx * B + 1)

    def stack(h):
        out = np.full((nf, NY, NX), 255, np.uint8)
        for i, n in enumerate(names):
            if n == 'elevation':
                out[i, :ny, :nx] = elev
            elif n in h.files:
                out[i, :ny, :nx] = h[n]
        return out

    def blocks(a):
        return np.lib.stride_tricks.sliding_window_view(a, (W, W), axis=(1, 2))[:, ::B, ::B]
    data = np.zeros((nby, nbx), bool)
    fc = [i for i, n in enumerate(names) if n != 'elevation']
    for st in {stamps[0], stamps[-1]}:
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            data |= (blocks(stack(h))[fc] != 255).any(axis=(0, 3, 4))
    by, bx = np.nonzero(data)
    if not len(by):
        return None
    folder = os.path.join(rdir, 'pt')
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(rdir, 'pt.tmp.npy')
    mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(by), len(stamps) * W, nf * W))
    for k, st in enumerate(stamps):
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            sel = blocks(stack(h))[:, by, bx]
        mm[:, k * W:(k + 1) * W, :] = sel.transpose(1, 2, 0, 3).reshape(len(by), W, nf * W)
    mm.flush()
    del mm
    keys = [f'{x}_{y}' for y, x in zip(by, bx)]
    step = -(-len(keys) // max(1, workers * 4))
    run_jobs(_pack_job, [(tmp, i, keys[i:i + step], folder) for i in range(0, len(keys), step)], workers)
    os.remove(tmp)
    pack.pack_folder(folder, keys, keep=png_too())
    return {'block': B, 'fields': names, 'stamps': stamps, 'tiles': keys}
FREE_MODELS = ('hrrr', 'gfs')
FREE_AHEAD = timedelta(hours=24)
FREE_SLACK = timedelta(hours=7)

def hour_token(model, run_id, stamp, secret=None):
    import hashlib
    import hmac
    secret = os.environ.get('DATA_PATH_SECRET', '') if secret is None else secret
    if not secret:
        return stamp
    tok = hmac.new(secret.encode(), f'free-hour|{model}|{run_id}|{stamp}'.encode(), hashlib.sha256).hexdigest()[:32]
    if os.environ.get('GITHUB_ACTIONS') == 'true' and tok not in _MASKED:
        print(f'::add-mask::{tok}', flush=True)
        _MASKED.add(tok)
    return tok
_MASKED: set = set()
FREE_FIELDS = {'wstar', 'depth', 'zi', 'cubase', 'cudepth', 'overdev', 'rain', 'ptype', 'cape', 'bs', 'densalt', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', *(f'cc{p}' for p in P.CC_LEVELS), 'u_sfc', 'v_sfc', 'u_blavg', 'v_blavg', 'bumpy', 'ceil', 'vis', 'fltcat', 't2', 'td2', 'sst', 'mslp', 'gust', 'pm25', *(f'{c}_a{ft}' for ft in WIND_ALTS_FT for c in ('u', 'v'))}
NOW_AHEAD = timedelta(hours=8)
NOW_FIELDS = FREE_FIELDS | {'twidth', 'sinkth', 'heat', 'ctop', 'wtrap', 'wlam', 'thml', 'thx', *(f'{k}_m{dz}' for dz in ML_TOP_DZ for k in ('dth', 'q')), *(f'wv{ft}' for ft in (8000, 10000, 12000, 14000, 18000, 24000)), *(f'dpd{ft}' for ft in (8000, 10000, 12000, 14000, 18000, 24000)), *(f't{ft}' for ft in (8000, 10000, 12000, 14000, 18000, 24000)), *(f'{c}_g{m}' for m in (100, 300) for c in ('u', 'v')), *(f'{c}_a{ft}' for ft in (2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000) for c in ('u', 'v'))}

def write_free(out, model, now=None, workers=1):
    if model not in FREE_MODELS:
        return None
    base = os.path.join(out, model)
    try:
        with open(os.path.join(base, 'latest.json')) as f:
            full = json.load(f)
    except FileNotFoundError:
        log(f'{model}: no full product yet, no free window')
        return None
    hours_dir = os.path.join(base, '_hours')
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    cutoff = now + FREE_AHEAD + FREE_SLACK
    stamps = sorted((s for d in full['days'].values() for s in d))
    times = [datetime.strptime(s, '%Y%m%d%H').replace(tzinfo=timezone.utc) for s in stamps]
    times = [t for t in times if t <= cutoff and os.path.exists(os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz'))]
    hourly = [t for t in times if t > now + FREE_AHEAD]
    manifest = _publish_open(out, os.path.join(out, 'free', model), model, full, times, FREE_FIELDS, cutoff, workers, points=True, hourly=hourly)
    if manifest:
        log(f'{model}: free window to {cutoff:%Y-%m-%d %HZ}: {len(times)} hours ({len(hourly)} by the hour), {len(manifest['fields'])} fields')
    now_times = [t for t in times if now - timedelta(hours=1) <= t <= now + NOW_AHEAD]
    nm = _publish_open(out, os.path.join(out, 'free', 'now', model), model, full, now_times, NOW_FIELDS, now + NOW_AHEAD, workers)
    if nm:
        log(f'{model}: 3D now window: {len(now_times)} hours, {len(nm['fields'])} fields')
    return manifest

def _publish_open(out, fbase, model, full, times, fields, cutoff, workers, points=False, hourly=()):
    base = os.path.join(out, model)
    hours_dir = os.path.join(base, '_hours')
    os.makedirs(fbase, exist_ok=True)
    if not times:
        log(f'{model}: no hours for {os.path.relpath(fbase, out)}')
        return None
    free_id = f'{full['runId']}-{cutoff:%Y%m%d%H}'
    rdir = os.path.join(fbase, free_id)
    shutil.rmtree(rdir, ignore_errors=True)
    tile = full['grid']['tile']
    levels = [(lv['k'], lv['grid']['tile']) for lv in full.get('levels', [])]
    reuse = (os.path.join(base, full['runId']), full.get('days', {}))
    hourly = set(hourly)
    daily = [t for t in times if t not in hourly]
    days, present, lpresent = write_tiles(rdir, hours_dir, daily, tile, times[-1], only=fields, levels=levels, workers=workers, reuse=reuse)
    pts = write_points(rdir, hours_dir, daily, times[-1], only=fields, workers=workers) if points else None
    hours, hpts = ([], {})
    if hourly:
        folder = lambda t: f'h/{hour_token(model, free_id, f'{t:%Y%m%d%H}')}'
        _, hp, hlp = write_tiles(rdir, hours_dir, sorted(hourly), tile, times[-1], only=fields, levels=levels, workers=workers, day_of=folder, elevation=False)
        for name, keys in hp.items():
            present.setdefault(name, set()).update(keys)
        for k, by_name in hlp.items():
            for name, keys in by_name.items():
                lpresent[k].setdefault(name, set()).update(keys)
        for t in sorted(hourly):
            st = f'{t:%Y%m%d%H}'
            hours.append(st)
            if points:
                p = write_points(os.path.join(rdir, folder(t)), hours_dir, [t], times[-1], only=fields, workers=workers)
                if p:
                    hpts[st] = {k: p[k] for k in ('fields', 'tiles')}
    last_open = max(daily) if daily else times[0] - timedelta(hours=1)
    qtimes = [q for q in stored_quarters(os.path.join(base, '_quarters'), times[0], cutoff) if q <= last_open] if model == 'hrrr' else []
    qdays, qpresent = write_quarters(rdir, os.path.join(base, '_quarters'), qtimes, tile, only=fields, workers=workers) if qtimes else ({}, {})
    manifest = {**full, 'runId': free_id, 'tier': 'free', 'freeUntil': cutoff.isoformat(), 'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'days': days, 'fields': {**{k: {**full['fields'][k], 'tiles': sorted(v)} for k, v in sorted(present.items()) if k in full['fields']}, **quarter_manifest(qdays, qpresent)}, 'quarters': qdays, 'levels': [{**lv, 'tiles': {k: sorted(v) for k, v in sorted(lpresent[lv['k']].items()) if k in full['fields']}} for lv in full.get('levels', [])], 'waveAltsFt': full.get('waveAltsFt', []) if any((k.startswith('wv') for k in present)) else [], 'points': pts, 'hours': hours, 'hourPoints': hpts, 'pack': 1}
    with open(os.path.join(fbase, 'latest.json'), 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))
    for old in os.listdir(fbase):
        if old != free_id and old != 'latest.json':
            shutil.rmtree(os.path.join(fbase, old), ignore_errors=True)
    open(os.path.join(out, '.changed'), 'w').close()
    return manifest
STORE_AHEAD_H = {'gfs': 60, 'icon': 60}
STORE_AFTER_FREE_H = {'hrrr': 12}

def prune_store(hours_dir, ahead_h, now=None):
    cut = (now or datetime.now(timezone.utc)) + timedelta(hours=ahead_h)
    n = 0
    for f in os.listdir(hours_dir):
        if not f.endswith('.npz') or f.endswith('.tmp.npz'):
            continue
        try:
            t = datetime.strptime(f[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if t > cut:
            for ext in ('.npz', '.run'):
                p = os.path.join(hours_dir, f[:10] + ext)
                if os.path.exists(p):
                    os.remove(p)
            n += 1
    return n

def store_fits(hours_dir, region, src_id=None):
    lons, lats = target_grid(region)
    keep = keep_mask(src_id, *np.meshgrid(lats, lons, indexing='ij')) if src_id else None
    for f in sorted(os.listdir(hours_dir)) if os.path.isdir(hours_dir) else []:
        if f.endswith('.npz') and (not f.endswith('.tmp.npz')):
            try:
                with np.load(os.path.join(hours_dir, f)) as h:
                    a = h[h.files[0]]
                    if a.shape != (len(lats), len(lons)):
                        return False
                    return keep is None or bool(np.all(a[~keep] == 255))
            except Exception:
                return False
    return True

def backfill_jobs(src, run, needed, hours_dir):
    have = {f[:-4] for f in os.listdir(hours_dir) if f.endswith('.npz') and (not f.endswith('.tmp.npz'))}
    jobs = []
    for t in needed:
        if (t > run or (t == run and src.fstep(0))) or f'{t:%Y%m%d%H}' in have:
            continue
        for r in candidate_runs(src, t, back_hours=24):
            fh = int((t - r).total_seconds() // 3600)
            if r < run and 0 <= fh <= src.max_fhour and src.fstep(fh):
                jobs.append((r, fh))
                break
    return jobs

def pick_run(src, needed):
    run, fhs = latest_complete_run(src, needed, log)
    if run is None:
        raise SystemExit(f'no complete {src.id} run found')
    return (run, fhs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=SOURCES)
    ap.add_argument('--out', default='dist-data')
    ap.add_argument('--days', type=int)
    ap.add_argument('--bbox', help='west,south,east,north override (testing)')
    ap.add_argument('--step', type=float)
    ap.add_argument('--max-hours', type=int, help='limit valid times (testing)')
    ap.add_argument('--keep-runs', type=int, default=1)
    ap.add_argument('--tile', type=int)
    ap.add_argument('--force', action='store_true', help='publish again even if this run is already published')
    ap.add_argument('--recompute', action='store_true', help='also recompute hours already computed from this run')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '3')))
    ap.add_argument('--free-only', action='store_true', help='only refresh the free 24 h window from stored hours')
    ap.add_argument('--no-quarters', action='store_true', help="skip HRRR's 15-minute output")
    ap.add_argument('--python-physics', action='store_true', help='physics.py only (never the C++ physics)')
    args = ap.parse_args()
    if args.free_only:
        write_free(args.out, args.model, workers=args.workers)
        hours_dir = os.path.join(args.out, args.model, '_hours')
        if args.model in STORE_AFTER_FREE_H and os.path.isdir(hours_dir):
            ahead = STORE_AFTER_FREE_H[args.model]
            log(f'{args.model}: {prune_store(hours_dir, ahead)} stored hours past +{ahead} h dropped (free window written)')
        return 0
    src = SOURCES[args.model]
    region = list(REGIONS[src.id])
    if args.bbox:
        region[:4] = [float(x) for x in args.bbox.split(',')]
    if args.step:
        region[4] = args.step
    needed = needed_times(src, args.days)
    run, fhs = pick_run(src, needed)
    if args.max_hours:
        mid = [f for f in fhs if (run + timedelta(hours=f)).hour in (18, 19, 20)] or fhs
        fhs = sorted(set(mid[:args.max_hours]))
    log(f'{src.id}: run {run:%Y-%m-%d %HZ}, {len(fhs)} valid times')
    run_id = f'{run:%Y%m%d%H}'
    base = os.path.join(args.out, src.id)
    latest = os.path.join(base, 'latest.json')
    if os.path.exists(latest) and (not args.force):
        with open(latest) as f:
            cur = json.load(f)
        if cur.get('runId') == run_id and cur.get('complete'):
            log(f'{src.id}: {run_id} already built, nothing to do')
            return 0
    rdir = os.path.join(base, run_id)
    build_dir = os.path.join(base, f'{run_id}.building')
    hours_dir = os.path.join(base, '_hours')
    for d in os.listdir(base) if os.path.isdir(base) else []:
        if d.endswith('.building'):
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    vfile = os.path.join(hours_dir, 'VERSION')
    kfile = os.path.join(hours_dir, 'KEEP')
    keep_now = repr(KEEP.get(src.id))
    if not os.path.exists(vfile) or open(vfile).read().strip() != str(HOURS_VERSION) or (not store_fits(hours_dir, region, src.id)) or (src.id in KEEP and (not os.path.exists(kfile) or open(kfile).read() != keep_now)):
        shutil.rmtree(hours_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(base, '_quarters'), ignore_errors=True)
    os.makedirs(hours_dir, exist_ok=True)
    with open(vfile, 'w') as f:
        f.write(str(HOURS_VERSION))
    if src.id in KEEP:
        with open(kfile, 'w') as f:
            f.write(keep_now)
    _CTX.update({'src': src, 'run': run, 'region': region, 'hours': hours_dir, 'holder': {}, 'recompute': args.recompute, 'pick_physics': not args.python_physics})
    results = [_process_hour(fhs[0])]
    rest = fhs[1:]
    back = backfill_jobs(src, run, needed, hours_dir) if not args.max_hours else []
    if back:
        log(f'{src.id}: backfilling {len(back)} earlier hours from runs ' + ', '.join(sorted({f'{r:%d %HZ}' for r, _ in back})))
    rest += back
    if rest:
        workers = max(1, min(args.workers, len(rest)))
        progress = Progress(src.id, len(rest))
        if workers == 1:
            for fh in rest:
                results.append(_process_hour(fh))
                progress.tick()
        else:
            results += process_lanes(rest, workers, progress)
    done = [t for t in results[:len(fhs)] if t is not None]
    if not done:
        raise SystemExit('no hours processed')
    qdir = os.path.join(base, '_quarters')
    if src.id == 'hrrr' and (not args.no_quarters):
        os.makedirs(qdir, exist_ok=True)
        _CTX['quarters'] = qdir
        lo, hi = (min(needed), max(needed))
        qjobs = [fh for fh in fhs if 1 <= fh <= QUARTER_MAX_FH and lo <= run + timedelta(hours=fh) - timedelta(minutes=45) <= hi]
        if qjobs:
            t0 = time.time()
            workers = max(1, min(args.workers, len(qjobs)))
            if workers == 1:
                qdone = [t for fh in qjobs for t in _process_quarters(fh)]
            else:
                import multiprocessing as mp
                with mp.get_context('fork').Pool(workers) as pool:
                    qdone = [t for part in pool.imap_unordered(_process_quarters, qjobs) for t in part]
            log(f'{src.id}: {len(qdone)} quarter hours from the 15-minute output in {_fmt_dur(time.time() - t0)}')
    window = {t.strftime('%Y%m%d%H') for t in needed}
    times = []
    for f in sorted(os.listdir(hours_dir)):
        stamp = f[:-4]
        if not f.endswith('.npz') or f.endswith('.tmp.npz'):
            continue
        if stamp in window:
            times.append(datetime.strptime(stamp, '%Y%m%d%H').replace(tzinfo=timezone.utc))
        else:
            os.remove(os.path.join(hours_dir, f))
            if os.path.exists(os.path.join(hours_dir, f'{stamp}.run')):
                os.remove(os.path.join(hours_dir, f'{stamp}.run'))
    lons, lats = target_grid(region)
    tile = args.tile or TILE[src.id]
    levels = [(k, LEVEL_TILE) for k in OVERVIEW.get(src.id, ())]
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    days, present, lpresent = write_tiles(build_dir, hours_dir, times, tile, done[0], levels=levels, workers=args.workers)
    qtimes = stored_quarters(qdir, min(needed), max(needed), prune=True) if src.id == 'hrrr' else []
    qdays, qpresent = write_quarters(build_dir, qdir, qtimes, tile, workers=args.workers) if qtimes else ({}, {})
    log(f'{src.id}: tiles for {len(times)} hours in {_fmt_dur(time.time() - t0)}')
    t0 = time.time()
    points = write_points(build_dir, hours_dir, times, done[0], workers=args.workers)
    log(f'{src.id}: point packs ({(len(points['tiles']) if points else 0)}) in {_fmt_dur(time.time() - t0)}')
    ntx = -(-len(lons) // tile)
    nty = -(-len(lats) // tile)
    grid = {'west': float(lons[0]), 'north': float(lats[0]), 'step': region[4], 'nx': len(lons), 'ny': len(lats), 'tile': tile, 'ntx': ntx, 'nty': nty}
    manifest = {'model': src.id, 'label': src.label, 'run': run.isoformat(), 'runId': run_id, 'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'complete': not args.max_hours and len(done) == len(fhs), 'grid': grid, 'levels': overview_manifest(grid, levels, lpresent), 'days': days, 'quarters': qdays, 'fields': {**{k: {'min': ENC[k][0], 'max': ENC[k][1], 'transform': ENC[k][2], 'tiles': sorted(v)} for k, v in sorted(present.items())}, **quarter_manifest(qdays, qpresent)}, 'heights': 'agl', 'sunGate': True, 'windAltsFt': WIND_ALTS_FT, 'waveAltsFt': WAVE_ALTS_FT if 'wv12000' in present else [], 'notes': src.notes, 'points': points, 'pack': 1}
    shutil.rmtree(rdir, ignore_errors=True)
    os.replace(build_dir, rdir)
    with open(latest + '.tmp', 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))
    os.replace(latest + '.tmp', latest)
    open(os.path.join(args.out, '.changed'), 'w').close()
    runs = sorted((d for d in os.listdir(base) if d.isdigit()))
    for old in runs[:-args.keep_runs]:
        shutil.rmtree(os.path.join(base, old), ignore_errors=True)
    if src.id in STORE_AHEAD_H and (not args.max_hours):
        log(f'{src.id}: {prune_store(hours_dir, STORE_AHEAD_H[src.id])} stored hours past +{STORE_AHEAD_H[src.id]} h dropped (published already)')
    log(f'{src.id}: run {run_id}: computed {len(done)} hours, published {len(times)} in {len(days)} days')
if __name__ == '__main__':
    sys.exit(main())
