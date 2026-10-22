from __future__ import annotations
import math
import numpy as np
from scipy import ndimage
R_EARTH = 6371000.0
STEP = 0.25
SMOOTH_KM = 90.0
WIND_SMOOTH_KM = 150.0
GRAD_MIN = 0.4
TFP_MIN = 0.3
MIN_LEN_KM = 400.0
MIN_PART_KM = 500.0
MOVE_MS = 1.5
ALONG_KM = 300.0
JOIN_KM = 200.0
JOIN_TURN = 50.0
CENTRE_KM = 800.0
CENTRE_HPA = 2.0
CENTRE_SMOOTH_KM = 100.0
LEVEL = 850
COORD = 0.1
SIMPLIFY_KM = 12.0
UNDERGROUND_HPA = 10.0
EDGE_KM = 150.0
TGRAD_MIN = 0.3
MIN_SPAN_DEG = 12.0
SUB_FROM, SUB_TO = (36.0, 30.0)
SUB_LEN = 2.0
SUB_RATIO = 0.6
LINE_GRAD = 0.6
WIND_SFC = 0.5
OCC_LOW_KM = 400.0
OCC_DEPTH = 8.0
OCC_PART_KM = 150.0

def theta_e(t_c, td_c, p_hpa):
    t = np.asarray(t_c, np.float64) + 273.15
    td = np.minimum(np.asarray(td_c, np.float64), np.asarray(t_c, np.float64)) + 273.15
    e = 6.112 * np.exp(17.67 * (td - 273.15) / (td - 29.65))
    r = 0.622 * e / np.maximum(p_hpa - e, 1.0)
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0
    th = t * (1000.0 / p_hpa) ** (0.2854 * (1 - 0.28 * r))
    return th * np.exp((3.376 / tl - 0.00254) * r * 1000 * (1 + 0.81 * r))

def theta_w(t_c, td_c, p_hpa):
    te = theta_e(t_c, td_c, p_hpa)
    x = te / 273.15
    a = (7.101574, -20.68208, 16.11182, 2.574631, -5.205688)
    b = (-3.552497, 3.781782, -0.6899655, -0.592934)
    num = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3 + a[4] * x ** 4
    den = 1 + b[0] * x + b[1] * x ** 2 + b[2] * x ** 3 + b[3] * x ** 4
    return np.where(te >= 173.15, te - np.exp(num / den), te)

def block_mean(a, k):
    if k <= 1:
        return np.asarray(a, np.float64)
    ny, nx = (a.shape[0] // k * k, a.shape[1] // k * k)
    b = np.asarray(a[:ny, :nx], np.float64).reshape(ny // k, k, nx // k, k)
    ok = np.isfinite(b)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0).sum(axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

def smooth(a, sy, sx, fill=True):
    ok = np.isfinite(a)
    w = ndimage.gaussian_filter(ok.astype(np.float64), (sy, sx), mode='nearest')
    v = ndimage.gaussian_filter(np.where(ok, a, 0.0), (sy, sx), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(w > 0.05, v / w, np.nan)
    return out if fill else np.where(ok, out, np.nan)

class Grid:

    def __init__(self, lats, lons):
        self.lats = np.asarray(lats, np.float64)
        self.lons = np.asarray(lons, np.float64)
        self.dlat = abs(self.lats[1] - self.lats[0])
        self.dlon = abs(self.lons[1] - self.lons[0])
        self.dy = R_EARTH * math.radians(self.dlat)
        self.dx = (R_EARTH * math.radians(self.dlon) * np.cos(np.radians(self.lats)))[:, None]

    def grad(self, a):
        gy, gx = np.gradient(a)
        return (gx / self.dx, -gy / self.dy)

    def sigma(self, km):
        sy = km * 1000 / self.dy
        c = max(0.2, math.cos(math.radians(float(np.mean(self.lats)))))
        return (sy, km * 1000 / (self.dy * self.dlon / self.dlat * c))

    def lonlat(self, fy, fx):
        return (self.lons[0] + fx * (self.lons[1] - self.lons[0]), self.lats[0] + fy * (self.lats[1] - self.lats[0]))

    def cell(self, lon, lat):
        return ((lat - self.lats[0]) / (self.lats[1] - self.lats[0]), (lon - self.lons[0]) / (self.lons[1] - self.lons[0]))

def sample(a, fy, fx):
    fy, fx = (np.asarray(fy, np.float64), np.asarray(fx, np.float64))
    ny, nx = a.shape
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    ty, tx = (fy - y0, fx - x0)
    v = a[y0, x0] * (1 - ty) * (1 - tx) + a[y0, x0 + 1] * (1 - ty) * tx + a[y0 + 1, x0] * ty * (1 - tx) + a[y0 + 1, x0 + 1] * ty * tx
    inside = (fy >= 0) & (fy <= ny - 1) & (fx >= 0) & (fx <= nx - 1)
    return np.where(inside, v, np.nan)

def km_between(lon0, lat0, lon1, lat1):
    p0, p1 = (np.radians(lat0), np.radians(lat1))
    dl = np.radians(np.asarray(lon1) - np.asarray(lon0))
    h = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH / 1000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def zero_lines(f, keep):
    ny, nx = f.shape
    fin = np.isfinite(f)
    s = np.where(fin, f > 0, False)

    def cross(a, b, ok):
        with np.errstate(invalid='ignore', divide='ignore'):
            t = a / (a - b)
        return (ok & ((a > 0) != (b > 0)), t)
    hc, ht = cross(f[:, :-1], f[:, 1:], fin[:, :-1] & fin[:, 1:])
    vc, vt = cross(f[:-1, :], f[1:, :], fin[:-1, :] & fin[1:, :])
    cell_ok = keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, :-1] & keep[1:, 1:]
    del s
    H = ny * nx
    pts = {}

    def point(eid):
        if eid not in pts:
            if eid < H:
                y, x = divmod(eid, nx)
                pts[eid] = (float(y), x + float(ht[y, x]))
            else:
                y, x = divmod(eid - H, nx)
                pts[eid] = (y + float(vt[y, x]), float(x))
        return pts[eid]
    adj = {}
    ys, xs = np.nonzero(cell_ok & (hc[:-1, :] | hc[1:, :] | vc[:, :-1] | vc[:, 1:]))
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges = []
        if hc[y, x]:
            edges.append(y * nx + x)
        if vc[y, x + 1]:
            edges.append(H + y * nx + x + 1)
        if hc[y + 1, x]:
            edges.append((y + 1) * nx + x)
        if vc[y, x]:
            edges.append(H + y * nx + x)
        if len(edges) == 2:
            pairs = [edges]
        elif len(edges) == 4:
            c = f[y, x] + f[y, x + 1] + f[y + 1, x] + f[y + 1, x + 1]
            top_pos = f[y, x] > 0
            pairs = [[edges[0], edges[3]], [edges[1], edges[2]]] if (c > 0) != top_pos else [[edges[0], edges[1]], [edges[2], edges[3]]]
        else:
            continue
        for a, b in pairs:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    lines, seen = ([], set())
    starts = [e for e, n in adj.items() if len(n) == 1] + list(adj)
    for e0 in starts:
        if e0 in seen:
            continue
        line, prev, cur = ([e0], None, e0)
        seen.add(e0)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = (cur, nxt[0])
            seen.add(cur)
            line.append(cur)
        if len(line) >= 2:
            lines.append([point(e) for e in line])
    return lines

def simplify(xy, tol):
    n = len(xy)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = (xy[i], xy[j])
        ab = b - a
        L = math.hypot(*ab)
        seg = xy[i + 1:j] - a
        d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return list(np.nonzero(keep)[0])

def _runs(types, seglen, min_km):
    types = list(types)
    for _ in range(len(types)):
        runs, i = ([], 0)
        while i < len(types):
            j = i
            while j + 1 < len(types) and types[j + 1] == types[i]:
                j += 1
            runs.append((i, j, float(sum(seglen[i:j + 1]))))
            i = j + 1
        short = [r for r in runs if r[2] < min_km]
        if not short or len(runs) == 1:
            break
        i, j, _ = min(short, key=lambda r: r[2])
        k = runs.index((i, j, _))
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        nb = max([r for r in (left, right) if r], key=lambda r: r[2])
        for m in range(i, j + 1):
            types[m] = types[nb[0]]
    return types

def join_lines(lines, km=None, max_turn=None):
    km = JOIN_KM if km is None else km
    max_turn = JOIN_TURN if max_turn is None else max_turn
    lines = [ln for ln in lines if len(ln) >= 2]

    def heading(ln, end):
        a, b = (ln[-min(4, len(ln))], ln[-1]) if end else (ln[min(3, len(ln) - 1)], ln[0])
        c = math.cos(math.radians(b[1]))
        v = np.array([(b[0] - a[0]) * c, b[1] - a[1]])
        n = math.hypot(*v)
        return v / n if n > 0 else v
    cos_max = math.cos(math.radians(max_turn))
    while len(lines) > 1:
        ends = np.array([p for ln in lines for p in (ln[0], ln[-1])])
        d = km_between(ends[:, None, 0], ends[:, None, 1], ends[None, :, 0], ends[None, :, 1])
        own = np.arange(len(ends)) // 2
        d[own[:, None] == own[None, :]] = np.inf
        ii, jj = np.nonzero(np.triu(d <= km))
        joined = False
        for k in np.argsort(d[ii, jj], kind='stable'):
            i, j = (int(ii[k]), int(jj[k]))
            a, ea, b, eb = (lines[i // 2], i % 2, lines[j // 2], j % 2)
            if float(np.dot(heading(a, ea), -heading(b, eb))) < cos_max:
                continue
            a = a if ea else a[::-1]
            b = b if not eb else b[::-1]
            lines = [ln for n, ln in enumerate(lines) if n not in (i // 2, j // 2)] + [np.vstack([a, b])]
            joined = True
            break
        if not joined:
            break
    return lines

def centres(grid, mslp):
    if mslp is None or not np.isfinite(mslp).any():
        return []
    p = smooth(mslp, *grid.sigma(CENTRE_SMOOTH_KM))
    sy, sx = grid.sigma(CENTRE_KM)
    size = (max(3, int(round(2 * sy)) | 1), max(3, int(round(2 * sx)) | 1))
    filled = np.where(np.isfinite(p), p, np.nanmean(p))
    mean = ndimage.uniform_filter(filled, size, mode='nearest')
    out = []
    ny, nx = p.shape
    for kind, ext, sign in (('L', ndimage.minimum_filter, -1), ('H', ndimage.maximum_filter, 1)):
        e = ext(np.where(np.isfinite(p), p, np.inf if sign < 0 else -np.inf), size=size, mode='nearest')
        ys, xs = np.nonzero((p == e) & np.isfinite(p) & (sign * (p - mean) >= CENTRE_HPA))
        for y, x in zip(ys.tolist(), xs.tolist()):
            if y < 2 or x < 2 or y > ny - 3 or (x > nx - 3):
                continue
            lon, lat = grid.lonlat(y, x)
            out.append((kind, float(lon), float(lat), float(p[y, x])))
    return out

def _ramp(lat):
    return float(np.clip((SUB_FROM - abs(float(lat))) / max(SUB_FROM - SUB_TO, 1e-06), 0, 1))

def _surround(g, pm):
    p = smooth(pm, *g.sigma(CENTRE_SMOOTH_KM))
    sy, sx = g.sigma(CENTRE_KM)
    size = (max(3, int(round(2 * sy)) | 1), max(3, int(round(2 * sx)) | 1))
    return ndimage.uniform_filter(np.where(np.isfinite(p), p, np.nanmean(p)), size, mode='nearest')

def _occluded(g, lon, lat, lows, pmean):
    ny, nx = pmean.shape
    out = np.zeros(len(lon), bool)
    for _, x, y, hpa in lows:
        fy, fx = g.cell(x, y)
        if float(pmean[int(np.clip(round(float(fy)), 0, ny - 1)), int(np.clip(round(float(fx)), 0, nx - 1))]) - hpa >= OCC_DEPTH:
            out |= km_between(lon, lat, x, y) <= OCC_LOW_KM
    return out

def detect(lats, lons, t_c, td_c, u, v, ps=None, mslp=None, p_hpa=LEVEL, u10=None, v10=None):
    lats, lons = (np.asarray(lats, np.float64), np.asarray(lons, np.float64))
    if abs(lats[-1] - lats[0]) < MIN_SPAN_DEG or abs(lons[-1] - lons[0]) < MIN_SPAN_DEG:
        return {'fronts': [], 'centres': []}
    step = abs(lats[1] - lats[0])
    k = max(1, int(round(STEP / step)))
    tw = theta_w(t_c, td_c, p_hpa)
    t_c = np.asarray(t_c, np.float64)
    under = np.zeros(tw.shape, bool)
    if ps is not None:
        under = ~(np.asarray(ps) > p_hpa + UNDERGROUND_HPA)
        tw, t_c = (np.where(under, np.nan, tw), np.where(under, np.nan, t_c))
        u = np.where(under, np.nan, u)
        v = np.where(under, np.nan, v)
    tw, t_c, u, v = (block_mean(np.asarray(a, np.float64), k) for a in (tw, t_c, u, v))
    if WIND_SFC > 0 and u10 is not None and (v10 is not None):
        u10, v10 = (block_mean(np.asarray(a, np.float64), k) for a in (u10, v10))
        u = np.where(np.isfinite(u), (1 - WIND_SFC) * u + WIND_SFC * u10, u10)
        v = np.where(np.isfinite(v), (1 - WIND_SFC) * v + WIND_SFC * v10, v10)
    g = Grid(lats[:tw.shape[0] * k:k] + (lats[1] - lats[0]) * (k - 1) / 2, lons[:tw.shape[1] * k:k] + (lons[1] - lons[0]) * (k - 1) / 2)
    under = block_mean(under.astype(np.float64), k) > 0.5
    ey, ex = g.sigma(EDGE_KM)
    near_under = ndimage.binary_dilation(under, np.ones((2 * int(round(ey)) + 1, 2 * int(round(ex)) + 1), bool)) if under.any() else under
    pm = block_mean(np.asarray(mslp, np.float64), k) if mslp is not None else None
    th = smooth(tw, *g.sigma(SMOOTH_KM))
    tt = smooth(t_c, *g.sigma(SMOOTH_KM))
    us, vs = (smooth(u, *g.sigma(WIND_SMOOTH_KM)), smooth(v, *g.sigma(WIND_SMOOTH_KM)))
    gx, gy = g.grad(th)
    G = np.hypot(gx, gy)
    with np.errstate(invalid='ignore', divide='ignore'):
        nx_, ny_ = (gx / G, gy / G)
    Gx, Gy = g.grad(G)
    tfp = -(Gx * nx_ + Gy * ny_)
    Tx, Ty = g.grad(tfp)
    loc = Tx * nx_ + Ty * ny_
    tgx, tgy = g.grad(tt)
    tn = tgx * nx_ + tgy * ny_
    hk = 100000.0
    keep = (tfp * hk * hk > TFP_MIN) & (G * hk > GRAD_MIN) & (tn * hk > TGRAD_MIN) & ~near_under
    cs = centres(g, pm) if pm is not None else []
    lows = [c for c in cs if c[0] == 'L']
    pmean = _surround(g, pm) if lows else None
    fronts = []
    lines = [np.array(g.lonlat(np.array([p[0] for p in ln]), np.array([p[1] for p in ln]))).T for ln in zero_lines(loc, keep)]
    for ll in join_lines(lines):
        lon, lat = (ll[:, 0], ll[:, 1])
        seg = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:])
        if seg.sum() < MIN_LEN_KM * (SUB_LEN if np.median(np.abs(lat)) < SUB_TO else 1.0):
            continue
        fy0, fx0 = g.cell(lon, lat)
        line_g = float(np.nanmean(sample(G, fy0, fx0))) * hk
        if line_g < LINE_GRAD:
            continue
        ramp = _ramp(np.median(lat))
        if ramp > 0 and float(np.nanmean(sample(tn, fy0, fx0))) * hk < SUB_RATIO * ramp * line_g:
            continue
        if len(lon) >= 5:
            lon = np.r_[lon[0], np.convolve(lon, np.ones(3) / 3, 'valid'), lon[-1]]
            lat = np.r_[lat[0], np.convolve(lat, np.ones(3) / 3, 'valid'), lat[-1]]
        fy, fx = g.cell(lon, lat)
        n_x, n_y = (sample(nx_, fy, fx), sample(ny_, fy, fx))
        across = sample(us, fy, fx) * n_x + sample(vs, fy, fx) * n_y
        seg = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:])
        dist = np.r_[0, np.cumsum(seg)]
        sm = np.array([np.nanmean(across[np.abs(dist - d) <= ALONG_KM / 2]) for d in dist])
        types = np.where(sm > MOVE_MS, 'c', np.where(sm < -MOVE_MS, 'w', 's'))
        seg_types = _runs(types[:-1], seg, MIN_PART_KM)
        if lows:
            occ = _occluded(g, lon, lat, lows, pmean)
            seg_types = _runs(np.where(occ[:-1] & occ[1:], 'o', np.array(seg_types)), seg, OCC_PART_KM)
        i = 0
        while i < len(seg_types):
            j = i
            while j + 1 < len(seg_types) and seg_types[j + 1] == seg_types[i]:
                j += 1
            plon, plat = (lon[i:j + 2], lat[i:j + 2])
            pnx, pny = (n_x[i:j + 2], n_y[i:j + 2])
            dxs = np.diff(plon) * np.cos(np.radians(plat[:-1]))
            dys = np.diff(plat)
            left = np.nansum(-dys * pnx[:-1] + dxs * pny[:-1])
            c = math.cos(math.radians(float(plat.mean())))
            keep_i = simplify(np.c_[plon * c * 111.2, plat * 111.2], SIMPLIFY_KM)
            plon, plat = (plon[keep_i], plat[keep_i])
            if left < 0:
                plon, plat = (plon[::-1], plat[::-1])
            fronts.append((str(seg_types[i]), list(zip(plon.tolist(), plat.tolist()))))
            i = j + 1
    return {'fronts': fronts, 'centres': cs}

def merge(results):
    out = {'fronts': [], 'centres': []}
    for r in results:
        if r:
            out['fronts'] += r['fronts']
            out['centres'] += r['centres']
    return out

def encode(res):
    f = []
    for kind, pts in res['fronts']:
        q = [(round(lon / COORD), round(lat / COORD)) for lon, lat in pts]
        dedup = [q[0]] + [b for a, b in zip(q, q[1:]) if b != a]
        if len(dedup) < 2:
            continue
        flat = [kind, dedup[0][0], dedup[0][1]]
        for a, b in zip(dedup, dedup[1:]):
            flat += [b[0] - a[0], b[1] - a[1]]
        f.append(flat)
    p = [[kind, round(lon / COORD), round(lat / COORD), round(hpa)] for kind, lon, lat, hpa in res['centres']]
    return {'f': f, 'p': p}

def decode(enc):
    fronts = []
    for line in enc.get('f', []):
        kind, x, y, rest = (line[0], line[1], line[2], line[3:])
        pts = [(x * COORD, y * COORD)]
        for i in range(0, len(rest) - 1, 2):
            x, y = (x + rest[i], y + rest[i + 1])
            pts.append((x * COORD, y * COORD))
        fronts.append((kind, pts))
    return {'fronts': fronts, 'centres': [(k, x * COORD, y * COORD, h) for k, x, y, h in enc.get('p', [])]}

def from_part(s, lv, levels, lats, lons):
    levels = list(levels)
    if LEVEL not in levels or lv.get('t') is None or lv.get('td') is None or (lv.get('u') is None):
        return None
    i = levels.index(LEVEL)
    return detect(lats, lons, lv['t'][i], lv['td'][i], lv['u'][i], lv['v'][i], ps=s.get('ps'), mslp=s.get('mslp'), u10=s.get('u10'), v10=s.get('v10'))
