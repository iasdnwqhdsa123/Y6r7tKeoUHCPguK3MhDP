import base64
import json
import math
import os
import sys
import tempfile
import time
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fronts
import physics
BOX = (-175.0, 15.0, -15.0, 72.0)
INNER = (-130.0, 20.25, -52.0, 56.0)
R_KM = 75.0
CLIP = 6.0
ALAT = (20.0, 72.0)
CLS = 'cwso'
GFS = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{c:%Y%m%d}/{c:%H}/atmos/gfs.t{c:%H}z.pgrb2.0p25.f{fh:03d}'
WANT = {('PRMSL', 'mean sea level'): 'mslp', ('PRES', 'surface'): 'ps', ('HGT', 'surface'): 'zs', ('TMP', '2 m above ground'): 't2', ('DPT', '2 m above ground'): 'td2', ('UGRD', '10 m above ground'): 'u10', ('VGRD', '10 m above ground'): 'v10', ('TMP', '850 mb'): 't850', ('RH', '850 mb'): 'rh850', ('UGRD', '850 mb'): 'u850', ('VGRD', '850 mb'): 'v850'}

def charts():
    with open(os.path.join(HERE, 'fronts_wpc.json')) as f:
        raw = json.load(f)
    return {datetime.strptime(k, '%Y%m%d%H').replace(tzinfo=timezone.utc): [(kind, np.array(p, float).reshape(-1, 2)) for kind, p in v] for k, v in raw.items()}

def get(url, rng=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
SOURCE = os.environ.get('FRONTS_SOURCE', 'gfs')
ECMWF = 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{c:%Y%m%d}/{c:%H}z/ifs/0p25/oper/{c:%Y%m%d%H}0000-{fh}h-oper-fc.{ext}'
ECMWF_WANT = {('msl', None): 'mslp', ('sp', None): 'ps', ('2t', None): 't2', ('2d', None): 'td2', ('10u', None): 'u10', ('10v', None): 'v10', ('t', '850'): 't850', ('r', '850'): 'rh850', ('u', '850'): 'u850', ('v', '850'): 'v850'}
ECMWF_GROUND = datetime(2026, 10, 23, tzinfo=timezone.utc)
_GROUND = {}

def fields(t):
    return ecmwf_fields(t) if SOURCE == 'ecmwf' else gfs_fields(t)

def _grib(url, r):
    import eccodes
    h = eccodes.codes_new_from_message(get(url, r))
    try:
        ni, nj = (eccodes.codes_get(h, 'Ni'), eccodes.codes_get(h, 'Nj'))
        return (eccodes.codes_get_values(h).reshape(nj, ni), eccodes.codes_get_array(h, 'distinctLatitudes'), eccodes.codes_get_array(h, 'distinctLongitudes'))
    finally:
        eccodes.codes_release(h)

def _ecmwf_ranges(c, fh, want):
    rows = [json.loads(ln) for ln in get(ECMWF.format(c=c, fh=fh, ext='index'), tries=10).decode().splitlines() if ln.strip()]
    out = {}
    for d in rows:
        key = (d['param'], str(d['levelist']) if d.get('levtype') == 'pl' else None)
        if key in want and want[key] not in out:
            out[want[key]] = (d['_offset'], d['_offset'] + d['_length'] - 1)
    return out

def ecmwf_fields(t):
    c = t.replace(hour=t.hour // 6 * 6)
    fh = int((t - c).total_seconds() // 3600)
    rngs = _ecmwf_ranges(c, fh, ECMWF_WANT)
    if len(rngs) != len(ECMWF_WANT):
        raise RuntimeError(f'{t}: missing {set(ECMWF_WANT.values()) - set(rngs)}')
    url = ECMWF.format(c=c, fh=fh, ext='grib2')
    out, lat, lon = ({}, None, None)
    for name, r in rngs.items():
        out[name], lat, lon = _grib(url, r)
    if 'zs' not in _GROUND:
        g = _ecmwf_ranges(ECMWF_GROUND, 0, {('z', None): 'zs'})['zs']
        _GROUND['zs'] = _grib(ECMWF.format(c=ECMWF_GROUND, fh=0, ext='grib2'), g)[0] / physics.G
    out['zs'] = _GROUND['zs']
    return _crop(out, lat, lon)

def gfs_fields(t):
    import eccodes
    c = t.replace(hour=t.hour // 6 * 6)
    url = GFS.format(c=c, fh=int((t - c).total_seconds() // 3600))
    idx = [ln.split(':') for ln in get(url + '.idx').decode().splitlines()]
    rngs = {}
    for i, e in enumerate(idx):
        key = (e[3], e[4])
        if key in WANT and WANT[key] not in rngs:
            rngs[WANT[key]] = (int(e[1]), int(idx[i + 1][1]) - 1)
    if len(rngs) != len(WANT):
        raise RuntimeError(f'{t}: missing {set(WANT.values()) - set(rngs)}')
    out, lat, lon = ({}, None, None)
    for name, r in rngs.items():
        h = eccodes.codes_new_from_message(get(url, r))
        try:
            ni, nj = (eccodes.codes_get(h, 'Ni'), eccodes.codes_get(h, 'Nj'))
            v = eccodes.codes_get_values(h).reshape(nj, ni)
            if lat is None:
                lat = eccodes.codes_get_array(h, 'distinctLatitudes')
                lon = eccodes.codes_get_array(h, 'distinctLongitudes')
        finally:
            eccodes.codes_release(h)
        out[name] = v
    return _crop(out, lat, lon)

def _crop(out, lat, lon):
    lon = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon)
    w, s, e, n = BOX
    if lat[0] < lat[-1]:
        lat, out = (lat[::-1], {k: v[::-1] for k, v in out.items()})
    yi = np.nonzero((lat >= s - 1e-06) & (lat <= n + 1e-06))[0]
    lon = lon[order]
    xi = np.nonzero((lon >= w - 1e-06) & (lon <= e + 1e-06))[0]
    return ({k: v[:, order][yi][:, xi].astype(np.float64) for k, v in out.items()}, lat[yi].astype(np.float64), lon[xi].astype(np.float64))

def inputs(f, lat, lon):
    t8 = f['t850'] - 273.15
    fronts.ALAT = ALAT
    return fronts.features(lat, lon, t8, physics.dewpoint_from_rh(t8, f['rh850']), f['u850'], f['v850'], f['ps'] / 100, f['mslp'] / 100, f['t2'] - 273.15, f['td2'] - 273.15, f['u10'], f['v10'], f['zs'])

def unit(lon, lat):
    lo, la = (np.radians(lon), np.radians(lat))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

def dense(p, km=10.0):
    out = []
    for (a, b), (c, d) in zip(p[:-1], p[1:]):
        n = max(1, int(fronts.km_between(a, b, c, d) // km))
        out += [(a + (c - a) * i / n, b + (d - b) * i / n) for i in range(n)]
    out.append(tuple(p[-1]))
    return np.array(out)

def labels(g, lines):
    from scipy.spatial import cKDTree
    la, lo = np.meshgrid(g.lats, g.lons, indexing='ij')
    cells = unit(lo.ravel(), la.ravel())
    best = np.full(cells.shape[0], np.inf)
    y = np.zeros(cells.shape[0], np.int8)
    chord = 2 * math.sin(R_KM / 6371.0 / 2)
    for k, kind in enumerate(CLS, 1):
        pts = [dense(p) for kk, p in lines if kk == kind and len(p) >= 2]
        if not pts:
            continue
        d, _ = cKDTree(unit(*np.vstack(pts).T)).query(cells, distance_upper_bound=chord)
        hit = d < best
        y[hit & np.isfinite(d)] = k
        best = np.minimum(best, d)
    return y.reshape(la.shape)

def weights(path=None):
    with open(path or os.path.join(HERE, fronts.CNN_FILE)) as f:
        return json.load(f)

def prep(shard, shards, out):
    os.makedirs(out, exist_ok=True)
    W = weights()
    mu, sd = (np.array(W['mu']), np.array(W['sd']))
    ts = sorted(charts())[shard::shards]
    ch = charts()

    def one(t):
        p = os.path.join(out, f'{t:%Y%m%d%H}.npz')
        if os.path.exists(p):
            return 0
        try:
            f, lat, lon = fields(t)
        except Exception as e:
            print(f'{t}: {e}', flush=True)
            return 0
        X, g, (nx, ny) = inputs(f, lat, lon)
        z = (X.transpose(1, 2, 0) - mu) / sd
        q = np.clip(np.rint((np.clip(z, -CLIP, CLIP) + CLIP) / (2 * CLIP) * 254), 0, 254).astype(np.uint8)
        y = labels(g, ch[t])
        tmp = p + '.tmp.npz'
        np.savez_compressed(tmp, Q=q, Y=y, lat=g.lats, lon=g.lons, nx=nx.astype(np.float32), ny=ny.astype(np.float32), zs=X[fronts.FEATURES.index('zs')].astype(np.float32))
        os.replace(tmp, p)
        return 1
    t0, n = (time.time(), 0)
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(one, ts)):
            n += r
            if i % 25 == 0:
                print(f'{i + 1}/{len(ts)} {time.time() - t0:.0f}s', flush=True)
    print(f'prep {shard}/{shards}: {n} new of {len(ts)} in {time.time() - t0:.0f}s')

def split(ts):
    days = sorted({t.date() for t in ts})
    test = set(days[::3])
    rest = [d for d in days if d not in test]
    val = set(rest[::4])
    return ([t for t in ts if t.date() not in test and t.date() not in val], [t for t in ts if t.date() in val], [t for t in ts if t.date() in test])

def cases(d):
    return sorted((datetime.strptime(os.path.basename(p)[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc) for p in os.listdir(d) if p.endswith('.npz') and (not p.endswith('.tmp.npz'))))

def load(d, t):
    return np.load(os.path.join(d, f'{t:%Y%m%d%H}.npz'))

def train(d, out, epochs=12, lr=0.001, batch=4):
    os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=true')
    import jax
    import jax.numpy as jnp
    import optax
    W0 = weights()
    dils = [int(x) for x in W0['dils']]
    n_conv = 2 + len(dils)
    P = [(jnp.array(W0[f'C{i}'], jnp.float32), jnp.array(W0[f'c{i}'], jnp.float32)) for i in range(n_conv)]
    tr, va, te = split(cases(d))
    print(f'cases: {len(tr)} train, {len(va)} validation, {len(te)} test', flush=True)
    Xa = np.stack([load(d, t)['Q'] for t in tr])
    Ya = np.stack([load(d, t)['Y'] for t in tr])
    Xv = np.stack([load(d, t)['Q'] for t in va])
    Yv = np.stack([load(d, t)['Y'] for t in va])
    mask = np.zeros(Ya.shape[1:], np.float32)
    mask[4:-4, 4:-4] = 1
    mask = jnp.array(mask)
    deq = lambda q: q.astype(np.float32) / 254 * 2 * CLIP - CLIP

    def forward(P, x):
        conv = lambda h, Wt, dd: jax.lax.conv_general_dilated(h, Wt, (1, 1), 'SAME', rhs_dilation=(dd, dd), dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        h = jax.nn.relu(conv(x, P[0][0], 1) + P[0][1])
        for (Wt, b), dd in zip(P[1:-1], dils):
            h = h + jax.nn.relu(conv(h, Wt, dd) + b)
        return conv(h, P[-1][0], 1) + P[-1][1]

    def loss(P, x, y):
        ce = optax.softmax_cross_entropy_with_integer_labels(forward(P, x), y)
        return (ce * mask).sum() / (mask.sum() * x.shape[0])
    steps = max(1, epochs * (len(Xa) // batch))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=0.0001))
    st = opt.init(P)

    @jax.jit
    def step(P, st, x, y):
        val, g = jax.value_and_grad(loss)(P, x, y)
        u, st = opt.update(g, st, P)
        return (optax.apply_updates(P, u), st, val)
    vloss = jax.jit(loss)

    def validate(P):
        return sum((float(vloss(P, jnp.array(deq(Xv[i:i + batch])), jnp.array(Yv[i:i + batch].astype(np.int32)))) * len(Xv[i:i + batch]) for i in range(0, len(Xv), batch))) / len(Xv)
    best = validate(P)
    print(f'start: validation loss {best:.4f} (the current weights)', flush=True)
    bestP, rng = (jax.tree_util.tree_map(np.asarray, P), np.random.default_rng(0))
    for ep in range(epochs):
        t0, tl = (time.time(), [])
        idx = rng.permutation(len(Xa))
        for i in range(0, len(idx) - batch + 1, batch):
            b = idx[i:i + batch]
            P, st, val = step(P, st, jnp.array(deq(Xa[b])), jnp.array(Ya[b].astype(np.int32)))
            tl.append(float(val))
        vl = validate(P)
        print(f'epoch {ep}: train {np.mean(tl):.4f} validation {vl:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best:
            best, bestP = (vl, jax.tree_util.tree_map(np.asarray, P))
    r = lambda a: np.vectorize(lambda x: float(f'{x:.6g}'))(np.asarray(a, np.float64)).tolist()
    Wn = {'names': W0['names'], 'dils': dils, 'R_km': R_KM, 'mu': W0['mu'], 'sd': W0['sd'], 'alat': list(ALAT)}
    for i, (Wt, b) in enumerate(bestP):
        Wn[f'C{i}'], Wn[f'c{i}'] = (r(Wt), r(b))
    with open(out, 'w') as f:
        json.dump(Wn, f, separators=(',', ':'))
    print(f'best validation loss {best:.4f}; weights in {out}', flush=True)
    return te

def _arrays_for(W):
    tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    json.dump(W, tmp)
    tmp.close()
    old = (fronts.CNN_FILE, dict(fronts._CNN))
    try:
        fronts.CNN_FILE = tmp.name
        fronts._CNN.clear()
        return fronts._weights()
    finally:
        fronts.CNN_FILE = old[0]
        fronts._CNN.clear()
        fronts._CNN.update(old[1])
        os.unlink(tmp.name)

def _pts(lines, km=25.0):
    P, K = ([], [])
    for kind, p in lines:
        p = np.asarray(p, float)
        if len(p) < 2:
            continue
        q = dense(p, km)
        P.append(q)
        K += [kind] * len(q)
    return (np.vstack(P) if P else np.zeros((0, 2)), np.array(K))

def _tally(ours, theirs, D=250.0):
    from scipy.spatial import cKDTree
    OP, OK = _pts(ours)
    WP, WK = _pts(theirs)
    w, s, e, n = BOX
    iw, is_, ie, in_ = INNER
    reg = lambda P: np.where((P[:, 0] > iw + 1) & (P[:, 0] < ie - 1) & (P[:, 1] > is_ + 1) & (P[:, 1] < in_ - 1), 'inner', np.where((P[:, 0] > w + 1) & (P[:, 0] < e - 1) & (P[:, 1] > s + 1) & (P[:, 1] < n - 1), 'outer', ''))
    chord = 2 * math.sin(D / 6371.0 / 2)
    out = {}
    ro, rw = (reg(OP) if len(OP) else np.array([]), reg(WP) if len(WP) else np.array([]))
    for region in ('inner', 'outer'):
        o, wm = (ro == region, rw == region)
        c = {'O': int(o.sum()), 'W': int(wm.sum())}
        if len(OP) and len(WP):
            to, tw = (cKDTree(unit(*OP.T)), cKDTree(unit(*WP.T)))
            dw, iw_ = to.query(unit(*WP[wm].T)) if wm.any() else (np.array([]), np.array([], int))
            do, io = tw.query(unit(*OP[o].T)) if o.any() else (np.array([]), np.array([], int))
            c['w_hit'] = int((dw <= chord).sum())
            c['o_hit'] = int((do <= chord).sum())
            c['w_same'] = int(sum((dw <= chord) & (OK[iw_] == WK[wm]) if wm.any() else []))
            c['o_same'] = int(sum((do <= chord) & (WK[io] == OK[o]) if o.any() else []))
        else:
            c.update(w_hit=0, o_hit=0, w_same=0, o_same=0)
        out[region] = c
    return out

def score(d, paths, ts=None, procs=4):
    ts = ts if ts is not None else split(cases(d))[2]
    tot = {}
    import multiprocessing
    with ProcessPoolExecutor(procs, mp_context=multiprocessing.get_context('spawn')) as ex:
        for r in ex.map(_score_tally, [(d, t, paths) for t in ts]):
            for name, regs in r.items():
                for region, c in regs.items():
                    a = tot.setdefault((name, region), {})
                    for k, v in c.items():
                        a[k] = a.get(k, 0) + v
    print(f"\nWPC's fronts on {len(ts)} test charts, 250 km (each chart alone, as the job draws one hour):")
    for (name, region), c in sorted(tot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        pod, sr = (c['w_hit'] / max(1, c['W']), c['o_hit'] / max(1, c['O']))
        pods, srs = (c['w_same'] / max(1, c['W']), c['o_same'] / max(1, c['O']))
        csi = lambda p, q: 1 / (1 / p + 1 / q - 1) if p > 0 and q > 0 else 0.0
        print(f'  {region:5s} {name:8s} any POD {pod:.2f} SR {sr:.2f} CSI {csi(pod, sr):.3f} | same type POD {pods:.2f} SR {srs:.2f} CSI {csi(pods, srs):.3f} | ours {c['O'] * 25} km, WPC {c['W'] * 25} km')
    return tot

def _score_tally(args):
    d, t, paths = args
    z = load(d, t)
    ch = charts()[t]
    out = {}
    for name, path in paths.items():
        W = weights(path)
        Wa = _arrays_for(W)
        mu, sd = (np.array(W['mu'], np.float32), np.array(W['sd'], np.float32))
        X = ((z['Q'].astype(np.float32) / 254 * 2 * CLIP - CLIP) * sd + mu).transpose(2, 0, 1)
        if name == 'before':
            X[fronts.FEATURES.index('alat')] = np.minimum(X[fronts.FEATURES.index('alat')], 56.0)
        p = fronts.probabilities(Wa, X)
        g = fronts.Grid(z['lat'], z['lon'])
        lines = fronts.ml_lines(g, p, z['nx'].astype(np.float64), z['ny'].astype(np.float64), zs=z['zs'] * 1000.0)
        out[name] = _tally(lines, ch)
    return out

def emit(path):
    with open(path, 'rb') as f:
        s = base64.b64encode(zlib.compress(f.read(), 9)).decode()
    print('=====WEIGHTS BEGIN', len(s))
    for i in range(0, len(s), 3000):
        print(s[i:i + 3000])
    print('=====WEIGHTS END')
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif cmd == 'train':
        te = train(sys.argv[2], sys.argv[3], epochs=int(os.environ.get('EPOCHS', '12')), batch=int(os.environ.get('BATCH', '4')))
        score(sys.argv[2], {'before': os.path.join(HERE, fronts.CNN_FILE), 'after': sys.argv[3]}, te)
        emit(sys.argv[3])
    elif cmd == 'score':
        score(sys.argv[2], {'before': os.path.join(HERE, fronts.CNN_FILE), 'after': sys.argv[3]})
    else:
        sys.exit(__doc__)
