import ctypes
import os
import shutil
import unittest
import numpy as np
import physics as P
import physics_native as N

def columns(ny=23, nx=31, levels=(1000, 975, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300), seed=0, shuffle=False, nan_cells=True):
    rng = np.random.default_rng(seed)
    shape = (ny, nx)
    f = lambda a: np.asarray(a, np.float32)
    p = np.array(levels, np.float32)
    zs = f(rng.random(shape) ** 2 * 3500)
    ps = f(1013.25 * (1 - zs / 44330) ** 5.255)
    lapse = f(4 + rng.random(shape) * 5.5)
    t2 = f(28 - 6.5 * zs / 1000 + rng.normal(0, 3, shape))
    td2 = f(t2 - rng.random(shape) * 18)
    zl = f(44330 * (1 - (p[:, None, None] / 1013.25) ** (1 / 5.255)) + rng.normal(0, 25, (len(p),) + shape))
    t = f(t2[None] + 6.5 * zs[None] / 1000 - lapse[None] * zl / 1000)
    td = f(t - rng.random((len(p),) + shape) * 25)
    ang = f(rng.random(shape) * 2 * np.pi)
    spd = f(3 + zl / 400 + rng.random((len(p),) + shape) * 4)
    u, v = (f(spd * np.cos(ang + zl / 8000)), f(spd * np.sin(ang + zl / 8000)))
    u10, v10 = (f(u[0] * 0.4), f(v[0] * 0.4))
    uh, vh = (f(u[0] * 0.7), f(v[0] * 0.7))
    if nan_cells:
        for a in (zs, ps, t2, td2, u10, v10):
            a[0, :3] = np.nan
        for a in (zl, t, td, u, v):
            a[:, 1, 5:8] = np.nan
        uh[2, 2:6] = np.nan
        t[3, 4, 4] = np.nan
    if shuffle:
        order = rng.permutation(len(p))
        p, zl, t, td, u, v = (p[order], zl[order], t[order], td[order], u[order], v[order])
    return dict(zs=zs, ps=ps, t2=t2, td2=td2, u10=u10, v10=v10, z=zl, p=p, t=t, td=td, u=u, v=v, uh=uh, vh=vh)

def profiles(c, hub=True, hub_m=80.0):
    args = (c['zs'], c['ps'], c['t2'], c['td2'], c['u10'], c['v10'], c['z'], c['p'], c['t'], c['td'], c['u'], c['v'])
    extra = (c['uh'], c['vh'], hub_m) if hub else ()
    return (P.Profile(*args, *extra), N.Profile(*args, *extra))
HAVE_CXX = shutil.which(os.environ.get('CXX', 'g++')) is not None

class Build(unittest.TestCase):

    @unittest.skipUnless(HAVE_CXX, 'no C++ compiler')
    def test_builds(self):
        self.assertTrue(N.available(), N.error)

def same(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)

@unittest.skipUnless(N.available(), f'C++ physics not available: {N.error}')
class NativeProfile(unittest.TestCase):

    def test_columns_bit_for_bit(self):
        for seed in range(4):
            for hub in (True, False):
                c = columns(seed=seed, shuffle=seed % 2 == 1)
                py, cx = profiles(c, hub)
                self.assertTrue(cx.native)
                for k in ('z', 'lnp', 't', 'td', 'zw', 'u', 'v', 'ztop', 'zs'):
                    self.assertTrue(same(getattr(py, k), getattr(cx, k)), f'{k} (seed {seed}, hub {hub})')

    def test_at_bit_for_bit(self):
        c = columns(seed=5)
        py, cx = profiles(c)
        rng = np.random.default_rng(9)
        zs = c['zs']
        for zq in (zs + 2, zs + 300, np.full_like(zs, 3000), (zs + rng.random(zs.shape) * 13000).astype(np.float32), zs - 100, np.full_like(zs, np.nan), np.float32(1500) + np.zeros_like(zs)):
            for names in (('t',), ('u', 'v'), ('t', 'td', 'lnp', 'u', 'v')):
                a, b = (py.at(zq, *names), cx.at(zq, *names))
                a, b = (a, b) if isinstance(a, list) else ([a], [b])
                for n, x, y in zip(names, a, b):
                    self.assertTrue(same(x, y), n)

    def test_other_inputs_fall_back(self):
        c = columns(seed=2)
        py, cx = profiles(c)
        for zq in (1500.0, c['zs'].astype(np.float64) + 300):
            a, b = (py.at(zq, 't'), cx.at(zq, 't'))
            self.assertTrue(same(a, b))
        c64 = {k: v.astype(np.float64) for k, v in c.items()}
        py, cx = profiles(c64)
        self.assertFalse(cx.native)
        self.assertTrue(same(py.at(c64['zs'] + 300, 't'), cx.at(c64['zs'] + 300, 't')))

    def test_column_at(self):
        c = columns(seed=3)
        zq = c['zs'] + 800
        self.assertTrue(same(P.column_at(c['z'], c['t'], zq), N.column_at(c['z'], c['t'], zq)))

    def test_block_copy(self):
        c = columns(ny=41, nx=37, seed=4)
        py, cx = profiles(c)
        zq = (c['zs'] + np.linspace(-50, 12000, c['zs'].size).reshape(c['zs'].shape)).astype(np.float32)
        outs = [np.empty_like(zq) for _ in range(5)]
        cp = cx.c()
        N.lib().cx_block_at(ctypes.byref(cp), N.ptr(zq), *[N.ptr(o) for o in outs])
        ref = py.at(zq, 't', 'td', 'lnp', 'u', 'v')
        ref[2] = np.log(ref[2])
        for n, x, y in zip(('t', 'td', 'lnp', 'u', 'v'), outs, ref):
            self.assertTrue(same(x, y), n)

@unittest.skipUnless(N.available(), f'C++ physics not available: {N.error}')
class NativeConvection(unittest.TestCase):

    @staticmethod
    def cloudbase(c, seed):
        rng = np.random.default_rng(seed + 100)
        zs, shape = (c['zs'], c['zs'].shape)
        lcl = np.maximum(0, rng.normal(1200, 900, shape)).astype(np.float32)
        lcl.flat[::7] = np.nan
        active = rng.random(shape) < 0.75
        return (zs + 2 + lcl, c['t2'] - P.DRY_LAPSE * lcl, active)

    def check_cumulus(self, c, hub, seed, **kw):
        py, cx = profiles(c, hub)
        self.assertTrue(cx.native)
        zb, tb, active = self.cloudbase(c, seed)
        for t in (tb, tb + 15, tb - 10):
            a, b = (P.cumulus_top(py, zb, t, active, **kw), N.cumulus_top(cx, zb, t, active, **kw))
            self.assertTrue(same(a, b), f'seed {seed}, hub {hub}, {kw}')
        return (py, zb, tb, active)

    def test_cumulus_top(self):
        for seed in range(4):
            for hub in (True, False):
                c = columns(seed=seed, shuffle=seed % 2 == 1)
                py, zb, tb, active = self.check_cumulus(c, hub, seed)
        top, warm = (P.cumulus_top(py, zb, tb, active), P.cumulus_top(py, zb, tb + 15, active))
        rose = active & np.isfinite(zb)
        self.assertTrue(((top > zb) & (top < py.ztop) & rose).any())
        self.assertTrue(((warm >= py.ztop) & rose).any())
        self.assertTrue((active & np.isnan(zb)).any() and (~active & np.isnan(zb)).any())

    def test_cumulus_top_entraining(self):
        for seed in range(4):
            for hub in (True, False):
                c = columns(seed=seed, shuffle=seed % 2 == 1)
                rng = np.random.default_rng(seed + 300)
                width = np.where(rng.random(c['zs'].shape) < 0.2, np.nan, rng.uniform(100, 3000, c['zs'].shape)).astype(np.float32)
                mix = P.cumulus_mix(width)
                py, zb, tb, active = self.check_cumulus(c, hub, seed, mix=mix)
        undilute, mixed = (P.cumulus_top(py, zb, tb + 3, active), P.cumulus_top(py, zb, tb + 3, active, mix=mix))
        rose = active & np.isfinite(zb)
        self.assertTrue((mixed[rose] <= undilute[rose]).all())
        self.assertTrue((mixed[rose] < undilute[rose]).any())

    def test_cumulus_top_columns_not_ascending(self):
        for seed in range(2):
            c = columns(seed=seed)
            z = c['z']
            z[[5, 6], 2:9] = z[[6, 5], 2:9]
            z[9, 10:14] -= 2500
            py, *_ = self.check_cumulus(c, seed == 0, seed)
            self.assertTrue((np.diff(py.z, axis=0) < 0).any())

    def test_cumulus_top_steps(self):
        c = columns(seed=6)
        py, zb, tb, active = self.check_cumulus(c, True, 6, max_above=3000.0)
        top = P.cumulus_top(py, zb, tb + 15, active, max_above=3000.0)
        self.assertTrue((active & (top >= zb + 2999) & (top < py.ztop)).any())
        for kw in (dict(step=100.0), dict(step=500), dict(max_above=3000.0), dict(step=250.0, max_above=2600), dict(max_above=100.0), dict(max_above=0.0), dict(step=300.0, max_above=-900.0)):
            self.check_cumulus(c, True, 6, **kw)

    def test_cumulus_top_tiny_grids(self):
        for ny, nx in ((1, 1), (2, 3)):
            for seed in range(3):
                c = columns(ny=ny, nx=nx, seed=seed, nan_cells=False)
                self.check_cumulus(c, seed % 2 == 0, seed)

    def test_cumulus_top_falls_back(self):
        c = columns(seed=7)
        py, cx = profiles(c)
        zb, tb, active = self.cloudbase(c, 7)
        for args, kw in (((zb, tb, active), dict(step=np.float64(200))), ((zb.astype(np.float64), tb, active), {}), ((zb, tb.astype(np.float64), active), {}), ((zb, tb, active.astype(np.float32)), {}), ((zb, tb, np.ones((1,) + zb.shape[1:], bool)), {})):
            self.assertTrue(same(P.cumulus_top(py, *args, **kw), N.cumulus_top(cx, *args, **kw)))
        c64 = {k: v.astype(np.float64) for k, v in c.items()}
        py, cx = profiles(c64)
        zb, tb, active = self.cloudbase(c64, 7)
        self.assertTrue(same(P.cumulus_top(py, zb, tb, active), N.cumulus_top(cx, zb, tb, active)))

    def test_exp_is_numpys(self):
        mode = N._np_exp_mode()
        self.assertIsNotNone(mode)
        x = np.concatenate([np.linspace(-15, 8, 400001, dtype=np.float32), np.float32([np.nan, np.inf, -np.inf, 0, 100, -120])])
        out = np.empty_like(x)
        N.cfunc('cx_np_exp', [ctypes.c_int64, N._F, ctypes.c_int32, N._F])(x.size, N.ptr(x), mode, N.ptr(out))
        with np.errstate(over='ignore'):
            self.assertTrue(same(np.exp(x), out))

    def check_layer(self, py, cx, zb, zt, names, **kw):
        a, b = (P.layer_mean(py, zb, zt, names, **kw), N.layer_mean(cx, zb, zt, names, **kw))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(same(x, y), f'{names} {kw}')

    def test_layer_mean(self):
        for seed in range(4):
            for hub in (True, False):
                c = columns(seed=seed, shuffle=seed % 2 == 0)
                py, cx = profiles(c, hub)
                zs = c['zs']
                zb, tb, active = self.cloudbase(c, seed)
                d = np.maximum(np.where(active, zb - zs - 2, np.float32(800)), 100)
                for names in (['u', 'v'], ['t', 'td'], ('v', 't', 'u', 'td', 'u')):
                    for kw in ({}, dict(n=2), dict(n=5), dict(n=17)):
                        self.check_layer(py, cx, zs + 2, zs + 2 + d, names, **kw)
                self.check_layer(py, cx, zs + 3000, zs + 2, ['u', 't'])
                self.check_layer(py, cx, np.float32(1500), zs + 2 + d, ['v'])

    def test_layer_mean_tiny_grids(self):
        for ny, nx in ((1, 1), (2, 3)):
            c = columns(ny=ny, nx=nx, seed=ny, nan_cells=False)
            py, cx = profiles(c, ny == 1)
            self.check_layer(py, cx, c['zs'] + 2, c['zs'] + 1500, ['u', 'v'])

    def test_layer_mean_falls_back(self):
        c = columns(seed=8)
        py, cx = profiles(c)
        zs = c['zs']
        self.check_layer(py, cx, zs + 2, zs + 1000, ['td', 'lnp'])
        self.check_layer(py, cx, zs + 2, zs + 1000, ['u'], n=np.int64(8))
        self.check_layer(py, cx, zs + 2, (zs + 1000).astype(np.float64), ['u', 'v'])
        self.check_layer(py, cx, zs + 2, 1000.0, ['t'])
        with self.assertRaises(ZeroDivisionError):
            N.layer_mean(cx, zs + 2, zs + 1000, ['u'], n=1)

def sink_inputs(c, seed):
    rng = np.random.default_rng(1000 + seed)
    shape = c['zs'].shape
    f = lambda a: np.asarray(a, np.float32)
    some = lambda p: rng.random(shape) < p
    up = np.argsort(-np.asarray(c['p']))
    wz = np.ascontiguousarray(c['z'][up])
    w = f(rng.normal(0, 0.5, wz.shape))
    w[rng.random(wz.shape) < 0.03] = np.nan
    w[rng.random(wz.shape) < 0.01] = np.inf
    ws = rng.random(shape) * 3.5
    ws[some(0.2)] = 0
    ws[some(0.05)] = -0.5
    ws[some(0.05)] = np.nan
    ws[some(0.02)] = np.inf
    zi = f(rng.random(shape) * 3500)
    zi[some(0.1)] = 0
    zi[some(0.05)] = np.nan
    zi[some(0.02)] = np.inf
    depth = f(rng.random(shape) * 4500)
    depth[some(0.15)] = 0
    depth[some(0.1)] = rng.random() * 300
    depth[some(0.05)] = np.nan
    depth[some(0.02)] = np.inf
    rain = f(rng.exponential(2, shape) * some(0.4))
    rain[some(0.05)] = np.nan
    rain[some(0.03)] = -1
    rain[some(0.01)] = np.inf
    refd = f(rng.uniform(-25, 70, shape))
    refd[some(0.05)] = np.nan
    refd[some(0.01)] = np.inf
    lee = rng.random(shape) * 2 * some(0.5) - 0.1 * some(0.1)
    lee[some(0.05)] = np.nan
    lee[some(0.02)] = np.inf
    lee[some(0.02)] = -np.inf
    snow = f(rng.choice([0, 0.5, 1], shape))
    snow[some(0.02)] = np.nan
    return dict(wz=wz, w=w, ws=ws, zi=zi, depth=depth, rain=rain, refd=refd, lee=lee, snow=snow)

@unittest.skipUnless(N.available(), f'C++ physics not available: {N.error}')
class NativeSinkRate(unittest.TestCase):

    def check(self, py, cx, s, w=True, ws64=True, lee='f64', refd=True, parts=True, n=None, native=True, msg=''):
        ws = s['ws'] if ws64 else s['ws'].astype(np.float32)
        lee_a = {None: None, 'f32': s['lee'].astype(np.float32), 'f64': s['lee']}[lee]
        args = (s['wz'] if w else None, s['w'] if w else None, ws, s['zi'], s['depth'], s['rain'], s['refd'] if refd else None, lee_a)
        kw = {'parts': parts} if n is None else {'n': n, 'parts': parts}
        kw['snow'] = s.get('snow')
        with np.errstate(all='ignore'):
            ref = P.sink_rate(py, *args, **kw)
            real = P.sink_rate
            if native:
                P.sink_rate = lambda *a, **k: self.fail(f'fell back to physics.py ({msg})')
            try:
                got = N.sink_rate(cx, *args, **kw)
            finally:
                P.sink_rate = real
        ref, got = (ref, got) if parts else ((ref,), (got,))
        self.assertEqual(len(ref), len(got), msg)
        for name, a, b in zip(('sink', 'between', 'why'), ref, got):
            self.assertTrue(same(a, b), f'{name}: {msg} ({a.dtype} {b.dtype})')
        return ref

    def test_every_variant(self):
        import itertools
        for seed, hub in ((0, True), (1, False)):
            c = columns(seed=seed, shuffle=seed == 1)
            py, cx = profiles(c, hub)
            s = sink_inputs(c, seed)
            for w, ws64, lee, refd, parts in itertools.product((True, False), (True, False), (None, 'f32', 'f64'), (True, False), (True, False)):
                self.check(py, cx, s, w, ws64, lee, refd, parts, msg=f'seed {seed} hub {hub} w {w} ws64 {ws64} lee {lee} refd {refd} parts {parts}')

    def test_seeds_and_grids(self):
        for seed in range(6):
            for ny, nx in ((23, 31), (1, 1), (2, 3), (41, 37)):
                for hub in (True, False):
                    c = columns(ny=ny, nx=nx, seed=seed, shuffle=seed % 3 == 2, nan_cells=min(ny, nx) > 3)
                    py, cx = profiles(c, hub)
                    s = sink_inputs(c, seed)
                    why = self.check(py, cx, s, msg=f'seed {seed} {ny}x{nx} hub {hub}')[2]
                    self.check(py, cx, s, w=False, lee=None, parts=False, msg=f'no w, seed {seed} {ny}x{nx}')
                    if ny * nx > 100 and seed == 0:
                        self.assertEqual(set(np.unique(why)), {0.0, 1.0, 2.0, 3.0})

    def test_band_samples(self):
        c = columns(seed=7)
        py, cx = profiles(c)
        s = sink_inputs(c, 7)
        for n in (2, 3, 4, 17):
            self.check(py, cx, s, n=n, msg=f'n {n}')
            self.check(py, cx, s, ws64=False, lee='f32', n=n, parts=False, msg=f'float32, n {n}')

    def test_other_inputs_fall_back(self):
        c = columns(seed=8)
        py, cx = profiles(c)
        s = sink_inputs(c, 8)
        for n in (np.int64(10), 1, 300, True):
            try:
                self.check(py, cx, s, n=n, native=False, msg=f'n {n!r}')
            except ZeroDivisionError:
                self.assertEqual(n, 1)
        for k, v in (('zi', s['zi'].astype(np.float64)), ('depth', s['depth'][:1]), ('rain', 0.0), ('ws', 1.0), ('ws', s['ws'].astype(np.float16)), ('lee', s['lee'][:, :1]), ('w', s['w'].astype(np.float64)), ('refd', s['refd'].astype(np.float64)), ('ws', np.asfortranarray(s['ws']))):
            self.check(py, cx, dict(s, **{k: v}), native=False, msg=f'{k} {getattr(v, 'dtype', type(v))}')
        self.check(py, py, s, native=False, msg='physics.Profile')

def _derivs(shape, grid_step=0.03, lat0=38.0):
    from scipy.ndimage import uniform_filter
    lat2d = lat0 + np.arange(shape[0])[:, None] * -grid_step + np.zeros(shape)
    dy = grid_step * 111320.0
    dx = dy * np.cos(np.radians(lat2d))
    if min(shape) >= 2:
        ddx = lambda a: np.gradient(a, axis=1) / dx
        ddy = lambda a: -np.gradient(a, axis=0) / dy
    else:
        ddx = lambda a: a / dx
        ddy = lambda a: -a / dy
    k = max(1, round(0.25 / grid_step))
    smooth = (lambda a: uniform_filter(a, k, mode='nearest')) if k > 1 else None
    return (ddx, ddy, smooth)

@unittest.skipUnless(N.available(), f'C++ physics not available: {N.error}')
class NativeAviation(unittest.TestCase):
    GRIDS = ((23, 31), (1, 1), (1, 5), (2, 3), (41, 37))

    def assertSame(self, a, b, msg=''):
        a, b = (a, b) if isinstance(a, tuple) else ((a,), (b,))
        self.assertEqual(len(a), len(b), msg)
        for j, (x, y) in enumerate(zip(a, b)):
            self.assertTrue(same(x, y), f'{msg} output {j}')

    def cases(self):
        for seed, (ny, nx) in enumerate(self.GRIDS * 2):
            c = columns(ny=ny, nx=nx, seed=seed, shuffle=seed % 2 == 1, nan_cells=ny >= 3 and nx >= 8)
            if seed == 5:
                for k in ('z', 't', 'td', 'u', 'v'):
                    c[k][:, 0, 0] = np.nan
                c['t'][:, 0, 1] = -5
                c['t2'][0, 1] = -5
                c['t'][:, 1, 1] = 30
                c['t2'][1, 1] = 30
            yield (seed, c)

    def test_freezing_level(self):
        for seed, c in self.cases():
            for hub in (True, False):
                py, cx = profiles(c, hub)
                self.assertTrue(cx.native)
                self.assertSame(P.freezing_level(py), N.freezing_level(cx), f'seed {seed} hub {hub}')

    def test_icing_potential(self):
        rng = np.random.default_rng(3)
        for seed, c in self.cases():
            order = np.argsort(-c['p'])
            z, t, td = (c['z'][order], c['t'][order], c['td'][order])
            t = (t - 10 * (seed % 3)).astype(np.float32)
            rh = P.rh_from_td(t, np.minimum(td, t))
            clw = (rng.random(z.shape) ** 3 * 0.0004).astype(np.float32)
            clw[rng.random(z.shape) < 0.05] = np.nan
            for cl in (None, clw):
                for zmax in ((), (7300.0,), (4000,), (np.float64(3000.0),), (np.inf,), (-np.inf,)):
                    a = P.icing_potential(z, t, rh, cl, c['zs'], *zmax)
                    b = N.icing_potential(z, t, rh, cl, c['zs'], *zmax)
                    self.assertSame(a, b, f'seed {seed} clw {cl is not None} zmax {zmax}')

    def test_ellrod_turbulence(self):
        for seed, c in self.cases():
            shape = c['zs'].shape
            ddx, ddy, smooth = _derivs(shape, grid_step=0.03 if seed % 2 else 0.25)
            zi = (np.random.default_rng(seed).random(shape) * 3000).astype(np.float32)
            zi.ravel()[::7] = np.nan
            for hub in (True, False):
                py, cx = profiles(c, hub)
                for kw in ({}, {'smooth': smooth}, {'smooth': smooth, 'zi': zi}, {'zi': zi, 'zmax': 6000.0, 'dz': 250.0}, {'zmin_agl': 0.0, 'zmax': 1000.0}):
                    a = P.ellrod_turbulence(py, ddx, ddy, **kw)
                    b = N.ellrod_turbulence(cx, ddx, ddy, **kw)
                    self.assertSame(a, b, f'seed {seed} hub {hub} {sorted(kw)}')

    def test_other_inputs_fall_back(self):
        c = columns(seed=11)
        order = np.argsort(-c['p'])
        z, t, td = (c['z'][order], c['t'][order], c['td'][order])
        rh = P.rh_from_td(t, td)
        for args in ((z.astype(np.float64), t, rh, None, c['zs']), (z, t, rh, None, c['zs'], 7300.1), (z, t, rh.astype(np.float64), None, c['zs']), (z[:, :, ::2], t[:, :, ::2], rh[:, :, ::2], None, c['zs'][:, ::2])):
            self.assertSame(P.icing_potential(*args), N.icing_potential(*args))
        c64 = {k: v.astype(np.float64) for k, v in c.items()}
        py, cx = profiles(c64)
        self.assertFalse(cx.native)
        self.assertSame(P.freezing_level(py), N.freezing_level(cx))
        ddx, ddy, smooth = _derivs(c['zs'].shape)
        self.assertSame(P.ellrod_turbulence(py, ddx, ddy, smooth), N.ellrod_turbulence(cx, ddx, ddy, smooth))

@unittest.skipUnless(N.available(), f'C++ physics not available: {N.error}')
class NativeWave(unittest.TestCase):

    def check(self, c, hub=True):
        py, cx = profiles(c, hub)
        a, b = (P.wave_environment(py), N.wave_environment(cx))
        for x, y, name in zip(a, b, ('score', 'wavelength')):
            self.assertTrue(same(x, y), name)
        return a

    def test_bit_for_bit(self):
        for seed in range(6):
            score, _ = self.check(columns(seed=seed, shuffle=seed % 3 == 2), hub=seed % 2 == 0)
            self.assertTrue(np.isfinite(score).any())

    def test_small_grids_and_chunks(self):
        for ny, nx in ((1, 1), (1, 5), (2, 3)):
            self.check(columns(ny=ny, nx=nx, seed=ny * 10 + nx, nan_cells=False))
        old = N._WAVE_CHUNK
        try:
            N._WAVE_CHUNK = 7
            self.check(columns(ny=9, nx=11, seed=3))
        finally:
            N._WAVE_CHUNK = old

    def test_strong_and_calm_winds(self):
        c = columns(seed=8)
        for k in ('u', 'v'):
            c[k] = (c[k] * 6).astype(np.float32)
        self.check(c)
        c = columns(seed=9)
        for k in ('u', 'v', 'u10', 'v10', 'uh', 'vh'):
            c[k] = np.zeros_like(c[k])
        self.check(c)

    def test_python_profile_falls_back(self):
        c = columns(seed=4)
        py, _ = profiles(c)
        a, b = (P.wave_environment(py), N.wave_environment(py))
        self.assertTrue(same(a[0], b[0]) and same(a[1], b[1]))

@unittest.skipUnless(N.available(), f'C++ physics not available: {N.error}')
class NativeTiles(unittest.TestCase):

    def codes(self, shape, seed=0):
        rng = np.random.default_rng(seed)
        smooth = np.cumsum(rng.integers(-3, 4, shape), axis=-1) % 255
        c = np.where(rng.random(shape) < 0.5, smooth, rng.integers(0, 256, shape)).astype(np.uint8)
        c[..., :shape[-2] // 3, :shape[-1] // 4] = 255
        c[..., -2:, :] = 254
        return c

    def test_downsample(self):
        import tiles
        for shape, k in (((3, 37, 53), 2), ((2, 41, 30), 4), ((17, 9), 4), ((1, 1), 2), ((5, 4, 4), 2)):
            c = self.codes(shape, seed=k + shape[-1])
            for none, rms in ((None, False), (254, False), (None, True), (254, True)):
                got, ref = (tiles.downsample(c, k, none=none, rms=rms), tiles.downsample_np(c, k, none, rms))
                self.assertEqual(got.dtype, np.uint8)
                np.testing.assert_array_equal(got, ref, err_msg=f'{shape} k={k} none={none} rms={rms}')
        self.assertIsNotNone(tiles._fns.get('cx_downsample'), 'the C++ was used, not the numpy fallback')

    def test_png(self):
        import io
        import tiles
        from PIL import Image
        for shape in ((300, 100), (1, 1), (7, 13), (2400, 100)):
            img = self.codes(shape, seed=shape[0])
            np.testing.assert_array_equal(tiles.png_filter(img), tiles.png_filter_np(img))
            self.assertIsNotNone(tiles._fns.get('cx_png_filter'))
            if tiles.deflate is not None:
                np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(tiles.png_bytes(img)))), img)

    def test_savez(self):
        import tempfile
        import tiles
        arrays = {'wstar': self.codes((40, 60)), 'u_a2000': self.codes((40, 60), 1), 'empty': np.zeros((0,), np.uint8)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'h.npz')
            tiles.savez(path, arrays)
            with np.load(path) as z:
                self.assertEqual(z.files, list(arrays))
                for k, v in arrays.items():
                    np.testing.assert_array_equal(z[k], v)
if __name__ == '__main__':
    unittest.main()
