import copy
import hashlib
import json
import unittest
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import aircorr
import blend as B
import build
import sky
MEMBERS = ['hrrr', 'gfs', 'ecmwf', 'icon']
BASE_REGION = (-100.0, 35.0, -98.5, 36.5, 0.03)
RUN = datetime(2026, 7, 1, 12)
VALID = datetime(2026, 7, 1, 19)
BINS = {'lead': [0, 6, 12, 24, 48, 72], 'solar': ['night', [0, 9], [9, 12], [12, 15], [15, 18], [18, 24]], 'regime': list(B.REGIMES)}
CU_CODE, BLUE_CODE = (100, 50)

def tree(feat, thr, left, right):
    return [[feat, thr, 1, 1, 2], [left], [right]]

def random_trees(rng, n_feat, n_trees=6, depth=3):
    trees = []
    for _ in range(n_trees):
        nodes = []

        def grow(d):
            i = len(nodes)
            nodes.append(None)
            if d == depth:
                nodes[i] = [float(rng.normal(0, 0.3))]
                return i
            f = int(rng.integers(n_feat))
            thr = float(rng.uniform(-5, 40))
            l = grow(d + 1)
            r = grow(d + 1)
            nodes[i] = [f, thr, int(rng.integers(2)), l, r]
            return i
        grow(0)
        trees.append(nodes)
    return trees

def make_bundle(seed=1, n_min=500, low=None):
    rng = np.random.default_rng(seed)
    nc = B.n_cells(BINS)
    fields = {}
    kinds = {'t2': ('add', 3.0), 'td2': ('add', 3.0), 'wind': ('log', [0.5, 2.0]), 'cu': ('logit', 3.0), 'cubase': ('add', 600.0)}
    for f, (kind, clamp) in kinds.items():
        table = {}
        for s in B.subsets(MEMBERS):
            table[B.subset_key(s)] = rng.dirichlet(np.ones(len(s)), nc).tolist()
        n = {m: [int(x) for x in rng.integers(n_min, 5 * n_min + 1, nc)] for m in MEMBERS}
        for m, c in (low or {}).items():
            n[m] = [c] * nc
        feats = ['x', 'sun_elev', 'dz_terrain', 'spread_t2', 'regime', 'lead_h', 'run_age_h']
        shift = {'t2': (0.8, -0.4), 'td2': (-0.5, 0.6), 'wind': (0.1, -0.15), 'cu': (0.5, -0.5), 'cubase': (150.0, -100.0)}[f]
        debias = {m: {'features': feats, 'base': 0.0, 'trees': [tree(0, {'t2': 25.0, 'td2': 15.0, 'wind': 5.0, 'cu': 0.5, 'cubase': 1800.0}[f], *shift)] + random_trees(rng, len(feats), 3, 2)} for m in ('gfs', 'ecmwf')}
        fields[f] = {'kind': kind, 'features': feats, 'clamp': clamp, 'debias': debias, 'weights': {'bins': copy.deepcopy(BINS), 'table': table, 'n_min': n_min, 'n': n}, 'gate': None, 'spread': {'features': ['x', 'lead_h'], 'base': 1.0, 'trees': [tree(0, 20.0, 0.2, -0.2)]}, 'pass': True}
    return {'schema': B.SCHEMA, 'id': 'bl1-20261123-0a1b2c3d', 'code': 'test', 'pairs': 'p1', 'regions': {'conus': {'base': 'hrrr', 'members': MEMBERS, 'grid_step': 0.125, 'max_age_h': {'gfs': 12, 'ecmwf': 22, 'icon': 12}, 'groups': {'surface': ['t2', 'td2', 'wind'], 'cumulus': ['cubase', 'cu']}, 'fields': fields}}, 'cv': {}}

def codes_of(v):
    return {k: build.quantize(a, *build.ENC[k]) for k, a in v.items()}

def hour_values(lat, lon, rng, offset=0.0, cu_shift=0.0):
    t2 = 24 + offset + 3 * np.sin(lat * 3) + rng.normal(0, 0.5, lat.shape)
    cu = lat + 0.2 * np.sin(lon * 5) > 35.75 + cu_shift
    v = {'t2': t2, 'td2': t2 - 9 - rng.uniform(0, 2, lat.shape), 'u_sfc': 4 + offset + rng.normal(0, 1, lat.shape), 'v_sfc': 1 + rng.normal(0, 1, lat.shape), 'wstar': np.full(lat.shape, 2.0 + 0.1 * offset), 'zi': np.full(lat.shape, 1600.0), 'depth': np.full(lat.shape, 1500.0), 'cubase': np.where(cu, 1400 + 50 * offset + rng.normal(0, 40, lat.shape), np.nan), 'cucover': np.where(cu, 0.3, 0.0), 'elevation': 400 + 60 * np.cos(lon * 4) + 20 * offset}
    q = codes_of(v)
    q['cuclass'] = np.where(cu, CU_CODE, BLUE_CODE).astype(np.uint8)
    return q

def native_grid(step, rng, offset, cu_shift=0.0, pad=0.5):
    w, s, e, n = BASE_REGION[:4]
    lats = np.arange(n + pad, s - pad - step / 2, -step)
    lons = np.arange(w - pad, e + pad + step / 2, step)
    lat, lon = (np.repeat(lats[:, None], len(lons), 1), np.repeat(lons[None, :], len(lats), 0))
    return (lats, lons, hour_values(lat, lon, rng, offset, cu_shift))

def base_hour(rng, **kw):
    lat, lon = B.grid2d(BASE_REGION)
    q = hour_values(lat, lon, rng)
    q['cudepth'] = build.quantize(np.where(q['cubase'] != 255, 800.0, np.nan), *build.ENC['cudepth'])
    q['overdev'] = build.quantize(np.zeros(lat.shape), *build.ENC['overdev'])
    q['cusky'] = np.full(lat.shape, 2, np.uint8)
    q['ptype'] = np.full(lat.shape, 7, np.uint8)
    q['spot1'] = np.full(lat.shape, 9, np.uint8)
    q['fltcat'] = np.full(lat.shape, 1, np.uint8)
    h = {'model': 'hrrr', 'run': RUN, 'valid': VALID, 'region': BASE_REGION, 'codes': q, 'aircorr': False}
    h.update(kw)
    return h

def members(rng, which=('gfs', 'ecmwf', 'icon'), runs=None):
    wreg = B.weight_region(BASE_REGION, 0.125)
    spec = {'gfs': (0.25, 1.5, 0.1), 'ecmwf': (0.1, -1.0, -0.1), 'icon': (0.125, 0.5, 0.05)}
    out = {}
    for m in which:
        step, off, cs = spec[m]
        lats, lons, q = native_grid(step, rng, off, cs)
        run = (runs or {}).get(m, RUN - timedelta(hours=6))
        out[m] = B.member(m, run, q, lats, lons, BASE_REGION, wreg, aircorr_on=False)
    return out

class Bundle(unittest.TestCase):

    def test_a_good_bundle_loads_and_its_sha_is_checked(self):
        data = json.dumps(make_bundle()).encode()
        b = B.load(data, hashlib.sha256(data).hexdigest())
        T = b['regions']['conus']['fields']['t2']['weights']['_T']
        self.assertEqual(T.shape, (16, B.n_cells(BINS), 4))
        with self.assertRaises(B.BundleError):
            B.load(data, '0' * 64)

    def test_bad_bundles_are_refused(self):

        def bad(change):
            b = make_bundle()
            change(b)
            with self.assertRaises(B.BundleError):
                B.validate(b)
        f = lambda b: b['regions']['conus']['fields']['t2']
        bad(lambda b: b.update(schema='blend-bundle/2'))
        bad(lambda b: b.update(id='bl1-2026-x'))
        bad(lambda b: f(b)['weights']['table'].pop('gfs+icon'))
        bad(lambda b: f(b)['weights']['table']['gfs+icon'][3].__setitem__(0, -0.1))
        bad(lambda b: f(b)['weights']['table']['hrrr+gfs'][0].__setitem__(0, 0.9))
        bad(lambda b: f(b).update(gate={'trees': []}))
        bad(lambda b: f(b)['debias']['gfs']['features'].append('doy'))
        bad(lambda b: f(b)['weights']['n'].pop('icon'))
        bad(lambda b: f(b)['weights']['bins'].update(regime=['blue', 'cumulus']))
        bad(lambda b: f(b).update(kind='mul'))
        bad(lambda b: b['regions']['conus'].update(base='nam'))
        bad(lambda b: f(b)['debias']['gfs']['trees'][0][0].__setitem__(3, 99))
        bad(lambda b: f(b)['weights']['bins']['solar'].__setitem__(2, 9))
        bad(lambda b: f(b)['debias']['gfs']['trees'][0][0].__setitem__(3, 0))
        bad(lambda b: f(b)['debias']['gfs']['trees'][0][1].__setitem__(0, float('nan')))
        bad(lambda b: f(b)['debias']['gfs']['features'].append('cape'))
        bad(lambda b: f(b)['weights']['n']['gfs'].__setitem__(0, 1.5))
        bad(lambda b: f(b)['weights']['n']['gfs'].__setitem__(0, -1))
        bad(lambda b: b['regions']['conus']['max_age_h'].update(gfs='soon'))
        bad(lambda b: b['regions']['conus']['groups'].update(surface=['t2', 'mslp']))
        bad(lambda b: f(b).update(clamp=None))
        bad(lambda b: f(b).update(clamp=[3.0, -3.0]))
        bad(lambda b: f(b).update(clamp=[0.5, 2.0]))
        bad(lambda b: f(b).update(clamp=-1.0))
        bad(lambda b: b['regions']['conus']['fields']['cu'].update(clamp=[0.3, 0.7]))
        bad(lambda b: b['regions']['conus']['fields']['wind'].update(clamp=[1.2, 2.0]))
        with self.assertRaises(B.BundleError):
            b = make_bundle()
            f(b)['weights']['n']['gfs'][0] = 'many'
            B.load(json.dumps(b).encode())

class Weights(unittest.TestCase):

    def setUp(self):
        self.b = B.prepare(B.validate(make_bundle()))
        self.rng = np.random.default_rng(5)

    def cond_and_members(self, which=('gfs', 'ecmwf', 'icon'), runs=None):
        r = self.b['regions']['conus']
        mem = members(self.rng, which, runs)
        bh = base_hour(self.rng)
        lat, lon = B.grid2d(BASE_REGION)
        wreg = B.weight_region(BASE_REGION, 0.125)
        mem['hrrr'] = {'model': 'hrrr', 'run': RUN, 'codes': bh['codes'], 'wcodes': B.regrid_codes(bh['codes'], B.regridder(lat, lon, wreg))}
        return (B.conditions(mem, VALID, RUN, wreg, r), mem)

    def test_they_sum_to_one_and_are_never_negative(self):
        cond, mem = self.cond_and_members()
        for f in B.FIELDS:
            w = B.weights(mem, cond, self.b, f)
            st = np.stack([w[m] for m in MEMBERS])
            self.assertTrue((st >= 0).all(), f)
            tot = st.sum(0)
            ok = tot > 0
            self.assertTrue(ok.any(), f)
            np.testing.assert_allclose(tot[ok], 1.0, atol=1e-12)

    def test_an_absent_member_takes_the_subsets_own_row(self):
        cond, mem = self.cond_and_members(which=('gfs', 'icon'))
        wt = self.b['regions']['conus']['fields']['t2']['weights']
        w = B.weights(mem, cond, self.b, 't2')
        cells = B.cell_index(cond, BINS)
        y, x = (3, 4)
        row = wt['table']['hrrr+gfs+icon'][cells[y, x]]
        self.assertEqual(w['ecmwf'][y, x], 0.0)
        np.testing.assert_allclose([w['hrrr'][y, x], w['gfs'][y, x], w['icon'][y, x]], row, rtol=1e-12)

    def test_too_old_and_under_n_min_are_absent(self):
        cond, mem = self.cond_and_members(runs={'gfs': RUN - timedelta(hours=13)})
        self.assertNotIn('gfs', cond['members'])
        w = B.weights(mem, cond, self.b, 'td2')
        self.assertTrue((w['gfs'] == 0).all())
        b = B.prepare(B.validate(make_bundle(low={'ecmwf': 10})))
        cond, mem = self.cond_and_members()
        w = B.weights(mem, cond, b, 't2')
        self.assertTrue((w['ecmwf'] == 0).all())
        cells = B.cell_index(cond, BINS)
        np.testing.assert_allclose([w['hrrr'][0, 0], w['gfs'][0, 0], w['icon'][0, 0]], b['regions']['conus']['fields']['t2']['weights']['table']['hrrr+gfs+icon'][cells[0, 0]], rtol=1e-12)

    def test_the_cells_bins(self):
        cond, _ = self.cond_and_members()
        self.assertAlmostEqual(cond['lead_h'], 7.0)
        si = B.sun_band(np.array([13.0, 13.0, 7.0]), np.array([60.0, -5.0, 20.0]), BINS['solar'])
        self.assertEqual(si.tolist(), [3, 0, 1])
        self.assertTrue((cond['regime'] >= 0).all())
        self.assertEqual(set(np.unique(cond['regime'])) - {0, 1}, set(), 'blue south, cumulus north')

    def test_regimes_from_the_codes(self):
        lat, lon = B.grid2d(BASE_REGION)
        q = hour_values(lat, lon, np.random.default_rng(0))
        q['u_sfc'][0, 0] = build.quantize(np.array(12.0), *build.ENC['u_sfc'])
        q['cuclass'][0, 1] = sky.CLASS_CODE0 + sky.CB - sky.CB
        q['cuclass'][0, 2] = sky.CLASS_CODE0 + sky.SC - sky.CB
        q['cuclass'][0, 3] = sky.CLASS_CODE0 + sky.AS - sky.CB
        q['wstar'][0, 4] = 255
        r = B.regime_of(q)
        self.assertEqual(r[0, :5].tolist(), [B.REGIMES.index(n) for n in ('wind', 'overdeveloping', 'stratiform', 'frontal')] + [-1])

    def test_every_class_has_a_regime(self):
        codes = np.arange(sky.CLASS_CODE0 + sky.ACC - sky.CB + 1, dtype=np.uint8)[None, :]
        q = {'wstar': build.quantize(np.full(codes.shape, 1.5), *build.ENC['wstar']), 'cuclass': codes, 'cubase': build.quantize(np.full(codes.shape, 1200.0), *build.ENC['cubase'])}
        r = B.regime_of(q)
        self.assertTrue((r >= 0).all(), codes[r < 0].tolist())

        def code(c):
            return np.array([[sky.CLASS_CODE0 + c - sky.CB]], np.uint8)
        for c, name in ((sky.SHOWERS, 'overdeveloping'), (sky.NS, 'frontal'), (sky.CB, 'overdeveloping')):
            self.assertEqual(int(B.regime_of({**{k: v[:, :1] for k, v in q.items()}, 'cuclass': code(c)})[0, 0]), B.REGIMES.index(name), c)

    def test_times_are_utc_and_the_issue_time_counts(self):
        ny = datetime(2026, 7, 1, 15, tzinfo=ZoneInfo('America/New_York'))
        wreg = B.weight_region(BASE_REGION, 0.125)
        r = self.b['regions']['conus']
        cond, mem = self.cond_and_members()
        a = B.conditions(mem, VALID, RUN, wreg, r)
        b = B.conditions(mem, ny, RUN, wreg, r)
        np.testing.assert_allclose(a['solar_h'], b['solar_h'])
        np.testing.assert_allclose(a['sun_elev'], b['sun_elev'])
        self.assertAlmostEqual(a['lead_h'], b['lead_h'])
        c = B.conditions(mem, VALID, RUN + timedelta(hours=3), wreg, r)
        self.assertAlmostEqual(c['lead_h'], 4.0)
        cond, mem = self.cond_and_members(runs={'gfs': RUN - timedelta(hours=10)})
        self.assertIn('gfs', B.available(mem, RUN, r))
        self.assertNotIn('gfs', B.available(mem, RUN + timedelta(hours=3), r))

class Blend(unittest.TestCase):

    def setUp(self):
        self.b = B.prepare(B.validate(make_bundle()))

    def run_blend(self, seed=7, b=None, native=True, **kw):
        rng = np.random.default_rng(seed)
        mem = members(rng)
        bh = base_hour(rng, **kw)
        return (bh, mem, B.blend(bh, mem, b or self.b, native=native))

    def test_it_stays_inside_the_debiased_members(self):
        bh, mem, res = self.run_blend()
        self.assertEqual(set(res['fields_blended']), set(B.FIELDS))
        for f in ('t2', 'td2', 'wind', 'cubase'):
            xs = np.stack([a for a in res['debiased'][f].values()])
            v = res['values'][f]
            ok = np.isfinite(v) & (res['base_weights'][f].sum(0) > 0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                lo, hi = (np.nanmin(xs, 0), np.nanmax(xs, 0))
            self.assertTrue(ok.any(), f)
            self.assertTrue((v[ok] >= lo[ok] - 1e-09).all() and (v[ok] <= hi[ok] + 1e-09).all(), f)
            k = {'wind': None, 'cubase': None}.get(f, f)
            if k:
                step = (build.ENC[k][1] - build.ENC[k][0]) / 254
                d = B._dec(res['codes'], k)
                okc = ok & np.isfinite(d)
                self.assertTrue((d[okc] <= hi[okc] + step).all(), f)
                if f == 't2':
                    self.assertTrue((d[okc] >= lo[okc] - step).all(), f)
        u, v = (B._dec(res['codes'], k) for k in ('u_sfc', 'v_sfc'))
        np.testing.assert_allclose(np.hypot(u, v), res['values']['wind'], atol=0.5)
        for f, W in res['base_weights'].items():
            t = W.sum(0)
            np.testing.assert_allclose(t[t > 0], 1.0, atol=1e-12)
            self.assertTrue((W >= 0).all())

    def test_the_codes_round_trip(self):
        bh, mem, res = self.run_blend()
        for f, k in (('t2', 't2'), ('td2', 'td2')):
            step = (build.ENC[k][1] - build.ENC[k][0]) / 254
            d = B._dec(res['codes'], k)
            v = res['values'][f]
            ok = np.isfinite(v) & (v > build.ENC[k][0]) & (v < build.ENC[k][1])
            if k == 'td2':
                ok &= B._dec(res['codes'], 't2') > v + 0.5
            self.assertLessEqual(np.max(np.abs(d[ok] - v[ok])), step / 2 + 0.0001, f)
        b = make_bundle()
        for f in b['regions']['conus']['fields'].values():
            f['debias'] = {}
        b = B.prepare(B.validate(b))
        rng = np.random.default_rng(2)
        bh = base_hour(rng)
        lat, lon = B.grid2d(BASE_REGION)
        wreg = B.weight_region(BASE_REGION, 0.125)
        same = {m: B.member(m, RUN, {k: v for k, v in bh['codes'].items() if k in B.INPUTS}, lat[:, 0], lon[0], BASE_REGION, wreg, aircorr_on=False) for m in ('gfs', 'ecmwf')}
        res = B.blend(bh, same, b)
        for k in ('t2', 'td2', 'cubase'):
            self.assertTrue(np.array_equal(res['codes'][k], bh['codes'][k]), k)
        for k in ('u_sfc', 'v_sfc'):
            self.assertLessEqual(int(np.abs(res['codes'][k].astype(int) - bh['codes'][k]).max()), 1, k)

    def test_the_class_fields(self):
        bh, mem, res = self.run_blend()
        q = res['codes']
        for k in ('ptype', 'spot1', 'fltcat'):
            self.assertTrue(np.array_equal(q[k], bh['codes'][k]), k)
        has = res['values']['cu'] >= B.CU_AT
        self.assertTrue(np.array_equal(q['cubase'] != 255, has), 'a cloud base just where the blend has cumulus')
        W = res['base_weights']['cu']
        allm = {**mem, 'hrrr': {'codes': bh['codes']}}
        for y, x in [(0, 0), (10, 10), (25, 30), (49, 49)]:
            agree = [(W[i, y, x], m) for i, m in enumerate(MEMBERS) if m in allm and (allm[m]['codes']['cubase'][y, x] != 255) == has[y, x] and (W[i, y, x] > 0)]
            if agree:
                lead = max(agree)[1]
                self.assertEqual(q['cuclass'][y, x], allm[lead]['codes']['cuclass'][y, x], (y, x))
        self.assertTrue(set(np.unique(q['cuclass'])) <= {CU_CODE, BLUE_CODE} | set(np.unique(sky.cloud_overview(q, build.ENC, True)['cuclass'])))

    def test_an_air_corrected_member_is_refused(self):
        rng = np.random.default_rng(1)
        mem = members(rng)
        mem['gfs']['aircorr'] = True
        del mem['icon']['aircorr']
        res = B.blend(base_hour(rng), mem, self.b)
        self.assertEqual(sorted(res['refused']), ['gfs', 'icon'])
        self.assertNotIn('gfs', [m['model'] for m in res['members']])
        for W in res['base_weights'].values():
            self.assertTrue((W[MEMBERS.index('gfs')] == 0).all() and (W[MEMBERS.index('icon')] == 0).all())
        with self.assertRaises(B.Refused) as e:
            B.blend(base_hour(rng, aircorr=True), members(rng), self.b)
        self.assertEqual(e.exception.model, 'hrrr')
        bh = base_hour(rng)
        del bh['aircorr']
        with self.assertRaises(B.Refused):
            B.blend(bh, members(rng), self.b)
        with self.assertRaises(TypeError):
            B.member('gfs', RUN, {}, [36.0], [-99.0], BASE_REGION, B.weight_region(BASE_REGION, 0.125))

    def test_rain_cloud_still_blends(self):
        rng = np.random.default_rng(8)
        mem = members(rng)
        bh = base_hour(rng)
        for c, h in ((sky.SHOWERS, bh), (sky.NS, mem['gfs']), (sky.SHOWERS, mem['ecmwf']), (sky.NS, mem['icon'])):
            for k in ('codes', 'wcodes'):
                if k in h:
                    h[k]['cuclass'] = np.full(h[k]['cuclass'].shape, sky.CLASS_CODE0 + c - sky.CB, np.uint8)
        res = B.blend(bh, mem, self.b)
        self.assertEqual(set(res['fields_blended']), set(B.FIELDS))
        self.assertTrue((res['base_weights']['t2'].sum(0) > 0).all())
        self.assertFalse(np.array_equal(res['codes']['t2'], bh['codes']['t2']))

    def test_no_weight_anywhere_is_the_bases(self):
        b = make_bundle(low={m: 10 for m in MEMBERS})
        b['regions']['conus']['fields']['t2']['weights']['n'] = make_bundle()['regions']['conus']['fields']['t2']['weights']['n']
        b = B.prepare(B.validate(b))
        bh, mem, res = self.run_blend(b=b)
        self.assertEqual(res['fields_blended'], ['t2'])
        self.assertEqual(res['fields_from_base']['wind'], 'no member had a weight anywhere')
        for k in ('u_sfc', 'v_sfc', 'cubase'):
            self.assertTrue(np.array_equal(res['codes'][k], bh['codes'][k]), k)

    def test_where_no_member_is_left_the_wind_is_the_bases(self):
        b = make_bundle()
        wt = b['regions']['conus']['fields']['wind']['weights']
        ri = B.REGIMES.index('cumulus')
        for m in MEMBERS:
            wt['n'][m] = [5 if c % len(B.REGIMES) == ri else n for c, n in enumerate(wt['n'][m])]
        b = B.prepare(B.validate(b))
        bh, mem, res = self.run_blend(b=b)
        none = res['base_weights']['wind'].sum(0) == 0
        self.assertTrue(none.any() and (~none).any())
        for k in ('u_sfc', 'v_sfc'):
            self.assertLessEqual(int(np.abs(res['codes'][k][none].astype(int) - bh['codes'][k][none]).max()), 1, k)
        u, v = (B._dec(res['codes'], k) for k in ('u_sfc', 'v_sfc'))
        np.testing.assert_allclose(np.hypot(u, v)[none], res['values']['wind'][none], atol=0.5)

    def test_the_cloud_base_stands_on_the_bases_ground(self):
        b = make_bundle()
        for f in b['regions']['conus']['fields'].values():
            f['debias'] = {}
        b = B.prepare(B.validate(b))
        rng = np.random.default_rng(2)
        bh = base_hour(rng)
        lat, lon = B.grid2d(BASE_REGION)
        wreg = B.weight_region(BASE_REGION, 0.125)
        same = {m: B.member(m, RUN, {k: v for k, v in bh['codes'].items() if k in B.INPUTS}, lat[:, 0], lon[0], BASE_REGION, wreg, aircorr_on=False) for m in ('gfs', 'ecmwf')}
        bh['dem'] = B._dec(bh['codes'], 'elevation') + 200 + 50 * np.sin(lon * 7)
        res = B.blend(bh, same, b)
        self.assertTrue(np.array_equal(res['codes']['cubase'], bh['codes']['cubase']))

    def test_td2_held_under_t2_and_said_so(self):
        b = make_bundle()
        b['regions']['conus']['fields']['td2']['pass'] = False
        b = B.prepare(B.validate(b))
        rng = np.random.default_rng(9)
        mem = members(rng)
        bh = base_hour(rng)
        bh['codes']['td2'] = np.minimum(bh['codes']['td2'].astype(int) + 60, 254).astype(np.uint8)
        res = B.blend(bh, mem, b)
        t2, td2 = (B._dec(res['codes'], 't2'), B._dec(res['codes'], 'td2'))
        self.assertTrue((td2 <= t2 + 1e-06).all())
        self.assertIn('held under the blended t2', res['fields_from_base']['td2'])

    def test_a_failing_cloud_base_follows_the_blended_presence(self):
        b = make_bundle()
        b['regions']['conus']['fields']['cubase']['pass'] = False
        b = B.prepare(B.validate(b))
        bh, mem, res = self.run_blend(b=b)
        self.assertIn('cu', res['fields_blended'])
        self.assertIn('blended cumulus presence', res['fields_from_base']['cubase'])
        has = res['values']['cu'] >= B.CU_AT
        self.assertTrue(np.array_equal(res['codes']['cubase'] != 255, has))
        both = has & (bh['codes']['cubase'] != 255)
        self.assertTrue(np.array_equal(res['codes']['cubase'][both], bh['codes']['cubase'][both]))

    def test_no_members_wave_bars_or_castellanus(self):
        rng = np.random.default_rng(7)
        mem = members(rng)
        bh = base_hour(rng)
        acc = sky.CLASS_CODE0 + sky.ACC - sky.CB
        for m in ('gfs', 'ecmwf', 'icon'):
            mem[m]['codes']['cuclass'][:, :5] = acc
            mem[m]['codes']['cuclass'][:, 5:10] = sky.WB_CODE
        bh['codes']['cuclass'][:3, 20:25] = sky.WB_CODE
        res = B.blend(bh, mem, self.b)
        q = res['codes']['cuclass']
        self.assertFalse((q[:, :10] == acc).any() or (q[3:, 5:10] == sky.WB_CODE).any())
        self.assertTrue((q[:3, 20:25] == sky.WB_CODE).all())

    def test_a_spread_coded_on_its_fields_range(self):
        b = make_bundle()
        b['regions']['conus']['fields']['cubase']['spread'] = None
        b['regions']['conus']['fields']['cu']['spread']['base'] = 0.3
        b = B.prepare(B.validate(b))
        bh, mem, res = self.run_blend(b=b)
        d = res['weights']['cumulus']
        self.assertEqual(d['bsprd_field'], 'cu')
        c = B.weight_codes(res['weights'])['bsprd_cumulus']
        np.testing.assert_allclose(B.sky.decode(c, *B.SPREAD_ENC['cu'])[np.isfinite(d['bsprd'])], d['bsprd'][np.isfinite(d['bsprd'])], atol=1 / 254)
        self.assertTrue((c[np.isfinite(d['bsprd'])] > 0).any())

    def test_a_failing_field_is_the_bases(self):
        b = make_bundle()
        b['regions']['conus']['fields']['wind']['pass'] = False
        b = B.prepare(B.validate(b))
        rng = np.random.default_rng(3)
        mem = members(rng)
        mem['ecmwf']['codes']['td2'] = mem['ecmwf']['codes']['td2'][:-1]
        bh = base_hour(rng)
        res = B.blend(bh, mem, b)
        self.assertIn('wind', res['fields_from_base'])
        self.assertIn('td2', res['fields_from_base'])
        self.assertIn('t2', res['fields_blended'])
        for k in ('u_sfc', 'v_sfc'):
            self.assertTrue(np.array_equal(res['codes'][k], bh['codes'][k]), k)
        t2, td2 = (B._dec(res['codes'], 't2'), B._dec(bh['codes'], 'td2'))
        keep = td2 <= t2
        self.assertTrue(np.array_equal(res['codes']['td2'][keep], bh['codes']['td2'][keep]))

    def test_an_absent_member_masked_at_the_base_cells(self):
        rng = np.random.default_rng(4)
        mem = members(rng)
        mem['icon']['codes']['t2'][:10] = 255
        res = B.blend(base_hour(rng), mem, self.b)
        W = res['base_weights']['t2']
        self.assertTrue((W[MEMBERS.index('icon'), :10] == 0).all())
        np.testing.assert_allclose(W[:, :10].sum(0), 1.0, atol=1e-12)

    def test_the_weight_fields(self):
        bh, mem, res = self.run_blend()
        wf = res['weights']
        self.assertEqual(set(wf), {'surface', 'cumulus'})
        for g, d in wf.items():
            st = np.stack([d['bw'][m] for m in MEMBERS])
            np.testing.assert_allclose(st.sum(0)[st.sum(0) > 0], 1.0, atol=1e-09)
            self.assertTrue(np.array_equal(d['blead'], st.argmax(0)))
            self.assertTrue(np.isfinite(d['bsprd']).any())
        c = B.weight_codes(wf)
        self.assertIn('bw_surface_ecmwf', c)
        self.assertEqual(c['blead_cumulus'].dtype, np.uint8)
        self.assertEqual(c['bw_surface_hrrr'].shape, (13, 13))

    def test_cpp_trees_are_numpys(self):
        if aircorr._native() is None:
            self.skipTest('no C++ library here')
        a = self.run_blend(native=True)[2]
        b = self.run_blend(native=False)[2]
        for k in a['codes']:
            self.assertTrue(np.array_equal(a['codes'][k], b['codes'][k]), k)
        for f in a['values']:
            self.assertTrue(np.array_equal(np.nan_to_num(a['values'][f]), np.nan_to_num(b['values'][f])), f)

    def test_the_debias_kinds(self):
        x = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(B.debias('add', x, np.array([5.0, -0.5, 0.0]), 3.0), [4.0, 1.5, 4.0])
        np.testing.assert_allclose(B.debias('log', x, np.log(np.array([4.0, 1.0, 0.1])), [0.5, 2.0]), [2.0, 2.0, 2.0])
        p = B.debias('logit', np.array([0.0, 1.0]), np.array([0.0, 0.0]), 3.0)
        np.testing.assert_allclose(p, [B.CU_EPS, 1 - B.CU_EPS])
if __name__ == '__main__':
    unittest.main()
