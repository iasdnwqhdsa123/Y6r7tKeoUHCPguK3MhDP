import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
import build
import opslake
import tiles
import verify
UTC = timezone.utc
REGION = (-135.0, 40.0, -110.0, 58.0, 0.5)
RUN = '2026062106'
STATIONS = [['KAAA', 45.0, -120.0, 500], ['KBBB', 52.5, -127.0, 100], ['KCCC', 41.0, -112.5, 1200], ['KDDD', 30.0, -100.0, 10], ['KEEE', 47.3, -115.2, 800]]
FIELDS = sorted(set(opslake.STATION_FIELDS + opslake.SLIM_FIELDS + opslake.BLEND_FIELDS) - {'overdev_nb'})

def made_up_store(hours_dir, rng, own_leads, borrowed, unstamped=()):
    os.makedirs(hours_dir)
    r0 = opslake.parse_run(RUN)
    lons, lats = build.target_grid(REGION)
    ny, nx = (len(lats), len(lons))
    cy, cx = (round((REGION[3] - 41.0) / REGION[4]), round((-112.5 - REGION[0]) / REGION[4]))
    assert (lats[cy], lons[cx]) == (41.0, -112.5)
    elev = rng.integers(0, 255, (ny, nx), dtype=np.uint8)
    for lead, stamp in [(ld, RUN) for ld in own_leads] + [(ld, '2026062100') for ld in borrowed] + [(ld, None) for ld in unstamped]:
        v = r0 + timedelta(hours=lead)
        q = {k: rng.integers(0, 255, (ny, nx), dtype=np.uint8) for k in FIELDS}
        q['t2'][cy, cx] = 255
        q['t2'][:3, -4:] = 255
        q['overdev'][rng.random((ny, nx)) < 0.2] = 255
        q['elevation'] = elev
        tiles.savez(os.path.join(hours_dir, f'{v:%Y%m%d%H}.npz'), q)
        if stamp:
            with open(os.path.join(hours_dir, f'{v:%Y%m%d%H}.run'), 'w') as f:
                f.write(stamp)
    return elev

class OpsRecord(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.hours = os.path.join(cls.tmp, 'gfs', '_hours')
        cls.own = [*range(0, 60), *range(60, 130, 3)]
        cls.borrowed = [130, 133, 136]
        cls.elev = made_up_store(cls.hours, np.random.default_rng(1), cls.own, cls.borrowed, unstamped=[139])
        cls.out = os.path.join(cls.tmp, 'out')
        os.environ.pop('BLEND_IN', None)
        cls.done = opslake.write(cls.hours, RUN, cls.out, build_id='123-1', published=True, region=REGION, stations=STATIONS, published_t='2026-06-21T09:41:00Z')
        cls.base = os.path.join(cls.out, 'ops', 'gfs', '20260621', '06', 'b123-1')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def hour(self, lead):
        v = opslake.parse_run(RUN) + timedelta(hours=lead)
        return np.load(os.path.join(self.hours, f'{v:%Y%m%d%H}.npz'))

    def leads(self, stamps):
        r0 = opslake.parse_run(RUN)
        return [int((datetime.strptime(str(s), '%Y%m%d%H').replace(tzinfo=UTC) - r0).total_seconds() // 3600) for s in stamps]

    def test_borrowed_hours_skipped(self):
        self.assertEqual(self.done['leads'], self.own)
        with np.load(os.path.join(self.base, 'stations.npz')) as z:
            got = self.leads(z['stamps'])
        self.assertFalse(set(got) & set(self.borrowed + [139]))

    def test_lead_schedule(self):
        with np.load(os.path.join(self.base, 'stations.npz')) as z:
            self.assertEqual(self.leads(z['stamps']), sorted(set(self.own) & opslake.STATION_LEADS))
            self.assertNotIn(49, self.leads(z['stamps']))
            self.assertIn(51, self.leads(z['stamps']))
        with np.load(os.path.join(self.base, 'slim.npz')) as z:
            self.assertEqual(self.leads(z['stamps']), list(range(48)))
        with np.load(os.path.join(self.out, 'cache', 'blendin', 'v=1', 'gfs', RUN, 'conus.npz')) as z:
            self.assertEqual(self.leads(z['stamps']), [*range(49), *range(51, 73, 3)])
            self.assertEqual(set((str(r) for r in z['runs'])), {RUN})
        self.assertEqual(sorted(opslake.STATION_LEADS)[-1], 240)
        self.assertIn(126, opslake.STATION_LEADS)
        self.assertNotIn(123, opslake.STATION_LEADS)

    def test_station_codes_are_the_stored_codes(self):
        with np.load(os.path.join(self.base, 'stations.npz')) as z:
            self.assertTrue({'stamps', 'idx', 'codes', 'fields', 'enc', 'model', 'run'} <= set(z.files))
            fields, idx, codes = ([str(f) for f in z['fields']], z['idx'], z['codes'])
            stamps = [str(s) for s in z['stamps']]
            self.assertEqual(str(z['model']), 'gfs')
            self.assertEqual(str(z['run']), RUN)
            enc = json.loads(str(z['enc']))
        self.assertEqual(fields[:len(verify.FIELDS)], list(verify.FIELDS))
        self.assertTrue(all((k in enc for k in fields)))
        self.assertEqual([STATIONS[i][0] for i in idx], ['KAAA', 'KBBB', 'KEEE'])
        all_idx, iy, ix = verify.station_cells('gfs', STATIONS, REGION)
        where = {int(i): (int(y), int(x)) for i, y, x in zip(all_idx, iy, ix)}
        for k, st in enumerate(stamps):
            with np.load(os.path.join(self.hours, f'{st}.npz')) as h:
                for s, i in enumerate(idx):
                    y, x = where[int(i)]
                    for j, f in enumerate(fields):
                        if f != 'overdev_nb':
                            self.assertEqual(codes[k, s, j], h[f][y, x], (st, f))
        only = os.path.join(self.tmp, 'only')
        os.makedirs(only)
        for st in stamps:
            shutil.copy(os.path.join(self.hours, f'{st}.npz'), only)
        keep = build.REGIONS['gfs']
        build.REGIONS['gfs'] = REGION
        try:
            vs, vidx, vcodes = verify.sample_hours('gfs', only, STATIONS)
        finally:
            build.REGIONS['gfs'] = keep
            shutil.rmtree(only)
        self.assertEqual(vs, stamps)
        np.testing.assert_array_equal(vidx, idx)
        np.testing.assert_array_equal(vcodes, codes[:, :, :len(verify.FIELDS)])

    def test_slim_mask(self):
        lons, lats = build.target_grid(REGION)
        lat2d, lon2d = np.meshgrid(lats, lons, indexing='ij')
        keep = build.keep_mask('gfs', lat2d, lon2d)
        self.assertTrue(keep.any() and (not keep.all()))
        self.assertFalse(keep[lats > 56.0][:, lons > -130.0].any())
        with np.load(os.path.join(self.base, 'slim.npz')) as z:
            fields = [str(f) for f in z['fields']]
            self.assertEqual(fields, list(opslake.SLIM_FIELDS))
            self.assertIn('u_blavg', fields)
            self.assertEqual(z['elev'].shape, (len(lats), len(lons)))
            np.testing.assert_array_equal(z['elev'][keep], self.elev[keep])
            np.testing.assert_array_equal(z['elev'][~keep], 255)
            self.assertTrue((self.elev[~keep] != 255).any())
            grid = json.loads(str(z['grid']))
            self.assertEqual((grid['nx'], grid['ny']), (len(lons), len(lats)))
            codes, stamps = (z['codes'], [str(s) for s in z['stamps']])
        self.assertEqual(codes.shape, (48, len(fields), len(lats), len(lons)))
        dark_seen = lit_seen = 0
        for k, st in enumerate(stamps):
            v = datetime.strptime(st, '%Y%m%d%H').replace(tzinfo=UTC)
            with np.load(os.path.join(self.hours, f'{st}.npz')) as h:
                off = (build.sun_elevation(v, lat2d, lon2d) < opslake.SUN_MIN) | (h['t2'] == 255) | ~keep
                self.assertTrue((h['t2'][~keep] != 255).any())
                for j, f in enumerate(fields):
                    np.testing.assert_array_equal(codes[k, j][off], 255)
                    np.testing.assert_array_equal(codes[k, j][~off], h[f][~off])
            dark_seen += off.all()
            lit_seen += (~off).any()
        self.assertTrue(dark_seen and lit_seen)

    def test_conus_crop(self):
        with np.load(os.path.join(self.out, 'cache', 'blendin', 'v=1', 'gfs', RUN, 'conus.npz')) as z:
            grid = json.loads(str(z['grid']))
            codes, fields = (z['codes'], [str(f) for f in z['fields']])
            self.assertEqual(fields, list(opslake.BLEND_FIELDS))
            self.assertFalse(bool(z['aircorr']))
            self.assertEqual(z['elev'].shape, codes.shape[2:])
        self.assertEqual((grid['west'], grid['north']), (-125.0, 50.0))
        self.assertEqual((grid['nx'], grid['ny']), (31, 21))
        y0, x0 = (16, 20)
        with self.hour(0) as h:
            for j, f in enumerate(fields):
                np.testing.assert_array_equal(codes[0, j], h[f][y0:y0 + 21, x0:x0 + 31])

    def test_no_blend_inputs_from_the_base_or_outside(self):
        self.assertIsNone(opslake.conus_window(build.REGIONS['icond2']))
        self.assertIsNone(opslake.conus_window(build.REGIONS['arome']))
        self.assertEqual(opslake.conus_window(build.REGIONS['hrrr']), (0, 868, 0, 1951))
        y0, y1, x0, x1 = opslake.conus_window(build.REGIONS['ecmwf'])
        self.assertLess((y1 - y0) * (x1 - x0), 160000)
        out = os.path.join(self.tmp, 'hrrr-out')
        opslake.write(self.hours, RUN, out, model='hrrr', build_id='1-1', region=REGION, stations=STATIONS, day_has_grids=True)
        self.assertFalse(os.path.exists(os.path.join(out, 'cache')))
        self.assertFalse(os.path.exists(os.path.join(out, 'ops', 'hrrr', 'grids')))

    def test_grids_due(self):
        self.assertTrue(opslake.grids_due('hrrr', '2026062112', False))
        self.assertFalse(opslake.grids_due('hrrr', '2026062106', False))
        self.assertTrue(opslake.grids_due('hrrr', '2026062115', False))
        self.assertFalse(opslake.grids_due('hrrr', '2026062115', True))
        self.assertTrue(opslake.grids_due('ecmwf', '2026062100', False))
        with open(os.path.join(self.out, 'ops', 'gfs', 'grids', '20260621.json')) as f:
            m = json.load(f)
        self.assertEqual((m['run'], m['build']), (RUN, '123-1'))
        self.assertTrue(m['key'].endswith('/b123-1/slim.npz'))

    def test_done_json(self):
        p = os.path.join(self.base, 'done.json')
        with open(p) as f:
            d = json.load(f)
        for k in ('stations_v', 'hours_v', 'phys', 'aircorr', 'acc_class', 'published', 'published_t', 'complete', 'bytes', 'sec'):
            self.assertIn(k, d)
        self.assertTrue(d['published'])
        self.assertTrue(d['complete'])
        self.assertEqual(d['published_t'], '2026-06-21T09:41:00Z')
        self.assertEqual(d['hours_v'], build.HOURS_VERSION)
        self.assertEqual(d['files']['stations.npz']['stations'], 3)
        self.assertGreater(d['bytes'], 0)
        last = os.stat(p).st_mtime_ns
        for dp, _, fs in os.walk(self.out):
            for f in fs:
                self.assertLessEqual(os.stat(os.path.join(dp, f)).st_mtime_ns, last, f)
        with open(os.path.join(self.out, 'ops', 'meta', f'stations-{d['stations_v']}.json')) as f:
            self.assertEqual(json.load(f)['stations'], STATIONS)

    def test_one_process_the_same(self):
        out = os.path.join(self.tmp, 'one')
        opslake.write(self.hours, RUN, out, build_id='123-1', published=True, region=REGION, stations=STATIONS, workers=1)
        for rel in ('ops/gfs/20260621/06/b123-1/stations.npz', 'ops/gfs/20260621/06/b123-1/slim.npz', f'cache/blendin/v=1/gfs/{RUN}/conus.npz'):
            with np.load(os.path.join(self.out, rel)) as a, np.load(os.path.join(out, rel)) as b:
                self.assertEqual(sorted(a.files), sorted(b.files))
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k], f'{rel} {k}')

    def test_phys_covers_the_cpp(self):
        names = [os.path.relpath(p, opslake.HERE) for p in opslake.phys_files()]
        for f in ('physics.py', 'build.py', 'physics_native.py', 'runwave.py', 'aircorr.py', 'opslake.py'):
            self.assertIn(f, names)
        self.assertTrue(any((n.startswith('native/') and n.endswith('.cpp') for n in names)))
        self.assertFalse(any((os.path.basename(n).startswith('test_') for n in names)))

    def test_blend_corner_rounded(self):
        reg = build.REGIONS['ecmwf']
        y0, y1, x0, x1 = opslake.conus_window(reg)
        lons, lats = build.target_grid(reg)
        out = os.path.join(self.tmp, 'corner')
        bl = np.zeros((1, len(opslake.BLEND_FIELDS), y1 - y0, x1 - x0), np.uint8)
        opslake.write_blendin(out, 'ecmwf', RUN, reg, lons, lats, (y0, y1, x0, x1), bl, [(opslake.parse_run(RUN), 0)], None)
        with np.load(os.path.join(out, 'cache', 'blendin', 'v=1', 'ecmwf', RUN, 'conus.npz')) as z:
            g = json.loads(str(z['grid']))
        self.assertEqual(g['north'], round(g['north'], 6))
        self.assertEqual(g['west'], round(g['west'], 6))
        self.assertAlmostEqual(g['north'], float(lats[y0]), places=6)

    def test_blend_fault_keeps_the_stations(self):
        out = os.path.join(self.tmp, 'fault')
        real = opslake.write_blendin
        opslake.write_blendin = lambda *a: (_ for _ in ()).throw(OSError('disk full'))
        try:
            d = opslake.write(self.hours, RUN, out, build_id='9-1', region=REGION, stations=STATIONS, day_has_grids=True, complete=True)
        finally:
            opslake.write_blendin = real
        self.assertIn('error', d['files']['blendin'])
        self.assertTrue(os.path.exists(os.path.join(out, 'ops', 'gfs', '20260621', '06', 'b9-1', 'stations.npz')))
        self.assertTrue(os.path.exists(os.path.join(out, 'ops', 'gfs', '20260621', '06', 'b9-1', 'done.json')))

    def test_built_in_part_then_whole(self):
        hours = os.path.join(self.tmp, 'part', 'gfs', '_hours')
        rng = np.random.default_rng(2)
        made_up_store(hours, rng, range(0, 22), range(22, 46))
        with open(os.path.join(os.path.dirname(hours), 'latest.json'), 'w') as f:
            json.dump({'runId': RUN, 'complete': False}, f)
        out1 = os.path.join(self.tmp, 'part', 'out1')
        d1 = opslake.write(hours, RUN, out1, build_id='1-1', region=REGION, stations=STATIONS)
        self.assertFalse(d1['complete'])
        self.assertEqual(d1['leads'], list(range(22)))
        self.assertEqual(d1['files']['slim.npz']['hours'], 22)
        self.assertEqual(d1['files']['blendin'], {'skipped': 'run built in part'})
        self.assertFalse(os.path.exists(os.path.join(out1, 'ops', 'gfs', 'grids')))
        self.assertFalse(os.path.exists(os.path.join(out1, 'cache')))
        shutil.rmtree(os.path.dirname(hours))
        made_up_store(hours, rng, range(0, 46), ())
        with open(os.path.join(os.path.dirname(hours), 'latest.json'), 'w') as f:
            json.dump({'runId': RUN, 'complete': True}, f)
        out2 = os.path.join(self.tmp, 'part', 'out2')
        d2 = opslake.write(hours, RUN, out2, build_id='2-1', region=REGION, stations=STATIONS, day_has_grids=False)
        self.assertTrue(d2['complete'])
        self.assertEqual(d2['leads'], list(range(46)))
        self.assertEqual(d2['files']['slim.npz']['hours'], 46)
        self.assertEqual(d2['files']['blendin']['hours'], 46)
        with open(os.path.join(out2, 'ops', 'gfs', 'grids', '20260621.json')) as f:
            self.assertEqual(json.load(f)['build'], '2-1')
        self.assertTrue(opslake.run_complete(hours, '2026062100'))
        self.assertTrue(opslake.run_complete(os.path.join(self.tmp, 'nowhere', '_hours'), RUN))

    def test_no_own_hours(self):
        self.assertIsNone(opslake.write(self.hours, '2026062018', os.path.join(self.tmp, 'none'), region=REGION, stations=STATIONS))

class OwnPathsOnly(unittest.TestCase):

    def test_no_pointer_writes(self):
        with open(opslake.__file__) as f:
            self.assertNotIn('control/', f.read())
if __name__ == '__main__':
    unittest.main()
