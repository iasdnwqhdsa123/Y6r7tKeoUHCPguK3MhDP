import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock
import numpy as np
import blend as B
import blendleg as L
import build
import opslake
import test_blend as T
UTC = timezone.utc
RUN = '2026070112'
R0 = datetime(2026, 7, 1, 12, tzinfo=UTC)
STATIONS = [['KAAA', 35.5, -99.5, 400], ['KBBB', 36.2, -98.9, 420], ['KFAR', 45.0, -80.0, 10]]

def write_hours(d, run, leads, rng):
    os.makedirs(d, exist_ok=True)
    r0 = L.parse_run(run)
    for ld in leads:
        v = r0 + timedelta(hours=ld)
        h = T.base_hour(rng)
        np.savez(os.path.join(d, f'{v:%Y%m%d%H}.npz'), **h['codes'])
        with open(os.path.join(d, f'{v:%Y%m%d%H}.run'), 'w') as f:
            f.write(f'{run}\n')

def write_member(d, model, run, rng, leads, aircorr=False, offset=0.0):
    step = {'gfs': 0.25, 'ecmwf': 0.1, 'icon': 0.125}[model]
    lats, lons, _ = T.native_grid(step, rng, offset)
    r0 = L.parse_run(run)
    stamps = [r0 + timedelta(hours=ld) for ld in leads]
    codes = np.full((len(stamps), len(opslake.BLEND_FIELDS), len(lats), len(lons)), 255, np.uint8)
    for i in range(len(stamps)):
        _, _, q = T.native_grid(step, rng, offset + 0.1 * i)
        for j, f in enumerate(opslake.BLEND_FIELDS):
            if f in q:
                codes[i, j] = q[f]
    _, _, q0 = T.native_grid(step, rng, offset)
    p = os.path.join(d, f'{model}-{run}.npz')
    np.savez(p, stamps=np.array([f'{v:%Y%m%d%H}' for v in stamps]), runs=np.array([run] * len(stamps)), fields=np.array(opslake.BLEND_FIELDS), codes=codes, elev=q0['elevation'], grid=json.dumps({'west': float(lons[0]), 'north': float(lats[0]), 'step': step, 'nx': len(lons), 'ny': len(lats)}), enc='{}', model=model, run=run, aircorr=aircorr)
    return p

class Base(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)

    def test_the_newest_whole_run(self):
        rng = np.random.default_rng(1)
        hd = os.path.join(self.d, '_hours')
        write_hours(hd, '2026070106', range(0, 49), rng)
        write_hours(hd, '2026070112', range(0, 49), rng)
        with mock.patch.dict(L.REGIONS['conus'], {'lead_max': 48}):
            self.assertEqual(L.runs_whole(hd, 'conus'), ['2026070112'])
        write_hours(hd, '2026070113', range(0, 19), rng)
        self.assertEqual(L.runs_whole(hd, 'conus'), [])
        self.assertEqual(L.main(['base', '--hours', os.path.join(self.d, 'none'), '--region', 'conus']), 0)

class Plan(unittest.TestCase):

    def test_newest_before_the_issue_and_young_enough(self):
        b = T.make_bundle()
        issued = datetime(2026, 7, 1, 15, 40, tzinfo=UTC)
        k = lambda m, r, reg='conus': f'cache/blendin/v=1/{m}/{r}/{reg}.npz'
        listing = [{'Key': k('gfs', '2026070100'), 'LastModified': '2026-07-01T05:00:00Z'}, {'Key': k('gfs', '2026070106'), 'LastModified': '2026-07-01T11:00:00Z'}, {'Key': k('gfs', '2026070112'), 'LastModified': '2026-07-01T16:30:00Z'}, {'Key': k('ecmwf', '2026063000'), 'LastModified': '2026-06-30T09:00:00Z'}, {'Key': k('ecmwf', '2026063012'), 'LastModified': '2026-06-30T21:00:00Z'}, {'Key': k('ecmwf', '2026070100'), 'LastModified': '2026-07-01T09:00:00Z'}, {'Key': k('icon', '2026070106', 'sh'), 'LastModified': '2026-07-01T10:00:00Z'}, {'Key': k('hrrr', '2026070112'), 'LastModified': '2026-07-01T14:00:00Z'}, {'Key': 'cache/blendin/v=1/gfs/2026070106/conus.npz.tmp', 'LastModified': '2026-07-01T11:00:00Z'}]
        got = L.plan(b, listing, 'conus', RUN, issued)
        self.assertEqual(got, {'gfs': k('gfs', '2026070106'), 'ecmwf': k('ecmwf', '2026070100')})

class Members(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)

    def test_three_hourly_read_between(self):
        rng = np.random.default_rng(2)
        p = write_member(self.d, 'gfs', '2026070106', rng, [48, 51])
        m = L.Member(p)
        a, b = (m.at(datetime(2026, 7, 3, 6, tzinfo=UTC)), m.at(datetime(2026, 7, 3, 9, tzinfo=UTC)))
        mid = m.at(datetime(2026, 7, 3, 7, tzinfo=UTC))
        t2a, t2b, t2m = (x['t2'].astype(float) for x in (a, b, mid))
        ok = (a['t2'] != 255) & (b['t2'] != 255)
        self.assertTrue(np.allclose(t2m[ok], np.rint(t2a[ok] * 2 / 3 + t2b[ok] / 3)))
        self.assertTrue(np.array_equal(mid['cuclass'], a['cuclass']))
        self.assertIsNone(m.at(datetime(2026, 7, 3, 12, tzinfo=UTC)))
        self.assertIn('elevation', mid)

class Run(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        p = mock.patch.object(L, 'region_grid', lambda region: (T.BASE_REGION, None))
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.dict(L.REGIONS['conus'], {'lead_max': 3})
        q.start()
        self.addCleanup(q.stop)

    def test_a_run_blended_into_the_record(self):
        rng = np.random.default_rng(3)
        hd = os.path.join(self.d, '_hours')
        write_hours(hd, RUN, range(0, 4), rng)
        md = os.path.join(self.d, 'members')
        os.makedirs(md)
        write_member(md, 'gfs', '2026070106', rng, range(0, 16), offset=1.5)
        write_member(md, 'ecmwf', '2026070100', rng, range(0, 22), offset=-1.0)
        b = B.prepare(B.validate(T.make_bundle()))
        out = os.path.join(self.d, 'out')
        issued = datetime(2026, 7, 1, 15, 40, tzinfo=UTC)
        paths = sorted((os.path.join(md, f) for f in os.listdir(md)))
        done = L.blend_run(hd, RUN, b, paths, 'conus', out, issued, workers=1, build_id='9-1', stations=STATIONS)
        self.assertIsNotNone(done)
        self.assertEqual(done['model'], 'blend')
        self.assertEqual(done['blend']['bundle'], b['id'])
        self.assertEqual(done['blend']['members'], ['ecmwf', 'gfs', 'hrrr'])
        self.assertEqual(len(done['blend']['hours']), 4)
        base = os.path.join(out, 'ops', 'blend', '20260701', '12', 'b9-1')
        with np.load(os.path.join(base, 'stations.npz')) as z:
            self.assertEqual(str(z['model']), 'blend')
            self.assertEqual(str(z['run']), RUN)
            self.assertEqual(list(z['idx']), [0, 1])
            self.assertEqual(z['codes'].shape[0], 4)
            t2 = z['codes'][:, :, list(z['fields']).index('t2')]
            self.assertTrue((t2 != 255).all())
        with open(os.path.join(base, 'done.json')) as f:
            self.assertIn('t2', json.load(f)['blend']['hours']['2026070112']['blended'])
        self.assertFalse(os.path.exists(os.path.join(out, '_blended')))

    def test_an_air_corrected_base_is_refused(self):
        rng = np.random.default_rng(4)
        hd = os.path.join(self.d, '_hours')
        write_hours(hd, RUN, range(0, 4), rng)
        b = B.prepare(B.validate(T.make_bundle()))
        with mock.patch.dict(os.environ, {'AIR_CORRECTIONS': '1'}):
            done = L.blend_run(hd, RUN, b, [], 'conus', os.path.join(self.d, 'out'), datetime(2026, 7, 1, 15, tzinfo=UTC), stations=STATIONS)
        self.assertIsNone(done)

class Regions(unittest.TestCase):

    def test_each_region_on_its_base(self):
        for name, cfg in L.REGIONS.items():
            reg, win = L.region_grid(name)
            full = build.REGIONS[cfg['base']]
            self.assertEqual(reg[4], full[4])
            if cfg['box'] is None:
                self.assertIsNone(win)
            else:
                w, s, e, n = cfg['box']
                self.assertTrue(reg[0] <= w + reg[4] and reg[2] >= e - reg[4] and (reg[1] <= s + reg[4]) and (reg[3] >= n - reg[4]))
        self.assertEqual({k: v[0] for k, v in opslake.BLEND_REGIONS.items()}, {k: v['base'] for k, v in L.REGIONS.items()})
        self.assertEqual(opslake.BLEND_REGIONS['sh'][1], L.REGIONS['sh']['box'])
        w = lambda m: sorted(opslake.blend_windows(m, build.REGIONS[m][:5]))
        self.assertEqual(w('gfs'), ['conus'])
        self.assertEqual(w('icon'), ['conus', 'sh'])
        self.assertEqual(w('ecmwf'), ['alps', 'conus'])
        self.assertEqual(w('arome'), ['alps'])
        self.assertEqual(w('icond2'), [])
        self.assertEqual(w('hrrr'), [])

class Pointers(unittest.TestCase):

    def test_the_leg_reads_the_pointer_and_never_writes_it(self):
        wf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.github', 'public', 'blend.yml')
        if not os.path.exists(wf):
            wf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.github', 'workflows', 'blend.yml')
        with open(wf) as f:
            lines = [ln for ln in f if 'control/' in ln and (not ln.lstrip().startswith('#'))]
        self.assertTrue(lines)
        for ln in lines:
            self.assertRegex(ln, 'aws|ops\\(\\)|s3 cp "s3://\\$OPS_S3_BUCKET/control/shadow/blend\\.json" /tmp/')
            self.assertNotIn('put-object', ln)
        with open(L.__file__) as f:
            self.assertNotIn('put_', f.read())
if __name__ == '__main__':
    unittest.main()
