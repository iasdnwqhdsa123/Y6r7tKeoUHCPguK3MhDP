import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
import numpy as np
import build
import verify_output

class StoredHoursOutlivingTheirFields(unittest.TestCase):

    def test_unknown_fields_left_out(self):
        tmp = tempfile.mkdtemp()
        try:
            hours, out = (os.path.join(tmp, 'hours'), os.path.join(tmp, 'out'))
            os.makedirs(hours)
            t = datetime(2026, 10, 28, 15, tzinfo=timezone.utc)
            g = np.full((20, 20), 100, np.uint8)
            np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), t2=g, wstar=g, wavelen=g, elevation=g)
            _, present, _ = build.write_tiles(out, hours, [t], 10, t, elevation=False)
            self.assertIn('t2', present)
            self.assertNotIn('wavelen', present)
            self.assertTrue(all((k in build.ENC for k in present)))
        finally:
            shutil.rmtree(tmp)

class DataEdge(unittest.TestCase):

    def test_edge_round_the_data(self):
        tmp = tempfile.mkdtemp()
        try:
            t = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
            el = np.full((40, 60), 255, np.uint8)
            el[5:30, 10:50] = 100
            el[30:35, 20:22] = 100
            np.savez(os.path.join(tmp, f'{t:%Y%m%d%H}.npz'), elevation=el)
            grid = {'west': -10.0, 'north': 50.0, 'step': 0.25}
            edge = build.manifest_edge(tmp, [t], grid)
            self.assertEqual(len(edge), 1)
            line = np.array(edge[0])
            self.assertEqual(line[0].tolist(), line[-1].tolist(), 'closed')
            self.assertAlmostEqual(line[:, 0].max(), 50 - 5 * 0.25 + 0.125)
            self.assertAlmostEqual(line[:, 0].min(), 50 - 34 * 0.25 - 0.125)
            self.assertAlmostEqual(line[:, 1].min(), -10 + 10 * 0.25 - 0.125)
            self.assertAlmostEqual(line[:, 1].max(), -10 + 49 * 0.25 + 0.125)
            self.assertLess(len(line), 30, 'straight runs simplified')
            self.assertIsNone(build.manifest_edge(tmp, [datetime(2026, 11, 14, 13, tzinfo=timezone.utc)], grid))
        finally:
            shutil.rmtree(tmp)

class ChartOverviews(unittest.TestCase):

    def test_saved_as_webp(self):
        import tempfile
        from PIL import Image
        import chart_tiles
        tmp = tempfile.mkdtemp()
        try:
            im = Image.new('RGBA', (8, 8), (200, 30, 30, 0))
            im.putpixel((2, 2), (200, 30, 30, 255))
            for fmt in ('jpeg', 'png'):
                path = os.path.join(tmp, fmt, '4', '1', '2.webp')
                chart_tiles.save(im, path, fmt)
                with Image.open(path) as back:
                    self.assertEqual((back.format, back.size), ('WEBP', (8, 8)))
                    if fmt == 'jpeg':
                        self.assertEqual(back.convert('RGBA').getpixel((6, 6))[3], 255)
        finally:
            shutil.rmtree(tmp)

class FrontsLabels(unittest.TestCase):

    def test_files_kept(self):
        import fronts_labels as fl
        feed = '<feed><entry><link href="https://www.data.jma.go.jp/developer/xml/data/20261114023831_0_VZSA60_010000.xml"/></entry><entry><link href="https://www.data.jma.go.jp/developer/xml/data/20261114050731_0_VZSA50_010000.xml"/></entry><entry><link href="https://www.data.jma.go.jp/developer/xml/data/20261114050731_0_VPWW53_010000.xml"/></entry></feed>'
        self.assertEqual([k for _, k in fl.jma_wanted(feed)], ['jma/202611/20261114023831_0_VZSA60_010000.xml.gz', 'jma/202611/20261114050731_0_VZSA50_010000.xml.gz'])
        listing = '<a href="Z__C_EDZW_20261113013630_tka01%2Cana_bwkman_dwdna_O_000000_000000_202611130000_WV12.png"><a href="Z__C_EDZW_20261113013630_tka01%2Cana_bwkman_dwdna_O_000000_000000_202611130000_WV12SW.png"><a href="Z__C_EDZW_20261201013630_tka01%2Cana_bwkman_dwdc_O_000000_000000_202612010000_WV12.png"><a href="Z__C_EDZW_20261113153600_nwv01%2Cico_hsy_NA_N_000000_300200_202611121200_WV1.png">'
        got = fl.dwd_wanted(listing)
        self.assertEqual([k for _, k in got], ['dwd/202611/Z__C_EDZW_20261113013630_tka01,ana_bwkman_dwdna_O_000000_000000_202611130000_WV12.png', 'dwd/202612/Z__C_EDZW_20261201013630_tka01,ana_bwkman_dwdc_O_000000_000000_202612010000_WV12.png'])
        self.assertTrue(got[0][0].startswith(fl.DWD_DIR) and '%2C' in got[0][0])
        have = fl.kept('2026-11-14 05:00:00     123 fronts-labels/jma/202611/a.xml.gz\n\n')
        self.assertEqual(have, {'jma/202611/a.xml.gz'})

class OutputWhole(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        fields = {f: {'tiles': ['0_0']} for f in ('t2', 'wstar', 'cloud', 'elevation')}
        m = {'runId': '2026102812', 'days': {'20261027': [], '20261028': []}, 'fields': fields, 'levels': []}
        os.makedirs(os.path.join(self.root, 'gfs'))
        with open(os.path.join(self.root, 'gfs', 'latest.json'), 'w') as f:
            json.dump(m, f)
        run = os.path.join(self.root, 'gfs', '2026102812')
        for day in m['days']:
            os.makedirs(os.path.join(run, day))
            for name in ('t2', 'wstar', 'cloud'):
                open(os.path.join(run, day, f'{name}.pack'), 'w').close()
        open(os.path.join(run, 'elevation.pack'), 'w').close()
        self.run = run

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_whole(self):
        self.assertEqual(verify_output.check_manifest(self.root, 'gfs'), [])

    def test_day_folder_gone(self):
        shutil.rmtree(os.path.join(self.run, '20261028'))
        self.assertEqual(verify_output.check_manifest(self.root, 'gfs'), ['gfs/2026102812/20261028: no folder'])

    def test_packs_gone(self):
        for name in ('t2', 'cloud'):
            os.remove(os.path.join(self.run, '20261027', f'{name}.pack'))
        self.assertEqual(len(verify_output.check_manifest(self.root, 'gfs')), 1)

    def test_no_terrain(self):
        os.remove(os.path.join(self.run, 'elevation.pack'))
        self.assertEqual(verify_output.check_manifest(self.root, 'gfs'), ['gfs/2026102812: no terrain'])

    def test_none_published(self):
        self.assertEqual(verify_output.check_manifest(self.root, 'icon'), [])

class AirCorrections(unittest.TestCase):
    FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib', 'fixtures', 'corrections_fixture.json')

    @unittest.skipUnless(os.path.exists(FIXTURE), "the calibration's fixture (not in the forecast job's copy)")
    def test_the_calibration_twins_numbers(self):
        import aircorr
        with open(self.FIXTURE) as f:
            fx = json.load(f)
        F, meta, corr = (fx['fields'], fx['meta'], aircorr.prepare(fx['corr']))
        names, nx, ny, nh = (F['names'], F['nx'], F['ny'], F['nh'])
        A = np.array([np.nan if x is None else x for x in F['data']], np.float64).reshape(nh, len(names), ny, nx)
        E = np.array([np.nan if x is None else x for x in fx['expected']], np.float64).reshape(nh, len(names), ny, nx)
        g = F['grid']
        lat = np.repeat((g['north'] - np.arange(ny) * g['step'])[:, None], nx, 1)
        lon = np.repeat((g['west'] + np.arange(nx) * g['step'])[None, :], ny, 0)
        for h in range(nh):
            out = aircorr.apply({n: A[h, i] for i, n in enumerate(names)}, lat, lon, meta['stampUtc'][F['stamps'][h]], meta['day'], meta.get('modelIndex', meta['model']), g['step'], corr)
            self.assertEqual(set(out), {'wstar', 'zi', 'depth', 'cubase', 'cudepth'})
            for k, a in out.items():
                e = E[h, names.index(k)]
                self.assertTrue(np.array_equal(np.isnan(a), np.isnan(e)), k)
                self.assertTrue(np.allclose(np.nan_to_num(a), np.nan_to_num(e), rtol=0.0001, atol=0.001), k)

    def test_the_cpp_trees_are_numpys(self):
        import aircorr
        if aircorr._native() is None:
            self.skipTest('no C++ library here')
        corr = aircorr.prepare(aircorr.load())
        rng = np.random.default_rng(3)
        for k, m in corr['fields'].items():
            X = rng.uniform(-50, 4000, (len(m['features']), 5000))
            X[rng.uniform(0, 1, X.shape) < 0.15] = np.nan
            self.assertTrue(np.array_equal(aircorr.predict(m, X), aircorr.predict(m, X, native=False)), k)

    def test_codes_corrected_where_there_are_thermals_and_masked_cells_kept(self):
        import aircorr
        leaf = lambda v: {'features': ['wstar'], 'base': v, 'trees': []}
        corr = aircorr.prepare({'version': 't', 'features': ['wstar'], 'model_ids': {'hrrr': 0}, 'clamps': aircorr.CLAMPS, 'cu_depth': 300, 'fields': {'ws': leaf(0.25), 'top': leaf(-0.1), 'zi': leaf(-0.1), 'cb': leaf(120.0), 'cu': leaf(0.0)}})
        shape = (2, 3)
        ws = np.array([[2.0, 3.0, np.nan], [1.0, 2.5, 4.0]])
        q = {'wstar': build.quantize(ws, *build.ENC['wstar']), 'zi': build.quantize(np.full(shape, 1500.0), *build.ENC['zi']), 'depth': build.quantize(np.full(shape, 1200.0), *build.ENC['depth']), 'cubase': build.quantize(np.array([[1400.0, np.nan, np.nan], [900.0, np.nan, 1600.0]]), *build.ENC['cubase']), 'cudepth': build.quantize(np.array([[500.0, np.nan, np.nan], [400.0, np.nan, 900.0]]), *build.ENC['cudepth']), 'u_blavg': build.quantize(np.zeros(shape), *build.ENC['u_blavg']), 'v_blavg': build.quantize(np.zeros(shape), *build.ENC['v_blavg'])}
        lat, lon = (np.full(shape, 40.0), np.full(shape, -100.0))
        out = aircorr.correct_codes(q, build.ENC, build.quantize, lat, lon, datetime(2026, 7, 1, 18), 'hrrr', 0.03, corr)
        dec = lambda k: aircorr._decode(out[k], *build.ENC[k])
        w = dec('wstar')
        self.assertAlmostEqual(w[0, 0], aircorr._decode(q['wstar'], *build.ENC['wstar'])[0, 0] * np.exp(0.25), delta=0.03)
        self.assertEqual(out['wstar'][0, 2], 255, 'no W*: no thermals there, nothing corrected')
        self.assertAlmostEqual(dec('zi')[0, 0], 1500 * np.exp(-0.1), delta=30)
        self.assertEqual(out['zi'][0, 2], q['zi'][0, 2], "the layer where there's no W* as it was")
        self.assertAlmostEqual(dec('cubase')[1, 0], 900 + 120, delta=30)
        self.assertEqual(out['cubase'][0, 1], 255, 'blue stays blue at p 0.5')
        for k in out:
            self.assertEqual(out[k].dtype, q[k].dtype)

    def test_off_unless_switched_on(self):
        import aircorr
        old = os.environ.pop('AIR_CORRECTIONS', None)
        try:
            self.assertFalse(aircorr.on())
            os.environ['AIR_CORRECTIONS'] = '1'
            self.assertTrue(aircorr.on())
        finally:
            os.environ.pop('AIR_CORRECTIONS', None)
            if old is not None:
                os.environ['AIR_CORRECTIONS'] = old
if __name__ == '__main__':
    unittest.main()
