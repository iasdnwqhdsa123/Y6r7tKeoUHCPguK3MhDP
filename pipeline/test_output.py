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
if __name__ == '__main__':
    unittest.main()
