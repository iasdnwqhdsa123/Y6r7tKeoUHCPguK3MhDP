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
