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

    def test_points_only(self):
        os.makedirs(os.path.join(self.root, 'ens'))
        with open(os.path.join(self.root, 'ens', 'latest.json'), 'w') as f:
            json.dump({'runId': '2026102800', 'days': {'20261028': []}, 'fields': {'t2': {'tiles': []}}}, f)
        self.assertEqual(verify_output.check_manifest(self.root, 'ens'), [])

    def test_none_published(self):
        self.assertEqual(verify_output.check_manifest(self.root, 'icon'), [])
if __name__ == '__main__':
    unittest.main()
