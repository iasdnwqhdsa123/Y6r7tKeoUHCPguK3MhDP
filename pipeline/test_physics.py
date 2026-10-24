import math
import os
import unittest
import numpy as np
import physics as P

def column(lapse=0.0098, top_ml=1500.0, speed=lambda z: 10.0):
    levels = np.array([1000, 950, 900, 850, 800, 750, 700, 600, 500, 400, 300], np.float32)
    z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
    t = np.where(z <= top_ml, 25 - lapse * z, 25 - lapse * top_ml - 0.0065 * (z - top_ml)).astype(np.float32)
    s = np.array([speed(h) for h in z], np.float32)
    shape = (len(levels), 1, 1)
    one = lambda v: np.full((1, 1), v, np.float32)
    return P.Profile(one(0), one(1013.25), one(25.5), one(12), one(-5), one(0), z.reshape(shape), levels, t.reshape(shape), (t - 12).reshape(shape), (-s).reshape(shape), np.zeros(shape, np.float32))

def tile_img(folder, key):
    import io
    import pack
    from PIL import Image
    return np.array(Image.open(io.BytesIO(pack.read_tile(folder + '.pack', key))))

def pt_img(rdir, m, key):
    import io
    import pack
    from PIL import Image
    return np.array(Image.open(io.BytesIO(pack.read_entry(os.path.join(rdir, 'pt.pack'), m['tiles'].index(key)))))

class PhysicsTest(unittest.TestCase):

    def test_wstar_textbook(self):
        w = P.wstar(np.float32(200), np.float32(1500), np.float32(25), np.float32(950))
        self.assertTrue(2.0 < float(w) < 2.4)

    def test_bs_ratio(self):
        bs = lambda w, u: float(P.bs_ratio(np.float32(w), np.float32(u)))
        self.assertAlmostEqual(bs(2, 5.144), 7.1, delta=0.01)
        self.assertAlmostEqual(bs(2, 7.3), 5.0, delta=0.01)
        self.assertEqual(bs(2, 0), 20)
        self.assertEqual(bs(0, 5), 0)
        self.assertEqual((P.CD_BL, P.BS_MAX), (0.003, 20.0))
        self.assertAlmostEqual(float(P.shear_tilt(np.float32(2), np.float32(4))), 8 / (0.0128 * 64), places=3)

    def test_profile_interpolation_is_monotonic(self):
        prof = column()
        t1, t2 = (prof.at(np.full((1, 1), 500.0), 't'), prof.at(np.full((1, 1), 3000.0), 't'))
        self.assertGreater(float(t1[0, 0]), float(t2[0, 0]))

    def test_wind_column_uses_the_models_hub_wind(self):
        levels = np.array([925, 850, 700, 500], np.float32)
        z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
        shape = (len(levels), 1, 1)
        one = lambda v: np.full((1, 1), v, np.float32)
        t = (15 - 0.0065 * z).reshape(shape)
        u = np.full(shape, 20.0, np.float32)
        args = (one(0), one(1013.25), one(15), one(5), one(1), one(0), z.reshape(shape), levels, t, t - 10, u, np.zeros(shape, np.float32))
        plain, hub = (P.Profile(*args), P.Profile(*args, one(15), one(0), 100.0))
        at = lambda prof, h: float(prof.at(np.full((1, 1), h, np.float32), 'u')[0, 0])
        self.assertAlmostEqual(at(hub, 10), 1, places=4)
        self.assertAlmostEqual(at(hub, 100), 15, places=4)
        self.assertLess(at(plain, 100), 4)
        self.assertAlmostEqual(at(hub, 3000), at(plain, 3000), places=4)
        self.assertAlmostEqual(float(hub.at(np.full((1, 1), 2.0, np.float32), 't')[0, 0]), 15, places=4)
        nan = P.Profile(*args, one(np.nan), one(np.nan), 100.0)
        self.assertAlmostEqual(at(nan, 100), at(plain, 100), places=4)

    def test_parcel_depth_finds_inversion(self):
        prof = column(top_ml=1500)
        d = float(P.parcel_ml_depth(prof, np.full((1, 1), 0.5, np.float32))[0, 0])
        self.assertTrue(800 < d < 2500, d)

    def test_wave_environment_likes_strong_stable_flow(self):
        good = column(lapse=0.004, top_ml=0, speed=lambda z: 12 + z / 800)
        calm = column(lapse=0.004, top_ml=0, speed=lambda z: 3)
        sg, _ = P.wave_environment(good)
        sc, _ = P.wave_environment(calm)
        self.assertGreater(float(sg[0, 0]), 0.3)
        self.assertLess(float(sc[0, 0]), 0.05)

    def test_local_relief(self):
        e = np.zeros((10, 10), np.float32)
        e[5, 5] = 400
        r = P.local_relief(e, 2)
        self.assertEqual(float(r[5, 7]), 400)
        self.assertEqual(float(r[0, 0]), 0)

    def test_relief_along_the_wind(self):
        e = np.zeros((12, 12), np.float32)
        e[:, 6] = 600
        west, south = (np.full_like(e, 10), np.zeros_like(e))
        across = P.relief_along(e, west, south, 3)
        along = P.relief_along(e, south, west, 3)
        self.assertEqual(float(across[5, 4]), 600)
        self.assertEqual(float(along[5, 4]), 0)
        self.assertEqual(float(along[5, 6]), 0)
        self.assertEqual(float(P.local_relief(e, 3)[5, 4]), 600)
        self.assertEqual(float(P.relief_along(e, south, south, 3)[5, 4]), 0)
        self.assertEqual(float(P.relief_along(e, west, west, 3)[5, 4]), 600)

class SoilWetness(unittest.TestCase):

    def test_each_models_fields(self):
        import build
        import physics as P
        np.testing.assert_allclose(P.soil_wetness(mstav=np.array([0.0, 55.0, 130.0])), [0, 0.55, 1], atol=1e-06)
        gfs = P.soil_wetness(soilw=np.array([0.05, 0.2, 0.4]), wilt=np.array([0.1, 0.1, 0.1]), fldcp=np.array([0.3, 0.3, 0.3]))
        np.testing.assert_allclose(gfs, [0, 0.5, 1], atol=1e-06)
        np.testing.assert_allclose(P.soil_wetness(soilm=np.array([0.05, 0.2, 0.35])), [0, 0.5, 1], atol=1e-06)
        self.assertIsNone(P.soil_wetness())
        self.assertEqual(build.ENC['soilm'], (0, 1, None))
        self.assertIn('soilm', build.NOW_FIELDS)

class WaterTemperature(unittest.TestCase):

    def test_open_water_only(self):
        import build
        s = {'tsfc': np.array([15.0, 20.0, -1.5, 25.0]), 'lsm': np.array([0.0, 1.0, 0.0, 0.0]), 'ice': np.array([0.0, 0.0, 0.9, np.nan])}
        w = build.water_temp(s)
        self.assertEqual(w[0], 15.0)
        self.assertTrue(np.isnan(w[1]), 'land')
        self.assertTrue(np.isnan(w[2]), 'ice-covered lake')
        self.assertEqual(w[3], 25.0, 'no ice value: open water')
        self.assertIsNone(build.water_temp({'tsfc': s['tsfc']}), 'no land-sea mask: no field')

class Ensemble(unittest.TestCase):

    def test_steps_and_files(self):
        from datetime import datetime, timezone
        import ens
        st = ens.steps()
        self.assertEqual(len(st), 65)
        self.assertIn(144, st)
        self.assertIn(150, st)
        self.assertNotIn(147, st)
        self.assertEqual(st[-1], 240)
        run = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        self.assertEqual(ens.SRC.idx_url(run, 150), 'https://data.ecmwf.int/forecasts/20261021/12z/ifs/0p25/enfo/20261021120000-150h-enfo-ef.index')
        self.assertTrue(ens.SRC.url(run, 0, 1, 'oper').endswith('/20261021/12z/ifs/0p25/oper/20261021120000-0h-oper-fc.grib2'))

    def test_heights_from_the_surface_pressure(self):
        import ens
        std = {925: 762.0, 850: 1457.0, 700: 3012.0, 600: 4206.0, 500: 5574.0}
        t_lv = np.array([[15 - 6.5 * std[p] / 1000] for p in ens.LEVELS], np.float32)
        z = ens.heights(np.zeros(1, np.float32), np.array([1013.25], np.float32), np.array([15.0], np.float32), t_lv)
        for i, p in enumerate(ens.LEVELS):
            self.assertAlmostEqual(float(z[i, 0]), std[p], delta=25)
        z = ens.heights(np.array([1500.0], np.float32), np.array([850.0], np.float32), np.array([10.0], np.float32), t_lv)
        self.assertLess(z[0, 0], 1500)
        self.assertAlmostEqual(float(z[2, 0]) - 1500, 3012 - 1457, delta=60)

    def test_usable_top_and_score_as_the_app(self):
        import ens
        ws = np.array([2.0, 2.0, 0.5, 2.0])
        zi = np.array([1500.0, 1500.0, 1500.0, 1200.0])
        base = np.array([np.nan, 1200.0, np.nan, 1500.0])
        top = ens.usable_top(ws, zi, base)
        self.assertAlmostEqual(top[0], 1500 * (0.6 + 0.4 * (1 - 1 / 2.6) / 0.8), delta=1)
        self.assertAlmostEqual(top[1], 1200, delta=1)
        self.assertEqual(top[2], 0)
        self.assertTrue(1200 < top[3] < 1500)
        s = ens.hour_score(np.array([2.2, 2.2, 2.2, 2.2, 0.0]), np.array([1500.0, 1500.0, 1500.0, 300.0, 1500.0]), np.array([0.0, 0.0, 0.6, 0.0, 0.0]), np.array([5.0, 13.0, 5.0, 5.0, 5.0]))
        self.assertEqual(s.tolist(), [4, 3, 0, 0, 0])

    def test_windows_and_days(self):
        from datetime import datetime, timedelta, timezone
        import ens
        f = {'2t': np.zeros((2, 1)), 'ssr': np.full((2, 1), 3600000.0), 'str': np.full((2, 1), -360000.0), 'tp': np.full((2, 1), 0.003), 'cp': np.full((2, 1), 0.001)}
        prev = {'ssr': np.zeros((2, 1)), 'str': np.zeros((2, 1)), 'tp': np.zeros((2, 1)), 'cp': np.zeros((2, 1))}
        ens.window(f, prev, 3)
        self.assertAlmostEqual(f['rnsw'][0, 0], 3600000.0 / 10800)
        self.assertAlmostEqual(f['rain'][0, 0], 1.0)
        self.assertAlmostEqual(f['conv'][0, 0], 1 / 3)
        run = datetime(2026, 10, 21, 0, tzinfo=timezone.utc)
        lon = np.array([0.0])
        days = ens.Days(run, run + timedelta(hours=48), lon)
        for h in range(0, 49, 3):
            score = np.array([[3], [4], [1], [0]], np.int8) if h in (12, 15) else np.zeros((4, 1), np.int8)
            days.add(run + timedelta(hours=h), lon, score, np.array([[h == 15], [False], [False], [False]]))
        d = days.fields(run + timedelta(hours=12), lon)
        self.assertEqual(d['day3'][0], 50)
        self.assertEqual(d['day1'][0], 75)
        self.assertEqual(d['daystorm'][0], 25)
        self.assertTrue(np.isnan(days.fields(run + timedelta(hours=48), lon)['day1'][0]))

    def test_point_packs_of_its_own_fields(self):
        import os
        import tempfile
        from datetime import datetime, timezone
        import build
        import ens
        with tempfile.TemporaryDirectory() as d:
            hd = os.path.join(d, 'h')
            os.makedirs(hd)
            t = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
            grids = {k: np.full((20, 20), 255, np.uint8) for k in ('t2_lo', 't2_mid', 't2_hi', 'elevation')}
            grids['t2_mid'][5:10, 5:10] = 100
            np.savez(os.path.join(hd, f'{t:%Y%m%d%H}.npz'), **grids)
            pts = build.write_points(d, hd, [t], t, fields=ens.FIELDS)
            self.assertEqual(pts['fields'], ['t2_lo', 't2_mid', 't2_hi', 'elevation'])
            self.assertTrue(os.path.exists(os.path.join(d, 'pt.pack')))
        self.assertEqual(build.STORE_AHEAD_H['ens'], 0)

class EcmwfNineKm(unittest.TestCase):

    def test_regular_grid_maps_as_the_kd_tree_does(self):
        import build
        lat1, lon1 = (np.arange(90, -90.5, -1.0), np.arange(-180, 180, 1.0))
        la, lo = np.meshgrid(lat1, lon1, indexing='ij')
        for reg in ((-170.0, -50.0, 179.0, 70.0, 1.0), (-10.0, 30.0, 40.0, 60.0, 1.0)):
            a, b = (build.Regridder(la, lo, reg), build.Regridder.regular(la, lo, reg))
            f = np.random.default_rng(1).random(la.shape)
            np.testing.assert_array_equal(a(f), b(f))
            np.testing.assert_array_equal(a.peak(f), b.peak(f))

    def test_heat_flux_from_net_radiation(self):
        import build
        rn = np.array([500.0, 500.0, 500.0, -80.0, 500.0])
        h = build.heat_from_net_radiation(rn, np.array([25.0, 25.0, 15.0, 10.0, 25.0]), np.array([5.0, 5.0, 14.0, 8.0, 5.0]), np.array([0.1, 0.4, 0.3, 0.3, 0.1]), np.array([1.0, 1.0, 1.0, 1.0, 0.0]))
        self.assertGreater(h[0], h[1])
        self.assertGreater(h[1], h[2])
        self.assertTrue(150 < h[0] < 350)
        self.assertLess(h[3], 0)
        self.assertEqual(h[4], 0)

    def test_older_runs_marked_where_they_take_over(self):
        from datetime import datetime, timedelta, timezone
        import build
        r6, r0 = (datetime(2026, 10, 21, 6, tzinfo=timezone.utc), datetime(2026, 10, 21, 0, tzinfo=timezone.utc))
        end = r6 + timedelta(hours=144)
        tail = [(end + timedelta(hours=h), r0) for h in (6, 12, 18)]
        times = [end - timedelta(hours=3), end] + [t for t, _ in tail[1:]]
        self.assertEqual(build.older_runs(tail, times), [{'from': tail[1][0].isoformat(), 'run': r0.isoformat()}])
        self.assertEqual(build.older_runs([], times), [])

    def test_snow_depth_and_sea_ice(self):
        from sources import ifs_derive
        s = ifs_derive({'sdwe': np.array([0.05, 0.0, 0.05]), 'rsn': np.array([100.0, 300.0, np.nan]), 'sithick': np.array([0.0, 0.5, np.nan]), 't2': np.zeros(3)})
        np.testing.assert_allclose(s['snod'], [0.5, 0.0, 0.2])
        np.testing.assert_array_equal(s['ice'], [0, 1, 0])
        self.assertFalse({'sdwe', 'rsn', 'sithick'} & set(s))

class EcmwfHourly(unittest.TestCase):

    def test_which_hours_are_built(self):
        from sources import SOURCES
        e, g = (SOURCES['ecmwf'], SOURCES['gfs'])
        self.assertTrue(all((e.builds(h) for h in range(0, 91))))
        self.assertEqual([h for h in range(88, 152) if e.builds(h)], [88, 89, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 150])
        self.assertEqual([h for h in range(118, 128) if g.builds(h)], [118, 119, 120, 123, 126])

    def test_complete_run_checks_its_last_file_step(self):
        from datetime import datetime, timedelta, timezone
        import sources
        src = sources.SOURCES['ecmwf']
        run = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        seen, orig = ([], sources.grib.exists)
        sources.grib.exists = lambda url: seen.append(url) or True
        try:
            r, fhs = sources.latest_complete_run(src, [run + timedelta(hours=h) for h in (1, 2, 3, 91, 92, 93)], log=lambda *a: None, now=run)
        finally:
            sources.grib.exists = orig
        self.assertEqual((r, fhs), (run, [1, 2, 3, 93]))
        self.assertEqual(seen, [src.idx_url(run, 93)])

    def test_an_hour_between_steps_blends_their_messages(self):
        from datetime import datetime, timezone
        from unittest import mock
        import build
        from sources import SOURCES
        src = SOURCES['ecmwf']
        run = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        t, pt = (('t', '850 mb'), src.surface['ptype'])
        snd_only = next((k for k in src.wanted() if k not in src.wanted(sounding=False)))
        steps = {6: {t: [270.0], pt: [1.0], snd_only: [1.0]}, 9: {t: [273.0], pt: [5.0], snd_only: [2.0]}}
        asked = []
        fake = lambda s, r, fh, wanted: asked.append((fh, wanted is None)) or steps[fh]
        with mock.patch.object(build, 'step_messages', fake), mock.patch.object(build.grib, 'decode', lambda m: (np.asarray(m, np.float32), None)):
            raw = build.fetch_hour(src, run, 7)
            self.assertEqual(asked, [(6, True), (9, True)])
            self.assertEqual(set(raw), {t, pt})
            np.testing.assert_allclose(build._decode(raw[t]), [271.0])
            np.testing.assert_allclose(build._decode(raw[pt]), [1.0])
            np.testing.assert_allclose(build._decode(build.fetch_hour(src, run, 8)[pt]), [5.0])
            self.assertIs(build.fetch_hour(src, run, 9), steps[9])

    def test_accumulations_between_steps_take_the_steps_window(self):
        from datetime import datetime, timedelta, timezone
        from unittest import mock
        import build
        from sources import SOURCES
        src = SOURCES['ecmwf']
        run = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        m = build.Blend([0.003], [0.006], 1 / 3, 6, 9)
        with mock.patch.object(build.grib, 'decode', lambda x: (np.asarray(x, np.float64), None)):
            kind, mean, hours, end = build.blend_window(src, 'prate', m, run)
        self.assertEqual((kind, hours, end), ('win', 3, run + timedelta(hours=9)))
        self.assertAlmostEqual(float(mean[0]) * 3600, 1.0, places=5)
        native = {'prate': ('win', mean, 3, end)}
        build.resolve_windows(src, run, 7, native, lambda *a: self.fail('no earlier file between steps'))
        self.assertEqual(native['prate'][0], 'win')

    def test_sunshine_over_a_window_that_ends_after_the_hour(self):
        from datetime import datetime, timedelta, timezone
        import build
        lat, lon = (np.array([[40.0]]), np.array([[-75.0]]))
        v = datetime(2026, 10, 21, 19, tzinfo=timezone.utc)
        np.testing.assert_allclose(build.instant_factor(v, 3, lat, lon, end=v), build.instant_factor(v, 3, lat, lon))
        f = float(build.instant_factor(v, 3, lat, lon, end=v + timedelta(hours=2))[0, 0])
        mean = np.mean([float(build.solar_estimate(v + timedelta(hours=2) - timedelta(hours=3 * k / 1200), lat, lon, 0)[0, 0]) for k in range(1201)])
        self.assertAlmostEqual(f, float(build.solar_estimate(v, lat, lon, 0)[0, 0]) / mean, places=2)
        self.assertGreater(f, 1)

    def test_soundings_at_the_files_steps_to_144_h(self):
        import build
        from sources import SOURCES
        e, g = (SOURCES['ecmwf'], SOURCES['gfs'])
        self.assertEqual([h for h in range(0, 13) if build.snd_hour(e, h)], [0, 3, 6, 9, 12])
        self.assertTrue(build.snd_hour(e, 144))
        self.assertFalse(build.snd_hour(e, 150))
        self.assertTrue(build.snd_hour(g, 47))
        self.assertFalse(build.snd_hour(g, 49))

    def test_open_meteo_values_in_load_hours_units(self):
        from datetime import datetime, timezone
        import build
        lat, lon = (np.array([[40.0]]), np.array([[-75.0]]))
        v = datetime(2026, 10, 21, 19, tzinfo=timezone.utc)
        np.testing.assert_allclose(build.om_convert(np.array([[283.15]]), 'K', v, lat, lon), [[10.0]])
        np.testing.assert_allclose(build.om_convert(np.array([[101300.0]]), 'Pa', v, lat, lon), [[1013.0]])
        np.testing.assert_allclose(build.om_convert(np.array([[2.5]]), None, v, lat, lon), [[2.5]])
        np.testing.assert_allclose(build.om_convert(np.array([[400.0]]), 'sun', v, lat, lon), 400 * build.instant_factor(v, 1, lat, lon))

    def test_a_step_is_fetched_once_for_the_hours_round_it(self):
        from datetime import datetime, timezone
        from unittest import mock
        import build
        from sources import SOURCES
        src = SOURCES['ecmwf']
        run = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        calls = []
        with mock.patch.object(type(src), 'fetch', lambda self, r, fh, sel: calls.append(fh) or {'fh': fh}), mock.patch.object(type(src), 'inventory', lambda self, r, fh, wanted=None: {}), mock.patch.object(build, '_STEPS', {}):
            for fh in (6, 9, 6, 9, 12, 15, 6):
                self.assertEqual(build.step_messages(src, run, fh, None), {'fh': fh})
            self.assertEqual(build.kept_step(src, run, 6), {'fh': 6})
            self.assertIsNone(build.kept_step(src, run, 9))
        self.assertEqual(calls, [6, 9, 12, 15, 6])

    def test_lanes_take_runs_of_hours(self):
        import build
        jobs = list(range(30))
        shares = build.lane_jobs(jobs, 2, 12)
        self.assertEqual([j for _, j in shares[0]], list(range(12)) + list(range(24, 30)))
        self.assertEqual([j for _, j in shares[1]], list(range(12, 24)))
        self.assertEqual(build.lane_jobs(jobs, 2, 1), [list(enumerate(jobs))[0::2], list(enumerate(jobs))[1::2]])

    def test_stalled_range_asked_of_the_next_mirror(self):
        import grib
        from unittest import mock
        asked = []

        class Ok:

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'data'

        def urlopen(req, timeout):
            asked.append((req.full_url, req.get_header('Range'), timeout))
            if 'one' in req.full_url:
                raise TimeoutError('stalled')
            return Ok()
        with mock.patch.object(grib.urllib.request, 'urlopen', urlopen), mock.patch.object(grib.time, 'sleep', lambda s: None):
            self.assertEqual(grib.http_get(['https://one/f', 'https://two/f'], (5, 9), tries=4, timeout=15), b'data')
        self.assertEqual(asked, [('https://one/f', 'bytes=5-9', 15), ('https://two/f', 'bytes=5-9', 15)])

    def test_build_pinned_to_the_shards_run(self):
        import build
        from datetime import datetime, timedelta, timezone
        from sources import SOURCES
        src = SOURCES['ecmwf']
        run = datetime(2026, 10, 22, 0, tzinfo=timezone.utc)
        needed = [run + timedelta(hours=h) for h in range(-6, 400)]
        got, fhs = build.pick_run(src, needed, '2026102200')
        self.assertEqual(got, run)
        self.assertEqual(fhs[0], 0)
        self.assertEqual(fhs[-1], 360)
        self.assertTrue(all((src.builds(f) for f in fhs)))

    def test_hour_shards_take_every_hour_once(self):
        import build
        fhs = list(range(0, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6))
        shares = [build.shard_hours(fhs, i, 3, 12) for i in range(3)]
        self.assertEqual(sorted((h for s in shares for h in s)), fhs)
        self.assertEqual(shares[1][:12], fhs[12:24])
        self.assertLessEqual(max(map(len, shares)) - min(map(len, shares)), 12)
        self.assertEqual(build.shard_hours([0, 3], 2, 3, 12), [])

    def test_hours_coded_per_window_as_whole(self):
        import build
        shape = (6, 8)
        windows = [((0, 4, 0, 5), (1, 3, 1, 4)), ((2, 6, 4, 8), (1, 4, 1, 4))]
        rng = np.random.default_rng(1)
        grids = [rng.uniform(-5, 40, (4, 5)).astype(np.float32), rng.uniform(-5, 40, (4, 4)).astype(np.float32)]
        grids[1][2, 2] = np.nan
        enc = build.ENC['t2']
        whole = build.quantize(build.paste(windows, grids, shape), *enc)
        codes = build.paste_codes(windows, [build.quantize(g, *enc) for g in grids], shape)
        np.testing.assert_array_equal(codes, whole)
        self.assertIsNone(build.paste_codes(windows, [None, None], shape))
        lat2d, lon2d = np.meshgrid(np.arange(6.0), np.arange(8.0), indexing='ij')
        np.testing.assert_array_equal(build.published_mask('icond2', lat2d, lon2d, {'t2': codes}), build.published_mask('icond2', lat2d, lon2d, {'t2': build.paste(windows, grids, shape)}))

class WindowedFields(unittest.TestCase):

    def setUp(self):
        import build
        from sources import SOURCES
        self.b, self.S = (build, SOURCES)

    def test_gfs_six_hour_mean_is_split(self):
        asked = []
        m, hrs = self.b._window_mean(self.S['gfs'], None, 126, 'shf', np.array([100.0]), lambda f, k: asked.append(f) or np.array([60.0]))
        self.assertEqual(float(m[0]), 140.0)
        self.assertEqual((hrs, asked), (3, [123]))
        m, hrs = self.b._window_mean(self.S['gfs'], None, 129, 'shf', np.array([80.0]), lambda f, k: None)
        self.assertEqual((float(m[0]), hrs), (80.0, 3))
        m, hrs = self.b._window_mean(self.S['gfs'], None, 123, 'shf', np.array([70.0]), lambda f, k: self.fail('no earlier file needed'))
        self.assertEqual((float(m[0]), hrs), (70.0, 3))
        m, hrs = self.b._window_mean(self.S['gfs'], None, 120, 'shf', np.array([100.0]), lambda f, k: np.array([96.0]))
        self.assertEqual((float(m[0]), hrs), (120.0, 1))

    def test_gfs_six_hourly_tail_is_its_own_window(self):
        m, hrs = self.b._window_mean(self.S['gfs'], None, 246, 'prate', np.array([2.0]), lambda f, k: self.fail('no earlier file needed'))
        self.assertEqual((float(m[0]), hrs), (2.0, 6))
        m, hrs = self.b._window_mean(self.S['gfs'], None, 240, 'shf', np.array([100.0]), lambda f, k: np.array([60.0]))
        self.assertEqual((float(m[0]), hrs), (140.0, 3))

    def test_gfs_hourly_to_120_then_3_hourly_to_240_then_6_hourly_to_384(self):
        gfs = self.S['gfs']
        self.assertTrue(all((gfs.fstep(f) for f in range(0, 121))))
        self.assertTrue(gfs.fstep(123) and gfs.fstep(168) and (not gfs.fstep(121)) and (not gfs.fstep(124)))
        self.assertTrue(gfs.fstep(237) and gfs.fstep(240) and (not gfs.fstep(243)) and gfs.fstep(246) and gfs.fstep(384))
        self.assertFalse(any((gfs.fstep(f) for f in range(241, 385) if f % 6)))
        self.assertEqual(sum((gfs.fstep(f) for f in range(385))), 185)
        self.assertEqual([self.b.prev_step(gfs, f) for f in (1, 6, 7, 120, 123, 126, 240, 246, 252)], [0, 5, 6, 119, 120, 123, 237, 240, 246])

    def test_gfs_hourly_rain_from_the_means_since_the_reset(self):
        rate = [0, 2.0, 0.5, 1.0, 3.0, 0.0, 4.0, 1.5, 0.2, 0.0, 6.0, 2.0, 1.0]
        ave = {f: np.array([np.mean(rate[(f - 1) // 6 * 6 + 1:f + 1]) / 3600]) for f in range(1, 13)}
        for fh in range(1, 13):
            asked = []
            native = {'prate': ave[fh], 'prate_now': np.array([50 / 3600.0])}
            self.b.resolve_windows(self.S['gfs'], None, fh, native, lambda f, k: asked.append((f, k)) or ave[f])
            kind, r, hours = native['prate']
            self.assertEqual((kind, hours), ('win', 1), fh)
            self.assertAlmostEqual(float(r[0]) * 3600, rate[fh], places=6, msg=f'f{fh:03d}')
            self.assertEqual(asked, [] if fh in (1, 7) else [(fh - 1, self.S['gfs'].surface['prate'])], fh)
        m, _ = self.b._window_mean(self.S['gfs'], None, 4, 'prate', np.array([1.5]), lambda f, k: np.array([1.0]))
        self.assertAlmostEqual(float(m[0]), 3.0)
        m, _ = self.b._window_mean(self.S['gfs'], None, 6, 'prate', np.array([2.0]), lambda f, k: np.array([1.8]))
        self.assertAlmostEqual(float(m[0]), 3.0)
        m, hrs = self.b._window_mean(self.S['gfs'], None, 6, 'sw', np.array([400.0]), lambda f, k: None)
        self.assertEqual((float(m[0]), hrs), (400.0, 6))
        native = {'prate': np.array([0.0]), 'prate_now': np.array([0.5 / 3600])}
        self.b.resolve_windows(self.S['gfs'], None, 0, native, lambda f, k: None)
        self.assertAlmostEqual(float(native['prate'][0]) * 3600, 0.5)

    def test_ecmwf_accumulation_is_differenced(self):
        m, hrs = self.b._window_mean(self.S['ecmwf'], None, 6, 'sw', np.array([30000000.0]), lambda f, k: np.array([20000000.0]))
        self.assertAlmostEqual(float(m[0]), 10000000.0 / 10800, places=3)
        self.assertEqual(self.b._window_mean(self.S['ecmwf'], None, 0, 'sw', np.array([0.0]), lambda f, k: None)[0], None)

    def test_ecmwf_rain_is_the_mean_rate_since_the_last_step(self):
        asked = []
        earlier = lambda f, k: asked.append((f, k)) or np.array([0.018])
        native = {'prate': np.array([0.03]), 'prate_now': np.array([50 / 3600.0])}
        self.b.resolve_windows(self.S['ecmwf'], None, 150, native, earlier)
        kind, rate, hours = native['prate']
        self.assertEqual((kind, hours), ('win', 6))
        self.assertAlmostEqual(float(rate[0]) * 3600, 2.0, places=5)
        self.assertEqual(asked, [(144, ('tp', 'sfc'))])
        native = {'prate': np.array([0.006])}
        self.b.resolve_windows(self.S['ecmwf'], None, 9, native, lambda f, k: np.array([0.003]))
        self.assertAlmostEqual(float(native['prate'][1][0]) * 3600, 1.0, places=5)
        native = {'prate': np.array([0.0]), 'prate_now': np.array([0.5 / 3600])}
        self.b.resolve_windows(self.S['ecmwf'], None, 0, native, lambda f, k: None)
        self.assertEqual(float(native['prate'][0]) * 3600, 0.5)
        self.assertNotIn('prate', self.b.SUNLIT)
        import grib
        rows = grib.parse_noaa_idx('1:0:d=x:PRATE:surface:6 hour fcst:\n2:100:d=x:PRATE:surface:0-6 hour ave fcst:\n3:200:d=x:TMP:surface:6 hour fcst:\n')
        sel = grib.select(rows, [self.S['gfs'].surface['prate'], self.S['gfs'].surface['prate_now']])
        self.assertEqual(sel[self.S['gfs'].surface['prate']]['offset'], 100)
        self.assertEqual(sel[self.S['gfs'].surface['prate_now']]['offset'], 0)
        self.assertEqual(self.S['gfs'].windowed['prate'], 'avg6')

    def test_hrrr_rain_is_the_last_hours_total(self):
        import grib
        hrrr = self.S['hrrr']
        idx = '83:0:d=x:PRATE:surface:5 hour fcst:\n84:100:d=x:APCP:surface:0-5 hour acc fcst:\n90:200:d=x:APCP:surface:4-5 hour acc fcst:\n96:300:d=x:TMP:surface:5 hour fcst:\n'
        sel = grib.select(grib.parse_noaa_idx(idx), [hrrr.surface['prate'], hrrr.surface['prate_now']])
        self.assertEqual(sel[hrrr.surface['prate']]['offset'], 200)
        self.assertEqual(sel[hrrr.surface['prate_now']]['offset'], 0)
        for idx, off in (('84:100:d=x:APCP:surface:0-1 hour acc fcst:\n', 100), ('84:100:d=x:APCP:surface:0-1 day acc fcst:\n90:200:d=x:APCP:surface:23-24 hour acc fcst:\n', 200), ('84:100:d=x:APCP:surface:0-2 day acc fcst:\n90:200:d=x:APCP:surface:47-48 hour acc fcst:\n', 200)):
            self.assertEqual(grib.select(grib.parse_noaa_idx(idx), [hrrr.surface['prate']])[hrrr.surface['prate']]['offset'], off)
        self.assertEqual(grib.select(grib.parse_noaa_idx('84:100:d=x:APCP:surface:0-0 day acc fcst:\n'), [hrrr.surface['prate']]), {})
        native = {'prate': np.array([3.0]), 'prate_now': np.array([40 / 3600.0])}
        self.b.resolve_windows(hrrr, None, 5, native, lambda f, k: self.fail('no earlier file needed'))
        self.assertAlmostEqual(float(native['prate'][1][0]) * 3600, 3.0, places=5)
        native = {'prate': None, 'prate_now': np.array([0.5 / 3600])}
        self.b.resolve_windows(hrrr, None, 0, native, lambda f, k: None)
        self.assertAlmostEqual(float(native['prate'][0]) * 3600, 0.5, places=5)

    def test_ecmwf_gust_none_at_step_0_and_found_at_every_step(self):
        native = {'gust': np.array([0.0])}
        self.b.resolve_windows(self.S['ecmwf'], None, 0, native, lambda f, k: None)
        self.assertIsNone(native['gust'])
        native = {'gust': np.array([12.0])}
        self.b.resolve_windows(self.S['ecmwf'], None, 3, native, lambda f, k: None)
        self.assertEqual(float(native['gust'][0]), 12.0)
        native = {'gust': np.array([3.0])}
        self.b.resolve_windows(self.S['gfs'], None, 0, native, lambda f, k: None)
        self.assertEqual(float(native['gust'][0]), 3.0)
        ec = self.S['ecmwf']
        import grib, json
        soil = grib.parse_ecmwf_idx(json.dumps({'_offset': 0, '_length': 10, 'param': 'vsw', 'levtype': 'sol', 'levelist': '1'}) + '\n\n' + json.dumps({'_offset': 10, '_length': 10, 'param': 't', 'levtype': 'pl', 'levelist': '850'}))
        self.assertEqual([r['level'] for r in soil], ['sol 1', '850 mb'])
        rows = grib.parse_ecmwf_idx(json.dumps({'_offset': 0, '_length': 10, 'param': '10fg3', 'levtype': 'sfc'}))
        for r in rows:
            r['var'] = ec.alias.get(r['var'], r['var'])
        self.assertIn(ec.surface['gust'], grib.select(rows, [ec.surface['gust']]))

    def test_ecmwf_fifteen_days(self):
        from sources import DAYS
        ec = self.S['ecmwf']
        self.assertEqual(DAYS['ecmwf'], 15)
        self.assertEqual(ec.max_fhour, 360)
        self.assertTrue(ec.fstep(144) and ec.fstep(150) and (not ec.fstep(147)) and ec.fstep(360))

    def test_waits_for_a_run_landing_late(self):
        from datetime import datetime, timezone
        from unittest import mock
        import check
        from sources import SOURCES
        hrrr = SOURCES['hrrr']
        at = lambda h, m=0: datetime(2026, 9, 30, h, m, tzinfo=timezone.utc)
        with mock.patch.object(check, 'latest_complete_run', lambda *a, **k: (at(0), [])):
            self.assertEqual(check.late_run(hrrr, at(8, 10)), (at(0), at(6)))
            self.assertEqual(check.late_run(hrrr, at(7, 0))[1], None)
            self.assertEqual(check.late_run(hrrr, at(11, 0))[1], None)
        with mock.patch.object(check, 'latest_complete_run', lambda *a, **k: (at(6), [])):
            self.assertEqual(check.late_run(hrrr, at(8, 10))[1], None)

    def test_map_data_rides_a_deploying_job_weekly(self):
        from datetime import timedelta as td
        from check import map_data_due
        d = lambda n: td(days=n)
        self.assertTrue(map_data_due([None, d(1)], []))
        self.assertFalse(map_data_due([d(3), d(3)], ['hrrr']))
        self.assertTrue(map_data_due([d(7.2), d(3)], ['hrrr', 'gfs']))
        self.assertFalse(map_data_due([d(7.2), d(7.2)], []))
        self.assertTrue(map_data_due([d(8.1), d(8.1)], ['hrrr', 'gfs']))

    def test_late_cron_stands_down_only_for_an_on_time_run_this_slot(self):
        from datetime import datetime, timezone
        import ontime
        cron = '10 2,8,14,20 * * *'
        at = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        run = lambda t, title=ontime.TITLE: {'display_title': title, 'created_at': t}
        self.assertEqual(ontime.last_slot(cron, at('2026-09-27T18:17:20')), at('2026-09-27T14:10:00'))
        self.assertEqual(ontime.last_slot(cron, at('2026-09-28T01:00:00')), at('2026-09-27T20:10:00'))
        self.assertTrue(ontime.stand_down([run('2026-09-27T14:10:03Z')], cron, at('2026-09-27T18:17:20')))
        self.assertFalse(ontime.stand_down([run('2026-09-27T17:57:25Z', 'Forecast data')], cron, at('2026-09-27T20:12:00')))
        self.assertFalse(ontime.stand_down([run('2026-09-27T14:10:03Z')], cron, at('2026-09-27T20:12:00')))
        self.assertFalse(ontime.stand_down([], cron, at('2026-09-27T20:12:00')))
        for c in ('cancelled', 'skipped'):
            self.assertFalse(ontime.stand_down([dict(run('2026-09-27T14:10:03Z'), conclusion=c)], cron, at('2026-09-27T18:17:20')))
        self.assertTrue(ontime.stand_down([dict(run('2026-09-27T14:10:03Z'), conclusion=None)], cron, at('2026-09-27T18:17:20')))

    def test_a_run_on_a_code_update_builds_only_what_isnt_published(self):
        import io
        import json
        import os
        import sys
        import tempfile
        from contextlib import redirect_stdout
        from datetime import datetime, timedelta, timezone
        from unittest import mock
        import check
        run = datetime(2026, 10, 12, 0, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        fresh = {'updated': now.isoformat()}

        def outputs(pub, *flags):
            with tempfile.NamedTemporaryFile('r', suffix='.txt') as f, mock.patch.dict(os.environ, {'GITHUB_OUTPUT': f.name}), mock.patch.object(sys, 'argv', ['check.py', '--data-url', 'https://d', *flags]), mock.patch.object(check, 'newest_run', lambda src, wait=True: run), mock.patch.object(check, 'published', lambda url: pub(url)), redirect_stdout(io.StringIO()):
                check.main()
                return dict((line.strip().split('=', 1) for line in f if '=' in line))

        def site(models=('hrrr', 'gfs', 'ecmwf', 'icon', 'icond2', 'ens'), free=True, maps=True, stale=False):

            def pub(url):
                if '/free/' in url:
                    return {'freeUntil': (now + timedelta(hours=29 if stale else 31)).isoformat()} if free else None
                if 'airspace' in url:
                    return dict(fresh, version=check.AVIATION_VERSION if 'aviation' in url else check.AIRSPACE_VERSION) if maps else None
                m = url.split('/')[-2]
                return {'runId': f'{(run - timedelta(hours=6) if stale else run):%Y%m%d%H}', 'complete': True} if m in models else None
            return pub
        self.assertEqual(outputs(site(stale=True))['any'], 'true')
        self.assertEqual(outputs(site(stale=True), '--missing-only')['any'], 'false')
        o = outputs(site(models=(), free=False, maps=False), '--missing-only')
        self.assertEqual((o['models'], o['free'], o['airspace']), ('hrrr gfs ecmwf icon icond2 ens', 'true', 'true'))
        from sources import PAUSED, REQUIRED, SOURCES
        self.assertEqual(o['active'], json.dumps([m for m in SOURCES if m not in PAUSED]))
        self.assertEqual(json.loads(o['required']), ['hrrr', 'gfs'])
        self.assertEqual(outputs(site(models=('hrrr', 'gfs', 'ecmwf', 'icond2', 'ens')), '--missing-only')['models'], 'icon')
        self.assertEqual(outputs(site(models=('hrrr', 'gfs', 'icon', 'icond2', 'ens')), '--missing-only')['models'], 'ecmwf')
        self.assertEqual(outputs(site(models=('hrrr', 'gfs', 'ecmwf', 'icon', 'icond2')), '--missing-only')['models'], 'ens')
        o = outputs(site(maps=False), '--missing-only')
        self.assertEqual((o['models'], o['airspace'], o['any']), ('', 'true', 'true'))

    def test_gfs_sixteen_days_and_ecmwf_at_9km(self):
        from datetime import datetime, timezone
        from sources import DAYS, PAUSED, needed_times
        gfs = self.S['gfs']
        self.assertEqual(DAYS['gfs'], 16)
        self.assertNotIn('ecmwf', PAUSED)
        ec = self.S['ecmwf']
        run = datetime(2026, 10, 21, 0, tzinfo=timezone.utc)
        self.assertTrue(ec.url(run, 24, 'pl').endswith('/20261021/00z/ifs/0p1/oper/20261021000000-24h-oper-fc-pl.grib2'))
        self.assertTrue(ec.idx_url(run, 24, 'sfc').endswith('-24h-oper-fc-sfc.index'))
        self.assertEqual([f for f in range(0, 361) if ec.fstep(f)][-3:], [348, 354, 360])
        self.assertTrue(ec.fstep(141) and (not ec.fstep(147)))
        self.assertEqual(ec.last_fh(run.replace(hour=6)), 144)
        self.assertEqual(ec.last_fh(run), 360)
        from sources import candidate_runs
        self.assertEqual(gfs.runs, (0, 6, 12, 18))
        self.assertEqual([r.hour for r in candidate_runs(gfs, datetime(2026, 9, 27, 23, 30, tzinfo=timezone.utc), back_hours=24)], [18, 12, 6, 0])
        now = datetime(2026, 9, 27, 2, 10, tzinfo=timezone.utc)
        run = datetime(2026, 9, 26, 18, tzinfo=timezone.utc)
        last = int((max(needed_times(gfs, now=now)) - run).total_seconds() // 3600)
        self.assertEqual(last, 399)
        built = max((f for f in range(last + 1) if f <= gfs.max_fhour and gfs.fstep(f)))
        self.assertEqual(built, 384)

    def test_instant_factor_follows_the_sun(self):
        from datetime import datetime, timezone
        lat, lon = (np.array([[39.0]]), np.array([[-105.0]]))
        f = lambda h, d=25: float(self.b.instant_factor(datetime(2026, 9, d, h, tzinfo=timezone.utc), 3, lat, lon)[0, 0])
        self.assertGreater(f(15), 1.5)
        self.assertLess(f(0, 26), 0.6)
        self.assertEqual(f(4, 26), 0.0)

    def test_heat_share_depends_on_humidity(self):
        h = self.b.heat_from_sunshine(np.array([800.0, 800.0, 800.0]), np.array([30.0, 20.0, 20.0]), np.array([5.0, 19.0, 19.0]), np.array([1.0, 1.0, 0.0]))
        self.assertGreater(h[0], 300)
        self.assertLess(h[1], 100)
        self.assertEqual(h[2], 0)

class CloudOverviewSmoothed(unittest.TestCase):

    def test_area_mean(self):
        import sky
        a = np.array([[1.0, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        m = sky.area_mean(a, 3)
        self.assertTrue(np.isnan(m[1, 1]))
        self.assertAlmostEqual(m[0, 0], (1 + 2 + 4) / 3)
        self.assertAlmostEqual(m[1, 0], (1 + 2 + 4 + 7 + 8) / 5)

    def test_a_lone_cell_takes_its_neighbours_class(self):
        import build, sky
        n = 9
        q = {k: np.full((n, n), 0, np.uint8) for k in sky.INPUTS}
        enc = build.ENC
        code = lambda name, v: np.uint8(round((v - enc[name][0]) / (enc[name][1] - enc[name][0]) * 254))
        q['wstar'][:] = code('wstar', 2.0)
        q['cudepth'][4, 4] = code('cudepth', 1500)
        cls = lambda out: sky.decode(out['cuclass'], 0, 254, None) / 50
        own = sky.cloud_overview(q, enc, True, smooth=False)
        smooth = sky.cloud_overview(q, enc, True)
        self.assertEqual(round(cls(own)[4, 4]), 3)
        self.assertLess(round(cls(smooth)[4, 4]), 3)
if __name__ == '__main__':
    unittest.main()

class FreeWindow(unittest.TestCase):

    def test_cutoff_and_fields(self):
        import json
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        out = tempfile.mkdtemp()
        hours = os.path.join(out, 'hrrr', '_hours')
        os.makedirs(hours)
        t0 = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
        stamps = []
        for h in range(0, 39, 3):
            t = t0 + timedelta(hours=h)
            stamps.append(f'{t:%Y%m%d%H}')
            q = np.full((5, 5), 100, np.uint8)
            np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, cape=q, sink=q, u_a2000=q, v_a2000=q, elevation=q)
        full = {'runId': '2026092512', 'run': t0.isoformat(), 'grid': {'tile': 100}, 'days': {'x': stamps}, 'fields': {k: {'min': 0, 'max': 1, 'transform': None, 'tiles': ['0_0']} for k in ('wstar', 'cape', 'sink', 'u_a2000', 'v_a2000', 'elevation')}}
        with open(os.path.join(out, 'hrrr', 'latest.json'), 'w') as f:
            json.dump(full, f)
        m = build.write_free(out, 'hrrr', now=t0 + timedelta(hours=1, minutes=40))
        published = sorted((s for d in m['days'].values() for s in d))
        stamp = lambda h: f'{t0 + timedelta(hours=h):%Y%m%d%H}'
        self.assertEqual(published[-1], stamp(24))
        self.assertEqual(m['hours'], [stamp(27), stamp(30)])
        rdir = os.path.join(out, 'free', 'hrrr', m['runId'])
        for h in (27, 30):
            self.assertTrue(os.path.exists(os.path.join(rdir, 'h', build.hour_token('hrrr', m['runId'], stamp(h)), 'wstar.pack')))
        self.assertIn('wstar', m['fields'])
        self.assertIn('u_a2000', m['fields'])
        self.assertIn('cape', m['fields'])
        self.assertNotIn('sink', m['fields'])
        self.assertEqual(m['tier'], 'free')
        with open(os.path.join(out, 'free', 'now', 'hrrr', 'latest.json')) as f:
            nm = json.load(f)
        now_hours = sorted((s for d in nm['days'].values() for s in d))
        self.assertEqual((now_hours[0], now_hours[-1]), ('2026092512', '2026092618'))
        self.assertIn('wstar', nm['fields'])
        self.assertTrue(build.NOW_FIELDS > build.FREE_FIELDS and 'twidth' in build.NOW_FIELDS)
        self.assertIsNone(build.write_free(out, 'ecmwf'))
        self.assertEqual([build.free_ahead(m).total_seconds() / 3600 for m in ('hrrr', 'gfs', 'icon', 'ecmwf')], [24, 48, 48, 24])

class Fronts(unittest.TestCase):

    @staticmethod
    def field(wind=(0.0, 0.0), ps=1000.0, flat=False):
        lats = np.arange(55, 24.99, -0.25)
        lons = np.arange(-120, -69.99, 0.25)
        la, lo = np.meshgrid(lats, lons, indexing='ij')
        t = 10.0 - (0 if flat else 4.0) * (np.tanh((la - 40.0) / 1.5) + 1) / 2 * 2
        td = t - 8.0
        u, v = (np.full(la.shape, wind[0]), np.full(la.shape, wind[1]))
        mslp = 1012 - 15 * np.exp(-(((la - 45) / 4) ** 2 + ((lo + 95) / 6) ** 2))
        return (lats, lons, t, td, u, v, np.full(la.shape, ps) if np.isscalar(ps) else ps, mslp)

    def test_theta_w(self):
        import fronts
        self.assertAlmostEqual(float(fronts.theta_w(20.0, 20.0, 1000.0)) - 273.15, 20.0, delta=0.3)
        self.assertLess(float(fronts.theta_w(20.0, 5.0, 1000.0)), float(fronts.theta_w(20.0, 15.0, 1000.0)))
        self.assertGreater(float(fronts.theta_w(5.0, 0.0, 850.0)) - 273.15, 5.0)

    def test_a_front_its_place_type_and_side(self):
        import fronts
        for wind, kind in (((0.0, 8.0), 'w'), ((0.0, -8.0), 'c'), ((8.0, 0.0), 's')):
            res = fronts.detect(*self.field(wind)[:6], ps=self.field()[6], mslp=self.field()[7])
            self.assertTrue(res['fronts'], wind)
            longest = max(res['fronts'], key=lambda f: len(f[1]))
            self.assertEqual(longest[0], kind, wind)
            pts = np.array(longest[1])
            self.assertTrue(np.all((pts[:, 1] < 40.5) & (pts[:, 1] > 37.0)), pts[:, 1])
            self.assertGreater(float(np.ptp(pts[:, 0])), 20)
            self.assertLess(pts[-1, 0], pts[0, 0])

    def test_none_without_a_zone_or_underground(self):
        import fronts
        f = self.field((0.0, -8.0), flat=True)
        self.assertEqual(fronts.detect(*f[:6], ps=f[6], mslp=f[7])['fronts'], [])
        f = self.field((0.0, -8.0), ps=800.0)
        self.assertEqual(fronts.detect(*f[:6], ps=f[6], mslp=f[7])['fronts'], [])
        lats, lons, t, td, u, v, ps, mp = self.field((0.0, -8.0))
        self.assertEqual(fronts.detect(lats[40:70], lons[:40], t[40:70, :40], td[40:70, :40], u[40:70, :40], v[40:70, :40], ps=ps[40:70, :40])['fronts'], [])

    def test_subtropics_a_moisture_edge_isnt_a_front(self):
        import fronts
        lats = np.arange(40, 9.99, -0.25)
        lons = np.arange(-120, -69.99, 0.25)
        la, lo = np.meshgrid(lats, lons, indexing='ij')
        step = (np.tanh((la - 24.0) / 1.5) + 1) / 2
        u, v = (np.zeros(la.shape), np.full(la.shape, -8.0))
        ps, mslp = (np.full(la.shape, 1000.0), np.full(la.shape, 1012.0))
        t = 24.0 - 8.0 * step
        self.assertTrue(fronts.detect(lats, lons, t, t - 8.0, u, v, ps=ps, mslp=mslp)['fronts'])
        t = np.full(la.shape, 24.0) - 1.0 * step
        td = 18.0 - 14.0 * step
        self.assertEqual(fronts.detect(lats, lons, t, td, u, v, ps=ps, mslp=mslp)['fronts'], [])
        self.assertGreater(fronts.SUB_FROM, 30)

    def test_occluded_next_to_a_deep_low(self):
        import fronts
        lats, lons, t, td, u, v, ps, _ = self.field((0.0, -8.0))
        la, lo = np.meshgrid(lats, lons, indexing='ij')
        deep = 1012 - 25 * np.exp(-(((la - 40.5) / 3) ** 2 + ((lo + 95) / 4) ** 2))
        res = fronts.detect(lats, lons, t, td, u, v, ps=ps, mslp=deep)
        occ = [p for k, p in res['fronts'] if k == 'o']
        self.assertTrue(occ, [k for k, _ in res['fronts']])
        pts = np.array([q for p in occ for q in p])
        self.assertLess(fronts.km_between(pts[:, 0], pts[:, 1], -95.0, 40.5).max(), fronts.OCC_LOW_KM + 50)
        self.assertTrue(any((k == 'c' for k, _ in res['fronts'])))
        shallow = 1012 - 4 * np.exp(-(((la - 40.5) / 3) ** 2 + ((lo + 95) / 4) ** 2))
        self.assertFalse(any((k == 'o' for k, _ in fronts.detect(lats, lons, t, td, u, v, ps=ps, mslp=shallow)['fronts'])))

    def test_typed_with_the_surface_wind_too(self):
        import fronts
        lats, lons, t, td, u, v, ps, mslp = self.field((0.0, -1.0))
        self.assertEqual(max(fronts.detect(lats, lons, t, td, u, v, ps=ps, mslp=mslp)['fronts'], key=lambda f: len(f[1]))[0], 's')
        u10, v10 = (np.zeros(u.shape), np.full(v.shape, -6.0))
        res = fronts.detect(lats, lons, t, td, u, v, ps=ps, mslp=mslp, u10=u10, v10=v10)
        self.assertEqual(max(res['fronts'], key=lambda f: len(f[1]))[0], 'c')

    def test_pressure_centres(self):
        import fronts
        f = self.field()
        cs = fronts.detect(*f[:6], ps=f[6], mslp=f[7])['centres']
        lows = [c for c in cs if c[0] == 'L']
        self.assertEqual(len(lows), 1)
        self.assertAlmostEqual(lows[0][1], -95, delta=1)
        self.assertAlmostEqual(lows[0][2], 45, delta=1)
        self.assertLess(lows[0][3], 1000)

    def test_encode_round_trip_and_size(self):
        import json
        import fronts
        f = self.field((0.0, -8.0))
        res = fronts.detect(*f[:6], ps=f[6], mslp=f[7])
        enc = fronts.encode(res)
        back = fronts.decode(enc)
        self.assertEqual([k for k, _ in back['fronts']], [k for k, _ in res['fronts']])
        for (_, a), (_, b) in zip(res['fronts'], back['fronts']):
            self.assertLess(np.abs(np.array(a)[[0, -1]] - np.array(b)[[0, -1]]).max(), fronts.COORD)
        self.assertLess(len(json.dumps(enc, separators=(',', ':'))), 2000)

    def test_fronts_drawn_once(self):
        import fronts
        lon = np.linspace(-100, -80, 41)
        strong = (1.0 * 900, lon[10:31], np.full(21, 40.0), 0.6)
        weak = (0.5 * 1700, lon, np.full(41, 41.35), 0.5)
        kept = fronts._undouble([strong, weak])
        self.assertEqual(len(kept), 3)
        spans = sorted(((float(k[1].min()), float(k[1].max())) for k in kept[1:]))
        self.assertLess(spans[0][1], -94)
        self.assertGreater(spans[1][0], -86)
        cross = (0.4 * 900, np.full(21, -90.0), np.linspace(36, 44, 21), 0.4)
        self.assertEqual(len(fronts._undouble([strong, cross])), 3)
        stub = (0.3 * 300, np.full(9, -90.0), np.linspace(39, 42, 9), 0.3)
        self.assertEqual(len(fronts._undouble([strong, stub])), 1)
        long_ = (0.3 * 900, np.full(31, -90.0), np.linspace(34, 46, 31), 0.3)
        self.assertEqual(len(fronts._undouble([strong, long_])), 3)
        a = np.c_[np.linspace(-100, -95, 6), np.full(6, 40.0)]
        b = np.c_[np.linspace(-94.9, -90, 6), np.full(6, 40.0)]
        c = np.c_[np.linspace(-89.9, -85, 6), np.full(6, 40.0)]
        j, = fronts.join_lines([c, a, b], km=50)
        self.assertEqual(len(j), 18)
        self.assertAlmostEqual(abs(j[0, 0] - j[-1, 0]), 15.0, places=6)
        go = np.c_[np.linspace(-100, -90, 21), np.full(21, 40.0)]
        back = np.c_[np.linspace(-90, -100, 21), np.full(21, 41.0)]
        pin = np.r_[go, [[-89.5, 40.5]], back]
        self.assertEqual(len(fronts._hairpins(pin)), 2)
        self.assertEqual(len(fronts._hairpins(go)), 1)

    def test_centre_pressure_from_the_field(self):
        import fronts
        g = fronts.Grid(np.arange(60, 29.99, -0.25), np.arange(-110, -69.99, 0.25))
        la, lo = np.meshgrid(g.lats, g.lons, indexing='ij')
        r2 = (la - 45) ** 2 + ((lo + 90) * 0.7) ** 2
        p = 1013 - 25 * np.exp(-r2 / 1.5)
        c, = fronts.centres(g, p)
        self.assertEqual(c[0], 'L')
        self.assertAlmostEqual(c[3], float(p.min()), places=3)
        self.assertLess(abs(c[2] - 45), 0.3)

    def test_network_weights_inputs_and_layers(self):
        import fronts
        W = fronts._weights()
        self.assertIsNotNone(W)
        lats, lons, t, td, u, v, ps, mslp = self.field((0.0, -8.0))
        X, g, (nx, ny) = fronts.features(lats, lons, t, td, u, v, ps, mslp, t + 2, td + 2, u, v, np.zeros(t.shape))
        self.assertEqual(X.shape, (len(fronts.FEATURES), len(lats), len(lons)))
        self.assertTrue(np.isfinite(X).all())
        self.assertLess(float(np.nanmean(ny[(g.lats > 38) & (g.lats < 42)])), -0.9)
        P = fronts.probabilities(W, X)
        self.assertEqual(P.shape, (5,) + X.shape[1:])
        self.assertTrue(np.allclose(P.sum(0), 1, atol=1e-05))
        la, lo = (np.arange(55, 24.99, -0.1), np.arange(-120, -69.99, 0.1))
        X2, g2, _ = fronts.features(la, lo, *(np.zeros((len(la), len(lo))) + c for c in (5, 0, 0, 0, 1000, 1012, 7, 2, 0, 0, 0)))
        self.assertAlmostEqual(abs(g2.lats[1] - g2.lats[0]), 0.25)
        self.assertAlmostEqual(g2.lons[1] - g2.lons[0], 0.25)
        rng = np.random.default_rng(0)
        h, Wc = (rng.normal(size=(2, 9, 11)).astype(np.float32), rng.normal(size=(3, 3, 2, 1)).astype(np.float32))
        hp = np.pad(h, ((0, 0), (2, 2), (2, 2)))
        want = sum((Wc[i, j, c, 0] * hp[c, 4 + 2 * (i - 1), 5 + 2 * (j - 1)] for i in range(3) for j in range(3) for c in range(2)))
        self.assertAlmostEqual(float(fronts._conv(h, Wc, 2)[0, 2, 3]), float(want), places=4)

    def test_network_fronts_both_hemispheres_and_fallback(self):
        import fronts
        lats, lons, t, td, u, v, ps, mslp = self.field((0.0, -8.0))
        zs = np.full(t.shape, 100.0)
        u10 = np.zeros(t.shape)
        v10 = np.repeat((3.0 - 11.0 * (np.tanh((lats - 40.0) / 0.5) + 1) / 2)[:, None], len(lons), 1)
        res = fronts.detect_ml(lats, lons, t, td, u, v, ps, mslp, t + 2, td + 2, u10, v10, zs)
        self.assertTrue(res['fronts'])
        longest = np.array(max(res['fronts'], key=lambda f: len(f[1]))[1])
        self.assertLess(float(np.median(np.abs(longest[:, 1] - 40))), 1.5, longest[:, 1])
        self.assertGreater(float(np.ptp(longest[:, 0])), 15)
        self.assertLess(longest[-1, 0], longest[0, 0])
        self.assertEqual([c[0] for c in res['centres']], ['L'])
        flip = lambda a: a[::-1]
        sres = fronts.detect_ml(-lats[::-1], lons, flip(t), flip(td), u, -flip(v), flip(ps), flip(mslp), flip(t) + 2, flip(td) + 2, u10, -flip(v10), zs)
        self.assertEqual(sorted((k for k, _ in sres['fronts'])), sorted((k for k, _ in res['fronts'])))
        s_long = np.array(max(sres['fronts'], key=lambda f: len(f[1]))[1])
        self.assertLess(float(np.median(np.abs(s_long[:, 1] + 40))), 1.5, s_long[:, 1])
        self.assertGreater(s_long[-1, 0], s_long[0, 0])
        eq = fronts.detect_ml(lats - 40, lons, t, td, u, v, ps, mslp, t + 2, td + 2, u10, v10, zs)
        self.assertFalse([p for _, pts in eq['fronts'] for p in pts if abs(p[1]) < fronts.ML_LAT[0] - 0.5])
        s = {'ps': ps, 'mslp': mslp, 't2': t + 2, 'td2': td + 2, 'u10': u10, 'v10': v10, 'zs': zs}
        lv = {'t': t[None], 'td': td[None], 'u': u[None], 'v': v[None]}
        keep_file, keep = (fronts.CNN_FILE, dict(fronts._CNN))
        try:
            fronts.CNN_FILE, fronts._CNN = ('no-such-file.json', {})
            self.assertEqual(fronts.from_part(s, lv, [850], lats, lons), fronts.detect(lats, lons, t, td, u, v, ps=ps, mslp=mslp, u10=u10, v10=v10))
        finally:
            fronts.CNN_FILE, fronts._CNN = (keep_file, keep)
        again = fronts.from_part(s, lv, [850], lats, lons)
        self.assertEqual((again['fronts'], again['centres']), (res['fronts'], res['centres']))
        st = fronts.unpack_states(fronts.pack_states([res['state'], sres['state']]))
        self.assertEqual([x['south'] for x in st], [False, True])
        self.assertLess(float(np.abs(st[0]['p'] - res['state']['p']).max()), 0.003)
        self.assertEqual(st[0]['key'], fronts.unpack_states(fronts.pack_states([res['state']]))[0]['key'])

    def test_steady_from_hour_to_hour(self):
        import fronts
        from datetime import datetime, timedelta, timezone
        lats, lons = (np.arange(55.0, 25.0, -0.25), np.arange(-115.0, -75.0, 0.25))
        g = fronts.Grid(lats, lons)
        ridge = np.exp(-((lats - 40.0) / 0.6) ** 2)[:, None] * np.ones(len(lons))
        nx_, ny_ = (np.zeros(ridge.shape), -np.ones(ridge.shape))

        def state(a, cold):
            p = np.stack([1 - a * ridge] + [a * ridge * f for f in (cold, 1 - cold, 0.0, 0.0)])
            return {'g': g, 'p': p.astype(np.float32), 'n': (nx_, ny_), 'south': False}
        t0 = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
        hs = {t0: state(0.7, 0.7), t0 + timedelta(hours=1): state(0.24, 0.45), t0 + timedelta(hours=2): state(0.7, 0.7)}
        mid = t0 + timedelta(hours=1)
        alone = fronts.ml_lines(g, hs[mid]['p'], nx_, ny_)
        self.assertFalse([k for k, _ in alone if k == 'c'])
        packed = {t: fronts.pack_states([s]) for t, s in hs.items()}
        out = fronts.day_fronts([(t, {'f': [], 'p': [['L', 1, 2, 990]]}) for t in sorted(hs)], lambda t: fronts.unpack_states(packed[t]))
        f = fronts.decode(out[mid])['fronts']
        self.assertTrue(f)
        self.assertEqual({k for k, _ in f}, {'c'})
        self.assertGreater(sum((fronts.km_between(*np.array(p[:-1]).T, *np.array(p[1:]).T).sum() for _, p in f)), 2500)
        self.assertEqual(out[mid]['p'], [['L', 1, 2, 990]])
        own = {'f': [['w', 1, 2, 3, 4]], 'p': []}
        self.assertEqual(fronts.day_fronts([(t0, own)], lambda t: None)[t0], own)

    def test_joined_across_a_gap(self):
        import fronts
        a = np.array([[-100.0, 40.0], [-98.0, 40.0], [-96.0, 40.0]])
        b = np.array([[-95.0, 40.1], [-93.0, 40.1]])
        c = np.array([[-95.0, 41.0], [-95.0, 43.0]])
        self.assertEqual(len(fronts.join_lines([a, b])), 1)
        self.assertEqual(len(fronts.join_lines([a, c])), 2)

    def test_published_by_the_day_never_as_tiles(self):
        import json
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        import fronts
        with tempfile.TemporaryDirectory() as d:
            hours, rdir = (os.path.join(d, '_hours'), os.path.join(d, 'run'))
            os.makedirs(hours)
            f = self.field((0.0, -8.0))
            enc = json.dumps(fronts.encode(fronts.detect(*f[:6], ps=f[6], mslp=f[7])), separators=(',', ':')).encode()
            t0 = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
            times = [t0 + timedelta(hours=h) for h in (0, 6, 24)]
            q = np.full((4, 4), 100, np.uint8)
            for t in times:
                np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, elevation=q, fronts=np.frombuffer(enc, np.uint8))
            self.assertTrue(build.write_fronts(rdir, hours, times))
            with open(os.path.join(rdir, build.day_key(t0), 'fronts.json')) as fh:
                day = json.load(fh)
            self.assertEqual(sorted(day), ['2026102212', '2026102218'])
            self.assertEqual(day['2026102212'], json.loads(enc))
            self.assertTrue(os.path.exists(os.path.join(rdir, build.day_key(times[2]), 'fronts.json')))
            _, present, _ = build.write_tiles(rdir, hours, times, 4, t0)
            self.assertNotIn('fronts', present)
            self.assertIn('wstar', present)

    def test_the_day_written_from_the_hours_together(self):
        import json
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        import fronts
        lats, lons = (np.arange(55.0, 25.0, -0.25), np.arange(-115.0, -75.0, 0.25))
        g = fronts.Grid(lats, lons)
        ridge = np.exp(-((lats - 40.0) / 0.6) ** 2)[:, None] * np.ones(len(lons))
        st = {'g': g, 'p': np.stack([1 - 0.7 * ridge, 0.7 * ridge, 0 * ridge, 0 * ridge, 0 * ridge]).astype(np.float32), 'n': (np.zeros(ridge.shape), -np.ones(ridge.shape)), 'south': False}
        with tempfile.TemporaryDirectory() as d:
            hours, rdir = (os.path.join(d, '_hours'), os.path.join(d, 'run'))
            os.makedirs(hours)
            t0 = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
            times = [t0 + timedelta(hours=h) for h in range(3)]
            enc = np.frombuffer(b'{"f":[],"p":[["H",1,2,1030]]}', np.uint8)
            for t in times:
                np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=np.zeros((4, 4), np.uint8), elevation=np.zeros((4, 4), np.uint8), fronts=enc, frontp=fronts.pack_states([st]))
            self.assertTrue(build.write_fronts(rdir, hours, times))
            with open(os.path.join(rdir, build.day_key(t0), 'fronts.json')) as fh:
                day = json.load(fh)
            self.assertEqual(sorted(day), [f'{t:%Y%m%d%H}' for t in times])
            for v in day.values():
                self.assertEqual({f[0] for f in v['f']}, {'c'})
                self.assertEqual(v['p'], [['H', 1, 2, 1030]])
            self.assertNotIn('frontp', build.write_tiles(rdir, hours, times, 4, t0)[1])

    def test_a_failing_part_costs_the_hour_nothing(self):
        import build

        class Rg:
            lats = np.arange(50, 24.99, -0.25)
            lons = np.arange(-120, -69.99, 0.25)
        broken = {'t': [np.zeros(3)], 'td': [np.zeros(3)], 'u': [np.zeros(3)], 'v': [np.zeros(3)]}
        self.assertIsNone(build.hour_fronts([({}, broken, (850,))], [Rg()]))
        out, states = build.hour_fronts([({}, {}, (1000, 850))], [Rg()])
        self.assertEqual(bytes(out), b'{"f":[],"p":[]}')
        self.assertIsNone(states)

class ReuseHours(unittest.TestCase):

    def test_kept_when_computed_from_same_run(self):
        import tempfile
        import build
        from datetime import datetime, timezone
        run = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            calls = []
            real = build.load_hour
            build.load_hour = lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError('download'))
            try:
                build._CTX.update({'src': None, 'run': run, 'region': None, 'hours': d, 'holder': {}, 'recompute': False})
                open(os.path.join(d, '2026092515.npz'), 'wb').close()
                with open(os.path.join(d, '2026092515.run'), 'w') as f:
                    f.write('2026092512')
                self.assertEqual(build._process_hour(3), datetime(2026, 9, 25, 15, tzinfo=timezone.utc))
                self.assertEqual(calls, [], 'same run: not downloaded again')
                with open(os.path.join(d, '2026092515.run'), 'w') as f:
                    f.write('2026092506')
                self.assertIsNone(build._process_hour(3))
                self.assertEqual(len(calls), 1)
                build._CTX['recompute'] = True
                with open(os.path.join(d, '2026092515.run'), 'w') as f:
                    f.write('2026092512')
                build._process_hour(3)
                self.assertEqual(len(calls), 2, '--recompute ignores the stored hour')
            finally:
                build.load_hour = real
                build._CTX['recompute'] = False

class PruneStore(unittest.TestCase):

    def test_hours_past_the_reach_go_the_rest_stay(self):
        import tempfile
        import build
        from datetime import datetime, timezone
        now = datetime(2026, 10, 11, 8, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            for stamp in ('2026101100', '2026101308', '2026101320', '2026101321', '2026102500'):
                open(os.path.join(d, f'{stamp}.npz'), 'wb').close()
                open(os.path.join(d, f'{stamp}.run'), 'w').close()
            open(os.path.join(d, 'VERSION'), 'w').close()
            self.assertEqual(build.prune_store(d, 60, now), 2)
            self.assertEqual(sorted(os.listdir(d)), ['2026101100.npz', '2026101100.run', '2026101308.npz', '2026101308.run', '2026101320.npz', '2026101320.run', 'VERSION'])

    def test_hrrr_store_kept_only_to_the_next_run_after_the_free_window(self):
        import tempfile
        from unittest import mock
        from datetime import datetime, timedelta, timezone
        import build
        self.assertEqual(build.STORE_AFTER_FREE_H, {'hrrr': 12})
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        with tempfile.TemporaryDirectory() as out:
            d = os.path.join(out, 'hrrr', '_hours')
            os.makedirs(d)
            stamps = [f'{now + timedelta(hours=h):%Y%m%d%H}' for h in (-20, 0, 11, 14, 40)]
            for stamp in stamps:
                open(os.path.join(d, f'{stamp}.npz'), 'wb').close()
            with mock.patch.object(build, 'write_free') as wf, mock.patch('sys.argv', ['build.py', '--model', 'hrrr', '--out', out, '--free-only']):
                self.assertEqual(build.main(), 0)
            wf.assert_called_once()
            self.assertEqual(sorted((f[:10] for f in os.listdir(d))), stamps[:3])

    def test_lanes_download_ahead_and_keep_order(self):
        import build
        seen = []
        real_p, real_h = (build._prefetch, build._process_hour)
        build._prefetch = lambda job: seen.append(('get', job)) or f'raw{job}'
        build._process_hour = lambda job, raw=None: (seen.append(('do', job, raw)), job)[1]
        try:

            class Q(list):
                put = list.append
            out = Q()
            build._lane([(0, 10), (1, 11), (2, 12), (3, 13)], out)
        finally:
            build._prefetch, build._process_hour = (real_p, real_h)
        self.assertEqual(out, [(0, 10), (1, 11), (2, 12), (3, 13)])
        self.assertEqual([s[1:] for s in seen if s[0] == 'do'], [(10, 'raw10'), (11, 'raw11'), (12, 'raw12'), (13, 'raw13')])
        self.assertEqual(sorted((s[1] for s in seen if s[0] == 'get')), [10, 11, 12, 13])

class CumulusOrBlue(unittest.TestCase):

    def test_model_low_cloud_decides(self):
        import build
        ws = np.array([2.0, 2.0, 2.0, 2.0, 0.3])
        zi = np.array([1500.0, 1500, 1500, 1500, 1500])
        lcl = np.array([1200.0, 1200, 1450, 1800, 1000])
        low = np.array([30.0, 0, 20, 90, 50])
        cu = build.cumulus_forms(ws, zi, lcl, low)
        self.assertEqual(cu.tolist(), [True, False, True, False, False])

    def test_scattered_low_cloud_over_the_top(self):
        import build
        ws = np.full(5, 2.0)
        zi = np.full(5, 1000.0)
        lcl = np.array([1200.0, 1200, 1200, 1600, 1100])
        low = np.array([35.0, 3, 60, 35, 20])
        self.assertEqual(build.cumulus_forms(ws, zi, lcl, low).tolist(), [True, False, False, False, True])
        far = build.cumulus_forms(np.full(1, 2.0), np.full(1, 1000.0), np.full(1, 1600.0), np.full(1, 35.0))
        self.assertFalse(far[0], 'LCL 600 m over the thermals: blue (zi + max(0.5 zi, CU_OVERSHOOT_MIN))')
        one = lambda v: np.full(1, float(v))
        self.assertTrue(build.cumulus_forms(one(0.57), one(331), one(772), one(6))[0])
        import sky
        self.assertAlmostEqual(float(sky.cu_reach(one(331), one(772))[0]), 1 - 441 / 500, places=6)
        self.assertEqual(float(sky.cu_reach(one(1500), one(1200))[0]), 1.0)

    def test_cover_from_the_band_at_the_base(self):
        import build
        import physics as P
        ps = np.full((1, 2), 780.0)
        p_base = np.array([[600.0, 700.0]])
        bands = [np.array([[2.0, 30]]), np.array([[40.0, 0]]), np.array([[0.0, 0]])]
        np.testing.assert_allclose(build.cumulus_cover(p_base, ps, P.COVER_TOPS_NCEP, bands), [[40, 30]])
        cc = np.zeros((len(P.CC_LEVELS), 1, 2), np.float32)
        cc[P.CC_LEVELS.index(600)] = 40
        cc[P.CC_LEVELS.index(700), 0, 1] = 30
        np.testing.assert_allclose(build.cumulus_cover(p_base, ps, P.COVER_TOPS_NCEP, bands, cc), [[40, 30]])
        cc[:] = 0
        cc[P.CC_LEVELS.index(500)] = 40
        out = build.cumulus_cover(p_base, ps, P.COVER_TOPS_NCEP, bands, cc)
        self.assertLess(out[0, 0], build.CU_MIN_LOW_CLOUD)

    def test_icon_bands(self):
        import physics as P
        lo, mid = P.band_tops(('icon', 0.7895, 0.3948), np.array([1000.0, 750.0]))
        np.testing.assert_allclose(lo, [789.5, 592.1], atol=0.2)
        np.testing.assert_allclose(mid, [394.8, 341.9], atol=0.2)

    def test_margin_without_low_cloud(self):
        import build
        ws = np.full(3, 2.0)
        zi = np.full(3, 1500.0)
        lcl = np.array([1450.0, 1250, 1000])
        self.assertEqual(build.cumulus_forms(ws, zi, lcl, None).tolist(), [False, True, True])
        self.assertEqual(build.cumulus_forms(ws, zi, lcl, np.full(3, np.nan)).tolist(), [False, True, True])

class RainDamp(unittest.TestCase):

    def test_steady_rain_stops_thermals_showers_dont(self):
        import build
        rain = np.array([2.0, 2.0, 0.08, 0.3, 2.0, 2.0])
        ws = np.array([0.8, 0.8, 0.8, 0.8, 2.5, 1.5])
        conv = np.array([False, True, False, False, False, False])
        d = build.rain_damp(rain, conv, ws)
        np.testing.assert_allclose(d, [0.0, 1.0, 1.0, 0.5, 1.0, 0.5])

    def test_convective_rain_matches_js(self):
        import build
        od = np.array([40.0, 0, 0, 0, 0])
        cd = np.array([0.0, 3000, 0, 0, 0])
        refc = np.array([0.0, 0, 47, 42, 42])
        rain = np.array([1.0, 1, 1, 3, 1])
        self.assertEqual(build.convective_rain(od, cd, refc, rain).tolist(), [True, True, True, True, False])
        self.assertEqual(build.convective_rain(od[2:], cd[2:], None, np.array([1.0, 7, 1])).tolist(), [False, True, False])

    def test_the_models_own_showers_decide(self):
        import build
        rain = np.array([2.0, 2.0, 8.0, 0.05])
        sh = build.showers_share(np.array([1.5, 0.4, 0.0, 0.05]), rain)
        np.testing.assert_allclose(sh[:3], [0.75, 0.2, 0.0])
        self.assertTrue(np.isnan(sh[3]), 'no share where it barely rains')
        z = np.zeros(4)
        self.assertEqual(build.convective_rain(z, z, None, rain, sh).tolist(), [True, False, False, False])
        self.assertIsNone(build.showers_share(None, rain))

    def test_hrrr_showers_from_the_hours_radar(self):
        import build
        rain = np.array([2.0, 2.0, 0.05])
        z = np.zeros(3)
        sh = build.showers_share(None, rain, z, z, np.array([48.0, 20, 50]))
        np.testing.assert_allclose(sh[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(sh[2]))

class BandsAboveGround(unittest.TestCase):

    def test_bands_by_height(self):
        import build
        import physics as P
        L = len(P.CC_LEVELS)
        ps = np.full((1, 3), 1000.0)
        h = np.array([110.0, 540, 990, 1460, 1950, 3010, 4200, 5570, 7180, 9160, 10360, 11800], np.float32)[:, None, None] * np.ones((1, 1, 3))
        cc = np.zeros((L, 1, 3), np.float32)
        cc[P.CC_LEVELS.index(700), 0, 0] = 80
        cc[P.CC_LEVELS.index(900), 0, 1] = 70
        lcc, mcc = (np.array([[80.0, 70, 40]]), np.array([[0.0, 0, 0]]))
        low, mid, high = build.bands_agl([lcc, mcc], cc, h, ps, P.COVER_TOPS_NCEP)
        np.testing.assert_allclose(low, [[0, 70, 40]])
        np.testing.assert_allclose(mid, [[80, 0, 0]])
        np.testing.assert_allclose(high, [[0, 0, 0]])

    def test_mid_band_cloud_high_up_is_high(self):
        import build
        import physics as P
        L = len(P.CC_LEVELS)
        ps = np.full((1, 3), 1000.0)
        h = np.array([110.0, 540, 990, 1460, 1950, 3010, 4200, 5570, 7500, 9160, 10360, 11800], np.float32)[:, None, None] * np.ones((1, 1, 3))
        cc = np.zeros((L, 1, 3), np.float32)
        cc[P.CC_LEVELS.index(400), 0, 0] = 70
        cc[P.CC_LEVELS.index(600), 0, 1] = 60
        cc[P.CC_LEVELS.index(400), 0, 2] = 20
        lcc, mcc, hcc = (np.array([[0.0, 0, 0]]), np.array([[70.0, 60, 20]]), np.array([[10.0, 30, 50]]))
        low, mid, high = build.bands_agl([lcc, mcc, hcc], cc, h, ps, P.COVER_TOPS_NCEP)
        np.testing.assert_allclose(low, [[0, 0, 0]])
        np.testing.assert_allclose(mid, [[0, 60, 0]])
        np.testing.assert_allclose(high, [[70, 30, 50]])

    def test_high_ground_mid_band_is_low(self):
        import build
        import physics as P
        L = len(P.CC_LEVELS)
        ps = np.full((1, 1), 780.0)
        h = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 750, 1900, 3300, 4900, 6900, 8100, 9500], np.float32)[:, None, None]
        cc = np.zeros((L, 1, 1), np.float32)
        cc[P.CC_LEVELS.index(600)] = 60
        low, mid, _ = build.bands_agl([np.array([[0.0]]), np.array([[60.0]])], cc, h, ps, P.COVER_TOPS_NCEP)
        np.testing.assert_allclose(low, [[60]])
        np.testing.assert_allclose(mid, [[0]])

class StormsAndDecks(unittest.TestCase):

    def test_rain_needs_a_deck_to_be_steady(self):
        import build
        z = np.zeros(3)
        rain = np.full(3, 1.5)
        tcc = np.array([45.0, 95, 70, 95])
        lcc = np.array([30.0, 90, 88, 60])
        z = np.zeros(4)
        rain = np.full(4, 1.5)
        self.assertEqual(build.convective_rain(z, z, None, rain, None, tcc, lcc, None).tolist(), [True, False, False, True])
        self.assertEqual(build.convective_rain(z, z, None, rain).tolist(), [False] * 4)

class ThermalWidth(unittest.TestCase):

    def test_published_bs_is_rasps(self):
        import build
        s, lv, levels, _ = grid_column(n=3, sfc_t=28, sfc_td=12, lapse=0.0085, shear=4.0)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(500), shf=one(80), hpbl=one(1200), u10=one(6), v10=one(0))
        out = build.analyse(dict(s), lv, levels, 0.25, np.full((3, 3), 40, np.float32))
        ws, bs = (float(out['wstar'][1, 1]), float(out['bs'][1, 1]))
        u_bl = float(np.hypot(out['u_blavg'][1, 1], out['v_blavg'][1, 1]))
        self.assertGreater(ws, 1)
        self.assertAlmostEqual(bs, ws / (np.sqrt(0.003) * u_bl), places=3)
        self.assertEqual(build.ENC['bs'], (0, 20, None))
        s2 = dict(s, prate=one(1.0), tcc=one(100), lcc=one(100), mcc=one(0), hcc=one(0), cape=one(0))
        wet = build.analyse(s2, lv, levels, 0.25, np.full((3, 3), 40, np.float32))
        self.assertLess(float(wet['wstar'][1, 1]), ws)
        self.assertAlmostEqual(float(wet['bs'][1, 1]), float(wet['wstar'][1, 1]) / (np.sqrt(0.003) * u_bl), places=3)

    def test_matches_allen_and_js(self):
        import build
        zi = np.array([2000.0, 1500, 1500, 150])
        w = build.thermal_width(zi, np.array([8.13, 9.5, 6.5, 9]), np.array([10.0, 20, 20, 10]))
        self.assertAlmostEqual(w[0], 2 * 0.102 * 0.6 ** (1 / 3) * 0.85 * 2000, delta=1)
        self.assertGreater(w[1], w[2])
        self.assertTrue(np.isnan(w[3]))
        blue = build.thermal_width(zi, np.array([8.13, 9.5, 6.5, 9]), np.array([10.0, 20, 20, 10]), np.array([True, False, False, False]))
        self.assertAlmostEqual(blue[0], w[0] * build.BLUE_WIDTH, delta=0.01)
        self.assertEqual(blue[1], w[1])

def grid_column(n=4, sfc_t=20.0, sfc_td=10.0, lapse=0.0065, rh_levels=60.0, shear=0.0, stretch=0.0):
    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300], np.float32)
    z1 = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
    t1 = (sfc_t - lapse * z1).astype(np.float32)
    L = len(levels)
    z = np.broadcast_to(z1[:, None, None], (L, n, n)).astype(np.float32)
    t = np.broadcast_to(t1[:, None, None], (L, n, n)).astype(np.float32)
    rh = np.full((L, n, n), rh_levels, np.float32)
    ix = np.arange(n, dtype=np.float32)[None, None, :]
    u = (10 + shear * z / 1000 + stretch * ix * (z / 5000)).astype(np.float32)
    v = np.zeros((L, n, n), np.float32)
    one = lambda x: np.full((n, n), x, np.float32)
    td = P.dewpoint_from_rh(t, rh)
    prof = P.Profile(one(0), one(1013.25), one(sfc_t), one(sfc_td), one(10), one(0), z, levels, t, td, u, v)
    s = {'zs': one(0), 't2': one(sfc_t), 'td2': one(sfc_td), 'u10': one(10), 'v10': one(0), 'tcc': one(80)}
    lv = {'z': z, 't': t, 'rh': rh, 'td': td, 'u': u, 'v': v}
    return (s, lv, levels, prof)

class RidgeFields(unittest.TestCase):

    def out(self, lapse=0.0065, step=0.25, w=None, hill=300.0):
        import build
        s, lv, levels, _ = grid_column(n=3, lapse=lapse)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(0), shf=one(0), hpbl=one(200))
        s['zs'] = np.array([[0, 0, 0], [0, hill, 0], [0, 0, 0]], np.float32)
        if w is not None:
            lv['w'] = np.full(lv['t'].shape, w, np.float32)
        return build.analyse(dict(s), lv, levels, step, one(40))

    def test_stability(self):
        n = lambda lapse: float(self.out(lapse)['nlow'][0, 0])
        self.assertTrue(0.015 < n(0.0) < 0.021, n(0.0))
        self.assertTrue(0.008 < n(0.0065) < 0.013, n(0.0065))
        self.assertLess(n(0.0098), 0.004)
        self.assertEqual(n(0.012), 0.0)
        self.assertTrue(np.isnan(self.out(hill=0)['nlow']).all())

    def test_wave_band_fine_grids_only(self):
        import build
        fine = self.out(step=0.05, w=0.5)['wlow']
        self.assertAlmostEqual(float(fine[0, 0]), 0.5, delta=0.01)
        self.assertNotIn('wlow', self.out(step=0.25, w=0.5))
        self.assertNotIn('wlow', self.out(step=0.05))
        self.assertTrue(np.isnan(self.out(step=0.05, w=0.5, hill=0)['wlow']).all())
        self.assertEqual(build.ENC['nlow'], (0, 0.04, None))
        self.assertEqual(build.ENC['wlow'], (-4, 4, None))
        for f in ('nlow', 'wlow'):
            self.assertNotIn(f, build.FREE_FIELDS)

class WaveExposure(unittest.TestCase):

    def test_ridge_across_the_wind(self):
        import math
        try:
            import wave_exposure as we
        except ImportError:
            self.skipTest('no wave_exposure.py here')
        m = 1000.0
        x = np.arange(400)
        a = 9000 / (2 * math.pi) / m
        slab = np.tile(500 / (1 + ((x - 200) / a) ** 2), (60, 1)).astype(np.float32)
        E = we.exposure(slab, m)
        j = we.LAMS.index(9000.0)
        east, west, north = (E[j * we.DIRS + d][30] for d in (2, 6, 0))
        self.assertAlmostEqual(int(np.argmax(east)) - 200, 4, delta=1)
        self.assertAlmostEqual(int(np.argmax(west)) - 200, -4, delta=1)
        self.assertLess(float(east[:195].max()), 0.002)
        self.assertLess(float(north.max()), 0.002)
        self.assertTrue(1.1 < 15 * float(east.max()) < 1.7, east.max())
        self.assertLess(float(east.max()), we.SCALE)

class LeeWaveFields(unittest.TestCase):

    def test_published_over_flat_ground(self):
        import build
        s, lv, levels, _ = grid_column(n=3, shear=3.0)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(0), shf=one(0), hpbl=one(200))
        out = build.analyse(dict(s), lv, levels, 0.25, one(40))
        self.assertIsNone(out['snowd'])
        snod = one(0.3)
        snod[0, 0] = -0.01
        snowd = build.analyse(dict(s, snod=snod), lv, levels, 0.25, one(40))['snowd']
        self.assertEqual((float(snowd[0, 0]), float(snowd[1, 1])), (0.0, np.float32(0.3)))
        self.assertLessEqual(float(snowd.max()), build.ENC['snowd'][1])
        self.assertTrue((out['wtrap'] > 5).all() and (out['wtrap'] <= 100).all(), out['wtrap'])
        self.assertTrue((out['wlam'] > 1000).all() and (out['wlam'] < 60000).all(), out['wlam'])
        self.assertTrue((out['wavepot'] == 0).all())
        for ft in build.WAVE_ALTS_FT:
            d = out[f'dpd{ft}']
            self.assertTrue((d >= 0).all() and (d < 60).all(), (ft, d))
            self.assertIn(f'dpd{ft}', build.NOW_FIELDS)
            t = out[f't{ft}']
            lo, hi, _ = build.ENC[f't{ft}']
            self.assertTrue((t > lo).all() and (t < hi).all(), (ft, t))
            self.assertIn(f't{ft}', build.NOW_FIELDS)
        self.assertTrue((out['t24000'] < out['t8000'] - 20).all())
        s, lv, levels, _ = grid_column(n=3, shear=3.0, lapse=0.0098)
        s.update(ps=one(1013.25), sw=one(0), shf=one(0), hpbl=one(200))
        neutral = build.analyse(dict(s), lv, levels, 0.25, one(40))
        self.assertTrue(np.isnan(neutral['wtrap']).all() and np.isnan(neutral['wlam']).all())
        sp = np.hypot(out['u_wl'], out['v_wl'])
        self.assertTrue((sp > 10).all() and (sp < 30).all(), sp)
        self.assertTrue(np.isnan(neutral['u_wl']).all() and np.isnan(neutral['v_wl']).all())
        self.assertEqual(build.ENC['u_wl'], (-60, 60, None))
        self.assertIn('u_wl', build.NOW_FIELDS)
        self.assertEqual(build.ENC['wtrap'], (0, 100, None))
        self.assertEqual(build.ENC['wlam'], (0, 60000, None))
        for f in ('wtrap', 'wlam'):
            self.assertNotIn(f, build.FREE_FIELDS)

class NightStorms(unittest.TestCase):

    def test_radar_core_without_thermals(self):
        import build
        s, lv, levels, _ = grid_column(n=3, shear=3.0)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(0), shf=one(0), hpbl=one(200), cape=one(800))
        s['refc'] = np.array([[55, 42, 42], [30, 0, 48], [0, 0, 0]], np.float32)
        s['cape'] = np.array([[0, 800, 50], [800, 800, 800], [0, 0, 0]], np.float32)
        od = build.analyse(dict(s), lv, levels, 0.25, one(40))['overdev']
        self.assertTrue((od[0, :2] >= 70).all(), od)
        self.assertTrue(od[0, 2] < 50 and od[1, 0] < 50 and (od[1, 1] < 50), od)
        self.assertTrue(od[1, 2] >= 70, od)
        self.assertTrue((build.analyse(dict(s, refc=None), lv, levels, 0.25, one(40))['overdev'] < 50).all())

class GeneralAviation(unittest.TestCase):

    def test_flight_category_boundaries(self):
        ft, sm = (P.FT, P.SM)
        ceil = np.array([5000 * ft, 2500 * ft, 800 * ft, 300 * ft, 5000 * ft, 5000 * ft, np.nan])
        vis = np.array([10 * sm, 10 * sm, 10 * sm, 10 * sm, 4 * sm, 0.5 * sm, np.nan])
        self.assertEqual(P.flight_category(ceil, vis)[:6].tolist(), [1, 2, 3, 4, 2, 4])
        self.assertTrue(np.isnan(P.flight_category(ceil, vis)[6]))

    def test_ceiling_msl_to_agl_and_none(self):
        c = P.ceiling_agl(np.array([2000.0, np.nan, 20000.0, 500.0]), np.array([1500.0, 0, 0, 0]), np.array([True, True, True, False]))
        self.assertEqual(c[0], 500)
        self.assertEqual(c[1], P.CEIL_NONE)
        self.assertEqual(c[2], P.CEIL_NONE)
        self.assertTrue(np.isnan(c[3]))

    def test_estimates_dont_invent_ifr(self):
        self.assertGreater(float(P.visibility_estimate(np.array([20.0]), np.array([18.9]), np.array([0.0]))[0]), 5 * P.SM)
        self.assertGreater(float(P.visibility_estimate(np.array([20.0]), np.array([18.9]), np.array([0.5]))[0]), 5 * P.SM)
        zs = np.zeros(1, np.float32)
        zc = np.array([[100.0], [1200.0]], np.float32)
        cc = np.array([[75.0], [80.0]], np.float32)
        rh = np.array([[95.0], [99.0]], np.float32)
        self.assertGreater(float(P.ceiling_from_cover(zc, cc, zs, np.array([93.0]), None, rh)[0]), 500)
        self.assertLess(float(P.ceiling_from_cover(zc, cc, zs, np.array([93.0]))[0]), 200)

    def test_visibility_estimate(self):
        dry = P.visibility_estimate(np.array([25.0]), np.array([5.0]), np.array([0.0]))
        fog = P.visibility_estimate(np.array([10.0]), np.array([9.9]), np.array([0.0]))
        rain = P.visibility_estimate(np.array([25.0]), np.array([5.0]), np.array([8.0]))
        snow = P.visibility_estimate(np.array([-2.0]), np.array([-5.0]), np.array([2.0]), np.array([True]))
        self.assertAlmostEqual(float(dry[0]), 10 * P.SM, delta=1)
        self.assertLess(float(fog[0]), P.SM)
        self.assertLess(float(rain[0]), 3 * P.SM)
        self.assertLess(float(snow[0]), P.SM)

    def test_freezing_level(self):
        _, _, _, warm = grid_column(sfc_t=20.0)
        _, _, _, cold = grid_column(sfc_t=-3.0, sfc_td=-6.0)
        self.assertAlmostEqual(float(P.freezing_level(warm)[0, 0]), 20 / 0.0065, delta=250)
        self.assertEqual(float(P.freezing_level(cold)[0, 0]), 0)

    def test_icing_needs_supercooled_moisture(self):
        s, lv, levels, _ = grid_column(sfc_t=12.0, rh_levels=98)
        order = np.argsort(-levels)
        pot, base, top = P.icing_potential(lv['z'][order], lv['t'][order], lv['rh'][order], None, s['zs'])
        self.assertGreater(float(pot[0, 0]), 50)
        self.assertTrue(float(base[0, 0]) > 12 / 0.0065 - 500)
        s, lv, levels, _ = grid_column(sfc_t=12.0, rh_levels=40)
        dry, _, _ = P.icing_potential(lv['z'][order], lv['t'][order], lv['rh'][order], None, s['zs'])
        self.assertEqual(float(dry[0, 0]), 0)
        clw = np.full(lv['t'].shape, 0.0003, np.float32)
        wet, _, _ = P.icing_potential(lv['z'][order], lv['t'][order], lv['rh'][order], clw, s['zs'])
        self.assertGreater(float(wet[0, 0]), 90)

    def test_turbulence_needs_shear_and_deformation(self):
        dx = lambda a: np.gradient(a, axis=1) / 25000.0
        dy = lambda a: -np.gradient(a, axis=0) / 25000.0
        _, _, _, calm = grid_column(shear=0, stretch=0)
        _, _, _, jet = grid_column(shear=8, stretch=6)
        ti_calm, _ = P.ellrod_turbulence(calm, dx, dy)
        ti_jet, z = P.ellrod_turbulence(jet, dx, dy)
        self.assertEqual(float(ti_calm.max()), 0)
        self.assertGreater(float(ti_jet[1, 1]), 8)
        self.assertTrue(300 < float(z[1, 1]) < 10500)

    def test_no_turbulence_from_a_mountain_front(self):
        from scipy.ndimage import uniform_filter
        n = 12
        levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300], np.float32)
        z1 = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
        z = np.broadcast_to(z1[:, None, None], (len(levels), n, n)).astype(np.float32)
        t = (20 - 0.0065 * z).astype(np.float32)
        u = (15 + 4 * z / 1000).astype(np.float32)
        zs = np.zeros((n, n), np.float32)
        zs[:, 7:] = 2800
        one = lambda x: np.full((n, n), x, np.float32)
        prof = P.Profile(zs, (1013.25 * (1 - zs / 44330) ** 5.255).astype(np.float32), one(15), one(5), one(3), one(0), z, levels, t, P.dewpoint_from_rh(t, np.full_like(t, 50)), u, np.zeros_like(u))
        dx = lambda a: np.gradient(a, axis=1) / 13000.0
        dy = lambda a: -np.gradient(a, axis=0) / 13000.0
        ti, _ = P.ellrod_turbulence(prof, dx, dy, lambda a: uniform_filter(a, 2, mode='nearest'), one(500))
        self.assertLess(float(ti.max()), 0.1)

    def test_turbulence_only_above_the_boundary_layer(self):
        dx = lambda a: np.gradient(a, axis=1) / 25000.0
        dy = lambda a: -np.gradient(a, axis=0) / 25000.0
        s, lv, levels, _ = grid_column()
        top = float(lv['z'][2, 0, 0])
        ix = np.arange(4, dtype=np.float32)[None, None, :]
        u = (10 + 6 * ix * np.minimum(lv['z'], top) / 1000).astype(np.float32)
        one = lambda x: np.full((4, 4), x, np.float32)
        low = P.Profile(one(0), one(1013.25), one(20), one(10), one(10), one(0), lv['z'], levels, lv['t'], lv['td'], u, lv['v'])
        ti_bl, _ = P.ellrod_turbulence(low, dx, dy, zi=one(top))
        self.assertLess(float(ti_bl.max()), 0.1)
        ti_night, z_night = P.ellrod_turbulence(low, dx, dy, zi=np.zeros((4, 4), np.float32))
        self.assertGreaterEqual(float(np.nanmin(z_night[ti_night > 0])), P.CAT_FLOOR_AGL + 500)
        ti_old, _ = P.ellrod_turbulence(low, dx, dy, zmin_agl=300.0)
        self.assertGreater(float(ti_old[1, 1]), P.TURB_BANDS[0])
        _, _, _, jet = grid_column(shear=8, stretch=6)
        ti, z = P.ellrod_turbulence(jet, dx, dy, zi=np.full((4, 4), 3000, np.float32))
        self.assertGreater(float(ti[1, 1]), P.TURB_BANDS[0])
        self.assertGreaterEqual(float(z[1, 1]), 3000 + 500)
        self.assertLessEqual(float(np.nanmax(z)), float(jet.ztop.max()) - 500)

    def test_turbulence_and_icing_bands(self):
        light, moderate, severe = P.TURB_BANDS
        self.assertEqual((light, moderate, severe), (6.0, 12.0, 18.0))
        self.assertEqual(P.ICE_BANDS[0], P.ICE_MIN)
        self.assertEqual(list(P.ICE_BANDS), sorted(P.ICE_BANDS))
        import build
        self.assertGreater(build.ENC['turb'][1], severe)

    def test_freezing_rain_is_not_an_icing_potential(self):
        import build
        s, lv, levels, prof = grid_column(sfc_t=1.0, sfc_td=0.5, rh_levels=50)
        s.update(crain=np.zeros_like(s['zs']), csnow=np.zeros_like(s['zs']), cfrzr=np.ones_like(s['zs']), cicep=np.zeros_like(s['zs']))
        rain = np.full_like(s['zs'], 1.0)
        dx = lambda a: np.gradient(a, axis=1) / 25000.0
        dy = lambda a: -np.gradient(a, axis=0) / 25000.0
        out = build.general_aviation(s, lv, levels, prof, rain, 0.25, dx, dy)
        self.assertEqual(float(out['ptype'][1, 1]), 5)
        self.assertLess(float(out['icing'][1, 1]), P.ICE_MIN)
        self.assertTrue(np.isnan(out['icebase'][1, 1]) and np.isnan(out['icetop'][1, 1]))

    def test_icing_single_level_has_base_equal_to_top(self):
        z = np.array([500, 1500, 3000, 5000], np.float32)[:, None, None]
        t = np.array([5, -1, -8, -30], np.float32)[:, None, None]
        rh = np.array([50, 50, 97, 50], np.float32)[:, None, None]
        pot, base, top = P.icing_potential(z, t, rh, None, np.zeros((1, 1), np.float32))
        self.assertGreaterEqual(float(pot[0, 0]), P.ICE_MIN)
        self.assertEqual(float(base[0, 0]), 3000)
        self.assertEqual(float(top[0, 0]), 3000)

    def test_wind_encoding_keeps_winter_jets(self):
        import build
        keys = [k for k in build.ENC if k.startswith(('u_', 'v_'))]
        self.assertIn('u_a30000', keys)
        self.assertIn('v_sfc', keys)
        for k in keys:
            aloft = k[2] == 'a' and int(k[3:]) >= build.WIND_ALOFT_FT
            self.assertEqual(build.ENC[k], (-100, 100, None) if aloft else (-60, 60, None), k)
        lo, hi, tr = build.ENC['u_a30000']
        decode = lambda q: lo + q / 254 * (hi - lo)
        jet = np.float32([75.0, -82.0])
        q = build.quantize(jet, lo, hi, tr)
        self.assertTrue(np.all(np.abs(decode(q.astype(np.float32)) - jet) <= (hi - lo) / 254 / 2 + 0.0001))
        self.assertEqual(build.quantize(np.float32([150.0]), lo, hi, tr)[0], 254)

    def test_precip_type(self):
        rain = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        flags = lambda *xs: np.array(xs, np.float32)
        code = P.precip_type(rain, crain=flags(1, 0, 1, 0, 1), csnow=flags(0, 1, 1, 0, 0), cfrzr=flags(0, 0, 0, 1, 0), cicep=flags(0, 0, 0, 0, 0))
        self.assertEqual(code.tolist(), [1, 2, 3, 5, 0])
        ec = P.precip_type(np.ones(5), ecmwf=np.array([1, 5, 7, 3, 8]))
        self.assertEqual(ec.tolist(), [1, 2, 3, 5, 4])

    def test_kuchera_ratio(self):
        levels = np.array([1000, 850, 700, 500, 400], np.float32)
        z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)[:, None, None]
        one = lambda v: np.full((1, 1), v, np.float32)

        def ratio(sfc, aloft, above=-40.0):
            t = np.array([sfc, aloft, aloft - 5, aloft - 15, above], np.float32)[:, None, None]
            prof = P.Profile(one(0), one(1013.25), one(sfc), one(sfc - 1), one(0), one(0), z, levels, t, t - 1, 0 * t, 0 * t)
            return round(float(P.kuchera_ratio(prof)[0, 0]), 2)
        self.assertEqual(ratio(-1.99, -5), 12.0)
        self.assertEqual(ratio(-7.99, -10), 18.0)
        self.assertEqual(ratio(-5, 1.01), 6.0)
        self.assertEqual(ratio(-30, -32), 30.0)
        self.assertEqual(ratio(3, 0), 3.0)
        self.assertEqual(ratio(-7.99, -10, above=5), 18.0)

    def test_snow_ratio_only_where_snow_falls(self):
        import build
        s, lv, levels, prof = grid_column(sfc_t=-6.0, sfc_td=-7.0, rh_levels=95)
        n = s['zs'].shape
        s['ptype'] = np.zeros(n, np.float32)
        s['ptype'][0, 0], s['ptype'][0, 1], s['ptype'][1, 0] = (5, 7, 1)
        dx = lambda a: np.gradient(a, axis=1) / 25000.0
        dy = lambda a: -np.gradient(a, axis=0) / 25000.0
        slr = build.general_aviation(s, lv, levels, prof, np.ones(n, np.float32), 0.25, dx, dy)['slr']
        self.assertTrue(np.isfinite(slr[0, 0]) and np.isfinite(slr[0, 1]))
        self.assertTrue(np.isnan(slr[1, 0]))
        self.assertAlmostEqual(float(slr[0, 0]), float(P.kuchera_ratio(prof)[0, 0]), 5)
        lo, hi, _ = build.ENC['slr']
        self.assertEqual((lo, hi), P.SLR_RANGE)
        self.assertIn('slr', build.POINT_FIELDS)

    def test_ecmwf_style_inputs(self):
        import build
        s, lv, levels, prof = grid_column(sfc_t=15.0, sfc_td=14.8, rh_levels=95)
        s['ptype'] = np.zeros_like(s['zs'])
        dx = lambda a: np.gradient(a, axis=1) / 25000.0
        dy = lambda a: -np.gradient(a, axis=0) / 25000.0
        out = build.general_aviation(s, lv, levels, prof, np.zeros_like(s['zs']), 0.25, dx, dy)
        for k in ('ceil', 'vis', 'fltcat', 'ptype', 'slr', 'refc', 'frzl', 'ctop', 'icing', 'icebase', 'icetop', 'turb', 'turbz', 'turbbase', 'turbtop', 'llws'):
            self.assertIn(k, out)
            self.assertEqual(out[k].shape, s['zs'].shape, k)
        self.assertLess(float(out['ceil'][0, 0]), 500)
        self.assertGreaterEqual(float(out['fltcat'][0, 0]), 3)
        q = {k: build.quantize(v, *build.ENC[k]) for k, v in out.items()}
        self.assertEqual(set(q), set(out))

class AviationSnapshot(unittest.TestCase):

    def test_airports_cells_idents_and_magvar_sign(self):
        import json
        import tempfile
        import aviation as A
        fake = {'airports.csv': [{'id': str(i), 'type': 'small_airport', 'name': f'Field {i}', 'latitude_deg': str(30 + i % 20), 'longitude_deg': str(-120 + i % 40), 'elevation_ft': '500', 'icao_code': '', 'gps_code': f'X{i:04d}', 'local_code': f'{i:04d}', 'ident': f'X{i:04d}', 'iata_code': '', 'municipality': 'Town', 'iso_region': 'US-XX'} for i in range(1200)] + [{'id': 'big', 'type': 'large_airport', 'name': 'Big Intl', 'latitude_deg': '39.4', 'longitude_deg': '-77.4', 'elevation_ft': '300', 'icao_code': 'KBIG', 'gps_code': 'KBIG', 'local_code': 'BIG', 'ident': 'KBIG', 'iata_code': 'BIG', 'municipality': 'City', 'iso_region': 'US-MD'}, {'id': 'gone', 'type': 'closed', 'name': 'Closed', 'latitude_deg': '39', 'longitude_deg': '-77', 'ident': 'KOLD'}, {'id': 'eu', 'type': 'large_airport', 'name': 'London', 'latitude_deg': '51.47', 'longitude_deg': '-0.46', 'ident': 'EGLL'}, {'id': 'as', 'type': 'large_airport', 'name': 'Far', 'latitude_deg': '35.55', 'longitude_deg': '139.78', 'ident': 'RJTT'}], 'runways.csv': [{'airport_ref': 'big', 'le_ident': '09', 'he_ident': '27', 'length_ft': '9000', 'width_ft': '150', 'surface': 'ASP', 'lighted': '1', 'closed': '0', 'le_heading_degT': '88'}], 'airport-frequencies.csv': [{'airport_ref': 'big', 'type': 'TWR', 'description': 'TOWER', 'frequency_mhz': '118.3'}], 'navaids.csv': [{'type': 'VORTAC', 'ident': 'WWW', 'name': 'West', 'latitude_deg': '40', 'longitude_deg': '-118', 'frequency_khz': '116800', 'magnetic_variation_deg': '-12.5', 'elevation_ft': '100', 'dme_channel': '115X'}, {'type': 'VOR', 'ident': 'EEE', 'name': 'East', 'latitude_deg': '40', 'longitude_deg': '-75', 'frequency_khz': '112100', 'magnetic_variation_deg': '11.0', 'elevation_ft': '100', 'dme_channel': ''}]}
        real = A.ourairports
        A.ourairports = lambda name: fake[name]
        try:
            with tempfile.TemporaryDirectory() as d:
                kinds = A.build_airports(d)
                self.assertEqual(kinds['L'], 2)
                self.assertNotIn('closed', json.dumps(kinds))
                cells = json.load(open(os.path.join(d, 'cells.json')))['cells']
                big = json.load(open(os.path.join(d, 'cells', A.cell_of(39.4, -77.4) + '.json')))
                rec = next((a for a in big['a'] if a[0] == 'KBIG'))
                self.assertEqual(rec[6], 1)
                self.assertEqual(rec[11][0][:3], ['09/27', 9000, 150])
                self.assertEqual(rec[7], 'BIG')
                ov = json.load(open(os.path.join(d, 'overview.json')))
                self.assertEqual(sorted((a[0] for a in ov['a'])), ['EGLL', 'KBIG'])
                self.assertEqual({n[0] for n in ov['n']}, {'WWW', 'EEE'})
                ids = json.load(open(os.path.join(d, 'idents.json')))
                self.assertIn('KBIG', ids)
                self.assertIn('BIG', ids)
                self.assertIn('EGLL', ids)
                self.assertNotIn('RJTT', ids)
                self.assertEqual(ids['WWW'][4], 'VORTAC')
                search = json.load(open(os.path.join(d, 'airports.json')))
                self.assertEqual(search['v'], 1)
                self.assertEqual(len(search['a']), 1202)
                big = next((a for a in search['a'] if a[0] == 'KBIG'))
                self.assertEqual(big, ['KBIG', 'Big Intl', 'L', 39.4, -77.4, 'City', 'US-MD', 'BIG'])
                self.assertNotIn('KOLD', json.dumps(search))
                self.assertTrue(all((c.count('_') == 1 for c in cells)))
                mv = json.load(open(os.path.join(d, 'magvar.json')))
                j, i = (round((40 - mv['south']) / mv['step']), round((-118 - mv['west']) / mv['step']))
                self.assertGreater(mv['values'][j][i], 5)
                j, i = (round((40 - mv['south']) / mv['step']), round((-75 - mv['west']) / mv['step']))
                self.assertLess(mv['values'][j][i], -5)
        finally:
            A.ourairports = real

    def test_tile_xy(self):
        import aviation as A
        self.assertEqual(A.tile_xy(0.0, 0.0, 1), (1, 1))
        x, y = A.tile_xy(39.1, -94.6, 8)
        self.assertEqual((x, y), (60, 97))

class Borders(unittest.TestCase):

    def test_lines_kept_in_region_and_simplified(self):
        import json
        import tempfile
        import borders as B
        line = [[-100 + i * 0.001, 40.0 + (1e-05 if i % 2 else 0)] for i in range(2001)]
        states = {'features': [{'geometry': {'type': 'LineString', 'coordinates': line}}] * 60 + [{'geometry': {'type': 'LineString', 'coordinates': [[10, 50], [11, 51]]}}]}
        countries = {'features': [{'geometry': {'type': 'MultiLineString', 'coordinates': [[[-120, 49], [-110, 49]]]}}]}
        real = B.get
        B.get = lambda url, **k: json.dumps(states if 'admin_1' in url else countries).encode()
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'borders.json')
                self.assertEqual(B.main(path=path), 61)
                out = json.load(open(path))
                self.assertEqual(out['s'][0], [-100, 40, -98, 40])
                self.assertEqual(out['c'], [[-120, 49, -110, 49]])
        finally:
            B.get = real

class RadarPeak(unittest.TestCase):

    def setUp(self):
        import build
        lat, lon = np.meshgrid(np.arange(40.0, 38.99, -0.01), np.arange(-80.0, -78.99, 0.01), indexing='ij')
        self.rg = build.Regridder(lat, lon, (-80.0, 39.0, -79.0, 40.0, 0.05))
        self.f = np.zeros(lat.shape, np.float32)
        self.f[52, 47] = 50

    def test_keeps_a_small_core(self):
        self.assertEqual(float(self.rg(self.f).max()), 0)
        p = self.rg.peak(self.f)
        self.assertEqual(float(p.max()), 50)
        self.assertEqual(int((p > 0).sum()), 1)
        self.assertEqual(p[10, 9], 50)

    def test_window_as_whole(self):
        whole = self.rg.peak(self.f)
        self.assertTrue(np.array_equal(self.rg.part(5, 15, 3, 12).peak(self.f), whole[5:15, 3:12]))

    def test_radar_takes_it_unsmoothed(self):
        import build
        d = build.radar({'radar': self.rg.peak(self.f), 'refd1': self.rg(self.f)}, step=0.05)
        self.assertEqual(float(d.max()), 50)

class QuarterHours(unittest.TestCase):

    def test_fields_like_the_hourly_ones(self):
        import build
        one = lambda v: np.full((2, 2), v, np.float32)
        s = {'t2': one(293.15), 'td2': one(283.15), 'zs': one(100), 'ceil': one(700), 'vis': one(20000), 'prate': one(2 / 3600), 'crain': one(1), 'csnow': one(0), 'refc': one(-5), 'u10': one(3), 'v10': one(-4)}
        f = build.quarter_fields(s)
        self.assertAlmostEqual(float(f['t2'][0, 0]), 20, places=4)
        self.assertAlmostEqual(float(f['ceil'][0, 0]), 600)
        self.assertEqual(float(f['vis'][0, 0]), 16100)
        self.assertAlmostEqual(float(f['rain'][0, 0]), 2, places=4)
        self.assertEqual(float(f['ptype'][0, 0]), 1)
        self.assertEqual(float(f['refc'][0, 0]), 0)
        self.assertTrue(set(f) <= set(build.QUARTER_FIELDS))

    def test_strips_and_manifest(self):
        import tempfile
        from datetime import datetime, timezone
        import build
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            qdir, out = (os.path.join(d, 'q'), os.path.join(d, 'out'))
            os.makedirs(qdir)
            times = [datetime(2026, 9, 26, 18, m, tzinfo=timezone.utc) for m in (15, 30, 45)]
            for i, t in enumerate(times):
                np.savez_compressed(os.path.join(qdir, f'{t:%Y%m%d%H%M}.npz'), t2=np.full((3, 4), 10 + i, np.uint8), vis=np.full((3, 4), 255, np.uint8))
            days, present = build.write_quarters(out, qdir, times, tile=4)
            self.assertEqual(days, {'20260926': ['202609261815', '202609261830', '202609261845']})
            self.assertEqual(sorted(present), ['q_t2'])
            strip = tile_img(os.path.join(out, '20260926', 'q_t2'), '0_0')
            self.assertEqual(strip.shape, (12, 4))
            self.assertEqual(int(strip[4, 0]), 11)
            self.assertEqual(int(strip[3, 0]), 255)
            man = build.quarter_manifest(days, present)
            self.assertEqual(man['q_t2']['min'], build.ENC['t2'][0])
            kept = build.stored_quarters(qdir, times[1], times[2], prune=True)
            self.assertEqual(kept, times[1:])
            self.assertFalse(os.path.exists(os.path.join(qdir, '202609261815.npz')))

class PointPacks(unittest.TestCase):

    def test_layout_overlap_and_blocks_left_out(self):
        import tempfile
        from datetime import datetime, timezone
        import build
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            hours, out = (os.path.join(d, 'h'), os.path.join(d, 'out'))
            os.makedirs(hours)
            ny, nx = (5, 9)
            times = [datetime(2026, 10, 8, h, tzinfo=timezone.utc) for h in (17, 18)]
            yy, xx = np.mgrid[:ny, :nx]
            for k, t in enumerate(times):
                t2 = (10 * yy + xx + 100 * k).astype(np.uint8)
                wstar = np.full((ny, nx), 255, np.uint8)
                wstar[0, 0] = 9
                np.savez_compressed(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), t2=t2, wstar=wstar, elevation=np.full((ny, nx), 7, np.uint8), sst=np.zeros((ny, nx), np.uint8))
            m = build.write_points(out, hours, times, times[0], block=4)
            self.assertEqual(m['block'], 4)
            self.assertEqual(m['fields'], ['t2', 'wstar', 'elevation'])
            self.assertEqual(m['stamps'], ['2026100817', '2026100818'])
            self.assertEqual(sorted(m['tiles']), ['0_0', '1_0'])
            self.assertFalse(os.path.exists(os.path.join(out, 'pt.tmp.npy')))
            self.assertEqual(os.path.exists(os.path.join(out, 'pt')), build.png_too())
            img = pt_img(out, m, '1_0')
            W = 5
            self.assertEqual(img.shape, (2 * W, 3 * W))
            self.assertEqual(int(img[1 * W + 2, 0 * W + 3]), 127)
            first = pt_img(out, m, '0_0')
            self.assertEqual(int(first[2, 4]), int(img[2, 0]))
            self.assertEqual(int(img[0, W + 1]), 255)
            self.assertEqual(int(first[0, W]), 9)
            self.assertEqual(int(img[W + 4, 2 * W + 4]), 7)
            free = build.write_points(os.path.join(d, 'free'), hours, times[:1], times[0], only={'wstar'}, block=4)
            self.assertEqual(free['fields'], ['wstar', 'elevation'])
            self.assertEqual(free['stamps'], ['2026100817'])
            self.assertEqual(free['tiles'], ['0_0'])

class OverviewLevels(unittest.TestCase):

    def test_mean_of_the_valid_codes_where_half_the_block_has_data(self):
        import build
        c = np.array([[10, 20, 100, 255, 7, 255], [30, 41, 101, 255, 255, 255]], np.uint8)
        out = build.downsample(c, 2)
        self.assertEqual(out.tolist(), [[25, 101, 255]])
        self.assertEqual(build.downsample(np.stack([c, c]), 2).tolist(), [[[25, 101, 255]]] * 2)
        self.assertEqual(build.downsample(np.full((2, 2), 255, np.uint8), 2).tolist(), [[255]])

    def test_storm_signals_keep_the_blocks_worst(self):
        import build
        c = np.full((4, 4), 25, np.uint8)
        c[1, 2] = 203
        self.assertEqual(build.downsample_peak(c, 4).tolist(), [[203]])
        self.assertLess(int(build.downsample(c, 4)[0, 0]), 40)
        self.assertEqual(build.downsample_peak(np.array([[255, 255], [255, 9]], np.uint8), 2).tolist(), [[255]])
        self.assertTrue({'overdev', 'refc', 'showers'} <= build.PEAK_FIELDS)

    def test_class_codes_take_the_cell_at_the_block_centre(self):
        import build
        c = np.arange(64, dtype=np.uint8).reshape(8, 8)
        self.assertEqual(build.downsample(c, 2, nearest=True).tolist(), c[0::2, 0::2].tolist())
        self.assertEqual(build.downsample(c, 4, nearest=True).tolist(), [[9, 13], [41, 45]])
        ptype = np.array([[1, 3], [3, 3]], np.uint8)
        self.assertEqual(int(build.downsample(ptype, 2, nearest=True)[0, 0]), 1)
        self.assertEqual(build.CLASS_FIELDS, {'ptype', 'fltcat', 'sinkby', 'cuclass', 'cusky'})
        for name in build.CLASS_FIELDS:
            self.assertEqual(build.ENC[name], (0, 254, None))

    def test_no_ceiling_is_never_averaged_in(self):
        import build
        none = build.quantize(np.array([P.CEIL_NONE]), *build.ENC['ceil'])[0]
        self.assertEqual(build.LEVEL_NONE, {'ceil': int(none)})
        low = int(build.quantize(np.array([300.0]), *build.ENC['ceil'])[0])
        c = np.array([[low, none, low, none, none, none, low, 255], [none, low, none, none, none, 255, 255, 255]], np.uint8)
        out = build.downsample(c, 2, none=none)
        self.assertEqual(out.tolist(), [[low, none, none, 255]])
        self.assertEqual(build.downsample(c, 2).tolist()[0][0], (2 * low + 2 * int(none) + 2) // 4)

    def test_sqrt_fields_average_the_amounts(self):
        import build
        c = np.array([[144, 0], [0, 0]], np.uint8)
        self.assertEqual(int(build.downsample(c, 2, rms=True)[0, 0]), 72)
        self.assertEqual(int(build.downsample(c, 2)[0, 0]), 36)
        self.assertTrue(build._sqrt_from_zero('rain') and (not build._sqrt_from_zero('t2')))

    def test_meteogram_fields_get_no_levels(self):
        import tempfile
        from datetime import datetime, timezone
        import build
        with tempfile.TemporaryDirectory() as d:
            hours, out = (os.path.join(d, 'h'), os.path.join(d, 'out'))
            os.makedirs(hours)
            t = datetime(2026, 9, 26, 18, tzinfo=timezone.utc)
            q = np.full((4, 4), 50, np.uint8)
            np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, cc850=q, elevation=q)
            _, present, lp = build.write_tiles(out, hours, [t], 4, t, levels=[(2, 2)])
            self.assertIn('cc850', present)
            self.assertEqual(sorted(lp[2]), ['elevation', 'wstar'])
            self.assertIn('cc850', build.LEVEL_SKIP)
            self.assertFalse(build.LEVEL_SKIP & (build.CLASS_FIELDS | {'wstar', 'ceil', 'vis', 'u_sfc', 'u_g300', 'wv12000'}))
            self.assertIn('vis', build.SAMPLED_FIELDS)

    def test_vertical_motion_levels_sample_not_average(self):
        import tempfile
        from datetime import datetime, timezone
        import build
        c = np.tile(np.array([[200, 54]], np.uint8), (4, 2))
        with tempfile.TemporaryDirectory() as d:
            hours, out = (os.path.join(d, 'h'), os.path.join(d, 'out'))
            os.makedirs(hours)
            t = datetime(2026, 9, 26, 18, tzinfo=timezone.utc)
            q = np.full((4, 4), 50, np.uint8)
            np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, wv10000=c, wlow=c, elevation=q)
            build.write_tiles(out, hours, [t], 4, t, levels=[(2, 2)])
            from PIL import Image
            for name in ('wv10000', 'wlow'):
                strip = tile_img(os.path.join(out, 'o2', '20260926', name), '0_0')
                self.assertEqual(strip.tolist(), [[200, 200], [200, 200]], name)
            mean = tile_img(os.path.join(out, 'o2', '20260926', 'wstar'), '0_0')
            self.assertEqual(mean.tolist(), [[50, 50], [50, 50]])
        self.assertEqual(int(build.downsample(c, 2)[0, 0]), 127)
        self.assertTrue({f'wv{ft}' for ft in build.WAVE_ALTS_FT} <= build.SAMPLED_FIELDS)
        self.assertFalse(build.SAMPLED_FIELDS & {'wstar', 'conv', 'wavemax', 'sink'})

    def test_partial_blocks_at_the_east_and_south_edges(self):
        import build
        c = np.full((5, 5), 255, np.uint8)
        c[:4, :4] = 50
        c[4, 4] = 90
        c[0:2, 4] = [60, 255]
        out = build.downsample(c, 2)
        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(int(out[2, 2]), 90)
        self.assertEqual(int(out[0, 2]), 60)
        self.assertEqual(int(out[1, 2]), 255)
        self.assertEqual(int(out[2, 0]), 255)
        n = np.arange(25, dtype=np.uint8).reshape(5, 5)
        self.assertEqual(build.downsample(n, 4, nearest=True).tolist(), [[6, 9], [21, 24]])

    def test_level_grid(self):
        import build
        hrrr = {'west': -125.0, 'north': 50.0, 'step': 0.05, 'nx': 1171, 'ny': 521, 'tile': 100, 'ntx': 12, 'nty': 6}
        g2, g4 = (build.level_grid(hrrr, 2), build.level_grid(hrrr, 4))
        self.assertEqual(g2, {'west': -124.975, 'north': 49.975, 'step': 0.1, 'nx': 586, 'ny': 261, 'tile': 200, 'ntx': 3, 'nty': 2})
        self.assertEqual(g4, {'west': -124.925, 'north': 49.925, 'step': 0.2, 'nx': 293, 'ny': 131, 'tile': 200, 'ntx': 2, 'nty': 1})
        lon = lambda g, i: g['west'] + i * g['step']
        self.assertAlmostEqual(lon(g4, 10), np.mean([lon(hrrr, i) for i in range(40, 44)]), places=9)
        self.assertEqual(build.OVERVIEW, {'hrrr': (2, 4, 8), 'icond2': (2, 4, 8)})
        lons, lats = build.target_grid(build.REGIONS['hrrr'])
        fine = {'west': -125.0, 'north': 50.0, 'step': 0.03, 'nx': len(lons), 'ny': len(lats), 'tile': 100, 'ntx': 20, 'nty': 9}
        self.assertEqual([build.level_grid(fine, k)['step'] for k in build.OVERVIEW['hrrr']], [0.06, 0.12, 0.24])
        self.assertEqual(build.level_grid(fine, 8), {'west': -124.895, 'north': 49.895, 'step': 0.24, 'nx': 244, 'ny': 109, 'tile': 200, 'ntx': 2, 'nty': 1})

    def test_tiles_and_manifest(self):
        import tempfile
        from datetime import datetime, timezone
        import build
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            hours, out = (os.path.join(d, 'h'), os.path.join(d, 'out'))
            os.makedirs(hours)
            times = [datetime(2026, 9, 26, 18 + i, tzinfo=timezone.utc) for i in range(2)]
            for i, t in enumerate(times):
                ws = np.full((5, 7), 10 * (i + 1), np.uint8)
                ws[:, 6] = 255
                np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=ws, ptype=np.full((5, 7), 3, np.uint8), elevation=np.full((5, 7), 40, np.uint8))
            days, present, lp = build.write_tiles(out, hours, times, 4, times[0], levels=[(2, 2)])
            self.assertEqual(sorted(present['wstar']), ['0_0', '0_1', '1_0', '1_1'])
            self.assertEqual(sorted(lp[2]['wstar']), ['0_0', '0_1', '1_0', '1_1'])
            self.assertEqual(sorted(lp[2]['elevation']), ['0_0', '0_1', '1_0', '1_1'])
            strip = tile_img(os.path.join(out, 'o2', '20260926', 'wstar'), '1_0')
            self.assertEqual(strip.shape, (4, 2))
            self.assertEqual(strip[:, 0].tolist(), [10, 10, 20, 20])
            self.assertEqual(strip[:, 1].tolist(), [255] * 4)
            self.assertTrue(len(tile_img(os.path.join(out, 'o2', 'elevation'), '0_0')))
            grid = {'west': -80.0, 'north': 42.0, 'step': 0.05, 'nx': 7, 'ny': 5, 'tile': 4, 'ntx': 2, 'nty': 2}
            lv, = build.overview_manifest(grid, [(2, 2)], lp)
            self.assertEqual((lv['k'], lv['path'], lv['grid']['nx'], lv['grid']['ny'], lv['grid']['ntx']), (2, 'o2', 4, 3, 2))
            self.assertEqual(sorted(lv['tiles']), ['elevation', 'ptype', 'wstar'])

    def test_free_window_gets_the_levels(self):
        import json
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        with tempfile.TemporaryDirectory() as out:
            hours = os.path.join(out, 'hrrr', '_hours')
            os.makedirs(hours)
            t0 = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
            stamps = []
            for h in range(3):
                t = t0 + timedelta(hours=h)
                stamps.append(f'{t:%Y%m%d%H}')
                q = np.full((6, 6), 100, np.uint8)
                np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, sink=q, elevation=q)
            grid = {'west': -80.0, 'north': 42.0, 'step': 0.05, 'nx': 6, 'ny': 6, 'tile': 4, 'ntx': 2, 'nty': 2}
            fields = {k: {'min': 0, 'max': 1, 'transform': None, 'tiles': ['0_0']} for k in ('wstar', 'sink', 'elevation')}
            levels = build.overview_manifest(grid, [(2, 2)], {2: {k: {'0_0'} for k in fields}})
            full = {'runId': '2026092512', 'run': t0.isoformat(), 'grid': grid, 'days': {'x': stamps}, 'fields': fields, 'levels': levels}
            with open(os.path.join(out, 'hrrr', 'latest.json'), 'w') as f:
                json.dump(full, f)
            m = build.write_free(out, 'hrrr', now=t0)
            lv, = m['levels']
            self.assertEqual(lv['grid'], levels[0]['grid'])
            self.assertEqual(sorted(lv['tiles']), ['elevation', 'wstar'])
            self.assertEqual(sorted(lv['tiles']['wstar']), ['0_0', '0_1', '1_0', '1_1'])
            self.assertTrue(len(tile_img(os.path.join(out, 'free', 'hrrr', m['runId'], 'o2', '20260925', 'wstar'), '1_1')))

    def test_free_window_links_the_full_days(self):
        import json
        import tempfile
        from datetime import datetime, timedelta, timezone
        from PIL import Image
        import build
        with tempfile.TemporaryDirectory() as out:
            hours = os.path.join(out, 'hrrr', '_hours')
            os.makedirs(hours)
            t0 = datetime(2026, 9, 25, 10, tzinfo=timezone.utc)
            times = [t0 + timedelta(hours=h) for h in (0, 1, 24, 25, 26)]
            for i, t in enumerate(times):
                q = np.full((6, 6), 10 * i, np.uint8)
                np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, elevation=q)
            rdir = os.path.join(out, 'hrrr', '2026092510')
            days, present, lp = build.write_tiles(rdir, hours, times, 4, times[0], levels=[(2, 2)])
            grid = {'west': -80.0, 'north': 42.0, 'step': 0.05, 'nx': 6, 'ny': 6, 'tile': 4, 'ntx': 2, 'nty': 2}
            fields = {k: {'min': 0, 'max': 1, 'transform': None, 'tiles': sorted(v)} for k, v in present.items()}
            full = {'runId': '2026092510', 'run': t0.isoformat(), 'grid': grid, 'days': days, 'fields': fields, 'levels': build.overview_manifest(grid, [(2, 2)], lp)}
            with open(os.path.join(out, 'hrrr', 'latest.json'), 'w') as f:
                json.dump(full, f)
            m = build.write_free(out, 'hrrr', now=t0)
            free = os.path.join(out, 'free', 'hrrr', m['runId'])
            same = lambda a, b: os.path.samefile(a, b)
            self.assertTrue(same(os.path.join(free, '20260925', 'wstar.pack'), os.path.join(rdir, '20260925', 'wstar.pack')))
            self.assertTrue(same(os.path.join(free, 'o2', '20260925', 'wstar.pack'), os.path.join(rdir, 'o2', '20260925', 'wstar.pack')))
            self.assertFalse(same(os.path.join(free, '20260926', 'wstar.pack'), os.path.join(rdir, '20260926', 'wstar.pack')))
            self.assertEqual(tile_img(os.path.join(free, '20260926', 'wstar'), '0_0').shape, (4, 4))
            self.assertEqual(m['days'], {'20260925': days['20260925'], '20260926': days['20260926'][:1]})
            self.assertEqual(m['hours'], days['20260926'][1:])
            for st in m['hours']:
                tok = build.hour_token('hrrr', m['runId'], st, secret='s' * 20)
                self.assertNotEqual(tok, st)
                tok = build.hour_token('hrrr', m['runId'], st)
                strip = tile_img(os.path.join(free, 'h', tok, 'wstar'), '0_0')
                self.assertEqual(strip.shape, (4, 4))
                self.assertTrue(os.path.exists(os.path.join(free, 'o2', 'h', tok, 'wstar.pack')))
            self.assertEqual(sorted(m['fields']['wstar']['tiles']), sorted(present['wstar']))

class Packs(unittest.TestCase):

    def test_round_trip_and_empty_entries(self):
        import tempfile
        import pack
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.pack')
            pack.write_pack(p, [b'ab', b'', b'cde', b''], 2, 2)
            ntx, nty, ends, start = pack.read_index(p)
            self.assertEqual((ntx, nty, ends.tolist(), start), (2, 2, [2, 2, 5, 5], 12 + 16))
            self.assertEqual(pack.read_tile(p, '0_0'), b'ab')
            self.assertEqual(pack.read_tile(p, '1_0'), b'')
            self.assertEqual(pack.read_tile(p, '0_1'), b'cde')
            self.assertEqual(pack.read_tile(p, '5_5'), b'')
            self.assertEqual(pack.keys_in(p), {'0_0', '0_1'})
            with open(p, 'rb') as f:
                raw = f.read()
            self.assertEqual(raw[:4], b'APK1')
            self.assertEqual(raw[start + 2:start + 5], b'cde')
            with self.assertRaises(ValueError):
                pack.write_pack(p, [b'a'], 2, 2)

    def test_hour_segments(self):
        import io
        import tempfile
        import zlib
        from PIL import Image
        import build
        import pack
        import tiles
        rng = np.random.default_rng(3)
        base = np.cumsum(np.cumsum(rng.integers(-2, 3, (5, 9, 11)), 1), 2)
        day = np.clip(base - base.min(), 0, 254).astype(np.uint8)
        day[2] = 7
        day[3, :, 8:] = 255
        png, offs = tiles.png_hours(day)
        self.assertEqual(len(offs), 6)
        whole = np.array(Image.open(io.BytesIO(png)))
        np.testing.assert_array_equal(whole, day.reshape(-1, 11))

        def unfilter(raw, w, h):
            rows = np.frombuffer(raw, np.uint8).reshape(h, w + 1)
            out = np.zeros((h, w), np.int64)
            for y in range(h):
                f, x = (rows[y, 0], rows[y, 1:].astype(np.int64))
                prev = out[y - 1] if y else np.zeros(w, np.int64)
                if y == 0:
                    self.assertIn(f, (0, 1))
                for i in range(w):
                    a = out[y, i - 1] if i else 0
                    b, c = (prev[i], prev[i - 1] if i else 0)
                    p = a + b - c
                    pr = a if abs(p - a) <= abs(p - b) and abs(p - a) <= abs(p - c) else b if abs(p - b) <= abs(p - c) else c
                    out[y, i] = x[i] + (0, a, b, (a + b) // 2, pr)[f] & 255
            return out.astype(np.uint8)
        for k in range(5):
            seg = png[offs[k]:offs[k + 1]]
            raw = zlib.decompressobj(-15).decompress(seg + b'\x03\x00')
            np.testing.assert_array_equal(unfilter(raw, 11, 9), day[k])
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, '20261022', 'wstar')
            keys = build._write_strips(np.repeat(day, 2, axis=2), 11, folder)
            self.assertEqual(keys, {'0_0', '1_0'})
            p = folder + '.pack'
            at = pack.read_hours(p)
            self.assertEqual(at.shape, (2, 6))
            ntx, nty, ends, start = pack.read_index(p)
            first = np.array(Image.open(io.BytesIO(pack.read_tile(p, '1_0'))))
            np.testing.assert_array_equal(first.reshape(5, 11, 11)[:, :9], np.repeat(day, 2, axis=2)[:, :, 11:])
            self.assertTrue((first.reshape(5, 11, 11)[:, 9:] == 255).all())
            seg = pack.read_hour(p, 1, 4)
            raw = zlib.decompressobj(-15).decompress(seg + b'\x03\x00')
            np.testing.assert_array_equal(unfilter(raw, 11, 11)[:9], np.repeat(day, 2, axis=2)[4, :, 11:])
            build._write_strips(day[0], 11, os.path.join(d, 'elevation'))
            self.assertIsNone(pack.read_hours(os.path.join(d, 'elevation.pack')))
        was = build.HOUR_SEGS
        try:
            build.HOUR_SEGS = False
            with tempfile.TemporaryDirectory() as d:
                build._write_strips(day, 11, os.path.join(d, 'w'))
                self.assertIsNone(pack.read_hours(os.path.join(d, 'w.pack')))
        finally:
            build.HOUR_SEGS = was

    def test_pngs_too_until_the_cutoff(self):
        import tempfile
        from datetime import datetime, timezone
        import build
        import pack
        was = build.PNG_TOO_UNTIL
        try:
            for until, pngs in ((datetime(2100, 1, 1, tzinfo=timezone.utc), True), (datetime(2000, 1, 1, tzinfo=timezone.utc), False)):
                build.PNG_TOO_UNTIL = until
                with tempfile.TemporaryDirectory() as d:
                    folder = os.path.join(d, '20260926', 'wstar')
                    keys = build._write_strips(np.full((2, 5, 7), 9, np.uint8), 4, folder)
                    self.assertEqual(keys, {'0_0', '1_0', '0_1', '1_1'})
                    self.assertTrue(os.path.exists(folder + '.pack'))
                    self.assertEqual(os.path.isdir(folder), pngs)
                    if pngs:
                        with open(os.path.join(folder, '1_1.png'), 'rb') as f:
                            self.assertEqual(f.read(), pack.read_tile(folder + '.pack', '1_1'))
        finally:
            build.PNG_TOO_UNTIL = was

    def test_free_window_encodes_over_the_older_layout(self):
        import json
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        import tiles
        with tempfile.TemporaryDirectory() as out:
            hours = os.path.join(out, 'hrrr', '_hours')
            os.makedirs(hours)
            t0 = datetime(2026, 9, 25, 10, tzinfo=timezone.utc)
            times = [t0 + timedelta(hours=h) for h in (0, 1)]
            for t in times:
                q = np.full((4, 4), 30, np.uint8)
                np.savez(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=q, elevation=q)
            rdir = os.path.join(out, 'hrrr', '2026092510')
            os.makedirs(os.path.join(rdir, '20260925', 'wstar'))
            tiles.write_png(os.path.join(rdir, '20260925', 'wstar', '0_0.png'), np.full((8, 4), 99, np.uint8))
            grid = {'west': -80.0, 'north': 42.0, 'step': 0.05, 'nx': 4, 'ny': 4, 'tile': 4, 'ntx': 1, 'nty': 1}
            fields = {k: {'min': 0, 'max': 1, 'transform': None, 'tiles': ['0_0']} for k in ('wstar', 'elevation')}
            full = {'runId': '2026092510', 'run': t0.isoformat(), 'grid': grid, 'fields': fields, 'levels': [], 'days': {'20260925': [f'{t:%Y%m%d%H}' for t in times]}}
            with open(os.path.join(out, 'hrrr', 'latest.json'), 'w') as f:
                json.dump(full, f)
            m = build.write_free(out, 'hrrr', now=t0)
            free = os.path.join(out, 'free', 'hrrr', m['runId'])
            self.assertEqual(m['pack'], 1)
            strip = tile_img(os.path.join(free, '20260925', 'wstar'), '0_0')
            self.assertEqual(strip.shape, (8, 4))
            self.assertEqual(int(strip[0, 0]), 30)

class LowLevelWind(unittest.TestCase):

    def test_hrrr_winds_turned_to_true_north(self):
        import grib
        rot = (np.sin(np.radians(38.5)), 262.5)
        u, v = (np.float32([0, 0]), np.float32([10, 10]))
        eu, ev = grib.earth_winds(u, v, np.array([-97.5, -77.5]), rot)
        self.assertAlmostEqual(float(eu[0]), 0, places=5)
        a = np.radians(0.6225146 * 20)
        self.assertAlmostEqual(float(eu[1]), 10 * np.sin(a), places=3)
        self.assertAlmostEqual(float(ev[1]), 10 * np.cos(a), places=3)
        self.assertAlmostEqual(float(np.hypot(eu[1], ev[1])), 10, places=4)
        self.assertIs(grib.earth_winds(u, v, np.array([0.0, 0.0]), None)[0], u)

    def test_sources_fetch_the_hub_wind(self):
        from sources import SOURCES
        self.assertEqual(SOURCES['hrrr'].hub_m, 80)
        self.assertIn(('UGRD', '80 m above ground'), SOURCES['hrrr'].wanted())
        self.assertIn(('VGRD', '100 m above ground'), SOURCES['gfs'].wanted())
        self.assertEqual(SOURCES['ecmwf'].surface['uh'], ('100u', 'sfc'))

    def test_terrain_following_levels_published(self):
        import build
        s, lv, levels, _ = grid_column(n=3, shear=4)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(500))
        lat = one(40)
        out = build.analyse(dict(s), lv, levels, 0.25, lat)
        self.assertTrue(np.isfinite(out['u_g100']).all() and np.isfinite(out['v_g300']).all())
        hub = build.analyse(dict(s, uh=one(7), vh=one(1), hub_m=100.0), lv, levels, 0.25, lat)
        self.assertAlmostEqual(float(hub['u_g100'][1, 1]), 7, places=4)
        self.assertAlmostEqual(float(hub['v_g100'][1, 1]), 1, places=4)
        u925 = float(lv['u'][1, 1, 1])
        self.assertTrue(7 < float(hub['u_g300'][1, 1]) < u925)
        for k in ('u_g100', 'v_g100', 'u_g300', 'v_g300'):
            self.assertEqual(build.ENC[k], build.WIND_ENC)
            self.assertNotIn(k, build.FREE_FIELDS)

class Bumpiness(unittest.TestCase):

    def prof(self, lapse, speed):
        levels = np.array([1000, 950, 900, 850, 700, 500], np.float32)
        z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
        shape = (len(levels), 1, 1)
        one = lambda v: np.full((1, 1), v, np.float32)
        t = (15 - lapse * z).reshape(shape)
        u = np.array([speed(h) for h in z], np.float32).reshape(shape)
        return P.Profile(one(0), one(1013.25), one(15), one(5), one(speed(10)), one(0), z.reshape(shape), levels, t, t - 10, u, np.zeros(shape, np.float32))

    def test_thermals_and_gusts_bumpy_stable_jet_smooth(self):
        one = lambda v: np.full((1, 1), v, np.float32)
        mixed = self.prof(0.0098, lambda z: 6.0)
        b = lambda prof, **k: float(P.bumpiness(prof, one(k.get('ws', 0)), one(k.get('zi', 1500)), one(k['gust']) if 'gust' in k else None, one(k.get('u10', 3)), one(0), one(k['ustar']) if 'ustar' in k else None, one(k.get('relief', 0)))[0, 0])
        self.assertGreaterEqual(b(mixed, ws=2.5, gust=6, ustar=0.3), 0.5)
        self.assertLess(b(mixed, ws=1.2, gust=5, ustar=0.25), 0.5)
        self.assertGreaterEqual(b(mixed, ws=0, gust=16, u10=7, ustar=0.7), 0.5)
        self.assertLess(b(mixed, ws=0, gust=7, u10=6, ustar=0.25), 0.5)
        jet = self.prof(-0.004, lambda z: min(20.0, 1 + z / 50))
        self.assertLess(b(jet, ws=0, zi=150, gust=4, u10=1.5, ustar=0.08), 0.5)
        self.assertLess(b(mixed, ws=0, gust=5, u10=4), 0.5)
        self.assertGreaterEqual(b(mixed, ws=0, gust=19, u10=11), 0.5)

class Sink(unittest.TestCase):

    def col(self, w=0.0, spread=12.0, levels=(1000, 950, 900, 850, 800, 750, 700, 600, 500), zs=0.0, lapse=0.0065):
        levels = np.array(levels, np.float32)
        z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
        shape = (len(levels), 1, 1)
        one = lambda v: np.full((1, 1), v, np.float32)
        t = (20 - lapse * z).reshape(shape)
        prof = P.Profile(one(zs), one(1013.25 * (1 - zs / 44330) ** 5.255), one(20 - lapse * zs), one(20 - lapse * zs - spread), one(5), one(0), z.reshape(shape), levels, t, t - spread, np.full(shape, 10, np.float32), np.zeros(shape, np.float32))
        wcol = np.broadcast_to(np.asarray(w, np.float32).reshape(-1, 1, 1) if np.ndim(w) else np.float32(w), shape).astype(np.float32)
        return (prof, z.reshape(shape), wcol)

    def parts(self, prof, wz=None, w=None, ws=0.0, zi=0.0, depth=0.0, rain=0.0, refd=None, lee=None, snow=None):
        one = lambda v: None if v is None else np.full((1, 1), v, np.float32)
        return [float(a[0, 0]) for a in P.sink_rate(prof, wz, w, one(ws), one(zi), one(depth), one(rain), one(refd), one(lee), parts=True, snow=one(snow))]

    def sink(self, *a, **k):
        return self.parts(*a, **k)[0]

    def test_calm_night_is_zero(self):
        prof, wz, w = self.col(0.0)
        self.assertEqual(self.parts(prof, wz, w), [0, 0, 0])
        self.assertEqual(self.sink(prof, wz, w, refd=-10, lee=0), 0)

    def test_model_subsidence_and_rising_air(self):
        prof, wz, w = self.col(-0.8)
        self.assertAlmostEqual(self.sink(prof, wz, w), 0.8, delta=0.05)
        self.assertEqual(self.parts(prof, wz, w)[2], 2)
        prof, wz, w = self.col(+0.8)
        self.assertEqual(self.sink(prof, wz, w), 0)
        prof, wz, w = self.col([0, 0, 0, 0, 0, 0, 0, -2, -2])
        self.assertLess(self.sink(prof, wz, w, ws=2, zi=4000, depth=4000), 0.05)

    def test_lee_slope(self):
        prof, wz, w = self.col(-0.2)
        self.assertAlmostEqual(self.sink(prof, wz, w, lee=1.5), 1.5, delta=0.01)
        self.assertAlmostEqual(self.sink(prof, wz, self.col(-2.0)[2], lee=1.5), 2.0, delta=0.05)
        self.assertEqual(self.sink(prof, wz, self.col(+0.5)[2], lee=1.5), 0)
        s, th, why = self.parts(prof, wz, w, ws=2.0, zi=2000, depth=2000, lee=1.5)
        self.assertAlmostEqual(s, 1.5, delta=0.01)
        self.assertGreater(th, 0.5)
        self.assertEqual(why, 2)

    def test_coarse_levels_and_ground_above_the_lowest(self):
        levels = (1000, 925, 850, 700, 600, 500)
        prof, wz, w = self.col([-20, -20, -0.5, -0.5, -0.5, -0.5], levels=levels, zs=800.0)
        s = self.sink(prof, wz, w)
        self.assertTrue(0.2 < s < 0.55, s)

    def test_snow_sinks_harder_than_rain(self):
        dry, wz, w = self.col(0.0, spread=12.0, lapse=0.0085)
        rain = self.sink(dry, wz, w, rain=1.0)
        snow = self.sink(dry, wz, w, rain=1.0, snow=1.0)
        mix = self.sink(dry, wz, w, rain=1.0, snow=0.5)
        self.assertTrue(rain < mix < snow < 2 * rain, (rain, mix, snow))
        self.assertEqual(self.sink(dry, wz, w, rain=1.0, snow=0.0), rain)
        cold, _, _ = self.col(0.0, lapse=0.015)
        warm, _, _ = self.col(0.0)
        one = lambda v: np.full((1, 1), v, np.float32)
        share = lambda prof, t: float(P.snow_share(one(t), prof, one(3000))[0, 0])
        self.assertEqual([share(warm, t) for t in (1, 2, 3, 4, 5)], [0, 1, 0.5, 0.5, 0])
        self.assertEqual(share(warm, 0), 0)
        self.assertEqual(share(cold, 0), 1)

    def test_between_thermals(self):
        prof, wz, w = self.col(0.0)
        s, strong, why = self.parts(prof, wz, w, ws=3.0, zi=2000, depth=2000)
        self.assertEqual((s, why), (0, 1))
        self.assertTrue(0.8 < strong < 1.2, strong)
        self.assertAlmostEqual(self.parts(prof, wz, w, ws=1.5, zi=2000, depth=2000)[1], strong / 2, delta=0.01)
        self.assertEqual(self.parts(prof, wz, w, ws=3.0, zi=250, depth=250)[1], 0)
        prof, wz, up = self.col(+0.5)
        self.assertAlmostEqual(self.parts(prof, wz, up, ws=3.0, zi=2000, depth=2000)[1], strong - 0.5, delta=0.05)
        prof, wz, down = self.col(-0.6)
        s, th, why = self.parts(prof, wz, down, ws=3.0, zi=2000, depth=2000)
        self.assertAlmostEqual(s, 0.6, delta=0.05)
        self.assertAlmostEqual(th, strong, delta=0.01)
        self.assertEqual(why, 2)

    def test_virga_into_dry_air(self):
        dry, wz, w = self.col(0.0, spread=15, lapse=0.0095)
        wet, _, _ = self.col(0.0, spread=0.5, lapse=0.0095)
        virga = self.sink(dry, wz, w, rain=0, refd=35)
        self.assertTrue(1.5 < virga <= 3.0, virga)
        self.assertLess(self.sink(wet, wz, w, rain=0, refd=35), 0.5)
        self.assertLess(self.sink(dry, wz, w, rain=0, refd=10), 0.05)
        light = self.sink(dry, wz, w, rain=0.3)
        self.assertTrue(0.1 < light < virga, light)
        heavy = self.sink(dry, wz, w, rain=40, refd=55)
        self.assertTrue(virga < heavy <= 3.0, heavy)
        stable, _, _ = self.col(0.0, spread=15, lapse=0.005)
        self.assertTrue(0.2 < self.sink(stable, wz, w, rain=0, refd=35) <= 0.5)
        self.assertLessEqual(self.sink(stable, wz, w, rain=40, refd=55), 0.5)
        _, _, down = self.col(-3.5)
        self.assertAlmostEqual(self.sink(dry, wz, down, rain=0, refd=35), 3.5, delta=0.1)
        s, th, why = self.parts(dry, wz, w, ws=2.0, zi=2000, depth=2000, rain=0, refd=35)
        self.assertAlmostEqual(s + th, virga, delta=0.01)
        self.assertEqual(why, 3)

    def test_cause(self):
        one = lambda v: np.full((1, 1), v, np.float32)
        dry, wz, calm = self.col(0.0, lapse=0.0095)
        _, _, down = self.col(-1.0)
        why = lambda w, **k: int(P.sink_rate(dry, wz, w, one(k.get('ws', 0)), one(2000), one(2000), one(k.get('rain', 0)), parts=True)[2][0, 0])
        self.assertEqual(why(calm), 0)
        self.assertEqual(why(calm, ws=2.5), 1)
        self.assertEqual(why(down, ws=1.0), 2)
        self.assertEqual(why(calm, ws=1.0, rain=8), 3)
        self.assertEqual(P.SINK_CAUSES[3], 'precipitation')

    def test_glacier_sink(self):
        g = lambda snowd, t2, slope, day=1: float(P.glacier_sink(np.float32(snowd), np.float32(t2), np.float32(slope), np.float32(day)))
        self.assertAlmostEqual(g(1.0, 10, 0.2), P.SINK_GLACIER, places=5)
        self.assertEqual(g(0.0, 10, 0.2), 0)
        self.assertEqual(g(0.01, 10, 0.2), 0)
        self.assertEqual(g(1.0, -2, 0.2), 0)
        self.assertEqual(g(1.0, 10, 0.2, day=0), 0)
        self.assertLess(g(1.0, 10, 0.0), g(1.0, 10, 0.2))
        self.assertGreater(g(1.0, 10, 0.0), 0)
        self.assertLess(g(1.0, 3, 0.2), g(1.0, 10, 0.2))
        self.assertEqual(P.SINK_CAUSES[4], 'cold ground (snow or ice)')

    def test_lee_hold(self):
        f = lambda *a: np.array(a, np.float32)
        held = P.lee_hold(f(1.5), f(15), f(2), f(2000))
        self.assertAlmostEqual(float(held[0]), 120, delta=1)
        self.assertGreater(float(np.cbrt((2000 - held[0]) / 2000)), 0.97)
        self.assertEqual(float(P.lee_gate(f(4), f(2))[0]), 0)
        self.assertEqual(float(P.lee_gate(f(8), f(2))[0]), 1)
        self.assertEqual(float(P.lee_hold(f(0.4), f(4), f(2), f(2000))[0]), 0)
        self.assertAlmostEqual(float(P.lee_hold(f(8), f(20), f(1.5), f(1500))[0]), P.LEE_TOP_CAP * 1500, delta=1)
        self.assertEqual(float(P.lee_hold(f(-1), f(15), f(2), f(2000))[0]), 0)
        self.assertEqual(float(P.lee_hold(f(1.5), f(15), f(2), f(2000), ramp=0)[0]), 0)

    def test_air_motion_climb(self):
        f = lambda *a: np.array(a, np.float32)
        vm = P.vertical_motion(f(-0.1, 0.4, 0.2, -0.3, 0.0), f(0.3, 0.1, 0.5, 0.0, 0.1), f(2, 2, 3, 0, 5))
        np.testing.assert_allclose(vm, [-0.3, 0.3, 0.2, -0.3, -0.1], atol=1e-06)
        np.testing.assert_allclose(P.air_motion_climb(f(2, 2, 2, 2, 0), f(-0.2, -2, 0.4, 2, -1)), [1.9, 1.6, 2.2, 2.3, 0], atol=1e-06)

    def test_mountain_plain(self):
        n = 60
        x = np.arange(n, dtype=np.float32)
        zs = np.tile(400 + 2100 * np.clip((x - 37) / 3, 0, 1), (n, 1)).astype(np.float32)
        zs[:, 41:][:, ::3] += 400
        sun = np.full((n, n), 300.0, np.float32)
        calm, windy = (np.zeros((n, n), np.float32), np.full((n, n), 15.0, np.float32))
        day = P.mountain_plain(zs, sun, calm, calm, 0.25)
        sink, lift = (day['sink'], day['lift'])
        self.assertGreater(float(sink[30, 37]), 0.1)
        self.assertGreater(float(sink[30, 36]), 0.05)
        self.assertLess(float(sink[30, 35]), float(sink[30, 36]))
        self.assertEqual(float(sink[30, 2]), 0)
        self.assertEqual(float(sink[30, 50]), 0)
        self.assertEqual(float(sink[30, 39]), 0)
        self.assertGreater(float(lift[30, 50]), 0)
        self.assertEqual(float(P.mountain_plain(zs, sun, windy, calm, 0.25)['sink'][30, 37]), 0)
        self.assertEqual(float(P.mountain_plain(zs, np.zeros_like(sun), calm, calm, 0.25)['sink'][30, 37]), 0)
        self.assertGreater(float(day['u'][30, 37]), 1.0)
        self.assertLess(abs(float(day['v'][30, 37])), 0.1)
        self.assertEqual(float(day['u'][30, 2]), 0)
        self.assertEqual(float(day['depth'][30, 37]), P.PUMP_DEPTH[0])
        cold = P.night_cooling(np.full((n, n), -40.0, np.float32))
        self.assertEqual(float(cold[0, 0]), 1.0)
        night = P.mountain_plain(zs, np.zeros_like(sun), calm, calm, 0.25, cool=cold)
        self.assertLess(float(night['u'][30, 37]), -0.8)
        self.assertLess(abs(float(night['u'][30, 37])), float(day['u'][30, 37]))
        self.assertEqual(float(night['depth'][30, 37]), P.PUMP_DEPTH[1])
        self.assertGreater(float(night['sink'][30, 42]), 0)
        self.assertLess(float(night['sink'].max()), 0.1)
        self.assertEqual(float(night['sink'][30, 37]), 0)
        self.assertGreater(float(night['lift'][30, 37]), 0)
        self.assertEqual(float(P.night_cooling(np.full((2, 2), 30.0))[0, 0]), 0)
        z5 = np.repeat(np.repeat(zs, 5, 0), 5, 1)
        h5 = P.mountain_plain(z5, np.full(z5.shape, 300.0, np.float32), np.zeros(z5.shape, np.float32), np.zeros(z5.shape, np.float32), 0.05)
        self.assertGreater(float(h5['sink'].max()), 0.1)
        self.assertEqual(float(np.abs(h5['u']).max()), 0)
        self.assertEqual((P.pump_unresolved(0.05), P.pump_unresolved(0.25)), (0.25, 1.0))
        self.assertEqual((P.pump_wind_share(0.05), P.pump_wind_share(0.25)), (0.0, 1.0))
        self.assertEqual(P.SINK_CAUSES[5][:9], 'mountain–')

    def test_glacier_sink_published(self):
        import build
        from datetime import datetime, timezone
        s, lv, levels, _ = grid_column(n=3)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(800), sw=one(0), shf=one(0), hpbl=one(100), snod=one(2.0))
        s['zs'] = np.array([[3000, 3300, 3600]] * 3, np.float32)
        s['t2'] = one(9)
        valid = datetime(2026, 7, 15, 19, tzinfo=timezone.utc)
        out = build.analyse(dict(s), lv, levels, 0.03, one(46), one(-121), valid)
        self.assertGreater(float(out['sink'][1, 1]), 0.3)
        self.assertEqual(int(out['sinkby'][1, 1]), 4)
        night = build.analyse(dict(s), lv, levels, 0.03, one(46), one(-121), datetime(2026, 7, 15, 9, tzinfo=timezone.utc))
        self.assertNotEqual(int(night['sinkby'][1, 1]), 4)

    def test_published_like_the_other_fields(self):
        import build
        s, lv, levels, _ = grid_column(n=3)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(500), shf=one(250), hpbl=one(1800))
        s['t2'][0, 0] = np.nan
        lv['w'] = np.full(lv['t'].shape, -0.4, np.float32)
        out = build.analyse(dict(s), lv, levels, 0.25, one(40))
        self.assertTrue(np.isnan(out['sink'][0, 0]))
        self.assertAlmostEqual(float(out['sink'][1, 1]), 0.4, delta=0.05)
        self.assertGreater(float(out['sinkth'][1, 1]), 0.3 * float(out['wstar'][1, 1]))
        self.assertEqual(build.ENC['sink'], (0, 6, None))
        for f in ('sink', 'sinkth', 'sinkby'):
            self.assertNotIn(f, build.FREE_FIELDS)
        self.assertEqual(int(out['sinkby'][1, 1]), 2)
        self.assertEqual(build.quantize(np.float32([2]), *build.ENC['sinkby'])[0], 2)
        self.assertIn(('REFD', '1000 m above ground'), __import__('sources').SOURCES['hrrr'].wanted())

class CloudLevels(unittest.TestCase):

    def test_calibrate_cover(self):
        L = P.CC_LEVELS
        one = lambda v: np.full((1, 1), v, np.float32)
        ps = one(1010)
        cc = np.zeros((len(L), 1, 1), np.float32)
        rh = np.full((len(L), 1, 1), 50, np.float32)
        rh[L.index(850)] = 88
        cc[L.index(300)], cc[L.index(250)] = (70, 35)
        icon = ('p', 800.0, 400.0)
        out = P.calibrate_cover(cc, rh, ps, [one(40), one(0), one(90)], icon)
        self.assertEqual(float(out[L.index(850), 0, 0]), 40)
        self.assertEqual(float(out[L.index(900), 0, 0]), 0)
        self.assertAlmostEqual(float(out[L.index(300), 0, 0]), 90, places=3)
        self.assertAlmostEqual(float(out[L.index(250), 0, 0]), 45, places=3)
        self.assertEqual(float(np.nanmax(out[L.index(700):L.index(400), 0, 0])), 0)
        cc2 = np.zeros_like(cc)
        cc2[L.index(950)] = 100
        cc2[0] = np.nan
        out = P.calibrate_cover(cc2, rh, ps, [one(30), None, one(np.nan)], icon)
        self.assertAlmostEqual(float(out[L.index(950), 0, 0]), 30, places=3)
        self.assertTrue(np.isnan(out[0, 0, 0]))
        cc3 = np.zeros_like(cc)
        cc3[L.index(700)] = 50
        out = P.calibrate_cover(cc3, rh, one(850), [one(100), one(0), one(0)], ('sigma', 0.8, 0.45))
        self.assertAlmostEqual(float(out[L.index(700), 0, 0]), 100, places=3)

    def cf(self, rh, p=850.0, ps=1000.0, clw=None):
        one = lambda v: np.full((1,), v, np.float32)
        return float(P.cloud_fraction(one(rh), p, one(ps), None if clw is None else one(clw / 1000))[0])

    def test_saturated_full_dry_clear(self):
        for p in P.CC_LEVELS:
            self.assertAlmostEqual(self.cf(100, p, 1013), 100, places=3)
            self.assertEqual(self.cf(40, p, 1013), 0)
        self.assertEqual(self.cf(100 * P.CC_RHC[0] - 1, 1000, 1013), 0)
        self.assertAlmostEqual(self.cf(110, 300, 1013), 100, places=3)

    def test_below_ground_and_outside_are_nan(self):
        self.assertTrue(np.isnan(self.cf(100, 900, 850)))
        self.assertFalse(np.isnan(self.cf(100, 850, 850.5)))
        self.assertTrue(np.isnan(self.cf(100, 500, np.nan)))

    def test_monotonic_in_humidity_and_easier_aloft(self):
        rh = np.linspace(0, 100, 201, dtype=np.float32)
        for p in (1000, 850, 500, 300):
            cf = P.cloud_fraction(rh, p, np.float32(1013))
            self.assertTrue((np.diff(cf) >= 0).all(), p)
        self.assertGreater(self.cf(92, 500, 1013), self.cf(92, 1000, 1013))

    def test_cloud_water_counts_whatever_the_humidity(self):
        self.assertAlmostEqual(self.cf(60, clw=0.5), 100, places=3)
        self.assertTrue(0 < self.cf(60, clw=0.03) < 100)
        self.assertEqual(self.cf(60, clw=0.001), 0)
        self.assertGreaterEqual(self.cf(96, clw=0.01), self.cf(96))

    def test_humidity_against_ice_when_cold(self):
        t = np.float32([-40.0, 10.0])
        td = P.dewpoint_from_w(P.EPS * P.es_ice(t) / (500 - P.es_ice(t)), 500)
        self.assertAlmostEqual(float(P.rh_cloud(t, td)[0]), 100, delta=1)
        self.assertLess(float(P.rh_from_td(t, td)[0]), 70)
        self.assertAlmostEqual(float(P.rh_cloud(t, t - 5)[1]), float(P.rh_from_td(t, t - 5)[1]), places=3)

    def test_levels_the_model_lacks_are_interpolated_in_log_pressure(self):
        levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300], np.float32)
        one = lambda v: np.full((1, 1), v, np.float32)
        rh = np.full((len(levels), 1, 1), 80, np.float32)
        rh[1] = 100
        cc = P.cloud_cover_levels(levels, rh, one(1013), one(80))
        self.assertEqual(cc.shape, (len(P.CC_LEVELS), 1, 1))
        at = dict(zip(P.CC_LEVELS, cc[:, 0, 0]))
        self.assertEqual(at[1000], 0)
        self.assertTrue(0 < at[900] < 100 and 0 < at[950] < 100)
        w950 = math.log(950 / 1000) / math.log(925 / 1000)
        self.assertAlmostEqual(at[950], float(P.cloud_fraction(np.float32(80 + 20 * w950), 950, np.float32(1013))), places=3)
        self.assertEqual(at[850], 0)
        wet_sfc = P.cloud_cover_levels(levels, rh, one(970), one(100))
        self.assertTrue(np.isnan(wet_sfc[0, 0, 0]))
        self.assertGreater(float(wet_sfc[1, 0, 0]), at[950])

    def test_ceiling_from_cover_not_from_moist_air(self):
        one = lambda v: np.full((1,), v, np.float32)
        zc = np.array([100, 550, 1000, 1500, 2000], np.float32)[:, None]
        col = lambda *c: np.array(c, np.float32)[:, None]
        ceil = lambda cc, rh2=90, tcc=100: float(P.ceiling_from_cover(zc, cc, one(0), one(rh2), one(tcc))[0])
        self.assertEqual(ceil(col(10, 5, 0, 0, 0), rh2=95), P.CEIL_NONE)
        self.assertAlmostEqual(ceil(col(0, 0, 25, 100, 100)), 1000 + 500 * (25 / 75), places=1)
        self.assertEqual(ceil(col(0, 0, 25, 100, 100), tcc=30), P.CEIL_NONE)
        self.assertEqual(ceil(col(100, 100, 0, 0, 0), rh2=99), 30)
        self.assertEqual(ceil(col(0, 0, 0, 0, 0), rh2=99), P.CEIL_NONE)
        self.assertAlmostEqual(ceil(col(np.nan, 100, 0, 0, 0)), 275, places=1)
        heights = P.level_heights(np.float32([1000, 925, 850]), np.float32([100, 760, 1450])[:, None], (1000, 950, 850, 200))
        self.assertAlmostEqual(float(heights[0, 0]), 100)
        self.assertTrue(760 > float(heights[1, 0]) > 100 + (760 - 100) * 0.6)
        self.assertTrue(np.isnan(heights[3, 0]))

    def test_published_like_the_other_cloud_layers(self):
        import build
        s, lv, levels, _ = grid_column(n=3, rh_levels=100)
        one = lambda x: np.full((3, 3), x, np.float32)
        s.update(ps=one(1013.25), sw=one(500))
        s['t2'][0, 0] = np.nan
        s['ps'][2, 2] = 880.0
        out = build.analyse(dict(s), lv, levels, 0.25, one(40))
        for p in P.CC_LEVELS:
            self.assertEqual(build.ENC[f'cc{p}'], (0, 100, None))
            self.assertEqual(f'cc{p}' in build.FREE_FIELDS, 'cloudlow' in build.FREE_FIELDS)
            if p >= 300:
                self.assertAlmostEqual(float(out[f'cc{p}'][1, 1]), 100, places=3)
            else:
                self.assertTrue(np.isnan(out[f'cc{p}'][1, 1]))
            self.assertTrue(np.isnan(out[f'cc{p}'][0, 0]))
        self.assertTrue(np.isnan(out['cc900'][2, 2]) and np.isfinite(out['cc850'][2, 2]))
        self.assertEqual(build.quantize(out['cc1000'], *build.ENC['cc1000'])[0, 0], 255)
        full = np.array(sorted(set(P.CC_LEVELS) | {975, 925}, reverse=True), np.float32)
        own = np.stack([one(float(i)) for i in range(len(full))])
        s2 = {'t2': one(20), 'td2': one(10), 'ps': one(1013.25)}
        cc = build.cloud_levels(s2, {'cc': own, 'rh': np.full(own.shape, 100, np.float32)}, full)
        self.assertEqual(float(cc['cc300'][1, 1]), float(list(full).index(300)))
        self.assertIn(('TCDC', '500 mb'), __import__('sources').SOURCES['gfs'].wanted())

class EntrainingCumulus(unittest.TestCase):

    def test_mix_from_the_thermal_width(self):
        m = P.cumulus_mix(np.array([100.0, 200.0, 1000.0, 5000.0, np.nan], np.float32))
        self.assertEqual(m.dtype, np.float32)
        np.testing.assert_allclose(m, [0.2, 0.2, 0.04, 0.02, 0.04], rtol=1e-06)

    def test_mixing_lowers_the_top_in_dry_air_not_in_moist(self):
        z = np.arange(0, 12001, 250, dtype=np.float32)[:, None, None]
        t = (20 - 0.0062 * z).astype(np.float32)
        p = (1013.25 * (1 - z / 44330.8) ** 5.255).astype(np.float32)
        one = lambda a: np.ascontiguousarray(np.broadcast_to(a, (z.shape[0], 1, 2)).astype(np.float32))

        def prof(td):
            zs = np.zeros((1, 2), np.float32)
            return P.Profile(zs, one(p)[0] * 100, np.full((1, 2), 20, np.float32), np.full((1, 2), 15, np.float32), np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), one(z), one(p)[:, 0, 0], one(t), one(td), np.zeros_like(one(t)), np.zeros_like(one(t)))
        base, tb, act = (np.full((1, 2), 1000, np.float32), np.full((1, 2), 14.5, np.float32), np.ones((1, 2), bool))
        mix = P.cumulus_mix(np.full((1, 2), 300, np.float32))
        dry, wet = (prof(t - 20), prof(t))
        und = P.cumulus_top(dry, base, tb, act)
        self.assertLess(P.cumulus_top(dry, base, tb, act, mix=mix)[0, 0], und[0, 0])
        self.assertLessEqual(P.cumulus_top(dry, base, tb, act, mix=mix)[0, 0], P.cumulus_top(wet, base, tb, act, mix=mix)[0, 0])

class ThermalDrift(unittest.TestCase):

    def setUp(self):
        import build
        self.drift = build.thermal_drift
        n = 60
        self.zs = np.full((n, n), 500.0, np.float32)
        self.lat = np.full((n, n), 40.0, np.float32)
        self.t2 = np.full((n, n), 25.0, np.float32)
        self.ps = np.full((n, n), 950.0, np.float32)

    def test_lake_downwind_shore_and_open_water(self):
        n = self.zs.shape[0]
        heat = np.where(np.arange(n)[None, :] < 30, 250.0, 0.0) * np.ones((n, 1), np.float32)
        zi = np.where(heat > 0, 1500.0, 100.0).astype(np.float32)
        ws = np.where(heat > 0, 2.0, 0.0).astype(np.float32)
        u, v = (np.full_like(zi, 8.0), np.zeros_like(zi))
        zi2, ws2 = self.drift(self.zs, heat, zi, ws, u, v, self.t2, self.ps, self.lat, 0.03)
        row = ws2[30]
        self.assertGreater(row[31], 0.8, 'thermals carry over the downwind shore')
        self.assertLess(row[55], 0.25, 'none far out over the water')
        self.assertGreater(zi2[30, 31], 1000, 'with the depth they came with')
        self.assertTrue(np.array_equal(ws2[:, :30], ws[:, :30]), 'the land upwind as it was')

    def test_the_model_kept_where_nothing_drifts_in(self):
        n = self.zs.shape[0]
        ws = np.full((n, n), 2.0, np.float32)
        ws[:, 30:] = 0.8
        zi = np.full((n, n), 1500.0, np.float32)
        zi[:, 30:] = 300.0
        heat = np.where(ws > 1, 250.0, 60.0).astype(np.float32)
        u, v = (np.full_like(zi, 5.0), np.zeros_like(zi))
        zi2, ws2 = self.drift(self.zs, heat, zi, ws, u, v, self.t2, self.ps, self.lat, 0.03)
        self.assertGreater(ws2[30, 30], 0.8, "the flat's edge: drifting thermals")
        self.assertEqual(ws2[30, 58], ws[30, 58], "its middle: the model's")
        self.assertEqual(zi2[30, 58], zi[30, 58])
        sea = np.where(np.arange(n)[None, :] < 30, 0.0, 1.0) * np.ones((n, 1), np.float32)
        ws3, zi3 = (sea.astype(np.float32), np.where(sea > 0, 300.0, 200.0).astype(np.float32))
        zi4, ws4 = self.drift(self.zs, sea * 250, zi3, ws3, u, v, self.t2, self.ps, self.lat, 0.03)
        self.assertTrue(np.array_equal(zi4, zi3) and np.array_equal(ws4, ws3))

    def test_calm_changes_nothing(self):
        n = self.zs.shape[0]
        heat = np.full((n, n), 200.0, np.float32)
        zi = np.random.default_rng(1).uniform(200, 2000, (n, n)).astype(np.float32)
        ws = np.ones_like(zi)
        zi2, ws2 = self.drift(self.zs, heat, zi, ws, np.zeros_like(zi), np.zeros_like(zi), self.t2, self.ps, self.lat, 0.03)
        self.assertTrue(np.array_equal(zi2, zi))

class Regions(unittest.TestCase):

    def test_gfs_grid_covers_every_area_in_whole_tiles(self):
        import build
        lons, lats = build.target_grid(build.REGIONS['gfs'])
        self.assertEqual((len(lons) % build.TILE['gfs'], len(lats) % build.TILE['gfs']), (0, 0))
        self.assertEqual((lons[0], lats[0]), (-170.0, 70.0))
        self.assertLess(lons[-1] - lons[0], 360)
        for lat, lon in [(39.3, -77.6), (49.3, -123.1), (61.2, -149.9), (64.8, -147.7), (58.3, -134.4), (48.9, 2.4), (46.5, 9.0), (60.0, 10.7), (36.7, -4.4), (-31.4, -64.2), (-23.9, 17.9), (31.6, -8.0), (-44.5, 169.97)]:
            self.assertTrue(lons[0] <= lon <= lons[-1] and lats[-1] <= lat <= lats[0], (lat, lon))

    def test_hrrr_grid_about_native(self):
        import build
        region = build.REGIONS['hrrr']
        self.assertEqual(region[4], 0.03)
        lons, lats = build.target_grid(region)
        self.assertEqual((len(lons), len(lats)), (1951, 868))
        self.assertEqual((lons[0], lats[0]), (-125.0, 50.0))
        self.assertAlmostEqual(lons[-1], -66.5, places=6)
        self.assertLessEqual(region[4], build.WAVE_BAND_MAX_STEP)

    def test_stored_hours_on_the_005_grid_dont_fit_the_003(self):
        import tempfile
        import build
        old = (-125.0, 24.0, -66.5, 50.0, 0.05)
        lons, lats = build.target_grid(old)
        with tempfile.TemporaryDirectory() as d:
            build.tiles.savez(os.path.join(d, '2026101112.npz'), {'wstar': np.zeros((len(lats), len(lons)), np.uint8)})
            self.assertTrue(build.store_fits(d, old))
            self.assertFalse(build.store_fits(d, build.REGIONS['hrrr']))

    def test_stored_hours_on_another_grid_dont_fit(self):
        import tempfile
        import build
        region = (-10.0, 40.0, 10.0, 50.0, 0.25)
        lons, lats = build.target_grid(region)
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(build.store_fits(d, region))
            np.savez(os.path.join(d, '2026100112.npz'), wstar=np.zeros((len(lats), len(lons)), np.uint8))
            self.assertTrue(build.store_fits(d, region))
            self.assertFalse(build.store_fits(d, (-10.0, 40.0, 20.0, 50.0, 0.25)))

    def test_gfs_publishes_the_areas_not_the_oceans(self):
        import build
        kept = {'Virginia': (39.3, -77.6), 'Newfoundland': (48.9, -55.0), 'Paris': (48.9, 2.4), 'Alps': (46.5, 9.0), 'Iceland': (64.1, -21.9), 'Moscow': (55.8, 37.6), 'Marrakech': (31.6, -8.0), 'Córdoba': (-31.4, -64.2), 'São Paulo': (-22.9, -47.1), 'Patagonia': (-45.0, -70.0), 'Bariloche': (-41.1, -71.3), 'Tierra del Fuego': (-54.5, -67.5), 'Quixadá': (-4.97, -39.0), 'Roldanillo': (4.4, -76.15), 'Cusco': (-13.5, -72.0), 'Mérida': (8.6, -71.1), 'Bitterwasser': (-23.9, 17.9), 'Omarama': (-44.5, 169.97), 'Benalla': (-36.55, 146.0), 'Narromine': (-32.2, 148.25), 'Waikerie': (-34.2, 140.0), 'Kingaroy': (-26.55, 151.85), 'Charters Towers': (-20.1, 146.3), 'Alice Springs': (-23.8, 133.9), 'Narrogin': (-32.9, 117.2), 'Tasmania': (-41.5, 147.2), 'Pokhara': (28.2, 83.98), 'Bir Billing': (32.05, 76.72), 'Skardu': (35.3, 75.6), 'Paro': (27.4, 89.4)}
        out = {'mid-Atlantic': (40.0, -40.0), 'Greenland': (65.0, -42.0), 'Sahara': (25.0, 5.0), 'Arctic Canada': (66.0, -100.0), 'Nullarbor': (-30.0, 128.0), 'Darwin': (-12.4, 130.8), 'Lake Eyre': (-24.0, 138.0), 'mid-Pacific': (-20.0, -150.0), 'Amazon': (-8.0, -63.0), 'Manaus': (-3.1, -60.0), 'Antarctica': (-70.0, 0.0), 'the Ganges plain': (25.0, 82.0), 'the Tarim basin': (40.0, 83.0), 'east of 180': (-40.0, 185.0)}
        pts = {**kept, **out}
        lat = np.array([[p[0] for p in pts.values()]])
        lon = np.array([[p[1] for p in pts.values()]])
        keep = dict(zip(pts, build.keep_mask('gfs', lat, lon)[0]))
        self.assertEqual({k for k, v in keep.items() if v}, set(kept))
        self.assertIsNone(build.keep_mask('hrrr', lat, lon))

    def test_cells_without_a_forecast_are_no_data(self):
        import build
        lat, lon = np.meshgrid([44.0, 43.5], [5.0, 6.0], indexing='ij')
        t2 = np.array([[290.0, 291.0], [np.nan, 292.0]])
        self.assertTrue((build.published_mask('icond2', lat, lon, {'t2': t2}) == np.isfinite(t2)).all())
        self.assertIsNone(build.published_mask('icond2', lat, lon, {'t2': np.full((2, 2), 290.0)}))
        self.assertIsNone(build.published_mask('hrrr', lat, lon, {}))
        g = build.published_mask('gfs', np.array([[40.0, 40.0]]), np.array([[-40.0, 5.0]]), {'t2': np.array([[np.nan, 290.0]])})
        self.assertEqual(g.tolist(), [[False, True]])

    def test_kept_boxes_only_are_computed_and_match_the_whole_grid(self):
        import build
        from scipy.ndimage import maximum_filter, minimum_filter
        region = (-10.0, 30.0, 14.75, 54.75, 0.25)
        lons, lats = build.target_grid(region)
        lon2d, lat2d = np.meshgrid(lons, lats)
        rng = np.random.default_rng(1)
        zs = rng.random(lat2d.shape, dtype=np.float32) * 1000
        lv = {'t': np.ascontiguousarray(rng.random((3,) + lat2d.shape, dtype=np.float32)), 'cloud': (np.arange(3, dtype=np.float32), None)}
        seen = []

        def fake(s, lv, levels, step, la, lo, valid):
            seen.append(all((a.flags['C_CONTIGUOUS'] and a.dtype == np.float32 for a in (s['zs'], lv['t']))))
            dy, dx = np.gradient(s['zs'])
            relief = maximum_filter(s['zs'], 3, mode='nearest') - minimum_filter(s['zs'], 3, mode='nearest')
            return {'g': dx + dy + lv['t'][1], 'relief': relief, 'lat': la, 'none': None}
        old_keep, old_analyse = (build.KEEP, build.analyse)
        try:
            build.KEEP = {'test': ((-5.0, 35.0, 0.0, 40.0), (5.0, 45.0, 10.0, 50.0))}
            build.analyse = fake
            full = fake({'zs': zs}, lv, None, 0.25, lat2d, lon2d, None)
            got = build.analyse_kept({'zs': zs, 'hub_m': 80.0}, lv, None, 0.25, lat2d, lon2d, None, build.keep_windows('test', region))
            keep = build.keep_mask('test', lat2d, lon2d)
        finally:
            build.KEEP, build.analyse = (old_keep, old_analyse)
        self.assertTrue(all(seen))
        self.assertIsNone(got['none'])
        for k in ('g', 'relief', 'lat'):
            np.testing.assert_allclose(got[k][keep], full[k][keep], rtol=1e-06, err_msg=k)
            self.assertTrue(np.isnan(got[k][~keep]).all(), k)
        self.assertIsNone(build.keep_windows('hrrr', region))

class IconOpenMeteo(unittest.TestCase):

    def setUp(self):
        try:
            import omfiles
        except ImportError:
            self.skipTest('omfiles not installed')
        import tempfile
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(getattr(self, 'dir', ''), ignore_errors=True)

    def write(self, name, arrays, units=None):
        import omfiles
        path = os.path.join(self.dir, name)
        w = omfiles.OmFileWriter(path)
        kids = []
        for k, a in arrays.items():
            u = [w.write_scalar((units or {}).get(k, 'm'), name='unit')]
            kids.append(w.write_array(np.asarray(a, np.float32), chunks=[min(4, n) for n in np.shape(a)], scale_factor=100.0, name=k, children=u))
        w.close(w.write_group('', children=kids))
        return path

    def test_the_hour_is_read_onto_the_output_grid_north_first_inside_the_boxes(self):
        import om
        a = np.arange(120, dtype=np.float32).reshape(10, 12)
        hour = om.Hour([om.open_file(self.write('h.om', {'temperature_2m': a}, {'temperature_2m': '°C'}))], om.Grid(0.0, 10.0, 1.0, 10, 12))
        region = (12.0, 2.0, 15.0, 7.0, 1.0)
        whole = hour.read('temperature_2m', region)
        self.assertEqual(whole.shape, (6, 4))
        np.testing.assert_allclose(whole[0], a[7, 2:6], atol=0.01)
        np.testing.assert_allclose(whole[-1], a[2, 2:6], atol=0.01)
        part = hour.read('temperature_2m', region, boxes=[(1, 3, 0, 2)])
        self.assertTrue(np.isnan(part[0]).all() and np.isnan(part[:, 2:]).all())
        np.testing.assert_allclose(part[1:3, :2], whole[1:3, :2])
        np.testing.assert_allclose(om.Field(hour, 'temperature_2m').read(region), whole + 273.15, atol=0.01)
        with self.assertRaises(ValueError):
            hour.read('temperature_2m', (12.5, 2.0, 15.0, 7.0, 1.0))
        with self.assertRaises(ValueError):
            hour.read('temperature_2m', (12.0, 2.0, 15.0, 7.0, 0.5))

    def test_derived_fields(self):
        import om
        n = (1, 4)
        hs = np.array([[-999.0, 0.0, 500.0, 2500.0]])
        lv = {}
        for p in om.PS_LEVELS:
            z = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
            lv[f'geopotential_height_{p}hPa'] = np.full(n, z)
            lv[f'temperature_{p}hPa'] = np.full(n, 15 - 0.0065 * z)
        units = {k: '°C' if k.startswith('temperature') else 'm' for k in lv}
        main = self.write('h.om', {**lv, 'temperature_2m': np.full(n, 15.0), 'relative_humidity_2m': np.full(n, 50.0), 'soil_temperature_0cm': np.array([[-3.0, 10.0, 10.0, 10.0]]), 'sensible_heat_flux': np.full(n, -200.0), 'direct_radiation': np.full(n, 500.0), 'diffuse_radiation': np.full(n, 100.0), 'precipitation': np.array([[0.0, 2.0, 2.0, 2.0]]), 'snowfall_water_equivalent': np.array([[0.0, 0.0, 1.0, 2.0]]), 'weather_code': np.array([[0.0, 66.0, 71.0, 73.0]]), 'pressure_msl': np.full(n, 1013.0)}, {**units, 'temperature_2m': '°C', 'soil_temperature_0cm': '°C', 'relative_humidity_2m': '%', 'pressure_msl': 'hPa'})
        static = om.open_file(self.write('s.om', {'x': hs}))
        hour = om.Hour([om.open_file(main)], om.Grid(0.0, 0.0, 1.0, 1, 4), arrays={'HSURF': static.get_child_by_name('x')})
        region = (0.0, 0.0, 3.0, 0.0, 1.0)
        get = lambda k: om.Field(hour, k).read(region)[0]
        np.testing.assert_allclose(get('zs'), [0, 0, 500, 2500], atol=0.1)
        np.testing.assert_array_equal(get('lsm'), [0, 1, 1, 1])
        np.testing.assert_array_equal(get('ice'), [1, 0, 0, 0])
        self.assertAlmostEqual(float(get('td2')[0]) - 273.15, 4.6, delta=0.2)
        np.testing.assert_allclose(get('ps') / 100, [1013.25, 1013.25, 954.6, 746.9], atol=3)
        np.testing.assert_allclose(get('sw'), 600, atol=0.1)
        np.testing.assert_allclose(get('shf'), 200, atol=0.1)
        np.testing.assert_array_equal(get('ptype'), [1, 3, 7, 5])
        np.testing.assert_allclose(get('pressure_msl'), 101300, atol=1)
        self.assertEqual(om.om_name(('temperature', '850 mb')), 'temperature_850hPa')

    def test_surface_pressure_from_the_levels_the_hour_has(self):
        import om
        n = (1, 3)
        hs = np.array([[0.0, 500.0, 2500.0]])
        lv = {}
        for p in (1000, 975, 950, 850, 700, 600, 500):
            z = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
            lv[f'geopotential_height_{p}hPa'] = np.full(n, z)
            lv[f'temperature_{p}hPa'] = np.full(n, 15 - 0.0065 * z)
        units = {k: '°C' if k.startswith('temperature') else 'm' for k in lv}
        main = self.write('h.om', {**lv, 'temperature_2m': np.full(n, 15.0)}, {**units, 'temperature_2m': '°C'})
        static = om.open_file(self.write('s.om', {'x': hs}))
        hour = om.Hour([om.open_file(main)], om.Grid(0.0, 0.0, 1.0, 1, 3), arrays={'HSURF': static.get_child_by_name('x')})
        f = om.Field(hour, 'ps')
        self.assertTrue(f.available())
        np.testing.assert_allclose(f.read((0.0, 0.0, 2.0, 0.0, 1.0))[0] / 100, [1013.25, 954.6, 746.9], atol=3)
        self.assertEqual(om.om_name(('cloud_cover', '')), 'cloud_cover')
        f = om.Field(hour, 'wind_u_component_80m')
        self.assertFalse(f.available(), 'not in these files')

    def test_steps_days_and_files(self):
        from datetime import datetime, timezone
        from sources import DAYS, SOURCES
        src = SOURCES['icon']
        steps = [fh for fh in range(0, 400) if src.fstep(fh)]
        self.assertEqual(len(steps), 78 + 34)
        self.assertEqual((steps[0], steps[77], steps[78], steps[-1]), (1, 78, 81, 180))
        self.assertEqual(src.runs, (0, 6, 12, 18))
        self.assertEqual([src.last_fh(datetime(2026, 10, 12, h, tzinfo=timezone.utc)) for h in src.runs], [180, 120, 180, 120])
        self.assertEqual(DAYS['icon'], 8)
        run = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
        self.assertEqual(src.url(run, 81), 'https://openmeteo.s3.amazonaws.com/data_spatial/dwd_icon/2026/10/12/1200Z/2026-10-15T2100.om')
        self.assertTrue(src.idx_url(run, 1).endswith('/2026/10/12/1200Z/2026-10-12T1300_model-level.om'))
        import om
        files = {'cape', 'cloud_cover', 'cloud_cover_high', 'cloud_cover_low', 'cloud_cover_mid', 'convective_cloud_base', 'convective_cloud_top', 'diffuse_radiation', 'direct_radiation', 'freezing_level_height', 'latent_heat_flux', 'precipitation', 'pressure_msl', 'rain', 'relative_humidity_2m', 'sensible_heat_flux', 'showers', 'snow_depth', 'snowfall_water_equivalent', 'soil_moisture_3_to_9cm', 'soil_temperature_0cm', 'temperature_2m', 'weather_code', 'wind_gusts_10m', 'wind_u_component_10m', 'wind_v_component_10m', 'wind_u_component_80m', 'wind_v_component_80m', 'HSURF'}
        files |= {f'{v}_{p}hPa' for p in (1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200) for v in ('temperature', 'relative_humidity', 'wind_u_component', 'wind_v_component', 'geopotential_height')}
        for key in [*src.surface.values(), *((g, f'{lv} mb') for lv in src.levels + src.cloud_levels for g in src.level_vars.values())]:
            name = om.om_name(key)
            self.assertTrue(name in files or set(om.RECIPES.get(name, ((None,), None))[0]) <= files, key)
        for need in ('zs', 'ps', 't2', 'td2', 'u10', 'v10', 'mslp'):
            self.assertIn(need, src.surface)

    def test_windows_are_the_steps_own(self):
        import build
        from sources import SOURCES
        src = SOURCES['icon']
        m, hrs = build._window_mean(src, None, 81, 'prate', np.array([3.0]), lambda f, k: self.fail('no earlier file'))
        self.assertEqual(hrs, 3)
        self.assertAlmostEqual(float(m[0]) * 3600, 1.0)
        m, hrs = build._window_mean(src, None, 5, 'sw', np.array([400.0]), lambda f, k: self.fail('no earlier file'))
        self.assertEqual((float(m[0]), hrs), (400.0, 1))

    def test_the_runs_start_comes_from_the_run_before(self):
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        from sources import SOURCES
        run = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            jobs = build.backfill_jobs(SOURCES['icon'], run, [run - timedelta(hours=1), run, run + timedelta(hours=1)], d)
            self.assertEqual(jobs, [(run - timedelta(hours=6), 5), (run - timedelta(hours=6), 6)])
            self.assertEqual(build.backfill_jobs(SOURCES['gfs'], run, [run], d), [])

    def test_short_runs_take_days_5_to_7_from_the_run_before(self):
        import tempfile
        from datetime import datetime, timedelta, timezone
        import build
        from sources import SOURCES, latest_complete_run
        src = SOURCES['icon']
        run = datetime(2026, 10, 12, 6, tzinfo=timezone.utc)
        h = lambda n: run + timedelta(hours=n)
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, f'{h(123):%Y%m%d%H}.npz'), 'w').close()
            jobs = build.backfill_jobs(src, run, [h(119), h(120), h(123), h(174), h(177), h(180)], d)
            r0 = run - timedelta(hours=6)
            self.assertEqual(jobs, [(r0, 129), (r0, 180)])
        import sources
        seen = []
        orig = sources.grib.exists
        sources.grib.exists = lambda url: seen.append(url) or True
        try:
            r, fhs = latest_complete_run(src, [h(1), h(120), h(126)], log=lambda *a: None, now=h(1))
        finally:
            sources.grib.exists = orig
        self.assertEqual((r, max(fhs)), (run, 120))
        self.assertIn(src.idx_url(run, 120), seen)

    def test_grid_is_on_the_files_lines_and_kept_like_gfs(self):
        import build
        import om
        from sources import SOURCES
        region = build.REGIONS['icon']
        i_n, j_w, ny, nx = om.Grid(*SOURCES['icon'].grid_def).window(region)
        lons, lats = build.target_grid(region)
        self.assertEqual((ny, nx), (len(lats), len(lons)))
        self.assertEqual((ny % build.TILE['icon'], -(-nx // build.TILE['icon'])), (0, 14))
        self.assertEqual(build.KEEP['icon'], build.KEEP['gfs'])
        self.assertIn('icon', build.FREE_MODELS)
        self.assertEqual(build.free_ahead('icon'), build.free_ahead('gfs'))

class EuropeAirspace(unittest.TestCase):
    XML = b'<?xml version="1.0"?><OFMX-Snapshot version="0.1" origin="test">\n    <Ase><AseUid region="LF"><codeType>TMA</codeType><codeId>T1</codeId></AseUid><txtName>STRASBOURG 9</txtName><codeClass>D</codeClass>\n      <codeDistVerUpper>STD</codeDistVerUpper><valDistVerUpper>145</valDistVerUpper><uomDistVerUpper>FL</uomDistVerUpper>\n      <codeDistVerLower>ALT</codeDistVerLower><valDistVerLower>4500</valDistVerLower><uomDistVerLower>FT</uomDistVerLower></Ase>\n    <Abd><AbdUid><AseUid region="LF"><codeType>TMA</codeType><codeId>T1</codeId></AseUid></AbdUid>\n      <Avx><codeType>GRC</codeType><geoLat>49.00000000N</geoLat><geoLong>008.00000000E</geoLong></Avx>\n      <Avx><codeType>CWA</codeType><geoLat>49.00000000N</geoLat><geoLong>008.50000000E</geoLong><geoLatArc>48.75000000N</geoLatArc><geoLongArc>008.50000000E</geoLongArc></Avx>\n      <Avx><codeType>GRC</codeType><geoLat>48.50000000N</geoLat><geoLong>008.50000000E</geoLong></Avx>\n      <Avx><codeType>FNT</codeType><geoLat>48.50000000N</geoLat><geoLong>008.00000000E</geoLong></Avx></Abd>\n    <Ase><AseUid region="LF"><codeType>CLASS</codeType><codeId>T1_D</codeId></AseUid><codeClass>D</codeClass>\n      <codeDistVerUpper>STD</codeDistVerUpper><valDistVerUpper>145</valDistVerUpper><uomDistVerUpper>FL</uomDistVerUpper>\n      <codeDistVerLower>ALT</codeDistVerLower><valDistVerLower>4500</valDistVerLower><uomDistVerLower>FT</uomDistVerLower></Ase>\n    <Adg><AdgUid><AseUid region="LF"><codeType>CLASS</codeType><codeId>T1_D</codeId></AseUid></AdgUid>\n      <AseUidSameExtent region="LF"><codeType>TMA</codeType><codeId>T1</codeId></AseUidSameExtent></Adg>\n    <Ase><AseUid region="LF"><codeType>R</codeType><codeId>R1</codeId></AseUid><txtName>LF-R 1</txtName>\n      <codeDistVerUpper>HEI</codeDistVerUpper><valDistVerUpper>300</valDistVerUpper><uomDistVerUpper>M</uomDistVerUpper>\n      <codeDistVerLower>HEI</codeDistVerLower><valDistVerLower>0</valDistVerLower><uomDistVerLower>FT</uomDistVerLower><Att><codeWorkHr>H24</codeWorkHr></Att></Ase>\n    <Abd><AbdUid><AseUid region="LF"><codeType>R</codeType><codeId>R1</codeId></AseUid></AbdUid>\n      <Circle><geoLatCen>48.00000000N</geoLatCen><geoLongCen>007.00000000E</geoLongCen><valRadius>2</valRadius><uomRadius>NM</uomRadius></Circle></Abd>\n    <Ase><AseUid region="LF"><codeType>FIR</codeType><codeId>F</codeId></AseUid><codeClass>G</codeClass></Ase>\n    </OFMX-Snapshot>'

    def test_controlled_and_special_use(self):
        import europe_airspace as E
        ctl, sua = E.parse(self.XML)
        self.assertEqual(len(ctl), 1)
        p = ctl[0]['properties']
        self.assertEqual((p['c'], p['lo'], p['hi']), ('D', '4,500 ft', 'FL145'))
        ring = ctl[0]['geometry']['coordinates'][0]
        self.assertGreater(max((x for x, _ in ring)), 8.7)
        self.assertEqual(ring[0], ring[-1])
        self.assertEqual(len(sua), 1)
        s = sua[0]['properties']
        self.assertEqual((s['t'], s['lo'], s['hi'], s['time']), ('R', 'SFC', '984 ft AGL', 'H24'))
        lats = [y for _, y in sua[0]['geometry']['coordinates'][0]]
        self.assertAlmostEqual(max(lats) - 48.0, 2 * 1852 / 111320, places=2)

    def test_airac_cycles(self):
        import europe_airspace as E
        from datetime import date
        self.assertEqual(E.airac(date(2026, 1, 22)), '2601')
        self.assertEqual(E.airac(date(2026, 1, 21)), '2513')
        self.assertEqual(E.cycles(date(2026, 10, 1)), ['2610', '2611', '2609'])

    def test_openaip_records(self):
        import europe_airspace as E
        sq = {'type': 'Polygon', 'coordinates': [[[-1, 51], [0, 51], [0, 52], [-1, 52], [-1, 51]]]}
        items = [{'name': 'LONDON CTR', 'type': 4, 'icaoClass': 3, 'geometry': sq, 'lowerLimit': {'value': 0, 'unit': 1, 'referenceDatum': 0}, 'upperLimit': {'value': 2500, 'unit': 1, 'referenceDatum': 1}}, {'name': 'LONDON TMA', 'type': 7, 'icaoClass': 0, 'geometry': sq, 'lowerLimit': {'value': 2500, 'unit': 1, 'referenceDatum': 1}, 'upperLimit': {'value': 195, 'unit': 6, 'referenceDatum': 2}}, {'name': 'EG D123', 'type': 2, 'icaoClass': 8, 'geometry': sq, 'lowerLimit': {'value': 0, 'unit': 1, 'referenceDatum': 0}, 'upperLimit': {'value': 300, 'unit': 0, 'referenceDatum': 0}}, {'name': 'FIR', 'type': 10, 'icaoClass': 6, 'geometry': sq}]
        ctl, sua = E.oa_features(items)
        self.assertEqual([(f['properties']['c'], f['properties']['lo'], f['properties']['hi']) for f in ctl], [('D', 'SFC', '2,500 ft'), ('A', '2,500 ft', 'FL195')])
        self.assertEqual([(f['properties']['t'], f['properties']['hi']) for f in sua], [('D', '984 ft AGL')])

def storm_column(rh_aloft=40.0, cap_z=None, sfc_t=30.0, **surface):
    s, lv, levels, _ = grid_column(n=3, sfc_t=sfc_t, sfc_td=24, lapse=0.0075, rh_levels=85)
    one = lambda x: np.full((3, 3), x, np.float32)
    if cap_z is not None:
        lv['t'] = np.where(lv['z'] > cap_z, sfc_t - 0.0075 * cap_z + 0.002 * (lv['z'] - cap_z), lv['t']).astype(np.float32)
    lv['rh'] = (np.where(levels[:, None, None] < 850, rh_aloft, 85) * np.ones_like(lv['t'])).astype(np.float32)
    lv['td'] = P.dewpoint_from_rh(lv['t'], lv['rh'])
    s.update(ps=one(1013.25), sw=one(800), shf=one(350), hpbl=one(1800), prate=one(1.0), cape=one(1000))
    s.update({k: one(v) if np.isscalar(v) else v for k, v in surface.items()})
    return (s, lv, levels)

class StormsFromTheModelsConvection(unittest.TestCase):

    def at(self, **kw):
        import build
        s, lv, levels = storm_column(**kw)
        out = build.analyse(dict(s), lv, levels, 0.25, np.full((3, 3), 40, np.float32))
        return {k: float(out[k][1, 1]) for k in ('cubase', 'cudepth', 'overdev')}

    def test_hrrr_needs_an_echo(self):
        weak = self.at(refmax=30.0)
        self.assertLess(weak['overdev'], 50)
        self.assertEqual(weak['overdev'], 45)
        self.assertGreaterEqual(self.at(refmax=45.0)['overdev'], 60)

    def test_gfs_icon_need_their_schemes_rain(self):
        self.assertLess(self.at(cprate=0.2)['overdev'], 50)
        self.assertGreaterEqual(self.at(cprate=0.8)['overdev'], 60)

    def test_storm_says(self):
        import build
        rain = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(build.storm_says({'cprate': np.array([0.2, 0.5, 0.9])}, rain, None), [False, True, True])
        np.testing.assert_array_equal(build.storm_says({'refmax': np.array([30.0, 41, 0])}, rain, np.array([0.0, 0, 45])), [False, True, True])
        self.assertTrue(build.storm_says({}, rain, None).all())

    def test_cu_depth_under_a_storm_is_its_updrafts(self):
        import build
        s, lv, levels = storm_column(refc=45.0, cape=2000.0)
        out = build.analyse(dict(s), lv, levels, 0.25, np.full((3, 3), 40, np.float32))
        dry = self.at(prate=0.0)
        self.assertLess(dry['cudepth'], 5000)
        self.assertGreaterEqual(float(out['overdev'][1, 1]), 50)
        self.assertGreater(float(out['cudepth'][1, 1]), dry['cudepth'] + 1000)
        self.assertEqual(float(out['cudepth'][1, 1]), self.at(refmax=45.0)['cudepth'])
        self.assertEqual(self.at(refmax=30.0)['cudepth'], dry['cudepth'])

    def test_warm_topped_cumulus_is_no_cb(self):
        import build
        warm = self.at(prate=0.0, cap_z=5500.0, sfc_t=33.0, rh_aloft=60.0)
        self.assertGreater(warm['cudepth'], 4000)
        self.assertLess(warm['overdev'], 50)
        s, lv, levels = storm_column(prate=0.0, cap_z=5500.0, sfc_t=33.0, rh_aloft=60.0)
        prof = P.Profile(s['zs'], s['ps'], s['t2'], s['td2'], s['u10'], s['v10'], lv['z'], levels, lv['t'], lv['td'], lv['u'], lv['v'])
        t_top = float(prof.at(np.full((3, 3), warm['cubase'] + 2 + warm['cudepth'], np.float32), 't')[1, 1])
        self.assertTrue(build.GLACIATE_T < t_top < 0, t_top)
        cold = self.at(prate=0.0, rh_aloft=60.0)
        self.assertGreaterEqual(cold['overdev'], 90)
        self.assertGreaterEqual(self.at(cap_z=5500.0, sfc_t=33.0, rh_aloft=60.0, refc=55.0)['overdev'], 70)

class CeilingsByLayer(unittest.TestCase):

    def ceil(self, zc, cc, tcc=100.0, **kw):
        one = lambda v: np.full((1,), v, np.float32)
        col = lambda a: np.array(a, np.float32)[:, None]
        return float(P.ceiling_from_cover(col(zc), col(cc), one(0), one(80), one(tcc), **kw)[0])

    def test_layers_add_up(self):
        zc = [500, 1000, 2000, 3000, 4000]
        c = self.ceil(zc, [0, 40, 0, 40, 0])
        self.assertTrue(2000 < c <= 3000, c)
        self.assertEqual(self.ceil(zc, [0, 30, 30, 0, 0]), P.CEIL_NONE)
        self.assertEqual(self.ceil(zc, [0, 45, 0, 10, 0]), P.CEIL_NONE)
        self.assertAlmostEqual(self.ceil(zc, [45, 0, 10, 0, 20]), 4000, delta=1)
        self.assertAlmostEqual(self.ceil(zc, [0, 25, 100, 100, 0]), 1000 + 1000 * (25 / 75), places=1)

    def test_gfs_cover_under_a_mid_deck(self):
        import build
        levels = np.array([1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300], np.float32)
        one = lambda v: np.full((1, 1), v, np.float32)
        z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)[:, None, None]
        cc = np.zeros(z.shape, np.float32)
        cc[list(levels).index(900)] = 30
        cc[list(levels).index(500)] = 60
        s = {'zs': one(0), 'ps': one(1013.25), 't2': one(25), 'td2': one(15), 'tcc': one(72)}
        lv = {'z': z, 'cc': cc, 'rh': np.full(z.shape, 70, np.float32)}
        native = one(float(z[list(levels).index(900), 0, 0]))
        inside = np.ones((1, 1), bool)
        c = float(build.ceiling_own_cover(P.ceiling_agl(native, s['zs'], inside), s, lv, levels, inside)[0, 0])
        self.assertTrue(float(z[list(levels).index(550), 0, 0]) < c < float(z[list(levels).index(500), 0, 0]), c)
        lv['cc'] = np.zeros(z.shape, np.float32)
        c = float(build.ceiling_own_cover(P.ceiling_agl(native, s['zs'], inside), s, lv, levels, inside)[0, 0])
        self.assertAlmostEqual(c, float(native[0, 0]), places=0)

    def test_ceiling_layer_top(self):
        L = P.CC_LEVELS
        zc = np.array([110, 540, 990, 1460, 1950, 3010, 4200, 5570, 7180, 9160, 10360, 11800], np.float32)[:, None]
        cc = np.zeros((len(L), 1), np.float32)
        cc[L.index(900)], cc[L.index(850)], cc[L.index(800)] = (80, 100, 90)
        cc[L.index(300)] = 90
        zs = np.zeros(1, np.float32)
        ceil = P.ceiling_from_cover(zc, cc, zs, np.array([80.0]))
        self.assertTrue(540 < float(ceil[0]) < 990, ceil)
        top = float(P.ceiling_top(ceil, zs, zc, cc)[0])
        self.assertTrue(1950 < top < 3010, top)
        self.assertAlmostEqual(top, 1950 + (90 - 50) / 90 * (3010 - 1950), places=0)
        self.assertTrue(np.isnan(P.ceiling_top(np.array([P.CEIL_NONE], np.float32), zs, zc, cc)[0]))
        self.assertTrue(np.isnan(P.ceiling_top(ceil, zs, zc, np.full_like(cc, 100))[0]))
        cc2 = np.zeros((len(L), 1), np.float32)
        cc2[L.index(900)], cc2[L.index(700)] = (40, 40)
        ceil2 = P.ceiling_from_cover(zc, cc2, zs, np.array([80.0]))
        self.assertTrue(1950 < float(ceil2[0]) < 3010, ceil2)
        self.assertAlmostEqual(float(P.ceiling_top(ceil2, zs, zc, cc2)[0]), 3010, places=0)
        import build
        self.assertEqual(build.ENC['ceiltop'], (0, 16000, None))
        self.assertNotIn('ceiltop', build.POINT_FIELDS)

class TotalCoverOfTheBands(unittest.TestCase):

    def test_column_max_becomes_the_bands_together(self):
        import build
        import sources
        self.assertTrue(sources.SOURCES['hrrr'].tcc_column_max and sources.SOURCES['gfs'].tcc_column_max)
        self.assertFalse(sources.SOURCES['icon'].tcc_column_max)
        tcc = np.array([50.0, 95.0, np.nan, 20.0])
        bands = [np.array([50.0, 50, 50, 20]), np.array([50.0, 50, 50, 0]), np.array([50.0, 50, 50, np.nan])]
        out = build.total_of_bands(tcc, bands)
        np.testing.assert_allclose(out[[0, 1, 3]], [87.5, 95, 20])
        self.assertTrue(np.isnan(out[2]))

class IconCover(unittest.TestCase):

    def test_uncalibrate_undoes_dwds_calibration(self):
        rng = np.random.default_rng(3)
        n = 2000
        l, m, h = (rng.uniform(0, 100, n).astype(np.float32) for _ in range(3))
        hi = np.maximum.reduce([l, m, h])
        t = hi + rng.uniform(0, 1, n).astype(np.float32) * (100 - hi)
        for lsm in (np.ones(n), np.zeros(n)):
            tc, lc, mc = P.icon_calibrate(t, l, m, h, lsm)
            self.assertTrue((lc >= l - 0.001).all() and (mc >= m - 0.001).all() and (tc >= t - 0.001).all())
            tr, lr, mr = P.icon_uncalibrate(tc, lc, mc, h, lsm)
            np.testing.assert_allclose(lr, l, atol=0.01)
            np.testing.assert_allclose(mr, m, atol=0.01)
            np.testing.assert_allclose(tr, t, atol=0.01)
        one = lambda v: np.array([v], np.float32)
        self.assertAlmostEqual(float(P.icon_calibrate(one(10), one(10), one(0), one(0), one(1))[1][0]), 28.2, delta=0.1)
        self.assertAlmostEqual(float(P.icon_calibrate(one(10), one(10), one(0), one(0), one(0))[1][0]), 15.0, delta=0.1)
        self.assertAlmostEqual(float(P.icon_uncalibrate(one(28.2), one(28.2), one(0), one(0), one(1))[1][0]), 10, delta=0.1)
        self.assertTrue(np.isnan(P.icon_uncalibrate(one(np.nan), one(np.nan), one(0), one(0), one(1))[0][0]))
        import sources
        self.assertTrue(sources.SOURCES['icon'].cover_calibrated)
        self.assertFalse(sources.SOURCES['gfs'].cover_calibrated or sources.SOURCES['hrrr'].cover_calibrated)

    def test_humidity_over_water_brought_to_ice(self):
        import sources
        self.assertTrue(sources.SOURCES['icon'].rh_over_water)
        self.assertFalse(sources.SOURCES['gfs'].rh_over_water)
        t = np.array([-30.0, 10.0], np.float32)
        rh_w = np.array([P.es_ice(-30.0) / P.es(-30.0) * 100, 75.0], np.float32)
        self.assertTrue(70 < rh_w[0] < 78)
        rh_c = P.rh_cloud(t, P.dewpoint_from_rh(t, rh_w))
        self.assertAlmostEqual(float(rh_c[0]), 100, delta=1)
        self.assertAlmostEqual(float(rh_c[1]), 75, delta=0.01)
        ps = np.full(2, 1000.0, np.float32)
        self.assertEqual(float(P.cloud_fraction(rh_w, 250, ps)[0]), 0)
        self.assertGreater(float(P.cloud_fraction(rh_c, 250, ps)[0]), 50)
        import build
        self.assertIs(build.cloud_rh({'rh_cloud': rh_c, 'rh': rh_w}), rh_c)
        self.assertIs(build.cloud_rh({'rh': rh_w}), rh_w)

class AltocumulusCastellanus(unittest.TestCase):

    def layer(self, lapse, rh=95.0, t650=-5.0):
        import build
        levels = np.array([1000, 925, 850, 750, 700, 650, 600, 550, 500, 400, 300], np.float32)
        z = (44330 * (1 - (levels / 1013.25) ** 0.1903)).astype(np.float32)
        k650, k550 = (list(levels).index(650), list(levels).index(550))
        t = np.empty_like(z)
        t[:k650 + 1] = t650 + 0.0065 * (z[k650] - z[:k650 + 1])
        t[k650:k550 + 1] = t650 - lapse / 1000 * (z[k650:k550 + 1] - z[k650])
        t[k550:] = t[k550] - 0.0065 * (z[k550:] - z[k550])
        r = np.full_like(z, 50.0)
        r[k650] = r[list(levels).index(600)] = rh
        sh = lambda a: a[:, None, None].astype(np.float32)
        lv = {'z': sh(z), 't': sh(t), 'rh': sh(r), 'td': P.dewpoint_from_rh(sh(t), sh(r))}
        return float(build.accas_score(lv, levels, np.zeros((1, 1), np.float32), np.full((1, 1), 15, np.float32))[0, 0])

    def test_unstable_saturated_layer_scores(self):
        self.assertGreater(self.layer(8.0), 0)
        self.assertGreater(self.layer(9.0), self.layer(8.0))
        self.assertEqual(self.layer(5.0), 0)
        self.assertEqual(self.layer(8.0, rh=60.0), 0)
        import build
        self.assertEqual(build.ENC['accas'], (0, 100, None))
        self.assertIn('accas', build.POINT_FIELDS)

class OwnSoundings(unittest.TestCase):

    def test_quantize_round_trips_and_marks_none(self):
        import build
        enc = build.snd_enc(500)
        lo, hi = enc['t']
        self.assertTrue(lo < -21 < hi and abs((lo + hi) / 2 + 21.2) < 1)
        q = build.snd_quantize(np.array([-21.0, np.nan, 999.0], np.float32), lo, hi)
        self.assertEqual(int(q[1]), 255)
        self.assertEqual(int(q[2]), 254)
        self.assertLess(abs(lo + (hi - lo) * int(q[0]) / 254 + 21.0), 0.3)

    def test_wanted_has_the_extra_levels_once(self):
        import sources
        src = sources.SOURCES['hrrr']
        w = src.wanted()
        self.assertEqual(len(w), len(set(w)))
        for c in ('TMP', 'DPT', 'UGRD', 'VGRD', 'HGT', 'VVEL'):
            self.assertIn((c, '150 mb'), w)
            self.assertIn((c, '825 mb'), w)
        self.assertIn(('UGRD', '250 mb'), w)

    def test_the_packs_layout_is_the_apps(self):
        import tempfile
        from datetime import datetime, timezone
        from PIL import Image
        import io
        import build
        import pack
        import sources
        src = sources.SOURCES['hrrr']
        with tempfile.TemporaryDirectory() as d:
            hours, out = (os.path.join(d, 'h'), os.path.join(d, 'out'))
            os.makedirs(hours)
            nv, sy, sx = (5, 5, 9)
            times = [datetime(2026, 10, 8, h, tzinfo=timezone.utc) for h in (17, 18)]
            yy, xx = np.mgrid[:sy, :sx]
            for k, t in enumerate(times):
                snd = np.stack([(10 * yy + xx + 100 * k + v) % 250 for v in range(nv)]).astype(np.uint8)
                np.savez_compressed(os.path.join(hours, f'{t:%Y%m%d%H}.npz'), wstar=np.zeros((15, 27), np.uint8), elevation=np.zeros((15, 27), np.uint8), snd=snd)
            m = build.write_soundings(out, hours, times, src, block=4)
            self.assertEqual((m['block'], m['nx'], m['ny'], m['step']), (4, sx, sy, src.snd_step))
            day = m['days']['20261008']
            self.assertEqual(day['stamps'], ['2026100817', '2026100818'])
            self.assertEqual(sorted(day['tiles']), ['0_0', '1_0'])
            path = os.path.join(out, 'snd', '20261008.pack')
            img = np.array(Image.open(io.BytesIO(pack.read_entry(path, day['tiles'].index('1_0')))))
            W = 5
            self.assertEqual(img.shape, (2 * W, nv * W))
            self.assertEqual(int(img[1 * W + 2, 3 * W + 3]), 130)
            days, present, _ = build.write_tiles(os.path.join(d, 't'), hours, times, 27, times[0])
            self.assertNotIn('snd', present)
from test_aviation_world import *
from test_airspace_world import *
