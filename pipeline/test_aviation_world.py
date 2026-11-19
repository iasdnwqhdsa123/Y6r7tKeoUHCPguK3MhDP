import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
import aviation as A
import borders as B
import wmm
__all__ = ['AviationWorld', 'BordersWorld', 'IdentsKFallback', 'RunwayHeadings', 'OverviewHeading', 'MagneticModel', 'SpecialUseLimits']

def field(id_, lat, lon, ident, kind='small_airport', name='', icao='', local='', region='US-XX'):
    return {'id': id_, 'type': kind, 'name': name or f'Field {id_}', 'latitude_deg': str(lat), 'longitude_deg': str(lon), 'elevation_ft': '500', 'icao_code': icao, 'gps_code': icao or ident, 'local_code': local, 'ident': ident, 'iata_code': '', 'municipality': 'Town', 'iso_region': region}

def navaid(ident, lat, lon, variation, kind='VOR-DME', name=''):
    return {'type': kind, 'ident': ident, 'name': name or ident, 'latitude_deg': str(lat), 'longitude_deg': str(lon), 'frequency_khz': '114800' if kind != 'NDB' else '350', 'magnetic_variation_deg': str(variation), 'elevation_ft': '100', 'dme_channel': ''}
FILLERS = [field(str(i), 30 + i % 20, -120 + i % 40, f'X{i:04d}', local=f'{i:04d}') for i in range(1200)]
KBIG = field('big', 39.4, -77.4, 'KBIG', kind='large_airport', name='Big Intl', icao='KBIG', local='BIG', region='US-MD')
US_NAVAIDS = [navaid('WWW', 40, -118, 14.3, 'VORTAC', 'West'), navaid('EEE', 40, -75, -12.0, 'VOR', 'East')]

def snapshot(airports, navaids, runways=()):
    fake = {'airports.csv': airports, 'runways.csv': list(runways), 'airport-frequencies.csv': [], 'navaids.csv': navaids}
    real = A.ourairports
    A.ourairports = lambda name: fake[name]
    try:
        with tempfile.TemporaryDirectory() as d:
            out = {'kinds': A.build_airports(d)}
            for name in ('idents', 'airports', 'magvar', 'cells', 'overview'):
                with open(os.path.join(d, f'{name}.json')) as f:
                    out[name] = json.load(f)
            return out
    finally:
        A.ourairports = real

class AviationWorld(unittest.TestCase):

    def test_every_forecast_area_kept_and_nothing_outside(self):
        kept = [field('gru', -23.43, -46.47, 'SBGR', 'large_airport', 'Guarulhos', 'SBGR', 'SP0002', 'BR-SP'), field('akl', -37.01, 174.79, 'NZAA', 'large_airport', 'Auckland', 'NZAA', '', 'NZ-AUK'), field('ivc', -46.41, 168.31, 'NZNV', 'medium_airport', 'Invercargill', 'NZNV', '', 'NZ-STL'), field('jnb', -26.14, 28.25, 'FAOR', 'large_airport', 'Tambo', 'FAOR', '', 'ZA-GT'), field('cmn', 33.37, -7.59, 'GMMN', 'large_airport', 'Casablanca', 'GMMN', '', 'MA-06'), field('ush', -54.84, -68.3, 'SAWH', 'medium_airport', 'Ushuaia', 'SAWH', '', 'AR-V'), field('bla', -36.55, 146.01, 'YBLA', 'medium_airport', 'Benalla', 'YBLA', '', 'AU-VIC'), field('hnd', 35.55, 139.79, 'RJTT', 'large_airport', 'Haneda', 'RJTT', '', 'JP-13'), field('tki', 42.73, 143.22, 'RJCT', 'small_airport', 'Tokachi', 'RJCT', '', 'JP-01'), field('pek', 40.08, 116.58, 'ZBAA', 'large_airport', 'Beijing Capital', 'ZBAA', '', 'CN-11'), field('pkr', 28.2, 83.98, 'VNPK', 'medium_airport', 'Pokhara', 'VNPK', '', 'NP-P4')]
        dropped = [field('fih', -4.39, 15.44, 'FZAA', 'large_airport', 'Kinshasa', 'FZAA', '', 'CD-KN'), field('ipc', -27.16, -109.42, 'SCIP', 'medium_airport', 'Mataveri', 'SCIP', '', 'CL-VS'), field('hnl', 21.32, -157.92, 'PHNL', 'large_airport', 'Honolulu', 'PHNL', 'HNL', 'US-HI'), field('sjo', 9.99, -84.2, 'MROC', 'large_airport', 'Juan Santamaría', 'MROC', '', 'CR-A'), field('mao', -3.04, -60.05, 'SBEG', 'large_airport', 'Eduardo Gomes', 'SBEG', '', 'BR-AM'), field('nop', 95.0, 10.0, 'XXXX', 'small_airport', 'Nowhere', '', '', 'XX-XX')]
        out = snapshot(FILLERS + [KBIG] + kept + dropped, US_NAVAIDS + [navaid('BLA', -36.6, 146.0, 12.0, 'NDB'), navaid('HNL', 21.3, -157.9, 10.0, 'VORTAC')])
        ids, search = (out['idents'], {a[0] for a in out['airports']['a']})
        for code in ('SBGR', 'NZAA', 'NZNV', 'FAOR', 'GMMN', 'KBIG', 'SAWH', 'YBLA', 'RJTT', 'RJCT', 'ZBAA', 'VNPK'):
            self.assertIn(code, ids)
            self.assertIn(code, search)
        for code in ('FZAA', 'SCIP', 'PHNL', 'MROC', 'SBEG', 'XXXX', 'HNL'):
            self.assertNotIn(code, ids)
            self.assertNotIn(code, search)
        self.assertEqual(out['kinds']['L'], 7)
        self.assertIn(A.cell_of(-23.43, -46.47), out['cells']['cells'])
        self.assertEqual(A.cell_of(-23.43, -46.47), '-10_-5')
        self.assertIn(A.cell_of(-36.55, 146.01), out['cells']['cells'])
        self.assertNotIn(A.cell_of(21.32, -157.92), out['cells']['cells'])
        self.assertEqual(ids['BLA'][2:], ['N', 'BLA', 'NDB'])
        self.assertIn('YBLA', {r[0] for r in out['overview']['a']})
        self.assertEqual(A.region_of(-36.55, 146.01), len(A.REGIONS))
        self.assertIsNone(A.region_of(None, 146.0))

    def test_the_areas_are_the_forecasts_own(self):
        areas = A.forecast_areas()
        self.assertGreaterEqual(len(areas), 10)
        self.assertIn((128.5, 30.0, 146.0, 46.0), areas)
        self.assertTrue(A.in_forecast_area(-36.55, 146.0) and A.in_forecast_area(40.0, 116.4) and A.in_forecast_area(43.0, 141.4))
        self.assertFalse(A.in_forecast_area(21.3, -157.9) or A.in_forecast_area(-4.4, 15.4))
        self.assertTrue(A.in_forecast_area(-37.0, 180.5))
        self.assertTrue(A.in_forecast_area(-37.0, -179.5))
        self.assertFalse(A.in_forecast_area(-37.0, -177.0))

    def test_new_regions_never_take_an_existing_identifier(self):
        base = snapshot(FILLERS + [KBIG], US_NAVAIDS)['idents']
        argentina = [field('ar1', -35.2, -59.1, 'AR-0001', local='BIG', region='AR-B'), field('ar2', -34.0, -60.0, 'AR-0002', local='EEE', region='AR-B'), field('ar3', -33.0, -61.0, 'AR-0003', local='XQZ', region='AR-S')]
        out = snapshot(argentina + FILLERS + [KBIG], [navaid('WWW', -37.0, 174.8, 19.3, 'NDB')] + US_NAVAIDS)
        ids = out['idents']
        self.assertEqual({k: v for k, v in ids.items() if k in base}, base)
        self.assertEqual(ids['BIG'][:3], [39.4, -77.4, 'A'])
        self.assertEqual(ids['EEE'][:3], [40.0, -75.0, 'N'])
        self.assertEqual(ids['WWW'][:2], [40.0, -118.0])
        self.assertEqual(ids['XQZ'][:2], [-33.0, -61.0])
        self.assertEqual(ids['AR-0001'][:2], [-35.2, -59.1])
        self.assertEqual(list(ids)[:len(base)], list(base))

    def test_magnetic_variation_in_every_forecast_area(self):
        mv = snapshot(FILLERS, US_NAVAIDS)['magvar']
        self.assertEqual((mv['west'], mv['step']), (-180.0, 2.0))
        self.assertEqual(mv['west'] + (mv['nx'] - 1) * mv['step'], 180.0)
        self.assertEqual(len(mv['values']), mv['ny'])
        self.assertTrue(all((len(r) == mv['nx'] for r in mv['values'])))
        self.assertEqual(mv['model'], 'WMM-2025')

        def at(lat, lon):
            return A.magvar_at(mv, lat, lon)
        self.assertAlmostEqual(at(-36.55, 146.0), 11.9, delta=0.4)
        self.assertAlmostEqual(at(-37.0, 174.8), 20.4, delta=0.4)
        self.assertAlmostEqual(at(-23.5, -46.6), -21.9, delta=0.4)
        self.assertAlmostEqual(at(-26.1, 28.2), -20.7, delta=0.4)
        self.assertGreater(at(40.0, -118.0), 5)
        self.assertLess(at(40.0, -75.0), -5)
        for lat, lon in ((35.7, 139.7), (43.0, 141.4), (40.0, 116.4), (31.2, 121.5), (27.7, 85.3), (-54.8, -68.3)):
            self.assertIsNotNone(at(lat, lon), (lat, lon))
        for w, s, e, n in A.forecast_areas():
            for lat, lon in ((s - A.AREA_HALO, w - A.AREA_HALO), (n + A.AREA_HALO, e + A.AREA_HALO), ((s + n) / 2, (w + e) / 2)):
                self.assertAlmostEqual(at(lat, lon), wmm.declination(lat, lon, mv['year']), delta=0.6, msg=(lat, lon))
        self.assertIsNone(at(-40.0, -120.0))
        self.assertIsNone(at(21.3, -157.9))
        self.assertAlmostEqual(at(-44.0, 180.5), at(-44.0, -179.5), delta=1e-09)

class MagneticModel(unittest.TestCase):
    VALUES = [(2025.0, 28, 89, -121, -99.77), (2025.0, 48, 80, -96, -29.91), (2025.0, 65, 43, 93, 0.5), (2025.0, 51, -33, 109, -5.49), (2025.0, 39, -59, -8, -15.75), (2025.5, 22, -37, 140, 9.28), (2025.5, 44, 33, -118, 11.1), (2026.0, 46, -24, -122, 14.01), (2026.5, 12, -79, 115, -137.58), (2027.0, 57, -43, 50, -48.27), (2027.5, 0, -13, -59, -17.49), (2028.0, 30, -36, -64, -4.65), (2028.5, 11, 34, 0, 1.57), (2029.0, 49, -50, -179, 32.11), (2029.5, 77, -18, 138, 4.45)]

    def test_against_the_published_values(self):
        for year, km, lat, lon, d in self.VALUES:
            self.assertAlmostEqual(wmm.declination(lat, lon, year, km), d, delta=0.01, msg=(year, lat, lon))

    def test_grid_matches_the_model(self):
        g = wmm.grid(2026.5, step=10.0)
        self.assertEqual((g['nx'], g['ny'], g['year']), (37, 18, 2026.5))
        self.assertEqual(g['values'][0][0], g['values'][0][-1])
        self.assertAlmostEqual(A.magvar_at(g, -38.0, 140.0), wmm.declination(-38.0, 140.0, 2026.5), delta=0.05)
        self.assertAlmostEqual(wmm.decimal_year(datetime(2026, 7, 2, 12, tzinfo=timezone.utc)), 2026.5, delta=0.001)

class BordersWorld(unittest.TestCase):

    def test_country_lines_in_the_new_regions_state_lines_in_north_america(self):
        states = {'features': [{'geometry': {'type': 'LineString', 'coordinates': [[-100, 40], [-99, 41]]}}] * 60 + [{'geometry': {'type': 'LineString', 'coordinates': [[-62, -35], [-61, -36]]}}]}
        lines = {'argentina-chile': [[-70, -35], [-69.5, -36]], 'south africa-botswana': [[25, -25], [26, -24.5]], 'brazil-venezuela': [[-64, 4], [-63, 1], [-62, -0.5]], 'japan': [[139, 35], [140, 36]], 'congo': [[20, -5], [21, -6]]}
        countries = {'features': [{'geometry': {'type': 'LineString', 'coordinates': c}} for c in lines.values()]}
        real = B.get
        B.get = lambda url, **k: json.dumps(states if 'admin_1' in url else countries).encode()
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'borders.json')
                self.assertEqual(B.main(path=path), 63)
                with open(path) as f:
                    out = json.load(f)
                self.assertEqual(len(out['s']), 60)
                self.assertEqual(out['c'], [[-70, -35, -69.5, -36], [25, -25, 26, -24.5], [-64, 4, -63, 1, -62, -0.5]])
        finally:
            B.get = real

class IdentsKFallback(unittest.TestCase):

    def test_later_regions_dont_take_codes_reached_through_k(self):
        import aviation
        idents = {'KGDP': [31.9, -105.2, 'A', 'DELL CITY'], 'XYZ': [40.0, -100.0, 'N', 'US VOR']}
        later = {'GDP': [-6.8, -43.6, 'N', 'GUADALUPE'], 'KXYZ': [-30.0, -60.0, 'A', 'AR FIELD'], 'SBGR': [-23.4, -46.5, 'A', 'GUARULHOS'], 'XYZ': [-1, -1, 'N', 'X']}
        aviation.merge_later(idents, later)
        self.assertNotIn('GDP', idents)
        self.assertNotIn('KXYZ', idents)
        self.assertEqual(idents['XYZ'][3], 'US VOR')
        self.assertEqual(idents['SBGR'][3], 'GUARULHOS')

class RunwayHeadings(unittest.TestCase):

    def setUp(self):
        row = [14.0 if -180 + i * 2 < -100 else -11.0 for i in range(181)]
        self.grid = {'west': -180.0, 'south': -88.0, 'step': 2.0, 'nx': 181, 'ny': 89, 'values': [row] * 89}

    def rw(self, le='', heading='', ends=None):
        r = {'le_ident': le, 'le_heading_degT': heading}
        if ends:
            r.update(zip(('le_latitude_deg', 'le_longitude_deg', 'he_latitude_deg', 'he_longitude_deg'), map(str, ends)))
        return r

    def test_own_heading_first(self):
        self.assertEqual(A.runway_heading(self.rw('15', '143.6'), (40, -75), self.grid), 144)

    def test_from_the_ends(self):
        self.assertEqual(A.runway_heading(self.rw('18', ends=(40.0, -75.0, 40.01, -75.0)), (40, -75), self.grid), 0)
        self.assertEqual(A.runway_heading(self.rw('09', ends=(40.0, -75.01, 40.0, -75.0)), (40, -75), self.grid), 90)

    def test_number_turned_to_true(self):
        self.assertEqual(A.runway_heading(self.rw('15'), (40, -75), self.grid), 139)
        self.assertEqual(A.runway_heading(self.rw('9L'), (40, -118), self.grid), 104)
        self.assertEqual(A.runway_heading(self.rw('NE'), (40, -75), self.grid), 45)

    def test_none_for_helipads_and_odd_idents(self):
        for le in ('H1', 'ALL', '', '40', '00'):
            self.assertIsNone(A.runway_heading(self.rw(le), (40, -75), self.grid), le)

class OverviewHeading(unittest.TestCase):

    def rw(self, ref, le, he, length, surface, heading=''):
        return {'airport_ref': ref, 'le_ident': le, 'he_ident': he, 'length_ft': str(length), 'width_ft': '100', 'surface': surface, 'lighted': '1', 'closed': '0', 'le_heading_degT': heading}

    def test_main_runway_as_the_map_picks_it(self):
        med = field('med', 39.7, -77.7, 'KMED', kind='medium_airport', name='Med Regional', icao='KMED', region='US-MD')
        bare = field('bare', 39.9, -77.9, 'KBAR', kind='medium_airport', name='No Runways', icao='KBAR', region='US-MD')
        runways = [self.rw('big', '18', '36', 9000, 'ASP', '3.4'), self.rw('med', '09', '27', 5000, 'TURF', '88.6'), self.rw('med', '14', '32', 4000, 'ASPH', '137.2'), self.rw('small', '05', '23', 3000, 'ASP', '50')]
        ov = snapshot(FILLERS + [KBIG, med, bare], US_NAVAIDS, runways)['overview']
        recs = {r[0]: r for r in ov['a']}
        self.assertEqual(recs['KBIG'][7:], [3])
        self.assertEqual(recs['KMED'][7:], [137])
        self.assertEqual(len(recs['KBAR']), 7)
        self.assertTrue(all((isinstance(r[7], int) for r in ov['a'] if len(r) > 7)))

    def test_main_runway_heading(self):
        rw = lambda length, surface, h: ['', length, 50, surface, 0, h]
        self.assertEqual(A.main_runway_heading([rw(5000, 'TURF', 90), rw(3000, 'ASP', 140)]), 90)
        self.assertEqual(A.main_runway_heading([rw(5000, 'GRS', None), rw(2000, 'GRS', 10)]), 10)
        self.assertIsNone(A.main_runway_heading([rw(5000, 'ASP', None)]))
        self.assertIsNone(A.main_runway_heading([]))

class SpecialUseLimits(unittest.TestCase):
    MOA = {'NAME': 'WARRIOR 2 LOW MOA', 'TYPE_CODE': 'MOA', 'LOWER_VAL': '100', 'LOWER_UOM': 'FT', 'LOWER_CODE': 'SFC', 'UPPER_VAL': '10000', 'UPPER_UOM': 'FT', 'UPPER_CODE': 'MSL', 'EXCLUSION': '0', 'TIMESOFUSE': '0700 - 2200, MON - FRI; OTHER TIMES BY NOTAM', 'CONT_AGENT': 'FAA, HOUSTON ARTCC'}

    def test_heights_over_the_ground_coded_sfc(self):
        self.assertEqual(A.limit('100', 'FT', 'SFC'), '100 ft AGL')
        self.assertEqual(A.limit('1500', 'FT', 'SFC'), '1,500 ft AGL')
        self.assertEqual(A.limit('0', 'FT', 'SFC'), 'SFC')
        self.assertEqual(A.limit(None, None, 'SFC'), 'SFC')
        self.assertEqual(A.limit('10000', 'FT', 'MSL'), '10,000 ft')
        self.assertEqual(A.limit('700', 'FT', 'AGL'), '700 ft AGL')
        self.assertEqual(A.limit('180', 'FL', 'STD'), 'FL180')
        self.assertEqual(A.limit('-9998', None, 'UNLTD'), '-9998')

    def test_exclusions_marked(self):
        area = A.sua_props(self.MOA, 'MOA')
        self.assertEqual((area['lo'], area['hi']), ('100 ft AGL', '10,000 ft'))
        self.assertNotIn('x', area)
        excl = A.sua_props({**self.MOA, 'LOWER_VAL': '1500', 'EXCLUSION': '1', 'REMARKS': 'EXCLUDES AT AND BELOW 1500 FT AGL'}, 'MOA')
        self.assertEqual((excl['n'], excl['lo'], excl['x']), ('WARRIOR 2 LOW MOA', '1,500 ft AGL', 1))
