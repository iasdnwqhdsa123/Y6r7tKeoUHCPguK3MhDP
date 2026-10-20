import json
import os
import tempfile
import unittest
import aviation as A
import borders as B
__all__ = ['AviationWorld', 'BordersWorld', 'IdentsKFallback']

def field(id_, lat, lon, ident, kind='small_airport', name='', icao='', local='', region='US-XX'):
    return {'id': id_, 'type': kind, 'name': name or f'Field {id_}', 'latitude_deg': str(lat), 'longitude_deg': str(lon), 'elevation_ft': '500', 'icao_code': icao, 'gps_code': icao or ident, 'local_code': local, 'ident': ident, 'iata_code': '', 'municipality': 'Town', 'iso_region': region}

def navaid(ident, lat, lon, variation, kind='VOR-DME', name=''):
    return {'type': kind, 'ident': ident, 'name': name or ident, 'latitude_deg': str(lat), 'longitude_deg': str(lon), 'frequency_khz': '114800' if kind != 'NDB' else '350', 'magnetic_variation_deg': str(variation), 'elevation_ft': '100', 'dme_channel': ''}
FILLERS = [field(str(i), 30 + i % 20, -120 + i % 40, f'X{i:04d}', local=f'{i:04d}') for i in range(1200)]
KBIG = field('big', 39.4, -77.4, 'KBIG', kind='large_airport', name='Big Intl', icao='KBIG', local='BIG', region='US-MD')
US_NAVAIDS = [navaid('WWW', 40, -118, 14.3, 'VORTAC', 'West'), navaid('EEE', 40, -75, -12.0, 'VOR', 'East')]

def snapshot(airports, navaids):
    fake = {'airports.csv': airports, 'runways.csv': [], 'airport-frequencies.csv': [], 'navaids.csv': navaids}
    real = A.ourairports
    A.ourairports = lambda name: fake[name]
    try:
        with tempfile.TemporaryDirectory() as d:
            out = {'kinds': A.build_airports(d)}
            for name in ('idents', 'airports', 'magvar', 'cells'):
                with open(os.path.join(d, f'{name}.json')) as f:
                    out[name] = json.load(f)
            return out
    finally:
        A.ourairports = real

class AviationWorld(unittest.TestCase):

    def test_new_regions_kept_and_the_rest_of_the_world_not(self):
        world = [field('gru', -23.43, -46.47, 'SBGR', 'large_airport', 'Guarulhos', 'SBGR', 'SP0002', 'BR-SP'), field('akl', -37.01, 174.79, 'NZAA', 'large_airport', 'Auckland', 'NZAA', '', 'NZ-AUK'), field('ivc', -46.41, 168.31, 'NZNV', 'medium_airport', 'Invercargill', 'NZNV', '', 'NZ-STL'), field('jnb', -26.14, 28.25, 'FAOR', 'large_airport', 'Tambo', 'FAOR', '', 'ZA-GT'), field('cmn', 33.37, -7.59, 'GMMN', 'large_airport', 'Casablanca', 'GMMN', '', 'MA-06'), field('hnd', 35.55, 139.79, 'RJTT', 'large_airport', 'Haneda', 'RJTT', '', 'JP-13'), field('fih', -4.39, 15.44, 'FZAA', 'large_airport', 'Kinshasa', 'FZAA', '', 'CD-KN'), field('ush', -54.84, -68.3, 'SAWH', 'medium_airport', 'Ushuaia', 'SAWH', '', 'AR-V'), field('ipc', -27.16, -109.42, 'SCIP', 'medium_airport', 'Mataveri', 'SCIP', '', 'CL-VS')]
        out = snapshot(FILLERS + [KBIG] + world, US_NAVAIDS)
        ids, search = (out['idents'], {a[0] for a in out['airports']['a']})
        for code in ('SBGR', 'NZAA', 'NZNV', 'FAOR', 'GMMN', 'KBIG', 'SAWH'):
            self.assertIn(code, ids)
            self.assertIn(code, search)
        for code in ('RJTT', 'FZAA', 'SCIP'):
            self.assertNotIn(code, ids)
            self.assertNotIn(code, search)
        self.assertEqual(out['kinds']['L'], 5)
        self.assertIn(A.cell_of(-23.43, -46.47), out['cells']['cells'])
        self.assertEqual(A.cell_of(-23.43, -46.47), '-10_-5')

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

    def test_magnetic_variation_in_the_south(self):
        south = [navaid('AA', -37.0, 174.8, 19.3), navaid('BCO', -23.5, -46.6, -20.0), navaid('JSV', -26.1, 28.2, -17.5), navaid('ENO', -34.8, -58.5, -6.5)]
        mv = snapshot(FILLERS, US_NAVAIDS + south)['magvar']
        self.assertLessEqual(mv['south'], -48)
        self.assertGreaterEqual(mv['west'] + (mv['nx'] - 1) * mv['step'], 179.9)

        def at(lat, lon):
            return mv['values'][round((lat - mv['south']) / mv['step'])][round((lon - mv['west']) / mv['step'])]
        self.assertAlmostEqual(at(-37.0, 174.8), 19.3, delta=0.5)
        self.assertAlmostEqual(at(-23.5, -46.6), -20.0, delta=0.5)
        self.assertAlmostEqual(at(-26.1, 28.2), -17.5, delta=0.5)
        self.assertGreater(at(40.0, -118.0), 5)
        self.assertIsNone(at(-40.0, -120.0))

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
