import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock
__all__ = ['OpenaipAreas', 'ControlledKinds']
_SPOT = {'GB': (-1.0, 51.5), 'IE': (-6.3, 53.4), 'ES': (-3.7, 40.4), 'PT': (-9.1, 38.7), 'FR': (2.3, 48.9), 'MA': (-7.6, 33.6), 'DZ': (3.0, 36.7), 'TN': (10.2, 36.8), 'BR': (-47.9, -15.8), 'AR': (-58.4, -34.6), 'CL': (-70.7, -33.4), 'UY': (-56.2, -34.9), 'PY': (-57.6, -25.3), 'BO': (-68.1, -16.5), 'PE': (-77.0, -12.0), 'EC': (-78.5, -0.2), 'CO': (-74.1, 4.7), 'VE': (-66.9, 10.5), 'ZA': (28.0, -26.2), 'NA': (17.1, -22.6), 'BW': (25.9, -24.6), 'ZW': (31.0, -17.8), 'MZ': (32.6, -25.9), 'LS': (27.5, -29.3), 'SZ': (31.1, -26.3), 'ZM': (28.3, -15.4), 'NZ': (174.8, -41.3), 'NO': (10.8, 59.9), 'EE': (24.8, 59.4), 'LV': (24.1, 56.9), 'LT': (25.3, 54.7), 'LU': (6.1, 49.6), 'IS': (-21.9, 64.1), 'RS': (20.5, 44.8), 'BA': (18.4, 43.9), 'ME': (19.3, 42.4), 'MK': (21.4, 42.0), 'AL': (19.8, 41.3), 'XK': (21.2, 42.7), 'MD': (28.9, 47.0), 'CY': (33.4, 35.2), 'MT': (14.5, 35.9), 'TR': (32.9, 39.9), 'IL': (34.8, 32.1), 'CA': (-75.7, 45.4), 'MX': (-99.1, 19.4), 'GT': (-90.5, 14.6), 'BZ': (-88.8, 17.3), 'SV': (-89.2, 13.7), 'HN': (-87.2, 14.1), 'NI': (-86.3, 12.1), 'CR': (-84.1, 9.9), 'PA': (-79.5, 9.0), 'AU': (149.1, -35.3), 'JP': (139.7, 35.7)}
_OFMX = b'<?xml version="1.0"?><OFMX-Snapshot version="0.1" origin="test">\n<Ase><AseUid region="LS"><codeType>CTR</codeType><codeId>C1</codeId></AseUid><txtName>BERN CTR</txtName><codeClass>D</codeClass>\n  <codeDistVerUpper>ALT</codeDistVerUpper><valDistVerUpper>5500</valDistVerUpper><uomDistVerUpper>FT</uomDistVerUpper>\n  <codeDistVerLower>HEI</codeDistVerLower><valDistVerLower>0</valDistVerLower><uomDistVerLower>FT</uomDistVerLower></Ase>\n<Abd><AbdUid><AseUid region="LS"><codeType>CTR</codeType><codeId>C1</codeId></AseUid></AbdUid>\n  <Circle><geoLatCen>46.91000000N</geoLatCen><geoLongCen>007.50000000E</geoLongCen><valRadius>5</valRadius><uomRadius>NM</uomRadius></Circle></Abd>\n<Ase><AseUid region="LS"><codeType>R</codeType><codeId>R1</codeId></AseUid><txtName>LS-R 1</txtName>\n  <codeDistVerUpper>STD</codeDistVerUpper><valDistVerUpper>100</valDistVerUpper><uomDistVerUpper>FL</uomDistVerUpper>\n  <codeDistVerLower>HEI</codeDistVerLower><valDistVerLower>0</valDistVerLower><uomDistVerLower>FT</uomDistVerLower></Ase>\n<Abd><AbdUid><AseUid region="LS"><codeType>R</codeType><codeId>R1</codeId></AseUid></AbdUid>\n  <Circle><geoLatCen>46.50000000N</geoLatCen><geoLongCen>008.00000000E</geoLongCen><valRadius>2</valRadius><uomRadius>NM</uomRadius></Circle></Abd>\n</OFMX-Snapshot>'

def _square(lon, lat, d=0.2):
    return {'type': 'Polygon', 'coordinates': [[[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]]}

def _records(cc, at=None):
    lon, lat = at or _SPOT[cc]
    ft = lambda v, datum=1: {'value': v, 'unit': 1, 'referenceDatum': datum}
    return [{'name': f'{cc} CTR', 'type': 4, 'icaoClass': 3, 'geometry': _square(lon, lat), 'lowerLimit': ft(0, 0), 'upperLimit': ft(2500)}, {'name': f'{cc} GLIDING', 'type': 21, 'icaoClass': 8, 'geometry': _square(lon + 0.5, lat), 'lowerLimit': ft(0, 0), 'upperLimit': ft(6500)}]

def _fetch(answers=None, calls=None, tick=None):

    def fetch(cc, key, deadline=None):
        if calls is not None:
            calls.append(cc)
        if tick:
            tick()
        got = answers[cc] if answers and cc in answers else _records(cc)
        if isinstance(got, Exception):
            raise got
        return got
    return fetch

def _bad():
    return RuntimeError('HTTP Error 502: Bad Gateway')

def _build(out, fetch, key='k', regions=(), clock=None):
    import europe_airspace as E
    log = io.StringIO()
    patches = [mock.patch.dict(os.environ, {'OPENAIP_API_KEY': key}), mock.patch.object(E, 'REGIONS', list(regions)), mock.patch.object(E, 'snapshot', lambda folder, suffix, cyc: _OFMX), mock.patch.object(E, 'openaip', fetch)]
    if clock:
        patches.append(mock.patch.object(E, 'time', types.SimpleNamespace(monotonic=clock, sleep=lambda s: None)))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(contextlib.redirect_stdout(log))
        try:
            res = E.build(out)
        except RuntimeError as e:
            res = e
    return (res, log.getvalue())

def _text(path):
    with open(path) as f:
        return f.read()

def _read(out, area):
    paths = (os.path.join(out, 'airspace', f'{area}.json'), os.path.join(out, 'airspace', 'aviation', f'sua-{area}.json'))
    return tuple((json.loads(_text(p)) if os.path.exists(p) else None for p in paths))

def _write(out, area, controlled, sua):
    os.makedirs(os.path.join(out, 'airspace', 'aviation'), exist_ok=True)
    for name, feats in ((f'{area}.json', controlled), (os.path.join('aviation', f'sua-{area}.json'), sua)):
        with open(os.path.join(out, 'airspace', name), 'w') as f:
            json.dump({'type': 'FeatureCollection', 'features': feats, 'source': 'openAIP'}, f)

def _ccs(fc):
    return sorted({f['properties'].get('cc') for f in fc['features']})

class OpenaipAreas(unittest.TestCase):

    def test_each_area_gets_its_countries(self):
        import europe_airspace as E
        want = {'eu': sorted(['DZ', 'ES', 'FR', 'GB', 'IE', 'MA', 'PT', 'TN', 'NO', 'EE', 'LV', 'LT', 'LU', 'IS', 'RS', 'BA', 'ME', 'MK', 'AL', 'XK', 'MD', 'CY', 'MT', 'TR', 'IL']), 'sa': ['AR', 'BO', 'BR', 'CL', 'CO', 'EC', 'PE', 'PY', 'UY', 'VE'], 'af': ['BW', 'LS', 'MZ', 'NA', 'SZ', 'ZA', 'ZM', 'ZW'], 'nz': ['NZ'], 'ca': ['CA'], 'mx': sorted(['MX', 'GT', 'BZ', 'SV', 'HN', 'NI', 'CR', 'PA']), 'au': ['AU'], 'jp': ['JP']}
        self.assertEqual({a: sorted(c) for a, c in E.OPENAIP_AREAS.items()}, want)
        with tempfile.TemporaryDirectory() as d:
            res, log = _build(d, _fetch())
            self.assertEqual(res, {a: (len(c), len(c)) for a, c in want.items()})
            for area, ccs in want.items():
                ctl, sua = _read(d, area)
                self.assertEqual((_ccs(ctl), _ccs(sua)), (ccs, ccs))
                self.assertEqual((ctl['source'], sua['source']), ('open flightmaps + openAIP',) * 2 if area == 'eu' else ('openAIP',) * 2)
                self.assertIn(f'airspace/{area}.json: {len(ccs)} features, ', log)
                self.assertIn(f'airspace/aviation/sua-{area}.json: {len(ccs)} features, ', log)
            br = next((f for f in _read(d, 'sa')[0]['features'] if f['properties']['cc'] == 'BR'))
            lon, lat = br['geometry']['coordinates'][0][0]
            self.assertLess(abs(lon - _SPOT['BR'][0]) + abs(lat - _SPOT['BR'][1]), 1)

    def test_gliding_sectors(self):
        import europe_airspace as E
        sq = _square(-65.0, -31.0)
        items = [{'name': f'T{t}', 'type': t, 'icaoClass': 8, 'geometry': sq} for t in (21, 1, 2, 3, 8, 9, 13, 14, 0)]
        ctl, sua = E.oa_features(items, 'AR')
        self.assertEqual(ctl, [])
        self.assertEqual([f['properties']['t'] for f in sua], ['GLD', 'R', 'D', 'P', 'TRA', 'TSA', 'ATZ', 'MATZ'])
        self.assertEqual(sua[0]['properties'], {'t': 'GLD', 'n': 'T21', 'lo': '?', 'hi': '?', 'time': '', 'ctl': '', 'cc': 'AR'})

    def test_budget_skips_the_countries_left(self):
        import europe_airspace as E
        now, calls = ([0.0], [])

        def tick():
            now[0] += E.OPENAIP_BUDGET / 3
        with tempfile.TemporaryDirectory() as d:
            old_nz = E.oa_features(_records('NZ'), 'NZ')
            _write(d, 'nz', *old_nz)
            res, log = _build(d, _fetch(calls=calls, tick=tick), clock=lambda: now[0])
            self.assertEqual(calls, ['GB', 'IE', 'ES'])
            self.assertIn("openaip PT: skipped, openAIP's 10 min are up; kept 0+0 from the last eu files", log)
            self.assertIn("openaip BR: skipped, openAIP's 10 min are up; kept 0+0 from the last sa files", log)
            self.assertIn("openaip NZ: skipped, openAIP's 10 min are up; kept 1+1 from the last nz files", log)
            self.assertEqual(res, {'eu': (3, 3), 'nz': (1, 1)})
            self.assertEqual(_ccs(_read(d, 'eu')[0]), ['ES', 'GB', 'IE'])
            self.assertEqual(_read(d, 'sa'), (None, None))
            self.assertEqual(_read(d, 'nz')[0]['features'], old_nz[0])

    def test_paging_stops_at_the_deadline(self):
        import europe_airspace as E
        pages = []

        def page(url, key):
            pages.append(url)
            return {'items': _records('FR'), 'totalPages': 3}
        clock = iter([0.0, 20.0]).__next__
        with mock.patch.object(E, 'openaip_page', page), mock.patch.object(E, 'time', types.SimpleNamespace(monotonic=clock, sleep=lambda s: None)):
            with self.assertRaises(TimeoutError):
                E.openaip('FR', 'k', deadline=10.0)
        self.assertEqual(len(pages), 1)
        self.assertIn('country=FR&limit=1000&page=1', pages[0])

    def test_a_failed_country_keeps_its_last_features(self):
        import europe_airspace as E
        with tempfile.TemporaryDirectory() as d:
            last = [E.oa_features(_records(cc, at), cc) for cc, at in (('BR', None), ('CL', (-71.5, -33.0)), ('AR', None))]
            _write(d, 'sa', [f for c, _ in last for f in c], [f for _, s in last for f in s])
            res, log = _build(d, _fetch({'BR': _bad(), 'AR': []}))
            self.assertIn('openaip BR: HTTP Error 502: Bad Gateway; kept 1+1 from the last sa files', log)
            self.assertIn('openaip AR: no airspace records; kept 1+1 from the last sa files', log)
            ctl, sua = _read(d, 'sa')
            self.assertEqual(_ccs(ctl), ['AR', 'BO', 'BR', 'CL', 'CO', 'EC', 'PE', 'PY', 'UY', 'VE'])
            self.assertEqual([f for f in ctl['features'] if f['properties']['cc'] == 'BR'], last[0][0])
            self.assertEqual([f for f in sua['features'] if f['properties']['cc'] == 'AR'], last[2][1])
            cl = [f for f in ctl['features'] if f['properties']['cc'] == 'CL']
            self.assertEqual(cl, E.oa_features(_records('CL'), 'CL')[0])
            self.assertEqual(res['sa'], (10, 10))
        with tempfile.TemporaryDirectory() as d:
            res, log = _build(d, _fetch({'ZM': _bad()}))
            self.assertIn('openaip ZM: HTTP Error 502: Bad Gateway; kept 0+0 from the last af files', log)
            self.assertEqual(_ccs(_read(d, 'af')[0]), ['BW', 'LS', 'MZ', 'NA', 'SZ', 'ZA', 'ZW'])

    def test_europe_files_keep_their_format(self):
        with tempfile.TemporaryDirectory() as d:
            res, _ = _build(d, _fetch(), regions=[(('lsas',), 'ls')])
            import europe_airspace as E
            n = len(E.OPENAIP_AREAS['eu']) + 1
            self.assertEqual(res['eu'], (n, n))
            raw = _text(os.path.join(d, 'airspace', 'eu.json'))
            self.assertNotIn(', ', raw)
            for fc in _read(d, 'eu'):
                self.assertEqual(set(fc), {'type', 'features', 'source'})
                self.assertEqual((fc['type'], fc['source']), ('FeatureCollection', 'open flightmaps + openAIP'))
            ctl, sua = _read(d, 'eu')
            ofm, oa = (ctl['features'][0], ctl['features'][1])
            self.assertEqual((ofm['type'], ofm['geometry']['type']), ('Feature', 'Polygon'))
            self.assertEqual(ofm['properties'], {'c': 'D', 'n': 'BERN CTR', 'lo': 'SFC', 'hi': '5,500 ft'})
            self.assertEqual(oa['properties'], {'c': 'D', 'n': 'GB CTR', 'lo': 'SFC', 'hi': '2,500 ft', 'cc': 'GB'})
            self.assertEqual(sua['features'][0]['properties'], {'t': 'R', 'n': 'LS-R 1', 'lo': 'SFC', 'hi': 'FL100', 'time': '', 'ctl': ''})
            self.assertEqual(set(sua['features'][1]['properties']), {'t', 'n', 'lo', 'hi', 'time', 'ctl', 'cc'})
            for f in ctl['features'] + sua['features']:
                ring = f['geometry']['coordinates'][0]
                self.assertEqual(ring[0], ring[-1])
                self.assertTrue(all((round(v, 4) == v for p in ring for v in p)))

    def test_no_key_leaves_openaip_out(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            res, log = _build(d, _fetch(calls=calls), key='', regions=[(('lsas',), 'ls')])
            self.assertEqual((calls, res), ([], {'eu': (1, 1)}))
            self.assertEqual(_read(d, 'eu')[0]['source'], 'open flightmaps + openAIP')
            res, log = _build(d, _fetch(calls=calls), key='')
            self.assertIsInstance(res, RuntimeError)
            self.assertEqual(len(_read(d, 'eu')[0]['features']), 1)

    def test_new_zealand_runs_east_of_180(self):
        import europe_airspace as E
        chatham = _square(-176.5, -43.9)
        across = {'type': 'Polygon', 'coordinates': [[[179.0, -40.0], [-179.0, -40.0], [-179.0, -38.0], [179.0, -38.0], [179.0, -40.0]]]}
        items = [{'name': n, 'type': 4, 'icaoClass': 3, 'geometry': g} for n, g in (('CHATHAM', chatham), ('ACROSS', across))]
        ctl, _ = E.oa_features(items, 'NZ')
        lons = [[x for x, _ in f['geometry']['coordinates'][0]] for f in ctl]
        self.assertTrue(all((183.0 < x < 184.0 for x in lons[0])))
        self.assertEqual((min(lons[1]), max(lons[1])), (179.0, 181.0))
        ctl, _ = E.oa_features([{'name': 'EASTER ISLAND', 'type': 4, 'icaoClass': 3, 'geometry': _square(-109.4, -27.2)}], 'CL')
        self.assertLess(max((x for x, _ in ctl[0]['geometry']['coordinates'][0])), -109.0)

    def test_check_py_can_import_them_bare(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ('airspace.py', 'europe_airspace.py'):
            tree = ast.parse(_text(os.path.join(here, name)))
            mods = {a.name.split('.')[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
            mods |= {n.module.split('.')[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
            self.assertEqual(mods - set(sys.stdlib_module_names) - {'airspace'}, set(), name)
if __name__ == '__main__':
    unittest.main()

class ControlledKinds(unittest.TestCase):

    def test_other_type_with_class_b_to_d_is_controlled(self):
        import europe_airspace as E
        sq = {'type': 'Polygon', 'coordinates': [[[174.0, -41.0], [175.0, -41.0], [175.0, -40.0], [174.0, -41.0]]]}
        rec = lambda t, c: {'type': t, 'icaoClass': c, 'name': f'{t}/{c}', 'geometry': sq, 'lowerLimit': {'value': 0, 'unit': 1, 'referenceDatum': 0}, 'upperLimit': {'value': 95, 'unit': 6, 'referenceDatum': 2}}
        ctl, _ = E.oa_features([rec(0, 2), rec(0, 3), rec(0, 1), rec(0, 0), rec(0, 8), rec(4, 3)], 'NZ')
        self.assertEqual(sorted((f['properties']['c'] for f in ctl)), ['B', 'C', 'D', 'D'])

    def test_transponder_and_radio_mandatory_zones(self):
        import europe_airspace as E
        sq = {'type': 'Polygon', 'coordinates': [[[-1.0, 51.0], [-0.5, 51.0], [-0.5, 51.5], [-1.0, 51.0]]]}
        rec = lambda t, c, n: {'type': t, 'icaoClass': c, 'name': n, 'geometry': sq, 'lowerLimit': {'value': 0, 'unit': 1, 'referenceDatum': 0}, 'upperLimit': {'value': 65, 'unit': 6, 'referenceDatum': 2}}
        ctl, sua = E.oa_features([rec(5, 8, 'STANSTED TMZ'), rec(6, 6, 'RMZ LFXX'), rec(4, 3, 'CTR')], 'GB')
        self.assertEqual([(f['properties']['c'], f['properties']['n'], f['properties']['cc']) for f in ctl], [('TMZ', 'STANSTED TMZ', 'GB'), ('RMZ', 'RMZ LFXX', 'GB'), ('D', 'CTR', 'GB')])
        self.assertEqual(sua, [])
        ase = lambda kind, ident, name, cls='': f'<Ase><AseUid region="ED"><codeType>{kind}</codeType><codeId>{ident}</codeId></AseUid><txtName>{name}</txtName><codeClass>{cls}</codeClass>\n  <codeDistVerUpper>STD</codeDistVerUpper><valDistVerUpper>65</valDistVerUpper><uomDistVerUpper>FL</uomDistVerUpper>\n  <codeDistVerLower>HEI</codeDistVerLower><valDistVerLower>0</valDistVerLower><uomDistVerLower>FT</uomDistVerLower></Ase>\n<Abd><AbdUid><AseUid region="ED"><codeType>{kind}</codeType><codeId>{ident}</codeId></AseUid></AbdUid>\n  <Circle><geoLatCen>52.{ident[-1]}0000000N</geoLatCen><geoLongCen>009.50000000E</geoLongCen><valRadius>5</valRadius><uomRadius>NM</uomRadius></Circle></Abd>'
        xml = ('<OFMX-Snapshot>' + ase('TMZ', 'Z1', 'TMZ HANNOVER') + ase('CLASS', 'Z2', 'RMZ EDVK', 'E') + ase('RAS', 'Z3', 'TMZ-RMZ EDXY') + ase('CTR', 'Z4', 'BREMEN CTR RMZ', '') + ase('TMA', 'Z5', 'TMA HAMBURG TMZ', 'C') + ase('R', 'Z6', 'ED-R TMZ') + '</OFMX-Snapshot>').encode()
        from collections import Counter
        kinds = Counter()
        ctl, sua = E.parse(xml, kinds)
        self.assertEqual(sorted(((f['properties']['c'], f['properties']['n']) for f in ctl)), [('C', 'TMA HAMBURG TMZ'), ('D', 'BREMEN CTR RMZ'), ('RMZ', 'RMZ EDVK'), ('TMZ', 'TMZ HANNOVER'), ('TMZ', 'TMZ-RMZ EDXY')])
        self.assertEqual([f['properties']['t'] for f in sua], ['R'])
        self.assertEqual(kinds, Counter({'TMZ': 1, 'CLASS': 1, 'RAS': 1, 'CTR': 1, 'TMA': 1, 'R': 1}))

    def test_faa_features_out_in_the_western_pacific_dropped(self):
        from airspace import west_pacific
        box = lambda w, e: {'type': 'Polygon', 'coordinates': [[[w, 13.0], [e, 13.0], [e, 14.0], [w, 13.0]]]}
        self.assertTrue(west_pacific(box(144.5, 145.0)))
        self.assertFalse(west_pacific(box(-120.0, -119.0)))
        self.assertFalse(west_pacific(box(173.0, 174.0)))
        self.assertFalse(west_pacific({'type': 'MultiPolygon', 'coordinates': [[[[178.0, 52.0], [-179.0, 52.0], [-179.0, 53.0], [178.0, 52.0]]]]}))
        self.assertFalse(west_pacific(None))
