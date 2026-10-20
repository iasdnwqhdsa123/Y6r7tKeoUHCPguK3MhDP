import json
import os
from aviation import BORDER_REGION, NATURAL_EARTH, REGIONS, get, simplify_line
SOURCES = (('s', 'ne_10m_admin_1_states_provinces_lines.geojson'), ('c', 'ne_10m_admin_0_boundary_lines_land.geojson'))
OUT = os.path.join(os.path.dirname(__file__), '..', 'public', 'geo', 'borders.json')

def main(tol: float=0.004, path: str=OUT) -> int:
    boxes = {'s': [BORDER_REGION], 'c': list(REGIONS)}
    out = {}
    for kind, name in SOURCES:
        data = json.loads(get(f'{NATURAL_EARTH}/{name}', tries=3, timeout=300))
        out[kind] = []
        for f in data.get('features', []):
            g = f.get('geometry') or {}
            lines = g.get('coordinates', []) if g.get('type') == 'MultiLineString' else [g.get('coordinates', [])] if g.get('type') == 'LineString' else []
            for line in lines:
                pts = [(x, y) for x, y, *_ in line]
                if not pts or not any((w <= x <= e and s <= y <= n for x, y in pts for w, s, e, n in boxes[kind])):
                    continue
                pts = simplify_line(pts, tol)
                if len(pts) >= 2:
                    out[kind].append([v for x, y in pts for v in (round(x, 3), round(y, 3))])
    if len(out['s']) + len(out['c']) < 50:
        raise RuntimeError('too few border lines')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'borders: {len(out['s'])} state/province + {len(out['c'])} country lines, {os.path.getsize(path) / 1000.0:.0f} kB → {path}')
    return len(out['s']) + len(out['c'])
if __name__ == '__main__':
    main()
