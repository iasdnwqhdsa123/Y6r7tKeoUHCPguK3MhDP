from __future__ import annotations
import math
import sys
from datetime import datetime, timezone
COF = '\n    2025.0            WMM-2025     11/13/2024\n  1  0  -29351.8       0.0       12.0        0.0\n  1  1   -1410.8    4545.4        9.7      -21.5\n  2  0   -2556.6       0.0      -11.6        0.0\n  2  1    2951.1   -3133.6       -5.2      -27.7\n  2  2    1649.3    -815.1       -8.0      -12.1\n  3  0    1361.0       0.0       -1.3        0.0\n  3  1   -2404.1     -56.6       -4.2        4.0\n  3  2    1243.8     237.5        0.4       -0.3\n  3  3     453.6    -549.5      -15.6       -4.1\n  4  0     895.0       0.0       -1.6        0.0\n  4  1     799.5     278.6       -2.4       -1.1\n  4  2      55.7    -133.9       -6.0        4.1\n  4  3    -281.1     212.0        5.6        1.6\n  4  4      12.1    -375.6       -7.0       -4.4\n  5  0    -233.2       0.0        0.6        0.0\n  5  1     368.9      45.4        1.4       -0.5\n  5  2     187.2     220.2        0.0        2.2\n  5  3    -138.7    -122.9        0.6        0.4\n  5  4    -142.0      43.0        2.2        1.7\n  5  5      20.9     106.1        0.9        1.9\n  6  0      64.4       0.0       -0.2        0.0\n  6  1      63.8     -18.4       -0.4        0.3\n  6  2      76.9      16.8        0.9       -1.6\n  6  3    -115.7      48.8        1.2       -0.4\n  6  4     -40.9     -59.8       -0.9        0.9\n  6  5      14.9      10.9        0.3        0.7\n  6  6     -60.7      72.7        0.9        0.9\n  7  0      79.5       0.0       -0.0        0.0\n  7  1     -77.0     -48.9       -0.1        0.6\n  7  2      -8.8     -14.4       -0.1        0.5\n  7  3      59.3      -1.0        0.5       -0.8\n  7  4      15.8      23.4       -0.1        0.0\n  7  5       2.5      -7.4       -0.8       -1.0\n  7  6     -11.1     -25.1       -0.8        0.6\n  7  7      14.2      -2.3        0.8       -0.2\n  8  0      23.2       0.0       -0.1        0.0\n  8  1      10.8       7.1        0.2       -0.2\n  8  2     -17.5     -12.6        0.0        0.5\n  8  3       2.0      11.4        0.5       -0.4\n  8  4     -21.7      -9.7       -0.1        0.4\n  8  5      16.9      12.7        0.3       -0.5\n  8  6      15.0       0.7        0.2       -0.6\n  8  7     -16.8      -5.2       -0.0        0.3\n  8  8       0.9       3.9        0.2        0.2\n  9  0       4.6       0.0       -0.0        0.0\n  9  1       7.8     -24.8       -0.1       -0.3\n  9  2       3.0      12.2        0.1        0.3\n  9  3      -0.2       8.3        0.3       -0.3\n  9  4      -2.5      -3.3       -0.3        0.3\n  9  5     -13.1      -5.2        0.0        0.2\n  9  6       2.4       7.2        0.3       -0.1\n  9  7       8.6      -0.6       -0.1       -0.2\n  9  8      -8.7       0.8        0.1        0.4\n  9  9     -12.9      10.0       -0.1        0.1\n 10  0      -1.3       0.0        0.1        0.0\n 10  1      -6.4       3.3        0.0        0.0\n 10  2       0.2       0.0        0.1       -0.0\n 10  3       2.0       2.4        0.1       -0.2\n 10  4      -1.0       5.3       -0.0        0.1\n 10  5      -0.6      -9.1       -0.3       -0.1\n 10  6      -0.9       0.4        0.0        0.1\n 10  7       1.5      -4.2       -0.1        0.0\n 10  8       0.9      -3.8       -0.1       -0.1\n 10  9      -2.7       0.9       -0.0        0.2\n 10 10      -3.9      -9.1       -0.0       -0.0\n 11  0       2.9       0.0        0.0        0.0\n 11  1      -1.5       0.0       -0.0       -0.0\n 11  2      -2.5       2.9        0.0        0.1\n 11  3       2.4      -0.6        0.0       -0.0\n 11  4      -0.6       0.2        0.0        0.1\n 11  5      -0.1       0.5       -0.1       -0.0\n 11  6      -0.6      -0.3        0.0       -0.0\n 11  7      -0.1      -1.2       -0.0        0.1\n 11  8       1.1      -1.7       -0.1       -0.0\n 11  9      -1.0      -2.9       -0.1        0.0\n 11 10      -0.2      -1.8       -0.1        0.0\n 11 11       2.6      -2.3       -0.1        0.0\n 12  0      -2.0       0.0        0.0        0.0\n 12  1      -0.2      -1.3        0.0       -0.0\n 12  2       0.3       0.7       -0.0        0.0\n 12  3       1.2       1.0       -0.0       -0.1\n 12  4      -1.3      -1.4       -0.0        0.1\n 12  5       0.6      -0.0       -0.0       -0.0\n 12  6       0.6       0.6        0.1       -0.0\n 12  7       0.5      -0.1       -0.0       -0.0\n 12  8      -0.1       0.8        0.0        0.0\n 12  9      -0.4       0.1        0.0       -0.0\n 12 10      -0.2      -1.0       -0.1       -0.0\n 12 11      -1.3       0.1       -0.0        0.0\n 12 12      -0.7       0.2       -0.1       -0.1\n999999999999999999999999999999999999999999999999\n999999999999999999999999999999999999999999999999\n'
A_REF = 6371.2
WGS84_A, WGS84_F = (6378.137, 1 / 298.257223563)
E2 = WGS84_F * (2 - WGS84_F)
LIFE = 5.0

def _model(cof: str=COF):
    lines = [l.split() for l in cof.strip().splitlines()]
    epoch, name = (float(lines[0][0]), lines[0][1])
    terms = {}
    for p in lines[1:]:
        if p[0].startswith('9999'):
            break
        terms[int(p[0]), int(p[1])] = tuple((float(x) for x in p[2:6]))
    return (epoch, name, terms, max((n for n, _ in terms)))
EPOCH, NAME, TERMS, DEGREE = _model()

def decimal_year(when: datetime | None=None) -> float:
    when = when or datetime.now(timezone.utc)
    start = datetime(when.year, 1, 1, tzinfo=when.tzinfo)
    return when.year + (when - start).total_seconds() / (datetime(when.year + 1, 1, 1, tzinfo=when.tzinfo) - start).total_seconds()

def _latitude(lat: float, alt_km: float):
    phi = math.radians(lat)
    rc = WGS84_A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    p, z = ((rc + alt_km) * math.cos(phi), (rc * (1 - E2) + alt_km) * math.sin(phi))
    r = math.hypot(p, z)
    phic = math.asin(z / r)
    ct, st = (math.sin(phic), max(math.cos(phic), 1e-12))
    P = [[0.0] * (DEGREE + 1) for _ in range(DEGREE + 1)]
    dP = [[0.0] * (DEGREE + 1) for _ in range(DEGREE + 1)]
    P[0][0] = 1.0
    for n in range(1, DEGREE + 1):
        if n == 1:
            P[1][1], dP[1][1] = (st, ct)
        else:
            k = math.sqrt((2 * n - 1) / (2 * n))
            P[n][n] = k * st * P[n - 1][n - 1]
            dP[n][n] = k * (ct * P[n - 1][n - 1] + st * dP[n - 1][n - 1])
        for m in range(n):
            a, b = (math.sqrt(n * n - m * m), math.sqrt((n - 1) ** 2 - m * m))
            p2, d2 = (P[n - 2][m], dP[n - 2][m]) if n >= 2 else (0.0, 0.0)
            P[n][m] = ((2 * n - 1) * ct * P[n - 1][m] - b * p2) / a
            dP[n][m] = ((2 * n - 1) * (ct * dP[n - 1][m] - st * P[n - 1][m]) - b * d2) / a
    return (phi, phic, r, P, dP, st)

def _coefficients(year: float):
    dt = year - EPOCH
    return [(n, m, g + dt * gd, h + dt * hd) for (n, m), (g, h, gd, hd) in sorted(TERMS.items())]

def _declination(at, coefs, lon: float) -> float:
    phi, phic, r, P, dP, st = at
    lam = math.radians(lon)
    x = y = zc = 0.0
    for n, m, g, h in coefs:
        ar = (A_REF / r) ** (n + 2)
        cm, sm = (math.cos(m * lam), math.sin(m * lam))
        x += ar * (g * cm + h * sm) * dP[n][m]
        y += ar * m * (g * sm - h * cm) * P[n][m]
        zc -= ar * (n + 1) * (g * cm + h * sm) * P[n][m]
    y /= st
    north = x * math.cos(phic - phi) - zc * math.sin(phic - phi)
    return math.degrees(math.atan2(y, north))

def declination(lat: float, lon: float, year: float, alt_km: float=0.0) -> float:
    return _declination(_latitude(lat, alt_km), _coefficients(year), lon)

def grid(year: float | None=None, step: float=2.0, south: float=-88.0, north: float=88.0, keep=None) -> dict:
    year = decimal_year() if year is None else year
    if not EPOCH <= year <= EPOCH + LIFE:
        print(f'wmm: {NAME} is valid {EPOCH:g} to {EPOCH + LIFE:g}, asked for {year:.2f}: time for the next model', flush=True)
    nx, ny = (int(round(360 / step)) + 1, int((north - south) / step + 1e-09) + 1)
    coefs = _coefficients(year)
    values = []
    for j in range(ny):
        lat = south + j * step
        at = _latitude(lat, 0.0)
        values.append([round(_declination(at, coefs, -180.0 + i * step), 1) if keep is None or keep(lat, -180.0 + i * step) else None for i in range(nx)])
    return {'west': -180.0, 'south': south, 'step': step, 'nx': nx, 'ny': ny, 'values': values, 'eastPositive': True, 'model': NAME, 'year': round(year, 2)}
if __name__ == '__main__':
    if len(sys.argv) == 3:
        print(f'{declination(float(sys.argv[1]), float(sys.argv[2]), decimal_year()):.2f}')
    else:
        g = grid()
        flat = [v for row in g['values'] for v in row]
        print(f'{g['model']} {g['year']}: {g['nx']} x {g['ny']} every {g['step']}°, {min(flat)} to {max(flat)}')
