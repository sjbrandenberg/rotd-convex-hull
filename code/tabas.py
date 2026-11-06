"""Tabas (1978, TBS station; NGA-gm time_series_metadata_id 57394): RotD0, RotD50, and RotD100 of PGA, PGV,
and 5%-damped PSA and SD by three methods (exact convex hull, 180 one-degree angles of every point, and RCTC's
default subset), hull vertex counts, and timing. Writes results/tabas.json."""

import json
import os
import time

import numpy as np

import rctc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCT = [0, 50, 100]


def brute(x, y):
    """Peaks at 180 one-degree angles of every point, and R-type-7 percentiles (RCTC's quantile)."""
    theta = np.radians(np.arange(0, 180))
    peaks = np.array([np.max(np.abs(x * np.cos(t) + y * np.sin(t))) for t in theta])
    return np.quantile(peaks, np.array(PCT) / 100)


def main():
    d = np.load(os.path.join(ROOT, 'data', '57394.npz'))
    h1, h2, dt = d['h1'], d['h2'], float(d['dt'])
    periods = np.array(rctc.DEFAULT_PERIODS)
    t0 = time.perf_counter()
    exact = rctc.RotD_cal_PSA(h1, h2, dt, periods, combine_index=PCT)
    t_exact = time.perf_counter() - t0
    t0 = time.perf_counter()
    angles = rctc.RotD_cal_PSA(h1, h2, dt, periods, combine_index=PCT, method='angles')
    t_angles = time.perf_counter() - t0
    t0 = time.perf_counter()
    subset = rctc.RotD_cal_PSA(h1, h2, dt, periods, combine_index=PCT, fraction=0.7)
    t_subset = time.perf_counter() - t0

    # Brute force over every (interpolated) point at 180 angles, and hull vertex counts.
    factor = rctc.interpolation_factor(dt)
    fine1, fine2 = rctc.Interpft(h1, factor), rctc.Interpft(h2, factor)
    fine_dt = dt / factor
    vertices, brute_vals, n_points = [], [], len(fine1)
    t0 = time.perf_counter()
    for p in periods:
        if p == -1:
            x, y, s = rctc._cumtrapz(h1, dt) * rctc.G_CM, rctc._cumtrapz(h2, dt) * rctc.G_CM, 1.0
        elif p == 0:
            x, y, s = h1, h2, 1.0
        else:
            x = rctc.relative_displacement(fine1, p, 0.05, fine_dt) * rctc.G_CM
            y = rctc.relative_displacement(fine2, p, 0.05, fine_dt) * rctc.G_CM
            s = (2 * np.pi / p) ** 2 / rctc.G_CM
        brute_vals.append(brute(x, y) * s)
        vertices.append(len(rctc._hull_cones(x, y)[0]))
    t_brute = time.perf_counter() - t0

    out = {
        'periods': periods.tolist(),
        'exact': [r.tolist() for r in exact['PSA_RotDxx']],
        'angles': [r.tolist() for r in angles['PSA_RotDxx']],
        'subset': [r.tolist() for r in subset['PSA_RotDxx']],
        'brute': np.array(brute_vals).T.tolist(),
        'h1': exact['PSA_h1'].tolist(), 'h2': exact['PSA_h2'].tolist(),
        'hull_vertices': vertices, 'points': n_points, 'dt': dt, 'interp_factor': factor,
        'seconds': {'exact': t_exact, 'angles_hull': t_angles, 'rctc_subset': t_subset, 'brute_all_points': t_brute},
    }
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'results', 'tabas.json'), 'w'), indent=1)
    ex, an, sb, br = (np.array(out[k]) for k in ('exact', 'angles', 'subset', 'brute'))
    for i, xx in enumerate(PCT):
        print(f'RotD{xx}: 1-deg/exact ratio range {np.min(an[i]/ex[i]):.4f} to {np.max(an[i]/ex[i]):.4f}; '
              f'brute/exact {np.min(br[i]/ex[i]):.4f} to {np.max(br[i]/ex[i]):.4f}; '
              f'RCTC subset/exact {np.min(sb[i]/ex[i]):.4f} to {np.max(sb[i]/ex[i]):.4f}')
    print('hull vertices: min', min(vertices), 'median', int(np.median(vertices)), 'max', max(vertices), 'of', n_points, 'points')
    print('seconds:', {k: round(v, 2) for k, v in out['seconds'].items()})
    i = list(periods).index(0.0); j = list(periods).index(-1.0)
    print('PGA g RotD0/50/100 exact', [round(ex[k][i], 4) for k in range(3)], 'H1/H2', round(out['h1'][i], 4), round(out['h2'][i], 4))
    print('PGV cm/s RotD0/50/100 exact', [round(ex[k][j], 2) for k in range(3)])


if __name__ == '__main__':
    main()
