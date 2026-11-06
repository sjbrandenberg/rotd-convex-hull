"""RotD0, RotD50, and RotD100 of PGA, PGV, PSA (111 periods), Arias intensity, and band-limited CAV for a sample
of records (data/sample_ids.txt), by the exact methods and by 180 one-degree angles, and PSA also by RCTC's
default subset (fraction 0.7). Also hull vertex counts and timing. Writes results/sample.npz.

    python code/sample.py --workers 8
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import intensity
import rctc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCT = [0, 50, 100]
PERIODS = np.array(rctc.DEFAULT_PERIODS)
THETA = np.radians(np.arange(0, 180))


def by_angles(values):
    """R type-7 percentiles (RCTC's quantile) of values at the 180 one-degree angles."""
    return np.quantile(values, np.array(PCT) / 100)


def one(tsm):
    d = np.load(os.path.join(ROOT, 'data', f'{tsm}.npz'))
    h1, h2, dt = d['h1'], d['h2'], float(d['dt'])
    n = min(len(h1), len(h2))
    if n < 100:
        return None
    h1, h2 = h1[:n], h2[:n]
    t0 = time.perf_counter()
    exact = np.array(rctc.RotD_cal_PSA(h1, h2, dt, PERIODS, combine_index=PCT)['PSA_RotDxx'])
    t_exact = time.perf_counter() - t0
    t0 = time.perf_counter()
    angles = np.array(rctc.RotD_cal_PSA(h1, h2, dt, PERIODS, combine_index=PCT, method='angles')['PSA_RotDxx'])
    t_angles = time.perf_counter() - t0
    subset = np.array(rctc.RotD_cal_PSA(h1, h2, dt, PERIODS, combine_index=PCT, fraction=0.7)['PSA_RotDxx'])
    # Hull vertices and the bound on RotD0's one-degree error (tan of the half-angle of the edge nearest the origin).
    f = rctc.interpolation_factor(dt)
    fine1, fine2 = rctc.Interpft(h1, f), rctc.Interpft(h2, f)
    vertices = np.zeros(len(PERIODS), int)
    tan_beta = np.zeros(len(PERIODS))
    for i, p in enumerate(PERIODS):
        if p == -1:
            x, y = rctc._cumtrapz(h1, dt), rctc._cumtrapz(h2, dt)
        elif p == 0:
            x, y = h1, h2
        else:
            x = rctc.relative_displacement(fine1, p, 0.05, dt / f)
            y = rctc.relative_displacement(fine2, p, 0.05, dt / f)
        r, phi, a, b = rctc._hull_cones(x, y)
        vertices[i] = len(r)
        # The least support is at an edge normal; there the two vertices' angular distance from it is beta.
        far = np.maximum(np.abs(a - phi), np.abs(b - phi))
        k = np.argmin(r * np.cos(far))
        tan_beta[i] = np.tan(far[k])
    # Arias intensity and band-limited CAV: exact and at one-degree angles.
    ia_exact = intensity.rotd_ia(h1, h2, dt, PCT)
    ia_angles = by_angles([intensity.arias(h1 * np.cos(t) + h2 * np.sin(t), dt) for t in THETA])
    cav_exact = intensity.rotd_cav_bl(h1, h2, dt, PCT)
    g1, fdt = intensity.band_limited(h1 * intensity.G, dt)
    g2, _ = intensity.band_limited(h2 * intensity.G, dt)
    cav_angles = by_angles(intensity.trapezoid_abs(np.outer(np.cos(THETA), g1) + np.outer(np.sin(THETA), g2), fdt))
    return dict(tsm=tsm, n=n, dt=dt, exact=exact, angles=angles, subset=subset, vertices=vertices,
                tan_beta=tan_beta, ia_exact=ia_exact, ia_angles=ia_angles, cav_exact=cav_exact,
                cav_angles=cav_angles, t_exact=t_exact, t_angles=t_angles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    ids = [int(line) for line in open(os.path.join(ROOT, 'data', 'sample_ids.txt')) if line.strip()]
    ids = [i for i in ids if os.path.exists(os.path.join(ROOT, 'data', f'{i}.npz'))]
    rows = []
    with Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(one, ids, chunksize=4), start=1):
            if row is not None:
                rows.append(row)
            if k % 100 == 0:
                print(k, 'of', len(ids), flush=True)
    rows.sort(key=lambda r: r['tsm'])
    out = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(os.path.join(ROOT, 'results', 'sample.npz'), periods=PERIODS, **out)
    print(len(rows), 'records')


if __name__ == '__main__':
    main()
