"""Times the rotation step of RotD (after the oscillator responses, which every method shares) for a random sample
of records from data/sample_ids.txt, at PGA, PGV, and 111 periods: the exact convex hull method, and conventional
rotation of every time step at 1-degree (180 angles) and 0.1-degree (1,800 angles) increments, each written as a
matrix product followed by the peak absolute value and the percentiles. Single-threaded (set before numpy loads).
Writes results/timing.npz.

    python code/timing.py --records 100
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import convex_hull  # noqa: E402
import spectra  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERIODS = np.array(spectra.DEFAULT_PERIODS)
PCT = np.array([0, 50, 100])


def brute(x, y, step_deg, chunk=4_000_000):
    """RotD0, RotD50, RotD100 from the peaks of every point at angles step_deg apart (chunked to bound memory)."""
    theta = np.radians(np.arange(0, 180, step_deg))
    u = np.vstack([np.cos(theta), np.sin(theta)])
    p = np.column_stack([x, y])
    rows = max(1, chunk // len(theta))
    peaks = np.zeros(len(theta))
    for start in range(0, len(p), rows):
        np.maximum(peaks, np.max(np.abs(p[start:start + rows] @ u), axis=0), out=peaks)
    return np.quantile(peaks, PCT / 100)


def responses(h1, h2, dt):
    """The two components' series for PGV, PGA, and each period's relative displacement."""
    f = spectra.interpolation_factor(dt)
    fine1, fine2 = spectra.Interpft(h1, f), spectra.Interpft(h2, f)
    out = []
    for p in PERIODS:
        if p == -1:
            out.append((spectra._cumtrapz(h1, dt), spectra._cumtrapz(h2, dt)))
        elif p == 0:
            out.append((h1, h2))
        else:
            out.append((spectra.relative_displacement(fine1, p, 0.05, dt / f),
                        spectra.relative_displacement(fine2, p, 0.05, dt / f)))
    return out


def save(rows, name):
    """Writes results/timing.npz from the records done so far."""
    a = np.array(rows)
    np.savez(os.path.join(ROOT, 'results', name), tsm=a[:, 0], n=a[:, 1], t_response=a[:, 2],
             t_exact=a[:, 3], t_deg1=a[:, 4], t_deg01=a[:, 5], err_deg1=a[:, 6], err_deg01=a[:, 7])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-tenth', action='store_true', help='skip 0.1-degree increments (10 times 1 degree)')
    parser.add_argument('--out', default='timing.npz', help='file name in results/')
    args = parser.parse_args()
    ids = [int(line) for line in open(os.path.join(ROOT, 'data', 'sample_ids.txt')) if line.strip()]
    ids = [i for i in ids if os.path.exists(os.path.join(ROOT, 'data', f'{i}.npz'))]
    rng = np.random.default_rng(args.seed)
    rows = []
    for tsm in rng.choice(ids, size=min(args.records, len(ids)), replace=False):
        d = np.load(os.path.join(ROOT, 'data', f'{tsm}.npz'))
        n = min(len(d['h1']), len(d['h2']))
        if n < 100:
            continue
        h1, h2, dt = d['h1'][:n], d['h2'][:n], float(d['dt'])
        t0 = time.perf_counter()
        series = responses(h1, h2, dt)
        t_resp = time.perf_counter() - t0
        times = {}
        results = {}
        methods = [('exact', lambda x, y: convex_hull.rotd_percentiles(x, y, PCT)), ('deg1', lambda x, y: brute(x, y, 1.0))]
        if not args.no_tenth:
            methods.append(('deg01', lambda x, y: brute(x, y, 0.1)))
        for name, fn in methods:
            t0 = time.perf_counter()
            results[name] = np.array([fn(x, y) for x, y in series])
            times[name] = time.perf_counter() - t0
        # Sanity: brute force can't go below the exact RotD0 or above the exact RotD100.
        assert np.all(results['deg1'][:, 0] >= results['exact'][:, 0] * (1 - 1e-9))
        if 'deg01' in results:
            assert np.all(results['deg01'][:, 2] <= results['exact'][:, 2] * (1 + 1e-9))
        else:
            times['deg01'] = np.nan
            results['deg01'] = np.full_like(results['exact'], np.nan)
        n_fine = len(series[-1][0])
        rows.append((tsm, n_fine, t_resp, times['exact'], times['deg1'], times['deg01'],
                     np.max(results['deg1'][:, 0] / results['exact'][:, 0]),
                     np.max(results['deg01'][:, 0] / results['exact'][:, 0])))
        print(f'{len(rows)}: tsm {tsm} n {n_fine} response {t_resp:.2f} s, rotation: exact {times["exact"]:.2f} s, '
              f'1 deg {times["deg1"]:.2f} s, 0.1 deg {times["deg01"]:.2f} s', flush=True)
        save(rows, args.out)
    a = np.array(rows)
    print('totals (s): response %.1f, exact %.1f, 1 deg %.1f, 0.1 deg %.1f' % tuple(np.nansum(a[:, 2:6], axis=0)))
    print('median ratio to exact: 1 deg %.1f; median n %d' % (np.median(a[:, 4] / a[:, 3]), np.median(a[:, 1])))


if __name__ == '__main__':
    main()
