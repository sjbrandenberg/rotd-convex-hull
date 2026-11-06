"""Fig. 4: RotD0 errors in the sample of NGA-West3 records (results/sample.npz): (a) one-degree increments versus
RotD100/RotD0, with the bound (dtheta/2) RotD100/RotD0; (b) the RCTC subset versus period: percentiles of the ratio
to the exact value, and the fraction of records more than 5% low."""

import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8})
d = np.load(os.path.join(ROOT, 'results', 'sample.npz'))
per, ex, an, sb = d['periods'], d['exact'], d['angles'], d['subset']
err = an[:, 0] / ex[:, 0] - 1
pol = ex[:, 2] / ex[:, 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
ax1.loglog(pol.ravel(), np.maximum(err.ravel(), 1e-7), '.', color='0.45', ms=1.0, alpha=0.4, rasterized=True)
x = np.logspace(0, np.log10(pol.max() * 1.2), 100)
delta = np.radians(0.5)
ax1.loglog(x, np.cos(delta) + x * np.sin(delta) - 1, 'k-', lw=1.0, label=r'Bound, $(\Delta\theta/2)\,$RotD100/RotD0')
ax1.set_xlim(1, pol.max() * 1.2); ax1.set_ylim(1e-6, 1)
ax1.set_xlabel('RotD100/RotD0'); ax1.set_ylabel('Relative error of RotD0, 1° increments')
ax1.legend(frameon=False, fontsize=7.5, loc='lower right')
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes, va='top')

psa = per > 0
T = per[psa]
r = (sb[:, 0] / ex[:, 0])[:, psa]
for q, style in ((50, '-'), (5, '--'), (1, ':')):
    ax2.semilogx(T, np.percentile(r, q, axis=0), 'k' + style, lw=1.0, label=f'{q}th percentile')
ax2.set_ylim(0.5, 1.05)
ax2.set_xlabel('Period (s)'); ax2.set_ylabel('RotD0, RCTC subset / exact')
ax2.legend(frameon=False, fontsize=7.5, loc='center left', bbox_to_anchor=(0.0, 0.55))
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, va='top')
twin = ax2.twinx()
twin.semilogx(T, 100 * np.mean(r < 0.95, axis=0), color='0.6', lw=1.0)
twin.set_ylim(0, 12); twin.set_ylabel('Records more than 5% low (%)', color='0.45')
twin.tick_params(axis='y', colors='0.45')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(ROOT, 'figures', f'fig4_sample.{ext}'), dpi=300)
i = list(per).index(0.0); j = list(per).index(-1.0)
print('PGA subset<0.95: %.2f%%, PGV: %.2f%%; PSA max fraction %.1f%% at T=%g; median 1st pct %.3f' % (
    100 * np.mean(sb[:, 0, i] / ex[:, 0, i] < 0.95), 100 * np.mean(sb[:, 0, j] / ex[:, 0, j] < 0.95),
    100 * np.max(np.mean(r < 0.95, axis=0)), T[np.argmax(np.mean(r < 0.95, axis=0))], np.median(np.percentile(r, 1, axis=0))))
