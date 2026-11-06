"""Fig. 3: Pacoima Dam (1971 San Fernando; NGA-gm time_series_metadata_id 74), 5%-damped PSA versus rotation angle at
the period where one-degree increments overestimate RotD0 the most: (a) all angles; (b) near the minimum, where two
cosine arcs meet in a V between one-degree samples."""

import os

import matplotlib.pyplot as plt
import numpy as np

import rctc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8})
d = np.load(os.path.join(ROOT, 'data', '74.npz'))
h1, h2, dt = d['h1'], d['h2'], float(d['dt'])
per = np.array(rctc.NGAW2_PERIODS)
ex = rctc.RotD_cal_PSA(h1, h2, dt, per, combine_index=0)['PSA_RotDxx']
an = rctc.RotD_cal_PSA(h1, h2, dt, per, combine_index=0, method='angles')['PSA_RotDxx']
T = per[np.argmax(an / ex)]
f = rctc.interpolation_factor(dt)
scale = (2 * np.pi / T) ** 2
x = rctc.relative_displacement(rctc.Interpft(h1, f), T, 0.05, dt / f) * scale
y = rctc.relative_displacement(rctc.Interpft(h2, f), T, 0.05, dt / f) * scale
r, phi, a, b = rctc._hull_cones(x, y)
vx, vy = r * np.cos(phi), r * np.sin(phi)
grid = np.radians(np.linspace(0, 180, 180001))
D = np.max(np.abs(np.outer(np.cos(grid), vx) + np.outer(np.sin(grid), vy)), axis=1)
deg = np.arange(0, 180)
S = np.max(np.abs(np.outer(np.cos(np.radians(deg)), x) + np.outer(np.sin(np.radians(deg)), y)), axis=1)
rotd = rctc.rotd_percentiles(x, y, [0, 50, 100])
far = np.maximum(np.abs(a - phi), np.abs(b - phi))
tan_beta = np.tan(far[np.argmin(r * np.cos(far))])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
for ax, size in ((ax1, 1.8), (ax2, 3.5)):
    ax.plot(np.degrees(grid), D, 'k-', lw=1.0, label='Exact (convex hull)')
    ax.plot(deg, S, 'o', mfc='white', mec='0.35', ms=size, mew=0.6, label='1° increments')
    ax.axhline(rotd[0], color='0.3', lw=0.6, ls=':')
ax1.set_yscale('log')
ax1.set_xlim(0, 180); ax1.set_xticks(range(0, 181, 30))
ax1.set_xlabel(r'Rotation angle, $\theta$ (°)'); ax1.set_ylabel(f'5%-damped PSA at T = {T:g} s (g)')
ax1.text(0.97, 0.04, '(a)', transform=ax1.transAxes, ha='right')
ax1.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.9))
c = np.degrees(grid[np.argmin(D)])
ax2.set_xlim(c - 3, c + 3)
ax2.set_ylim(rotd[0] * 0.98, rotd[0] * 1.12)
ax2.set_xlabel(r'Rotation angle, $\theta$ (°)')
ax2.text(c + 0.1, rotd[0] * 0.987, f'Exact RotD0 = {rotd[0]:.4f} g', fontsize=7.5)
k = np.argmin(S)
ax2.annotate(f'1° increments: {S[k]:.4f} g\n(+{100 * (S[k] / rotd[0] - 1):.1f}%)', xy=(deg[k], S[k]),
             xytext=(c + 0.6, rotd[0] * 1.065), fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.6))
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, va='top')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(ROOT, 'figures', f'fig3_pacoima.{ext}'), dpi=300)
print(f'T={T} RotD0 exact {rotd[0]:.4f} 1deg {S.min():.4f} ratio {S.min()/rotd[0]:.4f} tan_beta {tan_beta:.2f} '
      f'bound {0.5*np.radians(1)*tan_beta:.4f} min at {c:.2f} deg vertices {len(r)}')
