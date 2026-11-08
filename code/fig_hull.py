"""Fig. 1: the convex hull method for Tabas (TBS) at T = 1.71 s. (a) Response orbit, its reflection, and their
convex hull, with the support line at one rotation angle; (b) SD versus rotation angle: the exact curve, one
cosine arc per hull vertex, the 180 one-degree samples, and the exact RotD0, RotD50, and RotD100."""

import os

import matplotlib.pyplot as plt
import numpy as np

import convex_hull
import spectra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8})

d = np.load(os.path.join(ROOT, 'data', '57394.npz'))
h1, h2, dt = d['h1'], d['h2'], float(d['dt'])
T = 1.71
f = spectra.interpolation_factor(dt)
x = spectra.relative_displacement(spectra.Interpft(h1, f), T, 0.05, dt / f) * spectra.G_CM
y = spectra.relative_displacement(spectra.Interpft(h2, f), T, 0.05, dt / f) * spectra.G_CM
r, phi, a, b = convex_hull.hull_cones(x, y)
vx, vy = r * np.cos(phi), r * np.sin(phi)
order = np.argsort(np.mod(a, 2 * np.pi))
rotd = convex_hull.rotd_percentiles(x, y, [0, 50, 100])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={'width_ratios': [1, 1.35]})
ax1.plot(-x, -y, color='0.82', lw=0.5)
ax1.plot(x, y, color='0.45', lw=0.5)
hx, hy = np.append(vx[order], vx[order][0]), np.append(vy[order], vy[order][0])
ax1.plot(hx, hy, 'k-', lw=1.0)
ax1.plot(vx, vy, 'ko', ms=1.6)
theta = np.radians(60)
u = np.array([np.cos(theta), np.sin(theta)])
support = np.max(np.abs(vx * u[0] + vy * u[1]))
ax1.annotate('', xy=support * u, xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=0.9))
tangent = np.array([-u[1], u[0]])
p0 = support * u
ax1.plot(*np.column_stack([p0 - 25 * tangent, p0 + 25 * tangent]), 'k--', lw=0.8)
ax1.text(*(0.6 * support * u + np.array([-24, 4])), r'$D(\theta)$', fontsize=9)
ax1.set_aspect('equal')
lim = 1.15 * np.max(r)
ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
ax1.set_xlabel('$u_1$ (cm)'); ax1.set_ylabel('$u_2$ (cm)')
ax1.axhline(0, color='0.7', lw=0.5); ax1.axvline(0, color='0.7', lw=0.5)
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes, va='top')

grid = np.radians(np.linspace(0, 180, 36001))
D = np.max(np.abs(np.outer(np.cos(grid), vx) + np.outer(np.sin(grid), vy)), axis=1)
deg = np.arange(0, 180)
samples = np.max(np.abs(np.outer(np.cos(np.radians(deg)), x) + np.outer(np.sin(np.radians(deg)), y)), axis=1)


def draw(ax, size):
    ax.plot(np.degrees(grid), D, 'k-', lw=1.0, label='Exact (convex hull)')
    ax.plot(deg, samples, 'o', mfc='white', mec='0.4', ms=size, mew=0.5, label='1° increments')


draw(ax2, 2.0)
for value, label in zip(rotd, ['RotD0', 'RotD50', 'RotD100']):
    ax2.axhline(value, color='0.3', lw=0.6, ls=':')
    ax2.text(181, value, f'{label} = {value:.1f}', va='center', fontsize=7.5)
ax2.set_xlim(0, 180); ax2.set_xticks(range(0, 181, 30))
ax2.set_ylim(16, 42)
ax2.set_xlabel(r'Rotation angle, $\theta$ (°)'); ax2.set_ylabel('5%-damped SD (cm)')
ax2.text(0.02, 0.04, '(b)', transform=ax2.transAxes, va='bottom')
ax2.legend(loc='upper center', frameon=False, fontsize=7.5, ncol=2)
fig.tight_layout()
fig.subplots_adjust(right=0.84)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(ROOT, 'figures', f'fig1_hull.{ext}'), dpi=300)
print('vertices', len(r), 'RotD', rotd.round(2))
