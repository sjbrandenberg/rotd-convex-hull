"""Fig. 2: Tabas (TBS) 5%-damped PSA: (a) exact RotD0, RotD50, and RotD100 and the as-recorded components;
(b) ratios of values at 1-degree increments and of the RCTC subset to the exact values. Reads results/tabas.json."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8})
r = json.load(open(os.path.join(ROOT, 'results', 'tabas.json')))
per = np.array(r['periods'])
psa = per > 0
T = per[psa]
ex, an, sb = (np.array(r[k])[:, psa] for k in ('exact', 'angles', 'subset'))
h1, h2 = np.array(r['h1'])[psa], np.array(r['h2'])[psa]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.0, 5.2), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})
ax1.loglog(T, h1, color='0.6', lw=0.9, ls='--', label='H1 (074°)')
ax1.loglog(T, h2, color='0.6', lw=0.9, ls=':', label='H2 (344°)')
for row, label, lw in zip(ex, ['RotD0', 'RotD50', 'RotD100'], [1.0, 1.4, 1.0]):
    ax1.loglog(T, row, 'k-', lw=lw)
    ax1.text(T[-1] * 1.08, row[-1], label, va='center', fontsize=7.5)
ax1.set_ylabel('5%-damped PSA (g)')
ax1.legend(frameon=False, fontsize=7.5, loc='lower left')
ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, va='top')

styles = {0: ('k', '-'), 1: ('0.45', '-'), 2: ('0.7', '-')}
for i, label in enumerate(['RotD0', 'RotD50', 'RotD100']):
    ax2.semilogx(T, an[i] / ex[i], color=styles[i][0], lw=1.0, label=f'{label}, 1° increments')
for i, label in enumerate(['RotD0']):
    ax2.semilogx(T, sb[0] / ex[0], color='k', lw=1.0, ls='--', label='RotD0, RCTC subset')
ax2.axhline(1, color='0.8', lw=0.6)
ax2.set_ylim(0.68, 1.06)
ax2.set_xlabel('Period (s)'); ax2.set_ylabel('Ratio to exact value')
ax2.legend(frameon=False, fontsize=7, loc='lower left')
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, va='top')
ax2.set_xlim(0.01, 30)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(ROOT, 'figures', f'fig2_tabas_spectra.{ext}'), dpi=300)
print('RotD0 subset/exact min', (sb[0] / ex[0]).min().round(3), 'at', T[np.argmin(sb[0] / ex[0])])
