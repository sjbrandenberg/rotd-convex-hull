# Exact RotD measures of ground motion by the convex hull

Code for computing the orientation-independent intensity measures RotD0, RotD50, RotD100, and any other RotDnn
(Boore, 2010) exactly over all rotation angles, rather than at a finite set of angles, and the analyses and
figures of the accompanying paper (Brandenberg, in preparation, *Earthquake Spectra*).

The peak of two horizontal components rotated to angle θ is the support function of the convex hull of their
two-component orbit and its reflection through the origin. It is therefore a piecewise cosine of θ, one arc per
hull vertex, so:

- **RotD100** is the distance from the origin to the farthest hull vertex;
- **RotD0** is the distance from the origin to the nearest hull edge;
- **any other percentile** follows from a closed-form expression for the fraction of angles below a level, solved
  by bisection.

The same idea gives exact RotDnn of **Arias intensity** (a closed form; RotD50 is the mean of the two components)
and of **band-limited cumulative absolute velocity** (the support function of a zonotope).

## Code

| File | What it does |
|---|---|
| `code/convex_hull.py` | The exact method: RotDnn of any two series over all rotation angles from the convex hull (`rotd_percentiles()`), and of sums of absolute values such as CAV from a zonotope (`abs_sum_percentiles()`). Needs only numpy and scipy. |
| `code/spectra.py` | Response spectra and RotD of PGA, PGV, and PSA (`RotD_cal_PSA()`, exact by default). For comparison, it also ports the calculations of the RCTC R package (Wang et al., 2017), which doesn't use the convex hull: `method='angles'` gives 180 one-degree angles and `fraction=0.7` RCTC's default subset. |
| `code/intensity.py` | Exact RotDnn of Arias intensity (`rotd_ia()`) and band-limited CAV (`rotd_cav_bl()`). |
| `code/fetch_records.py` | Downloads records from the NGA-gm database API (https://nga-gm.org; Buckreis et al., 2025) into `data/`. Needs an API token from a nga-gm.org account. |
| `code/tabas.py` | The Tabas record (1978 Tabas, Iran; time_series_metadata_id 57394): exact, one-degree, and RCTC-subset RotD of PGA, PGV, and PSA. |
| `code/sample.py` | The same, with Arias intensity and CAV, for the records in `data/sample_ids.txt`. |
| `code/timing.py` | Times the rotation step of the exact method against rotating every time step at 1-degree and 0.1-degree increments (single thread). |
| `code/fig_*.py` | The paper's figures, written to `figures/`. |

## Reproducing the results

```bash
git clone https://github.com/sjbrandenberg/rotd-convex-hull.git
cd rotd-convex-hull
pip install numpy scipy matplotlib
python code/fetch_records.py --token-file ~/token.txt 57394 74
python code/fetch_records.py --token-file ~/token.txt --ids-file data/sample_ids.txt
cd code
python tabas.py
python sample.py --workers 8
python timing.py --records 50 --seed 2 --no-tenth   # exact vs 1 degree
python timing.py --records 10                       # also 0.1 degree (slow)
python fig_hull.py && python fig_spectra.py && python fig_pacoima.py && python fig_sample.py
```

## Using the exact method

```python
import numpy as np
import convex_hull
import spectra

# h1, h2: two horizontal accelerations (g) at time step dt (s)
out = spectra.RotD_cal_PSA(h1, h2, dt, combine_index=[0, 50, 100])  # periods -1 (PGV, cm/s), 0 (PGA), and PSA (g)
rotd0, rotd50, rotd100 = out['PSA_RotDxx']

# Any two series, e.g. oscillator responses: exact percentiles of the peak over all angles
convex_hull.rotd_percentiles(x, y, [0, 50, 100])
```

## References

- Boore, D. M. (2010). Orientation-independent, nongeometric-mean measures of seismic intensity from two horizontal
  components of motion. *Bulletin of the Seismological Society of America*, 100(4), 1830–1835.
- Buckreis, T. E., et al. (2025). A global application programming interface–enabled earthquake ground motion
  relational database for engineering applications. *Earthquake Spectra*. https://doi.org/10.1177/87552930251344978
- Wang, P., Stewart, J. P., Bozorgnia, Y., Boore, D. M., and Kishida, T. (2017). R package for computation of
  earthquake ground-motion response spectra. PEER Report 2017/09.

## License

MIT (see `LICENSE`, which also carries the license of the RCTC package, parts of which `code/spectra.py` ports).
