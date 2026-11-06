"""Exact RotDxx of Arias intensity and band-limited CAV, from nga-gm's
bin/compute_metrics.py (commit 1263221), with the helpers they need.

Accelerations are in g here (as in rctc.py); Arias intensity and CAV are
returned in m/s.
"""

import numpy as np

import rctc

G = rctc.G_CM / 100  # m/s^2


def band_limited(acc, dt):
    """acc sinc interpolated as rctc.RotD_cal_PSA() interpolates it, and its time step."""
    factor = rctc.interpolation_factor(dt)
    return rctc.Interpft(acc, factor), dt / factor


def trapezoid_abs(acc, dt):
    """The trapezoidal sum of |acc|; acc may be 2-D, one series per row."""
    acc = np.abs(np.asarray(acc, dtype=float))
    return dt * (np.sum(acc, axis=-1) - 0.5 * (acc[..., 0] + acc[..., -1]))


def arias(acc_g, dt):
    """Arias intensity (m/s) of acc in g: pi / (2 g) times the integral of acc^2."""
    a = np.asarray(acc_g, dtype=float) * G
    return np.pi / (2 * G) * dt * np.sum(a * a, axis=-1)


def rotd_ia(h1_g, h2_g, dt, percentiles):
    """Exact RotDxx of Arias intensity (m/s) over all angles: rotated to theta, IA is
    k (A cos^2 + B sin^2 + 2 C sin cos), its mean plus a sinusoid in 2 theta of
    amplitude R, whose xx-th percentile is mean - R cos(pi xx / 100)."""
    h1, h2 = np.asarray(h1_g, float) * G, np.asarray(h2_g, float) * G
    k = np.pi * dt / (2 * G)
    a, b, c = k * np.sum(h1 * h1), k * np.sum(h2 * h2), k * np.sum(h1 * h2)
    mean, amplitude = 0.5 * (a + b), np.hypot(0.5 * (a - b), c)
    return mean - amplitude * np.cos(np.pi * np.asarray(percentiles, dtype=float) / 100)


def cav_bl(acc_g, dt):
    """Band-limited CAV (m/s): trapezoidal sum of |acc| of the sinc-interpolated acceleration."""
    fine, fine_dt = band_limited(np.asarray(acc_g, float) * G, dt)
    return trapezoid_abs(fine, fine_dt)


def rotd_cav_bl(h1_g, h2_g, dt, percentiles):
    """Exact RotDxx of band-limited CAV (m/s): the support function of a zonotope."""
    fine1, fine_dt = band_limited(np.asarray(h1_g, float) * G, dt)
    fine2, _ = band_limited(np.asarray(h2_g, float) * G, dt)
    weights = np.full(len(fine1), fine_dt)
    weights[0] = weights[-1] = fine_dt / 2
    return rctc.abs_sum_percentiles(weights * fine1, weights * fine2, percentiles)
