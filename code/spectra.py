"""Response spectra and RotD of two horizontal components, with the
calculations of the RCTC R package (Wang et al., 2017; MIT license,
https://github.com/wltcwpf/RCTC, version 0.2.4) for comparison. RCTC
doesn't use the convex hull: its oscillator (PS_cal_cpp()), sinc
interpolation (Interpft()), and rotation through 180 one-degree angles
(GM_RotD_cal(), with RCTC's subset of points, subset_select()) are ported
here so that the exact method can be compared with RCTC's own calculation.
Each follows the R or C++ original, so the results match RCTC's to
rounding, except that the oscillator recursion runs as an IIR filter
(scipy.signal.lfilter) rather than a loop.

RotD_cal_PSA() takes RCTC's arguments and by default (method='exact')
computes RotDxx over the continuous range of angles from 0 to 180 degrees
with the convex hull method of convex_hull.py. method='angles' takes RCTC's
180 one-degree angles, rotating only the hull's vertices (fraction=None),
which gives the same peaks at those angles as rotating every point.
fraction=0.7 reproduces RCTC's default, which rotates only the points whose
amplitude is at least 0.7 times the smaller component's peak. That can
leave out the point that sets an angle's peak, and makes RotD0 too low.

Accelerations are in g. Periods are in s.
"""

import numpy as np
from scipy.signal import lfilter, lfiltic

import convex_hull

# g in cm/s^2, as RCTC, compute_metrics.py (9.81 m/s^2), the site's VT2 and
# DT2 files, and the original processing's R code (velts * 9.81 * 100) use.
G_CM = 981.0

# RotD_cal_PSA()'s default periods: -1 for PGV, 0 for PGA, and the 111
# NGA-West2 periods.
NGAW2_PERIODS = [
    0.010, 0.020, 0.022, 0.025, 0.029, 0.030, 0.032, 0.035, 0.036, 0.040, 0.042,
    0.044, 0.045, 0.046, 0.048, 0.050, 0.055, 0.060, 0.065, 0.067, 0.070, 0.075,
    0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120, 0.130, 0.133, 0.140, 0.150,
    0.160, 0.170, 0.180, 0.190, 0.200, 0.220, 0.240, 0.250, 0.260, 0.280, 0.290,
    0.300, 0.320, 0.340, 0.350, 0.360, 0.380, 0.400, 0.420, 0.440, 0.450, 0.460,
    0.480, 0.500, 0.550, 0.600, 0.650, 0.667, 0.700, 0.750, 0.800, 0.850, 0.900,
    0.950, 1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900,
    2.000, 2.200, 2.400, 2.500, 2.600, 2.800, 3.000, 3.200, 3.400, 3.500, 3.600,
    3.800, 4.000, 4.200, 4.400, 4.600, 4.800, 5.000, 5.500, 6.000, 6.500, 7.000,
    7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 11.000, 12.000, 13.000, 14.000,
    15.000, 20.000,
]
DEFAULT_PERIODS = [-1.0, 0.0] + NGAW2_PERIODS


def _coefficients(period, damping, dt):
    """PS_cal_cpp()'s recursion coefficients for one oscillator."""
    omega_n = 2 * np.pi / period
    omega_d = omega_n * np.sqrt(1 - damping**2)
    k = omega_n**2
    e = np.exp(-damping * omega_n * dt)
    s = np.sin(omega_d * dt)
    c = np.cos(omega_d * dt)
    r = np.sqrt(1 - damping**2)
    A = e * (damping / r * s + c)
    B = e * (1 / omega_d * s)
    C = 1 / k * (2 * damping / omega_n / dt
                 + e * (((1 - 2 * damping**2) / omega_d / dt - damping / r) * s - (1 + 2 * damping / omega_n / dt) * c))
    D = 1 / k * (1 - 2 * damping / omega_n / dt
                 + e * (((2 * damping**2 - 1) / omega_d / dt) * s + 2 * damping / omega_n / dt * c))
    AA = -e * (omega_n / r * s)
    BB = e * (c - damping / r * s)
    CC = 1 / k * (-1 / dt + e * ((omega_n / r + damping / dt / r) * s + 1 / dt * c))
    DD = 1 / k / dt * (1 - e * (damping / r * s + c))
    return omega_n, A, B, C, D, AA, BB, CC, DD


def relative_displacement(data, period, damping, dt):
    """An oscillator's relative displacement history, in the units of data
    times s^2, as PS_cal_cpp() computes it: zero at the first point, then

        x[j] = A x[j-1] + B v[j-1] - C a[j-1] - D a[j]
        v[j] = AA x[j-1] + BB v[j-1] - CC a[j-1] - DD a[j]

    Eliminating v gives a second-order filter, which holds from j = 2.
    """
    a = np.asarray(data, dtype=float)
    n = len(a)
    x = np.zeros(n)
    if n < 2:
        return x
    _, A, B, C, D, AA, BB, CC, DD = _coefficients(period, damping, dt)
    x[1] = -C * a[0] - D * a[1]
    if n > 2:
        b = [-D, -B * DD - C + D * BB, -B * CC + C * BB]
        den = [1.0, -(A + BB), A * BB - B * AA]
        zi = lfiltic(b, den, y=[x[1], x[0]], x=[a[1], a[0]])
        x[2:], _ = lfilter(b, den, a[2:], zi=zi)
    return x


def PS_cal_cpp(data, period_t, damping, time_dt, type_return=1):
    """Response spectra of an acceleration time series (in g).

    With type_return 1, a 4 x len(period_t) array: the peak response
    acceleration (g), PSA (g), PSV (cm/s), and SD (cm). With type_return 2,
    the relative displacement histories (cm), periods by data points.
    """
    data = np.asarray(data, dtype=float)
    if type_return != 1:
        return np.array([relative_displacement(data, period, damping, time_dt) * G_CM for period in period_t])
    result = np.zeros((4, len(period_t)))
    for i, period in enumerate(period_t):
        omega_n = 2 * np.pi / period
        x = relative_displacement(data, period, damping, time_dt)
        # The response acceleration, from the relative velocity, which
        # the displacement filter doesn't return.
        v = np.zeros_like(x)
        if len(x) > 1:
            _, A, B, C, D, AA, BB, CC, DD = _coefficients(period, damping, time_dt)
            # v[j] - BB v[j-1] = AA x[j-1] - CC a[j-1] - DD a[j]
            forcing = AA * x[:-1] - CC * data[:-1] - DD * data[1:]
            v[1:] = lfilter([1.0], [1.0, -BB], forcing)
        response = -2 * damping * omega_n * v[1:] - omega_n**2 * x[1:]
        peak = np.max(np.abs(x)) if len(x) else 0.0
        result[0, i] = np.max(np.abs(response)) if len(response) else 0.0
        result[1, i] = omega_n**2 * peak
        result[2, i] = omega_n * peak * G_CM
        result[3, i] = peak * G_CM
    return result


def Interpft(data, interpolation_factor):
    """Sinc interpolation by FFT zero padding: interpolation_factor (a power
    of two) times the points of data."""
    data = np.asarray(data, dtype=float)
    if interpolation_factor == 1:
        return data
    npts = len(data)
    npw2_exp = int(np.floor(np.log(npts) / np.log(2)))
    npw2 = 2**npw2_exp if 2**npw2_exp == npts else 2 ** (npw2_exp + 1)
    padded = np.concatenate([data, np.zeros(npw2 - npts)])
    m = len(padded)
    nyquist = npw2 // 2  # zero-based index of the Nyquist term
    a = np.fft.fft(padded)
    b = np.concatenate([a[:nyquist + 1], np.zeros(m * interpolation_factor - m, dtype=complex), a[nyquist + 1:]])
    b[nyquist] = b[nyquist] / 2
    b[nyquist + m * interpolation_factor - m] = b[nyquist]
    y = np.real(np.fft.ifft(b) * interpolation_factor)
    return y[:npts * interpolation_factor]


def subset_select(data1, data2, fraction, length):
    """The points of two series whose vector amplitude is at least fraction
    times the smaller of the two peaks, and that threshold."""
    data1 = np.asarray(data1[:length])
    data2 = np.asarray(data2[:length])
    alevel = fraction * min(np.max(np.abs(data1)), np.max(np.abs(data2)))
    keep = np.sqrt(data1 * data1 + data2 * data2) >= alevel
    return data1[keep], data2[keep], alevel


def _cumtrapz(y, dt):
    """pracma::cumtrapz() on evenly spaced points: starts at 0."""
    out = np.zeros(len(y))
    out[1:] = np.cumsum((y[1:] + y[:-1]) * dt / 2)
    return out


def _rotated_peaks(series1, series2):
    """Peak absolute values of the two series rotated through angles 1 to 180
    degrees (index 0 to 179), as GM_RotD_cal() orders them: angle theta in
    1..90 gives series1 cos + series2 sin, and theta + 90 gives
    -series1 sin + series2 cos."""
    theta = np.radians(np.arange(1, 91))
    cos = np.cos(theta)[:, None]
    sin = np.sin(theta)[:, None]
    peaks = np.empty(180)
    peaks[:90] = np.max(np.abs(series1 * cos + series2 * sin), axis=1)
    peaks[90:] = np.max(np.abs(-series1 * sin + series2 * cos), axis=1)
    return peaks


def interpolation_factor(time_dt):
    """GM_RotD_cal()'s automatic sinc interpolation factor: the power of two
    that brings the time step toward 0.001 s, at most 8."""
    return 2 ** max(min(int(np.ceil(np.log(time_dt / 0.001) / np.log(2))), 3), 0)


def GM_RotD_cal(data1, data2, period_t, damping, time_dt, fraction=None, Interpolation_factor='auto'):
    """Peak responses of two horizontal components rotated through 180
    angles: RotD180 (angles by periods, in g), and pga_rot (g), pgv_rot
    (cm/s), and pgd_rot (cm) by angle. The GMRotD outputs aren't ported.

    fraction=None rotates the vertices of the response orbit's convex hull
    (exact); a number rotates RCTC's subset (see subset_select())."""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    vel_1 = _cumtrapz(data1, time_dt) * G_CM
    vel_2 = _cumtrapz(data2, time_dt) * G_CM
    disp_1 = _cumtrapz(vel_1, time_dt)
    disp_2 = _cumtrapz(vel_2, time_dt)
    gm = {
        'pga_rot': _rotated_peaks(data1, data2),
        'pgv_rot': _rotated_peaks(vel_1, vel_2),
        'pgd_rot': _rotated_peaks(disp_1, disp_2),
    }

    if Interpolation_factor == 'auto':
        interp_factor = interpolation_factor(time_dt)
    elif Interpolation_factor < 1 or Interpolation_factor & (Interpolation_factor - 1):
        raise ValueError('The interpolation factor must be a power of 2.')
    else:
        interp_factor = Interpolation_factor
    time_dt = time_dt / interp_factor
    data1 = Interpft(data1, interp_factor)
    data2 = Interpft(data2, interp_factor)
    number = min(len(data1), len(data2))
    data1 = data1[:number]
    data2 = data2[:number]

    rotd180 = np.zeros((180, len(period_t)))
    for i, period in enumerate(period_t):
        if period <= 0:
            continue  # RotD_cal_PSA() reads PGV and PGA from pga_rot and pgv_rot
        omega = 2 * np.pi / period
        rd1 = relative_displacement(data1, period, damping, time_dt) * G_CM
        rd2 = relative_displacement(data2, period, damping, time_dt) * G_CM
        if fraction is None:
            rot_rd1, rot_rd2 = convex_hull.hull_select(rd1, rd2)
        else:
            rot_rd1, rot_rd2, _ = subset_select(rd1, rd2, fraction, number)
        rotd180[:, i] = _rotated_peaks(rot_rd1, rot_rd2) * omega**2 / G_CM
    gm['RotD180'] = rotd180
    return gm


def _rotd_exact(data1, data2, time_dt, period, damping, percentiles, Interpolation_factor='auto'):
    """RotD_cal_PSA()'s values with method='exact': each component's peak,
    and RotDxx over all angles, by period (-1 for PGV, 0 for PGA). PGA and PGV
    are of the series as given, and PSA of the sinc-interpolated series, as
    in GM_RotD_cal()."""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    vel_1 = _cumtrapz(data1, time_dt) * G_CM
    vel_2 = _cumtrapz(data2, time_dt) * G_CM
    if Interpolation_factor == 'auto':
        interp_factor = interpolation_factor(time_dt)
    elif Interpolation_factor < 1 or Interpolation_factor & (Interpolation_factor - 1):
        raise ValueError('The interpolation factor must be a power of 2.')
    else:
        interp_factor = Interpolation_factor
    fine1 = Interpft(data1, interp_factor)
    fine2 = Interpft(data2, interp_factor)
    number = min(len(fine1), len(fine2))
    fine1, fine2 = fine1[:number], fine2[:number]
    fine_dt = time_dt / interp_factor

    h1 = np.zeros(len(period))
    h2 = np.zeros(len(period))
    rotdxx = np.zeros((len(percentiles), len(period)))
    for i, p in enumerate(period):
        if p == -1:
            x, y, scale = vel_1, vel_2, 1.0
        elif p == 0:
            x, y, scale = data1, data2, 1.0
        else:
            x = relative_displacement(fine1, p, damping, fine_dt) * G_CM
            y = relative_displacement(fine2, p, damping, fine_dt) * G_CM
            scale = (2 * np.pi / p)**2 / G_CM
        h1[i] = np.max(np.abs(x)) * scale
        h2[i] = np.max(np.abs(y)) * scale
        rotdxx[:, i] = convex_hull.rotd_percentiles(x, y, percentiles) * scale
    return h1, h2, rotdxx


def RotD_cal_PSA(data1, data2, dt, period=None, damping=0.05, combine_index=50, Interpolation_factor='auto',
                 fraction=None, method='exact'):
    """PGV (cm/s), PGA (g), and PSA (g) of two horizontal components: each
    as recorded (PSA_h1 and PSA_h2) and RotDxx for xx = combine_index, in
    order of period (-1 for PGV, 0 for PGA). combine_index may also be a
    list, which returns a RotDxx array for each, computing the responses
    once.

    method='exact' takes RotDxx over all angles (convex_hull.rotd_percentiles());
    method='angles' over RCTC's 180 one-degree angles, as R's quantile().
    fraction is as in GM_RotD_cal(): None (exact peaks) or RCTC's 0.7, which
    implies method='angles'. Each component's own values are the same
    either way.

    Returns a dict of numpy arrays: period, PSA_h1, PSA_h2, and PSA_RotDxx
    (a list of arrays when combine_index is a list).
    """
    period = np.asarray(DEFAULT_PERIODS if period is None else period, dtype=float)
    indexes = combine_index if isinstance(combine_index, (list, tuple)) else [combine_index]
    if method == 'exact' and fraction is None:
        h1, h2, rotdxx = _rotd_exact(data1, data2, dt, period, damping, indexes, Interpolation_factor)
        return {
            'period': period,
            'PSA_h1': h1,
            'PSA_h2': h2,
            'PSA_RotDxx': list(rotdxx) if isinstance(combine_index, (list, tuple)) else rotdxx[0],
        }
    if method not in ('exact', 'angles'):
        raise ValueError("method must be 'exact' or 'angles'.")
    res = GM_RotD_cal(data1, data2, period, damping, dt, fraction=fraction,
                      Interpolation_factor=Interpolation_factor)
    # By angle, for each period: index 179 is angle 180 (h1), and 89 is
    # angle 90 (h2).
    by_angle = res['RotD180'].copy()
    by_angle[:, period == -1] = res['pgv_rot'][:, None]
    by_angle[:, period == 0] = res['pga_rot'][:, None]
    # R's quantile() default (type 7) is numpy's linear method.
    rotdxx = [np.quantile(by_angle, index / 100, axis=0) for index in indexes]
    return {
        'period': period,
        'PSA_h1': by_angle[179],
        'PSA_h2': by_angle[89],
        'PSA_RotDxx': rotdxx if isinstance(combine_index, (list, tuple)) else rotdxx[0],
    }
