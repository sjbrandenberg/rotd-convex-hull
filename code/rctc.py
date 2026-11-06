"""A Python port of the parts of the RCTC R package (Wang et al., MIT
license, https://github.com/wltcwpf/RCTC, version 0.2.4) that
compute_metrics.py uses: PS_cal_cpp() and RotD_cal_PSA(), with the
functions they call (GM_RotD_cal(), Interpft(), and subset_select()), and
rotd_percentiles(), which isn't in RCTC.

Each function follows the R or C++ original, so the results match RCTC's to
rounding, with three changes:

- Speed: the oscillator recursion in PS_cal_cpp() runs as an IIR filter
  (scipy.signal.lfilter) rather than a loop.
- Exact peaks at each angle: GM_RotD_cal() takes fraction=None, which
  rotates only the vertices of the convex hull of the two components'
  response orbit (and its reflection). The peak at any angle is the hull's
  extent in that direction, so this gives the same peaks as rotating every
  point. RCTC's default, fraction=0.7, rotates only the points whose
  amplitude is at least 0.7 times the smaller component's peak. That can
  leave out the point that sets an angle's peak, and makes RotD00 too low
  (by up to 15% in tests). Pass fraction=0.7 to reproduce RCTC.
- Exact RotDxx over all angles: by default (method='exact'), RotD_cal_PSA()
  takes each percentile over the continuous range of angles from 0 to 180
  degrees, not over RCTC's 180 one-degree angles (rotd_percentiles()).
  Between changes of hull vertex the peak is a cosine of the angle, so the
  fraction of angles below any level can be added up exactly, and RotDxx
  is the level where it equals xx%, found by bisection. RotD0 (the nearest
  hull edge) and RotD100 (the farthest vertex) are exact outright. With
  one-degree angles, RotD0 of a strongly polarized response, whose minimum
  is a sharp V, can be off by a third, and RotD50 by up to 0.2%. Pass
  method='angles' for RCTC's one-degree angles.

Accelerations are in g. Periods are in s.
"""

import numpy as np
from scipy.signal import lfilter, lfiltic
from scipy.spatial import ConvexHull, QhullError

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


def _angle_bound(x, y, norms):
    """A lower bound on the peak at each of the 180 one-degree angles: the
    smallest over those angles of the peak of the 256 points farthest from
    the origin, rotated. A point closer to the origin than it can't be the
    peak at any of those angles. Zero for 512 points or fewer, which aren't
    worth pruning."""
    if len(x) <= 512:
        return 0.0
    far = np.argpartition(norms, -256)[-256:]
    theta = np.radians(np.arange(0, 180))
    return np.min(np.max(np.abs(np.outer(np.cos(theta), x[far]) + np.outer(np.sin(theta), y[far])), axis=1))


def hull_select(data1, data2):
    """Points of two series at which every rotated series' peak absolute
    value is reached, for the 180 one-degree rotations: the vertices of the
    convex hull of their orbit and its reflection through the origin. A
    rotated series' peak is the hull's extent in that direction."""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    norms = np.hypot(data1, data2)
    keep = norms >= _angle_bound(data1, data2, norms)
    points = np.column_stack([data1[keep], data2[keep]])
    try:
        vertices = ConvexHull(np.vstack([points, -points])).vertices
    except (QhullError, ValueError):
        # Degenerate orbits (all zero, or on a line) have no 2-D hull.
        return points[:, 0], points[:, 1]
    # A reflected vertex stands for its original point: the peaks are of
    # absolute values.
    vertices = np.unique(vertices % len(points))
    return points[vertices, 0], points[vertices, 1]


def _hull_cones(data1, data2):
    """The vertices of the convex hull of the points (data1, data2) and their
    reflections through the origin, as distance r and direction phi, and the
    range of directions [a, b] over which each vertex sets the peak: there the
    peak of the two series rotated to angle theta is r cos(theta - phi).

    Points that can't be a vertex are dropped first. _angle_bound() bounds
    the peak only at one-degree angles, and the least peak can lie between
    them, far below it for a strongly polarized orbit. The hull of the points
    kept gives a bound over all directions, its least peak, which is no more
    than the whole hull's: the points dropped that are farther from the
    origin than that are added to the hull's vertices, and the hull found
    again from those."""
    # Only the points kept are stacked as rows (x, y): the series are long.
    norms = np.hypot(data1, data2)
    bound = _angle_bound(data1, data2, norms)
    keep = norms >= bound
    vertices = _hull_vertices(np.column_stack([data1[keep], data2[keep]]))
    cones = _vertex_cones(vertices)
    least = _least_peak(*cones)
    if least < bound:
        missed = (norms >= least) & ~keep
        if np.any(missed):
            cones = _vertex_cones(_hull_vertices(np.vstack([vertices, np.column_stack([data1[missed], data2[missed]])])))
    return cones


def _least_peak(r, phi, a, b):
    """The least peak over all directions: at the end of each vertex's range
    farther from its direction."""
    return float(np.min(r * np.cos(np.maximum(np.abs(a - phi), np.abs(b - phi)))))


def _hull_vertices(points):
    """The vertices, counterclockwise, of the convex hull of points (rows
    (x, y)) and their reflections through the origin."""
    both = np.vstack([points, -points])
    try:
        # In 2-D, Qhull lists the vertices counterclockwise.
        return both[ConvexHull(both).vertices]
    except (QhullError, ValueError):
        # On a line through the origin (or all zero): the farthest point and
        # its reflection, each setting the peak over half the circle.
        far = points[np.argmax(np.hypot(points[:, 0], points[:, 1]))]
        return np.array([far, -far])


def _vertex_cones(vertices):
    """_hull_cones() of a hull's vertices, counterclockwise (_hull_vertices())."""
    r = np.hypot(vertices[:, 0], vertices[:, 1])
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])
    if len(vertices) == 2:
        a, b = phi - np.pi / 2, phi + np.pi / 2
    else:
        edges = np.roll(vertices, -1, axis=0) - vertices
        normal = np.arctan2(-edges[:, 0], edges[:, 1])  # outward normal of each counterclockwise edge
        a, b = np.roll(normal, 1), normal
    # Each range runs counterclockwise from a to b. A vertex's own direction
    # needn't lie in its range, but every direction in the range is within
    # 90 degrees of it, so phi is taken as the copy nearest the range's middle.
    b = a + np.mod(b - a, 2 * np.pi)
    phi = phi + 2 * np.pi * np.round((0.5 * (a + b) - phi) / (2 * np.pi))
    return r, phi, a, b


def _zonotope_cones(g1, g2):
    """The vertices of the zonotope sum_i [-g_i, g_i] (the Minkowski sum of
    the segments from -g_i to g_i), as distance r and direction phi, and the
    range of directions [a, b] over which each vertex sets its support
    function, sum_i |g_i . u(theta)|, over half a circle: there that sum is
    r cos(theta - phi).

    With every g_i turned into the upper half plane (which leaves |g_i . u|
    alone) and sorted by direction gamma_i, from theta = -90 degrees, where
    every g_i . u is negative, g_i . u turns positive at gamma_i - 90
    degrees. So the vertex from there to the next turn is -S + 2 (g_1 + ...
    + g_k), with S the sum of all the g_i."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    keep = (g1 != 0) | (g2 != 0)
    g1, g2 = g1[keep], g2[keep]
    flip = (g2 < 0) | ((g2 == 0) & (g1 < 0))
    g1 = np.where(flip, -g1, g1)
    g2 = np.where(flip, -g2, g2)
    gamma = np.arctan2(g2, g1)
    order = np.argsort(gamma, kind='stable')
    g1, g2, gamma = g1[order], g2[order], gamma[order]
    vx = 2 * np.concatenate([[0.0], np.cumsum(g1)]) - np.sum(g1)
    vy = 2 * np.concatenate([[0.0], np.cumsum(g2)]) - np.sum(g2)
    edges = np.concatenate([[0.0], gamma, [np.pi]]) - np.pi / 2
    a, b = edges[:-1], edges[1:]
    nonempty = b > a
    r, a, b = np.hypot(vx, vy)[nonempty], a[nonempty], b[nonempty]
    phi = np.arctan2(vy, vx)[nonempty]
    # As in _hull_cones(): the copy of phi nearest the range's middle.
    phi = phi + 2 * np.pi * np.round((0.5 * (a + b) - phi) / (2 * np.pi))
    return r, phi, a, b


def _fraction_below(levels, r, phi, a, b, span=2 * np.pi):
    """For each level m, the fraction of directions in which the support
    (the peak) is at most m: in each vertex's range, those where
    |theta - phi| >= arccos(m / r). span is the total angle of the ranges."""
    beta = np.arccos(np.clip(np.asarray(levels, dtype=float)[:, None] / r, -1.0, 1.0))
    above = np.clip(np.minimum(b, phi + beta) - np.maximum(a, phi - beta), 0, None)
    return np.sum((b - a) - above, axis=1) / span


def _cone_percentiles(r, phi, a, b, percentiles, span):
    """Percentiles over all directions of a support function given by its
    vertices and their ranges (_hull_cones() or _zonotope_cones())."""
    percentiles = np.asarray(percentiles, dtype=float)
    # Most at the farthest vertex, whose own direction lies in its range.
    least = _least_peak(r, phi, a, b)
    most = float(np.max(r))
    out = np.where(percentiles <= 0, least, most).astype(float)
    inner = (percentiles > 0) & (percentiles < 100)
    if np.any(inner):
        lo = np.full(np.sum(inner), least)
        hi = np.full(np.sum(inner), most)
        target = percentiles[inner] / 100
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = _fraction_below(mid, r, phi, a, b, span) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
            if np.all(hi - lo <= 4 * np.finfo(float).eps * most):
                break
        out[inner] = 0.5 * (lo + hi)
    return out


def rotd_percentiles(data1, data2, percentiles):
    """Percentiles over all angles from 0 to 180 degrees of the peak absolute
    value of data1 cos(theta) + data2 sin(theta): RotDxx for xx in
    percentiles, exact rather than over a set of angles (see the module
    docstring). data1 and data2 are the whole series: _hull_cones() drops
    the points that can't set a peak."""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    if len(data1) == 0 or not np.any(data1) and not np.any(data2):
        return np.zeros(len(percentiles))
    return _cone_percentiles(*_hull_cones(data1, data2), percentiles, 2 * np.pi)


def abs_sum_percentiles(g1, g2, percentiles):
    """Percentiles over all angles from 0 to 180 degrees of
    sum_i |g1_i cos(theta) + g2_i sin(theta)|, exact. That sum, such as a
    trapezoidal CAV with g the weighted samples, is the support function of
    a zonotope (_zonotope_cones()), a cosine between the angles where one
    term changes sign, so the percentiles follow as rotd_percentiles()'s
    do, with one vertex per term."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    if not np.any(g1) and not np.any(g2):
        return np.zeros(len(percentiles))
    return _cone_percentiles(*_zonotope_cones(g1, g2), percentiles, np.pi)


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
            rot_rd1, rot_rd2 = hull_select(rd1, rd2)
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
        rotdxx[:, i] = rotd_percentiles(x, y, percentiles) * scale
    return h1, h2, rotdxx


def RotD_cal_PSA(data1, data2, dt, period=None, damping=0.05, combine_index=50, Interpolation_factor='auto',
                 fraction=None, method='exact'):
    """PGV (cm/s), PGA (g), and PSA (g) of two horizontal components: each
    as recorded (PSA_h1 and PSA_h2) and RotDxx for xx = combine_index, in
    order of period (-1 for PGV, 0 for PGA). combine_index may also be a
    list, which returns a RotDxx array for each, computing the responses
    once.

    method='exact' takes RotDxx over all angles (rotd_percentiles());
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
