"""The exact RotD method: the peak absolute value of two series x and y
rotated to angle theta, max |x cos(theta) + y sin(theta)|, is the support
function of the convex hull of the points (x, y) and their reflections
through the origin. Between changes of hull vertex it is a cosine of theta,
r cos(theta - phi), with r and phi the vertex's distance and direction. So
RotD100 is the farthest vertex, RotD0 the nearest hull edge (the least
perpendicular distance from the origin to an edge's line), and any other
RotDxx the level at which the fraction of angles whose peak is below it,
added up exactly over the vertices' ranges, equals xx%, found by bisection
(rotd_percentiles()).

The same holds for a sum of absolute values, sum_i |g_i . u(theta)|, such
as a trapezoidal CAV, with the zonotope sum_i [-g_i, g_i] in place of the
hull (abs_sum_percentiles()).

Needs only numpy and scipy (Qhull).
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


# Points farther from the origin than this many are hulled first, to find
# the points that can't be vertices (hull_cones()).
FAR_POINTS = 1024


def hull_cones(data1, data2):
    """The vertices of the convex hull of the points (data1, data2) and their
    reflections through the origin, as distance r and direction phi, and the
    range of directions [a, b] over which each vertex sets the peak: there the
    peak of the two series rotated to angle theta is r cos(theta - phi).

    Qhull would find the vertices from every point, but most points lie well
    inside the hull, so they are dropped first, which saves most of the time.
    The hull of the FAR_POINTS points farthest from the origin lies inside the
    whole hull, and so does the circle about the origin that touches that
    hull's nearest edge (radius its least peak, _least_peak()). A point
    inside that circle is inside the whole hull, so it isn't a vertex."""
    norms = np.hypot(data1, data2)
    if len(norms) > 2 * FAR_POINTS:
        far = np.argpartition(norms, -FAR_POINTS)[-FAR_POINTS:]
        radius = _least_peak(*_vertex_cones(_hull_vertices(np.column_stack([data1[far], data2[far]]))))
        keep = norms >= radius
        data1, data2 = data1[keep], data2[keep]
    return _vertex_cones(_hull_vertices(np.column_stack([data1, data2])))


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
    """hull_cones() of a hull's vertices, counterclockwise (_hull_vertices())."""
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
    # As in hull_cones(): the copy of phi nearest the range's middle.
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
    vertices and their ranges (hull_cones() or _zonotope_cones())."""
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
    docstring). data1 and data2 are the whole series: hull_cones() drops
    the points that can't set a peak."""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    if len(data1) == 0 or not np.any(data1) and not np.any(data2):
        return np.zeros(len(percentiles))
    return _cone_percentiles(*hull_cones(data1, data2), percentiles, 2 * np.pi)


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
