import numpy as np
from scipy.spatial import cKDTree

def estimate_nn_spacing(points, k=2):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    return np.median(dists[:, 1])

def boundary_points_by_angular_gap(points, r=None, gap_threshold=np.pi):
    """
    Identify boundary points by missing angular coverage.

    Parameters
    ----------
    points : (N,2) array
    r : float or None
        Neighbor cutoff. If None, uses 1.3 * median nearest-neighbor spacing.
    gap_threshold : float
        Largest angular gap above this => boundary.
        pi is a good default.

    Returns
    -------
    boundary_idx : array
    max_gaps : array
    """
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)

    if r is None:
        a = estimate_nn_spacing(points)
        r = 1.3 * a

    neighbors = tree.query_ball_point(points, r=r)

    max_gaps = np.zeros(len(points))

    for i, nbrs in enumerate(neighbors):
        nbrs = [j for j in nbrs if j != i]

        if len(nbrs) < 2:
            max_gaps[i] = 2 * np.pi
            continue

        vecs = points[nbrs] - points[i]
        angles = np.sort(np.arctan2(vecs[:, 1], vecs[:, 0]))

        diffs = np.diff(angles)
        wrap = (angles[0] + 2 * np.pi) - angles[-1]
        gaps = np.concatenate([diffs, [wrap]])

        max_gaps[i] = np.max(gaps)

    boundary_idx = np.where(max_gaps > gap_threshold)[0]
    return points[boundary_idx], max_gaps, r
