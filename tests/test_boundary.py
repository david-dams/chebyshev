import numpy as np
import kwant

from generate_data import boundary_points_by_angular_gap

def test_boundary_graphene():
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(1, norbs=1)
    shape_fun = lambda pos : pos[0] ** 2 + pos[1] ** 2 <  10 ** 2
    syst[lat.shape(shape_fun, (0, 0))] = 0.
    syst[lat.neighbors()] = 1.
    syst.eradicate_dangling()
    fsyst = syst.finalized()

    pos = np.array([site.pos for site in fsyst.sites])
    dist = np.linalg.norm(pos - pos[:, None], axis = -1).round(3)
    nn = np.unique(dist)[1]

    border = pos[np.argwhere((np.abs(dist - nn).round(3) < 1e-5).sum(axis = -1) < 3)[:, 0], :]
    border2, _, _ = boundary_points_by_angular_gap(pos)

    for b in border:
        found = np.any(np.all(np.isclose(border2, b), axis=1))
        assert found, f"{b} not detected"
