# TODO: generate graphene tb data (gets first 100 chebyshev moments)
# TODO: fit against fourier shape descriptors

import kwant
import numpy as np
import matplotlib.pyplot as plt

N_MAX = 10

# just to check
def _plot_boundary(fsyst):
    pos = boundary_positions(fsyst)
    plt.scatter(x = pos[:, 0], y =  pos[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def circle(r):
    return lambda pos : pos[0] ** 2 + pos[1] ** 2 < r ** 2

# TODO:
def polygon(r, n):
    return < r

def get_shape_fun():
    n = np.random.randint(N_MAX + 1)
    is_large = np.random.randint(2)
    r_min, r_max = [R_MIN_LARGE, R_MAX_LARGE] if is_large else [R_MIN_SMALL, R_MAX_SMALL]
    r = r_min + r_max * np.random.rand()
    return circle(r) if N_MAX == n else polygon(r, n)    

def get_moments(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    moments = spectrum._moments()    
    return moments

def get_boundary_positions(fsyst):
    return np.array(
        [fsyst.pos(i) for i in range(fsyst.graph.num_nodes) if len(fsyst.graph.out_neighbors(i)) == 2]
    )

def get_system():
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(1, norbs=1)

    shape = get_shape_fun()

    syst[lat.shape(shape, (0, 0))] = 0.
    syst[lat.neighbors()] = 1.
    syst.eradicate_dangling()

    fsyst = syst.finalized()

    return fsyst

# TODO: make params unique
def get_data(N):    
    fsyst = get_system()
    pos = get_boundary_positions(fsyst)
    moments = get_moments()
    return {"pos" : pos, "moments" : moments}
    

if __name__ == '__main__':
    np.random.seed(0)
    data = get_data()    
    # fsyst = system().finalized()

# kwant.plot(fsyst)
# _plot_boundary(fsyst)
