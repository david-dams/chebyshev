import kwant
import numpy as np
from numpy.fft import fft
import itertools
import shapely
import matplotlib.pyplot as plt

R_MIN = 2
R_MAX = 200
R_STEPS = 500
N_MIN = 12
N_MAX = 40
MAX_FEATURES = 100

# visual tests
def _plot_boundary(fsyst):
    pos = get_boundary_positions(fsyst)
    plt.scatter(x = pos[:, 0], y =  pos[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    # from shapely import plotting
    # plotting.plot_polygon(polygon)
    # kwant.plot(fsyst)

def _plot_corner_order(pos):
    plt.scatter(x = pos[:, 0], y =  pos[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    for i, p in enumerate(pos):
        ax.annotate(f"{i}", p)
    plt.show()    

def get_shape_fun(r, n):
    if n == np.inf:
        return lambda pos : pos[0] ** 2 + pos[1] ** 2 < r ** 2
    
    points = [(np.cos(2*np.pi * i/n), np.sin(2*np.pi * i/n)) for i in np.arange(n)]
    coords = r * np.array( points + [points[0]]  )     
    polygon = shapely.Polygon(coords)
    return lambda pos : polygon.contains(shapely.Point(pos))

def get_moments(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    moments = spectrum._moments()    
    return moments

def get_boundary_positions(fsyst):
    return np.array(
        [fsyst.pos(i) for i in range(fsyst.graph.num_nodes) if len(fsyst.graph.out_neighbors(i)) == 2]
    )

def get_system(r : float, n : int):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(1, norbs=1)

    syst[lat.shape(get_shape_fun(r, n), (0, 0))] = 0.
    syst[lat.neighbors()] = 1.
    syst.eradicate_dangling()

    fsyst = syst.finalized()

    return fsyst

def _safe_key(r: float, n: int) -> str:
    return f"r_{r:.6f}_n_{n}".replace(".", "p")

def get_grid(n_min = N_MIN, n_max = N_MAX):
    corners = np.append(np.arange(n_min, n_max + 1), np.inf)
    radii = np.linspace(R_MIN, R_MAX, R_STEPS)
    return itertools.product(radii, corners)

def generate_data():
    for r, n in get_grid():
        fsyst = get_system(r, n)

        pos = get_boundary_positions(fsyst)        
        print(f"System with {fsyst.graph.num_nodes} atoms and r, n = {r, n}.")        
        moments = get_moments(fsyst)

        fname = f"{_safe_key(r, n)}.npz"
        np.savez_compressed(
            fname,
            r=np.array(r),
            pos=np.asarray(pos, dtype=np.float64),
            moments=np.asarray(moments),
        )

def extract_features():
    training_name = "training.npz"
    features = []
    moments = []
    names = []

    def pad(signal):
        diff = MAX_FEATURES - signal.shape[0]
        if diff > 0:
            return np.pad(signal, (0, diff))
        return signal[:MAX_FEATURES]
    
    for r, n in get_grid(3, 40):
        fname = f"{_safe_key(r, n)}.npz"

        try:
            data = np.load(fname)        

            # circular order of positions
            pos = data["pos"]
            c = pos.mean(axis=0)
            pos = pos -  c
            # _plot_corner_order(pos)
            ang = np.arctan2(pos[:,1], pos[:,0])
            pos = pos[np.argsort(ang)]
            # _plot_corner_order(pos)            
            
            # ft defined with 1/N as in https://pages.hmc.edu/ruye/e161/lectures/fd/node1.html vs https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details        
            enc = fft(pos[:, 0] + 1j*pos[:, 1], axis = 0) / pos.shape[0]
            
            # rotation invariance => take absolute value
            r = np.abs(enc)

            # position invariance => drop k = 0
            r = r[1:]
            
            # if r.size > MAX_FEATURES => truncate, else: pad
            r = pad(r)
                        
            features.append(r)
            moments.append(data["moments"])
            names.append(fname)            
            
        except FileNotFoundError:
            pass
        
    np.savez_compressed(
        training_name,
        features = features,
        moments = moments,
        names = names        
    )
        
if __name__ == '__main__':
    # np.random.seed(0)
    # generate_data()
    extract_features()
