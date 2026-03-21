import os
import itertools
import kwant
import numpy as np
from numpy.fft import fft
import shapely
import matplotlib.pyplot as plt

from chebyshev.utils import boundary_points_by_angular_gap

from train import TRAINING_DATA

R_MIN = 2
R_MAX = 200
R_STEPS = 500
N_MIN = 3
N_MAX = 40
MAX_FEATURES = 100

COEFFICIENTS_PATH=Path(__file__).parent.parent / "data" /"coefficients"
W90_FILES_DIR=Path(__file__).parent.parent / "data" /"wannier"

def _plot_boundary(fsyst):
    pos = get_boundary_positions(fsyst)
    plt.scatter(x = pos[:, 0], y =  pos[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

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

def radius_n_from_coefficients_file(fname):
    radius = float(fname.split(".")[0].split("_")[2].replace("p", "."))
    n = float(fname.split(".")[0].split("_")[4].split('p')[0])
    return radius, n
        
def coefficients_file(r: float, n: int, model_id : str) -> str:
    return f"{COEFFICIENTS_PATH}/{model_id}_r_{r:.6f}_n_{n}".replace(".", "p")

def get_grid():
    corners = np.append(np.arange(N_MIN, N_MAX + 1), np.inf)
    radii = np.linspace(R_MIN, R_MAX, R_STEPS)
    return itertools.product(radii, corners)

def get_boundary_positions(fsyst):
    pos = np.array([fsyst.pos(i) for i in range(fsyst.graph.num_nodes)])
    return boundary_points_by_angular_gap(pos)

def get_moments_limits(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    moments = spectrum._moments()    
    return moments, spectrum._a, spectrum._b

def get_systems():
    for w90_path in os.listdir(W90_FILES_DIR):
        parsed = Wannier90ToKwant(
            wout_path=w90_path / "wannier90.wout",
            hr_path=w90_path / "wannier90_hr.dat",
            n=2,
            rel_cutoff=1e-3,
        )
        for r, n in get_grid():
            shape_fun = get_shape_fun(r, n)
            fsyst = parsed.to_kwant_system(shape_fun)            
            yield coefficients_file(r, n, w90_path), fsyst
            
    # phenom tb for graphene
    # syst = kwant.Builder()
    # lat = kwant.lattice.honeycomb(1, norbs=1)
    # for r, n in get_grid():
    #     shape_fun = get_shape_fun(r, n)
    #     syst[lat.shape(shape_fun, (0, 0))] = 0.
    #     syst[lat.neighbors()] = 1.
    #     syst.eradicate_dangling()
    #     fsyst = syst.finalized()            
    #     yield coefficients_file(r, n, "graphene-1eV"), fsyst

def generate_data():
    for fname, fsyst in get_systems():
        pos = get_boundary_positions(fsyst)
        moments, a, b = get_moments_limits(fsyst)        
        print(f"Generated {fname}")
        np.savez_compressed(
            fname,
            pos=np.asarray(pos, dtype=np.float64),
            moments=np.asarray(moments),
            a = a,
            b = b
        )

def extract_features():
    features = []
    moments = []
    names = []
    radii = []
    corners = []
    lower_limits = []
    upper_limits = []

    def pad(signal):
        diff = MAX_FEATURES - signal.shape[0]
        if diff > 0:
            return np.pad(signal, (0, diff))
        return signal[:MAX_FEATURES]

    for fname in filter(lambda x : "npz" in x, os.listdir(COEFFICIENTS_PATH)):
        radius, n = radius_n_from_coefficients_file(fname)
        
        data = np.load(f"{COEFFICIENTS_PATH}/{fname}")        

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
        radii.append(radius)
        corners.append(n)
        lower_limits.append(data["a"])
        upper_limits.append(data["b"])
                    
    np.savez_compressed(
        TRAINING_DATA,
        features = features,
        moments = moments,
        names = names,
        radii = radii,
        corners = corners,
        lower_limits = lower_limits,
        upper_limits = upper_limits,
        model = fname.split("_")[0]
    )
        
if __name__ == '__main__':
    # np.random.seed(0)
    generate_data()
    extract_features()
