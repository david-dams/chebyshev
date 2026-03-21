# TODO: generate graphene tb data (gets first 100 chebyshev moments)
# TODO: fit against fourier shape descriptors

import kwant
import numpy as np
import matplotlib.pyplot as plt

R_MIN = 2
R_MAX = 200
R_STEPS = 500

# just to check
def _plot_boundary(fsyst):
    pos = boundary_positions(fsyst)
    plt.scatter(x = pos[:, 0], y =  pos[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def circle(r):
    return lambda pos : pos[0] ** 2 + pos[1] ** 2 < r ** 2

def get_moments(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    fsyst
    moments = spectrum._moments()    
    return moments

def get_boundary_positions(fsyst):
    return np.array(
        [fsyst.pos(i) for i in range(fsyst.graph.num_nodes) if len(fsyst.graph.out_neighbors(i)) == 2]
    )

def get_system(r):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(1, norbs=1)

    shape = circle(r)

    syst[lat.shape(shape, (0, 0))] = 0.
    syst[lat.neighbors()] = 1.
    syst.eradicate_dangling()

    fsyst = syst.finalized()

    return fsyst

def _safe_key(r: float) -> str:
    return f"r_{r:.6f}".replace(".", "p")

def generate_data():
    for r in np.linspace(R_MIN, R_MAX, R_STEPS):
        fsyst = get_system(r)
        pos = get_boundary_positions(fsyst)        
        print(f"System with {fsyst.graph.num_nodes} atoms.")        
        moments = get_moments(fsyst)

        fname = f"{_safe_key(r)}.npz"
        np.savez_compressed(
            fname,
            r=np.array(r),
            pos=np.asarray(pos, dtype=np.float64),
            moments=np.asarray(moments),
        )

        
# ffts boundary, returns 2D array with rows corresponding to abs, angle
def fft(pos, cut = 10):
    res = np.fft.fft(pos[:, 0] + 1j*pos[:, 1])
    return np.stack([np.abs(res), np.angle(res)])[:, :cut]

# annotates data with features : Nx2x10 shape dependent numbers
def extract_features():
    training_name = "training.npz"
    features = []
    moments = []
    names = []
    
    for r in np.linspace(R_MIN, R_MAX, R_STEPS):
        fname = f"{_safe_key(r)}.npz"
        

    
        try:
            data = np.load(fname)
            features.append(fft(data["pos"]))
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
    
    data = np.load(training_name)    
        
if __name__ == '__main__':
    np.random.seed(0)
    # generate_data()
    extract_features()
