import kwant
import numpy as np
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt

from generate_data import get_system, get_grid

N_SAMPLES = 4

def get_spectrum(s, energy, moments):
    energy = np.asarray(energy)
    e = (energy - s._b) / s._a
    g_e = (np.pi * np.sqrt(1 - e) * np.sqrt(1 + e))

    # factor 2 comes from the norm of the Chebyshev polynomials
    moments[1:] = 2 * moments[1:]

    return np.transpose(chebval(e, moments) / g_e)

def get_predictions(name):
    data = np.load(f"{name}_predictions.npz")
    
    y = data["y"]
    y_pred = data["y_pred"]
    radii = data["radii"]
    corners = data["corners"]

    # randomly plot some spectra
    idxs = np.random.choice(y.shape[0], N_SAMPLES)

    return y[idxs], y_pred[idxs], radii[idxs], corners[idxs]

def plot_predictions(name):
    
    y, y_pred, radii, corners = get_predictions(name)

    for i, r in enumerate(radii):        
        fsyst = get_system(float(r), int(corners[i]))        
        spectrum = kwant.kpm.SpectralDensity(fsyst)

        # analytical reference
        energies, densities = spectrum()
        plt.plot(energies, densities.real)

        # sanity check: these we saved and should give the same results as the reference
        densities_ref = get_spectrum(spectrum, energies, y[i])
        plt.plot(energies, densities_ref.real, '.-')
        
        # prediction
        densities_pred = get_spectrum(spectrum, energies, y_pred[i])
        plt.plot(energies, densities_pred.real, '--')

        # plt.savefig(f"{name}_{r}_{corners[i]}.pdf")
        plt.show()

if __name__ == "__main__":
    plot_predictions("cnn")    
