import kwant
import numpy as np
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt

from generate_data import get_system, get_grid

def get_spectrum(s, energy, moments):
    energy = np.asarray(energy)
    e = (energy - s._b) / s._a
    g_e = (np.pi * np.sqrt(1 - e) * np.sqrt(1 + e))

    # factor 2 comes from the norm of the Chebyshev polynomials
    moments[1:] = 2 * moments[1:]

    return np.transpose(chebval(e, moments) / g_e)

def get_predictions(name, n_samples = None):
    data = np.load(f"{name}_predictions.npz")
    
    y = data["y"]
    y_pred = data["y_pred"]
    radii = data["radii"]
    corners = data["corners"]

    # randomly plot some spectra
    if n_samples is not None:
        np.random.seed(0)    
        idxs = np.random.choice(y.shape[0], N_SAMPLES)
    # sort by size
    else:
        idxs = radii.argsort()        

    return y[idxs], y_pred[idxs], radii[idxs], corners[idxs]

def maybe_to_int(n):
    if n == np.inf:
        return n
    return int(n)

def plot_predictions(name):
    
    y, y_pred, radii, corners = get_predictions(name, n_samples = 4)

    fig, ax = plt.subplots(1,1)
    for i, r in enumerate(radii):        
        fsyst = get_system(float(r), maybe_to_int(corners[i]))     
        spectrum = kwant.kpm.SpectralDensity(fsyst)

        # sample energies from analytical reference
        energies = spectrum.energies
        # energies, densities = spectrum()
        # plt.plot(spectrum.energies, densities.real)

        # TODO: there are some slight deviations between ref and analytical => probably float stuff from loading
        # sanity check: these we saved and should give the same results as the reference
        densities_ref = get_spectrum(spectrum, energies, y[i])
        plt.plot(energies, densities_ref.real, '.-')
        
        # prediction
        densities_pred = get_spectrum(spectrum, energies, y_pred[i])
        ax.plot(energies, densities_pred.real, '--')

    plt.savefig(f"{name}.pdf")

def plot_predictions_2d(name):
    
    y, y_pred, radii, corners = get_predictions(name)

    spectra_ref, spectra_pred = [], []

    for i, r in enumerate(radii):        
        fsyst = get_system(float(r), maybe_to_int(corners[i]))     
        spectrum = kwant.kpm.SpectralDensity(fsyst)
        
        energies = spectrum.energies
        
        densities_ref = get_spectrum(spectrum, energies, y[i])
        densities_pred = get_spectrum(spectrum, energies, y_pred[i])

        spectra_ref.append(densities_ref)
        spectra_pred.append(densities_pred)
        
    fig, axs = plt.subplots(1, 2)

    im0 = axs[0].matshow(spectra_ref)
    axs[0].set_xlabel(r"$E$")
    axs[0].set_ylabel(r"$r (nm)$")
    axs[0].set_aspect("equal")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].matshow(spectra_pred)
    axs[1].set_xlabel(r"$E$")
    axs[1].set_ylabel(r"$r (nm)$")
    axs[1].set_aspect("equal")
    fig.colorbar(im1, ax=axs[1])

    plt.tight_layout()


    plt.savefig(f"2d_{name}.pdf")
    
if __name__ == "__main__":
    # plot_predictions("cnn")    
    # plot_predictions("linear_regression")    
    # plot_predictions("mlp")    

    plot_predictions_2d("cnn")    
    plot_predictions_2d("linear_regression")    
    plot_predictions_2d("mlp")    
