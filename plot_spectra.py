import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval

from train_new import PLOT_DIR, PREDICTIONS_DIR
USE_PRED_AB = False

def reconstruct_spectrum(energy, moments, a, b):
    """
    Reconstruct DOS from Chebyshev moments on a physical energy grid.

    Parameters
    ----------
    energy : (M,) array
        Physical energy values.
    moments : (N,) array
        Chebyshev moments mu_n.
    a, b : float
        Rescaling parameters so that x = (E - b)/a lies in [-1, 1].

    Returns
    -------
    rho : (M,) array
        Reconstructed density of states.
    """
    energy = np.asarray(energy, dtype=float)
    mu = np.array(moments, dtype=float, copy=True)

    x = (energy - b) / a
    inside = np.abs(x) < 1.0

    rho = np.full_like(x, np.nan, dtype=float)
    if not np.any(inside):
        return rho

    mu[1:] *= 2.0
    denom = np.pi * np.sqrt(np.clip(1.0 - x[inside] ** 2, 1e-15, None))
    rho[inside] = chebval(x[inside], mu) / denom
    return rho


def get_predictions(name, n_samples=None, seed=0):
    data = np.load(f"{PREDICTIONS_DIR}/{name}.npz")

    y_mu = data["y_mu"]
    y_pred_mu = data["y_pred_mu"]
    y_ab = data["y_ab"]
    y_pred_ab = data["y_pred_ab"]
    radii = data["radii"]
    corners = data["corners"]

    if n_samples is not None:
        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, y_mu.shape[0])
        idxs = rng.choice(y_mu.shape[0], size=n_samples, replace=False)
    else:
        idxs = np.argsort(radii)

    return (
        y_mu[idxs],
        y_pred_mu[idxs],
        y_ab[idxs],
        y_pred_ab[idxs],
        radii[idxs],
        corners[idxs],
    )


def common_energy_grid(ab, ab_pred=None, n_energy=400, pad=0.98):
    """
    Build a common energy grid from a collection of [a,b] pairs.

    pad < 1 avoids evaluating exactly at the Chebyshev singular endpoints.
    """
    pairs = [ab]
    if ab_pred is not None:
        pairs.append(ab_pred)

    all_ab = np.concatenate(pairs, axis=0)
    a_all = all_ab[:, 0]
    b_all = all_ab[:, 1]

    e_min = np.min(b_all - pad * np.abs(a_all))
    e_max = np.max(b_all + pad * np.abs(a_all))
    return np.linspace(e_min, e_max, n_energy)


def plot_predictions(name, n_samples=4, use_pred_ab=False, n_energy=400):
    y_mu, y_pred_mu, y_ab, y_pred_ab, radii, corners = get_predictions(name, n_samples=n_samples)

    energy = common_energy_grid(y_ab, y_pred_ab, n_energy=n_energy)

    fig, ax = plt.subplots(figsize=(7, 4))

    for i in range(len(radii)):
        a_ref, b_ref = y_ab[i]
        rho_ref = reconstruct_spectrum(energy, y_mu[i], a_ref, b_ref)

        if use_pred_ab:
            a_pred, b_pred = y_pred_ab[i]
        else:
            a_pred, b_pred = y_ab[i]

        rho_pred = reconstruct_spectrum(energy, y_pred_mu[i], a_pred, b_pred)

        label_ref = f"ref r={radii[i]:.2f}, n={corners[i]}"
        label_pred = f"pred r={radii[i]:.2f}, n={corners[i]}"

        ax.plot(energy, rho_ref, "-", linewidth=1.5, label=label_ref)
        ax.plot(energy, rho_pred, "--", linewidth=1.5, label=label_pred)

    ax.set_xlabel("E")
    ax.set_ylabel("DOS")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/prediction_{name}.pdf")
    plt.close(fig)


def plot_predictions_2d(name, use_pred_ab=False, n_energy=400):
    y_mu, y_pred_mu, y_ab, y_pred_ab, radii, corners = get_predictions(name, n_samples=None)

    energy = common_energy_grid(y_ab, y_pred_ab, n_energy=n_energy)

    spectra_ref = []
    spectra_pred = []

    for i in range(len(radii)):
        a_ref, b_ref = y_ab[i]
        rho_ref = reconstruct_spectrum(energy, y_mu[i], a_ref, b_ref)

        if use_pred_ab:
            a_pred, b_pred = y_pred_ab[i]
        else:
            a_pred, b_pred = y_ab[i]

        rho_pred = reconstruct_spectrum(energy, y_pred_mu[i], a_pred, b_pred)

        spectra_ref.append(rho_ref)
        spectra_pred.append(rho_pred)

    spectra_ref = np.stack(spectra_ref, axis=0)
    spectra_pred = np.stack(spectra_pred, axis=0)

    extent = [energy[0], energy[-1], 0, len(radii) - 1]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axs[0].imshow(
        spectra_ref,
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    axs[0].set_title("Reference")
    axs[0].set_xlabel("E")
    axs[0].set_ylabel("sample index")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(
        spectra_pred,
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    axs[1].set_title("Prediction")
    axs[1].set_xlabel("E")
    axs[1].set_ylabel("sample index")
    fig.colorbar(im1, ax=axs[1])

    fig.savefig(f"{PLOT_DIR}/2d_{name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_predictions("cnn", n_samples=4, use_pred_ab=USE_PRED_AB)
    plot_predictions("linear_regression", n_samples=4, use_pred_ab=USE_PRED_AB)
    plot_predictions("mlp", n_samples=4, use_pred_ab=USE_PRED_AB)

    plot_predictions_2d("cnn", use_pred_ab=USE_PRED_AB)
    plot_predictions_2d("linear_regression", use_pred_ab=USE_PRED_AB)
    plot_predictions_2d("mlp", use_pred_ab=USE_PRED_AB)
