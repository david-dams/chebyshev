from __future__ import annotations

import os
import itertools
from pathlib import Path

import kwant
import numpy as np
from numpy.fft import fft
from scipy.spatial import cKDTree
import shapely

from train import TRAINING_DATA
from w90 import Wannier90ToKwant

def _anchor(): # for emacs repl lol
    pass

BASE_DIR = Path(_anchor.__code__.co_filename).parent.parent
COEFFICIENTS_PATH = BASE_DIR / "data" / "coefficients"
W90_FILES_DIR = BASE_DIR / "data" / "wannier"

R_MIN = 10
R_MAX = 200
R_STEPS = 500
N_MIN = 3
N_MAX = 40

N_BOUNDARY_SAMPLES = 256
N_FOURIER_MODES = 100
N_ANGLE_BINS = 24

Z_ATOL = 1e-6
BOUNDARY_GAP_THRESHOLD = np.pi

# ------------------------------------------------------------
# shape / filenames
# ------------------------------------------------------------
def is_valid_w90_path(w90_path):
    return w90_path.is_dir() and "JVASP" in w90_path.name and not "JVASP-27971" in w90_path.name and not "JVASP-6145" in w90_path.name # w90 broken / missing

def get_shape_fun(r, n):
    if n == np.inf:
        return lambda pos: pos[0] ** 2 + pos[1] ** 2 < r ** 2

    points = [
        (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n))
        for i in np.arange(n)
    ]
    coords = r * np.array(points + [points[0]])
    polygon = shapely.Polygon(coords)
    return lambda pos: polygon.contains(shapely.Point(pos)) # shapely ignores z coordinate


def coefficients_file(r: float, n: int | float, model_id: str) -> Path:
    r_str = f"{r:.6f}".replace(".", "p")
    n_str = "inf" if n == np.inf else str(int(n))
    return COEFFICIENTS_PATH / f"{model_id}_r_{r_str}_n_{n_str}.npz"


def radius_n_from_coefficients_file(fname):
    stem = Path(fname).stem
    parts = stem.split("_")
    radius = float(parts[parts.index("r") + 1].replace("p", "."))
    n_raw = parts[parts.index("n") + 1]
    n = np.inf if n_raw == "inf" else int(n_raw)
    return radius, n


def get_grid():
    corners = np.append(np.arange(N_MIN, N_MAX + 1), np.inf)
    radii = np.linspace(R_MIN, R_MAX, R_STEPS)
    return itertools.product(radii, corners)


# ------------------------------------------------------------
# boundary utilities
# ------------------------------------------------------------

def estimate_nn_spacing(points, k=2):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 1.0

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    return float(np.median(dists[:, 1]))


def boundary_points_by_angular_gap(points, r=None, gap_threshold=np.pi):
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return points, np.empty(0), r

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
        wrap = angles[0] + 2 * np.pi - angles[-1]
        gaps = np.concatenate([diffs, [wrap]])

        max_gaps[i] = np.max(gaps)

    idx = np.where(max_gaps > gap_threshold)[0]
    return points[idx], max_gaps, r


def resample_closed_curve(pos, n_samples):
    pos = np.asarray(pos, dtype=float)

    if len(pos) < 2:
        raise ValueError("Need at least 2 boundary points")

    closed = np.vstack([pos, pos[0]])
    seg = np.diff(closed, axis=0)
    seglen = np.linalg.norm(seg, axis=1)

    keep = seglen > 1e-12
    if not np.all(keep):
        closed = np.vstack([closed[:-1][keep], closed[0]])
        seg = np.diff(closed, axis=0)
        seglen = np.linalg.norm(seg, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seglen)])
    L = s[-1]

    if L <= 0:
        raise ValueError("Boundary has zero length")

    t = np.linspace(0.0, L, n_samples, endpoint=False)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(seglen) - 1)

    alpha = (t - s[idx]) / seglen[idx]
    return closed[idx] + alpha[:, None] * seg[idx]


def order_boundary_points_cyclic(points_xy):
    c = points_xy.mean(axis=0)
    ang = np.arctan2(points_xy[:, 1] - c[1], points_xy[:, 0] - c[0])
    return points_xy[np.argsort(ang)]


# ------------------------------------------------------------
# system helpers
# ------------------------------------------------------------

def system_positions(fsyst):
    return np.asarray([fsyst.pos(i) for i in range(fsyst.graph.num_nodes)], dtype=float)


def system_species(fsyst):
    return np.asarray([str(site.family.name) for site in fsyst.sites], dtype=object)


def cluster_layers(z, atol=Z_ATOL):
    z = np.asarray(z, dtype=float)
    order = np.argsort(z)

    groups = []
    for idx in order:
        if not groups or abs(z[idx] - z[groups[-1][0]]) > atol:
            groups.append([idx])
        else:
            groups[-1].append(idx)

    z_values = np.asarray([np.mean(z[g]) for g in groups], dtype=float)
    return z_values, [np.asarray(g, dtype=int) for g in groups]


def boundary_indices_by_layer(atomic_system, z_atol=Z_ATOL):
    pos = system_positions(atomic_system)
    z_values, layer_groups = cluster_layers(pos[:, 2], atol=z_atol)

    boundary_groups = []

    for group in layer_groups:
        xy = pos[group, :2]

        if len(xy) < 3:
            boundary_groups.append(np.empty(0, dtype=int))
            continue

        boundary_xy, _, _ = boundary_points_by_angular_gap(
            xy,
            gap_threshold=BOUNDARY_GAP_THRESHOLD,
        )

        tree = cKDTree(xy)
        dist, local_idx = tree.query(boundary_xy, k=1)
        local_idx = np.unique(local_idx[dist < 1e-8])

        boundary_groups.append(group[local_idx])

    return z_values, boundary_groups


def primitive_frame_2d(prim_vec):
    prim_vec = np.asarray(prim_vec, dtype=float)

    a1 = prim_vec[:2, 0]
    a2 = prim_vec[:2, 1]

    e1 = a1 / np.linalg.norm(a1)
    a2_orth = a2 - np.dot(a2, e1) * e1

    if np.linalg.norm(a2_orth) < 1e-12:
        e2 = np.array([-e1[1], e1[0]])
    else:
        e2 = a2_orth / np.linalg.norm(a2_orth)

    return e1, e2


def neighbor_lists(fsyst):
    return [list(fsyst.graph.out_neighbors(i)) for i in range(fsyst.graph.num_nodes)]


# ------------------------------------------------------------
# feature functions
# ------------------------------------------------------------

def boundary_fourier_by_layer(
    atomic_system,
    n_samples=N_BOUNDARY_SAMPLES,
    n_modes=N_FOURIER_MODES,
):
    pos = system_positions(atomic_system)
    z_values, boundary_groups = boundary_indices_by_layer(atomic_system)

    rows = []

    for idx in boundary_groups:
        if len(idx) < 3:
            rows.append(np.zeros(2 * n_modes, dtype=float))
            continue

        xy = order_boundary_points_cyclic(pos[idx, :2])
        xy = resample_closed_curve(xy, n_samples=n_samples)

        # translation invariant only
        xy = xy - xy.mean(axis=0, keepdims=True)

        signal = xy[:, 0] + 1j * xy[:, 1]
        coeffs = fft(signal) / len(signal)

        coeffs = coeffs[1 : 1 + n_modes]

        if len(coeffs) < n_modes:
            coeffs = np.pad(coeffs, (0, n_modes - len(coeffs)))

        rows.append(np.concatenate([coeffs.real, coeffs.imag]))

    return z_values, np.asarray(rows, dtype=float)


def boundary_species_histogram_by_layer(
    atomic_system,
    species_vocab,
    normalize=True,
):
    species_vocab = list(species_vocab)
    species_to_idx = {s: i for i, s in enumerate(species_vocab)}

    species = system_species(atomic_system)
    z_values, boundary_groups = boundary_indices_by_layer(atomic_system)

    rows = []

    for idx in boundary_groups:
        hist = np.zeros(len(species_vocab), dtype=float)

        for s in species[idx]:
            hist[species_to_idx[str(s)]] += 1.0

        if normalize and hist.sum() > 0:
            hist /= hist.sum()

        rows.append(hist)

    return z_values, np.asarray(rows, dtype=float)


def boundary_bond_angle_histogram_by_layer(
    atomic_system,
    prim_vec,
    n_bins=N_ANGLE_BINS,
    normalize=True,
):
    pos = system_positions(atomic_system)
    z_values, boundary_groups = boundary_indices_by_layer(atomic_system)
    neighbors = neighbor_lists(atomic_system)

    e1, e2 = primitive_frame_2d(prim_vec)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    rows = []

    for idx in boundary_groups:
        hist = np.zeros(n_bins, dtype=float)

        for i in idx:
            for j in neighbors[i]:
                # in-plane bonds only
                if abs(pos[i, 2] - pos[j, 2]) > 10 * Z_ATOL:
                    continue

                v = pos[j, :2] - pos[i, :2]
                if np.linalg.norm(v) < 1e-12:
                    continue

                x = np.dot(v, e1)
                y = np.dot(v, e2)
                theta = np.arctan2(y, x)

                b = np.searchsorted(edges, theta, side="right") - 1
                b = np.clip(b, 0, n_bins - 1)
                hist[b] += 1.0

        if normalize and hist.sum() > 0:
            hist /= hist.sum()

        rows.append(hist)

    return z_values, np.asarray(rows, dtype=float)


def lattice_vector_features(prim_vec):
    prim_vec = np.asarray(prim_vec, dtype=float)

    a1 = prim_vec[:2, 0]
    a2 = prim_vec[:2, 1]

    l1 = np.linalg.norm(a1)
    l2 = np.linalg.norm(a2)

    cosang = np.dot(a1, a2) / (l1 * l2)
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.arccos(cosang)

    area = abs(np.cross(a1, a2))

    return np.asarray([l1, l2, angle, area], dtype=float)


def thickness_features(atomic_system):
    pos = system_positions(atomic_system)
    z_values, _ = cluster_layers(pos[:, 2], atol=Z_ATOL)

    if len(z_values) == 0:
        return np.zeros(3, dtype=float)

    thickness = z_values.max() - z_values.min()
    n_layers = len(z_values)

    if n_layers > 1:
        spacing = np.diff(np.sort(z_values))
        mean_spacing = spacing.mean()
    else:
        mean_spacing = 0.0

    return np.asarray([thickness, n_layers, mean_spacing], dtype=float)


# ------------------------------------------------------------
# KPM
# ------------------------------------------------------------

def get_moments_limits(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    moments = spectrum._moments()
    return moments, spectrum._a, spectrum._b


# ------------------------------------------------------------
# species vocabulary
# ------------------------------------------------------------

def collect_species_vocab():
    vocab = set()

    for w90_path in W90_FILES_DIR.iterdir():
        if not is_valid_w90_path(w90_path):
            continue

        try:
            wout_path = w90_path / "wannier90.wout"
            text = wout_path.read_text()
            atom_list = Wannier90ToKwant._parse_atom_list(text)
        except:
            print(wout_path)


        for atom_type, _ in atom_list:
            vocab.add(str(atom_type))
            
    return sorted(vocab)

# ------------------------------------------------------------
# system generation
# ------------------------------------------------------------
def plot_finite_spectrum(finalized_system, title):
    H = finalized_system.hamiltonian_submatrix(sparse=False)
    evals = la.eigvalsh(H)

    plt.figure()
    plt.plot(evals, ".")
    plt.xlabel("state index")
    plt.ylabel("Energy")
    plt.title(title)
    plt.show()

def get_systems():
    for w90_path in W90_FILES_DIR.iterdir():
        if not is_valid_w90_path(w90_path):
            continue

        parsed = Wannier90ToKwant(
            wout_path=w90_path / "wannier90.wout",
            hr_path=w90_path / "wannier90_hr.dat",
            n=2,
            rel_cutoff=1e-3,
        )

        for r, n in get_grid():
            shape_fun = get_shape_fun(r, n)

            # Larger Wannier cut before filtering by atomic tags.
            # Tune if needed.
            larger_shape_fun = get_shape_fun(1.10 * r, n)

            atomic_system, fsyst = parsed.to_kwant_systems(
                shape_fun=shape_fun,
                larger_shape_fun=larger_shape_fun,
            )
            
            plot_finite_spectrum(fsyst, w90_path.stem)                
            
            fname = coefficients_file(r, n, w90_path.name)

            yield {
                "fname": fname,
                "model": w90_path.name,
                "radius": r,
                "corners": n,
                "atomic_system": atomic_system,
                "wannier_system": fsyst,
                "prim_vec": parsed.prim_vec,
            }


# ------------------------------------------------------------
# stage 1: generate per-sample files
# ------------------------------------------------------------

def generate_data():
    COEFFICIENTS_PATH.mkdir(parents=True, exist_ok=True)

    species_vocab = collect_species_vocab()

    for sample in get_systems():
        fname = sample["fname"]
        atomic_system = sample["atomic_system"]
        fsyst = sample["wannier_system"]
        prim_vec = sample["prim_vec"]

        z_fourier, fourier = boundary_fourier_by_layer(atomic_system)

        z_species, species_hist = boundary_species_histogram_by_layer(
            atomic_system,
            species_vocab=species_vocab,
        )

        z_angles, bond_angle_hist = boundary_bond_angle_histogram_by_layer(
            atomic_system,
            prim_vec=prim_vec,
        )

        if not (
            np.allclose(z_fourier, z_species, atol=Z_ATOL, rtol=0.0)
            and np.allclose(z_fourier, z_angles, atol=Z_ATOL, rtol=0.0)
        ):
            raise ValueError(f"Layer mismatch for {fname}")

        moments, a, b = get_moments_limits(fsyst)

        np.savez_compressed(
            fname,
            fourier=fourier,
            species_hist=species_hist,
            bond_angle_hist=bond_angle_hist,
            z_values=z_fourier,
            lattice_features=lattice_vector_features(prim_vec),
            thickness_features=thickness_features(atomic_system),
            moments=np.asarray(moments),
            a=a,
            b=b,
            radius=sample["radius"],
            corners=sample["corners"],
            model=sample["model"],
            species_vocab=np.asarray(species_vocab, dtype=object),
        )

        print(f"Generated {fname}")


# ------------------------------------------------------------
# stage 2: aggregate files into TRAINING_DATA
# ------------------------------------------------------------

def extract_features():
    names = []
    models = []
    radii = []
    corners = []

    fourier = []
    species_hist = []
    bond_angle_hist = []
    z_values = []
    lattice_features = []
    thickness = []

    moments = []
    lower_limits = []
    upper_limits = []

    for fname in sorted(COEFFICIENTS_PATH.glob("*.npz")):
        data = np.load(fname, allow_pickle=True)

        names.append(fname.name)
        models.append(str(data["model"]))
        radii.append(float(data["radius"]))
        corners.append(float(data["corners"]))

        fourier.append(data["fourier"])
        species_hist.append(data["species_hist"])
        bond_angle_hist.append(data["bond_angle_hist"])
        z_values.append(data["z_values"])
        lattice_features.append(data["lattice_features"])
        thickness.append(data["thickness_features"])

        moments.append(data["moments"])
        lower_limits.append(float(data["a"]))
        upper_limits.append(float(data["b"]))

    np.savez_compressed(
        TRAINING_DATA,
        names=np.asarray(names, dtype=object),
        models=np.asarray(models, dtype=object),
        radii=np.asarray(radii, dtype=float),
        corners=np.asarray(corners, dtype=float),

        # variable number of z-layers => object arrays
        fourier=np.asarray(fourier, dtype=object),
        species_hist=np.asarray(species_hist, dtype=object),
        bond_angle_hist=np.asarray(bond_angle_hist, dtype=object),
        z_values=np.asarray(z_values, dtype=object),

        # fixed-size global features
        lattice_features=np.asarray(lattice_features, dtype=float),
        thickness_features=np.asarray(thickness, dtype=float),

        moments=np.asarray(moments, dtype=object),
        lower_limits=np.asarray(lower_limits, dtype=float),
        upper_limits=np.asarray(upper_limits, dtype=float),
    )


if __name__ == "__main__":
    generate_data()
    extract_features()
