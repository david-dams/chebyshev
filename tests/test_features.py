from pathlib import Path

import numpy as np
import pytest

import generate_data as g
from w90 import Wannier90ToKwant


# ------------------------------------------------------------
# pure feature-function tests (fast, no kwant system needed)
# ------------------------------------------------------------

def test_species_pairs_and_dim():
    vocab = ["C", "N", "O"]
    pairs = g.species_pairs(vocab)
    n = len(vocab)
    assert len(pairs) == n * (n + 1) // 2
    block = g.N_BASIS_U_BINS * g.N_BASIS_V_BINS + g.N_BASIS_Z_BINS
    assert g.basis_feature_dim(vocab) == len(pairs) * block


def _toy_basis():
    prim_vec = np.array([[2.46, -1.23], [0.0, 2.13], [0.0, 0.0]])  # (3, 2)
    atom_list = [
        ("C", np.array([0.0, 0.0, 0.0])),
        ("C", np.array([1.23, 0.71, 0.0])),
        ("N", np.array([0.0, 1.42, 1.5])),
    ]
    vocab = ["C", "N"]
    return atom_list, prim_vec, vocab


def test_basis_features_shape_and_finite():
    atom_list, prim_vec, vocab = _toy_basis()
    b = g.basis_features(atom_list, prim_vec, vocab)
    assert b.shape == (g.basis_feature_dim(vocab),)
    assert np.isfinite(b).all()
    assert (b >= 0).all()


def test_basis_features_permutation_invariant():
    atom_list, prim_vec, vocab = _toy_basis()
    b = g.basis_features(atom_list, prim_vec, vocab)
    b_rev = g.basis_features(atom_list[::-1], prim_vec, vocab)
    assert np.allclose(b, b_rev)


def test_basis_features_single_atom_is_zero():
    prim_vec = np.array([[2.46, -1.23], [0.0, 2.13], [0.0, 0.0]])
    b = g.basis_features([("C", np.zeros(3))], prim_vec, ["C"])
    assert b.shape == (g.basis_feature_dim(["C"]),)
    assert np.allclose(b, 0.0)


def test_basis_features_blocks_normalized():
    atom_list, prim_vec, vocab = _toy_basis()
    b = g.basis_features(atom_list, prim_vec, vocab)
    block_len = g.N_BASIS_U_BINS * g.N_BASIS_V_BINS + g.N_BASIS_Z_BINS
    uv = g.N_BASIS_U_BINS * g.N_BASIS_V_BINS
    for k in range(len(g.species_pairs(vocab))):
        off = k * block_len
        uv_block = b[off:off + uv]
        z_block = b[off + uv:off + block_len]
        # each populated histogram block is normalized to sum 1
        for hist in (uv_block, z_block):
            s = hist.sum()
            assert np.isclose(s, 0.0) or np.isclose(s, 1.0)


def test_assemble_features_length_and_dtype():
    atom_list, prim_vec, vocab = _toy_basis()
    basis = g.basis_features(atom_list, prim_vec, vocab)

    n_layers = 2
    fourier = np.random.rand(n_layers, 2 * g.N_FOURIER_MODES)
    species_hist = np.random.rand(n_layers, len(vocab))
    bond_angle_hist = np.random.rand(n_layers, g.N_ANGLE_BINS)
    lattice = np.arange(4.0)
    thickness = np.arange(3.0)

    feat = g.assemble_features(
        fourier, species_hist, bond_angle_hist, lattice, thickness, basis
    )

    expected = (
        2 * g.N_FOURIER_MODES
        + len(vocab)
        + g.N_ANGLE_BINS
        + 4
        + 3
        + g.basis_feature_dim(vocab)
    )
    assert feat.shape == (expected,)
    assert feat.dtype == np.float32


def test_assemble_features_mean_pools_layers():
    vocab = ["C", "N"]
    basis = np.zeros(g.basis_feature_dim(vocab))

    fourier = np.stack(
        [np.ones(2 * g.N_FOURIER_MODES), 3.0 * np.ones(2 * g.N_FOURIER_MODES)]
    )
    species_hist = np.zeros((2, len(vocab)))
    bond_angle_hist = np.zeros((2, g.N_ANGLE_BINS))

    feat = g.assemble_features(
        fourier, species_hist, bond_angle_hist, np.zeros(4), np.zeros(3), basis
    )
    # first block is the mean-pooled fourier -> (1 + 3) / 2 == 2
    assert np.allclose(feat[: 2 * g.N_FOURIER_MODES], 2.0)


# ------------------------------------------------------------
# integration test on a real (small) w90 material
# ------------------------------------------------------------

@pytest.fixture
def parsed():
    w90_path = Path(__file__).parent.parent / "data" / "wannier" / "JVASP-8879"
    return Wannier90ToKwant(
        wout_path=w90_path / "wannier90.wout",
        hr_path=w90_path / "wannier90_hr.dat",
        n=2,
        rel_cutoff=1e-3,
    )


def test_feature_pipeline_on_system(parsed):
    def shape(pos):
        x, y, _ = pos
        return x**2 + y**2 < 8**2

    atomic_system, _ = parsed.to_kwant_systems(shape, shape)

    vocab = sorted({str(a[0]) for a in parsed.atom_list})

    z_f, fourier = g.boundary_fourier_by_layer(atomic_system)
    z_s, species_hist = g.boundary_species_histogram_by_layer(
        atomic_system, species_vocab=vocab
    )
    z_a, bond_angle_hist = g.boundary_bond_angle_histogram_by_layer(
        atomic_system, prim_vec=parsed.prim_vec
    )

    # all feature extractors must agree on the layer structure
    assert len(z_f) == len(z_s) == len(z_a)
    assert np.allclose(z_f, z_s) and np.allclose(z_f, z_a)

    basis = g.basis_features(parsed.atom_list, parsed.prim_vec, vocab)
    feat = g.assemble_features(
        fourier,
        species_hist,
        bond_angle_hist,
        g.lattice_vector_features(parsed.prim_vec),
        g.thickness_features(atomic_system),
        basis,
    )

    expected = (
        2 * g.N_FOURIER_MODES
        + len(vocab)
        + g.N_ANGLE_BINS
        + 4
        + 3
        + g.basis_feature_dim(vocab)
    )
    assert feat.shape == (expected,)
    assert np.isfinite(feat).all()
