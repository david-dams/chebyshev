from pathlib import Path

import pytest

import kwant
import numpy as np

from chebyshev.w90 import Wannier90ToKwant

@pytest.fixture
def w90_path():
    return Path(__file__).parent.parent / "data" / "wannier" / "JVASP-8879"

def test_hamiltonian(w90_path):

    parsed = Wannier90ToKwant(
        wout_path=w90_path / "wannier90.wout",
        hr_path=w90_path / "wannier90_hr.dat",
        n=2,
        rel_cutoff=1e-3,
    )

    def shape(pos):
        x, y, _ = pos
        return x**2 + y**2 < 20**2

    fsyst = parsed.to_kwant_system(shape)

    ham_dict = parsed.ham_dict
    keys = set(ham_dict.keys())
    sites = fsyst.sites
    wannier_pos = np.array(parsed.wannier_list)
    cutoff = parsed.abs_hamiltonian_cutoff

    for i, si in enumerate(sites):
        tag_i = si.tag
        basis_i = si.family.offset
        wannier_idx_i = int(si.family.name)

        for j, sj in enumerate(sites):
            tag_j = sj.tag
            basis_j = sj.family.offset
            wannier_idx_j = int(sj.family.name)

            diff = tag_j - tag_i

            if diff in keys:
                h_ref = ham_dict[diff][wannier_idx_i, wannier_idx_j]
                if np.abs(h_ref) < cutoff:
                    continue
                h_kwant = fsyst.hamiltonian(i, j)

                assert np.allclose(h_kwant, h_ref), (
                    f"Mismatch for i={i}, j={j}, R={diff}\n"
                    f"kwant={h_kwant}\nref={h_ref}"
                )

