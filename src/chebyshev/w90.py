from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import kwant


class Wannier90ToKwant:
    """
    Parse wannier90.wout + wannier90_hr.dat into a Kwant polyatomic lattice
    and finite Kwant systems.

    Parameters
    ----------
    wout_path
        Path to wannier90.wout
    hr_path
        Path to wannier90_hr.dat
    n
        Dimensionality to keep (1, 2, or 3)
    abs_hamiltonian_cutoff
        Matrix elements with abs(value) < cutoff are dropped
    rel_cutoff
        if `abs_hamiltonian_cutoff` is None, set abs_hamiltonian_cutoff as
        rel_cutoff * MAX_ABS_VALUE

    Main public attributes
    ----------------------
    atom_list
        List[(atom_type, cartesian_pos)]
    wannier_list
        List[cartesian_pos], one per Wannier function, in file order
    lattice_vectors
        Full 3x3 lattice-vector matrix from the .wout
    prim_vec
        lattice_vectors[:, :n]
    ham_dict
        Dict[R_tuple, matrix], where R_tuple has length n
    lattice
        kwant.lattice.Polyatomic lattice for Wannier orbitals
    atomic_lattice
        kwant.lattice.Polyatomic lattice for atomic positions
    system
        Last Wannier Kwant system created
    atomic_system
        Last atomic dummy system created
    """

    def __init__(
        self,
        wout_path: str | Path,
        hr_path: str | Path,
        n: int,
        abs_hamiltonian_cutoff: float | None = None,
        rel_cutoff: float | None = None,
    ) -> None:
        if n not in (1, 2, 3):
            raise ValueError(f"`n` must be 1, 2, or 3, got {n}.")

        self.wout_path = Path(wout_path)
        self.hr_path = Path(hr_path)
        self.n = n
        self.abs_hamiltonian_cutoff = abs_hamiltonian_cutoff
        self.rel_cutoff = rel_cutoff

        if self.abs_hamiltonian_cutoff is None and self.rel_cutoff is None:
            raise ValueError(
                "Either `abs_hamiltonian_cutoff` or `rel_cutoff` must be provided."
            )

        self.atom_list: list[tuple[str, np.ndarray]] = []
        self.wannier_list: list[np.ndarray] = []
        self.lattice_vectors: np.ndarray | None = None
        self.prim_vec: np.ndarray | None = None
        self.ham_dict: dict[tuple[int, ...], np.ndarray] = {}

        self.num_wann: int | None = None
        self.nrpts: int | None = None

        self.system = None
        self.atomic_system = None

        self._parse_wout()
        self._parse_hr()

        self.lattice = self._build_polyatomic_lattice(
            basis=np.array(self.wannier_list, dtype=float),
            prim_vec=self.prim_vec,
            name=[str(i) for i in range(self.num_wann)],
        )

        self.atomic_lattice = self._build_polyatomic_lattice(
            basis=np.array([x[1] for x in self.atom_list], dtype=float),
            prim_vec=self.prim_vec,
            name=[x[0] for x in self.atom_list],
        )

    @staticmethod
    def _extract_floats(line: str) -> list[float]:
        return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]

    def _zero_R(self) -> tuple[int, ...]:
        return (0,) * self.n

    def _zero_tag(self) -> tuple[int, ...]:
        return (0,) * 3

    def _parse_wout(self) -> None:
        text = self.wout_path.read_text()

        self.lattice_vectors = self._parse_lattice_vectors(text)
        self.prim_vec = self.lattice_vectors[:, : self.n]

        self.atom_list = self._parse_atom_list(text)
        self.wannier_list = self._parse_wannier_list(text)

    def _parse_lattice_vectors(self, text: str) -> np.ndarray:
        m = re.search(
            r"Lattice Vectors \(Ang\)\s*\n"
            r"\s*a_1\s+([^\n]+)\n"
            r"\s*a_2\s+([^\n]+)\n"
            r"\s*a_3\s+([^\n]+)",
            text,
        )
        if m is None:
            raise ValueError("Could not find 'Lattice Vectors (Ang)' block in wout.")

        rows = []
        for i in range(1, 4):
            vals = [float(x) for x in m.group(i).split()]
            if len(vals) != 3:
                raise ValueError("Malformed lattice-vector block in wout.")
            rows.append(vals)

        return np.asarray(rows, dtype=float)

    def _parse_atom_list(self, text: str) -> list[tuple[str, np.ndarray]]:
        """
        Parse lines like:
        | Sb   1   0.00000   0.00000   0.07079   |    0.00000   0.00000   3.14938    |
        """
        atom_list: list[tuple[str, np.ndarray]] = []

        lines = text.splitlines()
        in_block = False

        atom_re = re.compile(
            r"^\s*\|\s*([A-Za-z][A-Za-z0-9_]*)\s+\d+\s+"
            r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\s+"
            r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\s+"
            r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\s+\|\s+"
            r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
            r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
            r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*\|"
        )

        for line in lines:
            if "Cartesian Coordinate (Ang)" in line:
                in_block = True
                continue

            if in_block and line.strip().startswith("*---"):
                break

            if not in_block:
                continue

            m = atom_re.match(line)
            if m:
                atom_type = m.group(1)
                cart = np.array(
                    [float(m.group(2)), float(m.group(3)), float(m.group(4))],
                    dtype=float,
                )
                atom_list.append((atom_type, cart))

        if not atom_list:
            raise ValueError("Could not parse atomic positions from wout.")

        return atom_list

    def _parse_wannier_list(self, text: str) -> list[np.ndarray]:
        """
        Parse Wannier centers from the final 'Final State' block.
        """
        final_state_positions = [
            m.start()
            for m in re.finditer(r"^\s*Final State\b", text, flags=re.MULTILINE)
        ]
        if not final_state_positions:
            raise ValueError("Could not find 'Final State' block in wout.")

        start = final_state_positions[-1]
        subtext = text[start:]

        wf_re = re.compile(
            r"WF centre and spread\s+(\d+)\s+\(\s*"
            r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*,\s*"
            r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*,\s*"
            r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*"
            r"\)"
        )

        tmp: dict[int, np.ndarray] = {}
        for m in wf_re.finditer(subtext):
            idx = int(m.group(1))
            pos = np.array(
                [float(m.group(2)), float(m.group(3)), float(m.group(4))],
                dtype=float,
            )
            tmp[idx] = pos

        if not tmp:
            raise ValueError(
                "Could not parse Wannier centres from final-state block in wout."
            )

        max_idx = max(tmp)
        return [tmp[i] for i in range(1, max_idx + 1)]

    def _parse_hr(self) -> None:
        lines = self.hr_path.read_text().splitlines()
        if len(lines) < 4:
            raise ValueError("hr.dat is too short.")

        self.num_wann = int(lines[1].strip())
        self.nrpts = int(lines[2].strip())
        n_deg_lines = math.ceil(self.nrpts / 15)
        data_start = 3 + n_deg_lines

        ham_accum: dict[tuple[int, ...], np.ndarray] = {}

        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) != 7:
                continue

            R_full = tuple(int(parts[i]) for i in range(3))
            R = R_full[: self.n]

            i = int(parts[3]) - 1
            j = int(parts[4]) - 1
            re_part = float(parts[5])
            im_part = float(parts[6])
            val = complex(re_part, im_part)

            if R not in ham_accum:
                ham_accum[R] = np.zeros((self.num_wann, self.num_wann), dtype=complex)

            ham_accum[R][i, j] += val

        if self.rel_cutoff is not None and self.abs_hamiltonian_cutoff is None:
            max_abs = max(np.abs(mat).max() for mat in ham_accum.values())
            self.abs_hamiltonian_cutoff = self.rel_cutoff * max_abs

        assert self.abs_hamiltonian_cutoff is not None

        for R, mat in ham_accum.items():
            mat[np.abs(mat) < self.abs_hamiltonian_cutoff] = 0

        self.ham_dict = {
            R: mat
            for R, mat in ham_accum.items()
            if np.any(np.abs(mat) >= self.abs_hamiltonian_cutoff)
        }

    @staticmethod
    def _build_polyatomic_lattice(
        basis: np.ndarray,
        prim_vec: np.ndarray,
        name="",
    ) -> kwant.lattice.general:
        norbs = [1] * basis.shape[0]
        return kwant.lattice.general(
            prim_vecs=prim_vec.T,
            basis=basis,
            name=name,
            norbs=norbs,
        )

    def _build_atomic_dummy_builder(
        self,
        shape_fun: Callable[[np.ndarray], bool],
    ) -> kwant.Builder:
        """
        Build a dummy atomic builder containing only sites inside `shape_fun`.
        This is used purely to determine which lattice tags (unit cells) are
        present in the target atomic region.
        """
        syst = kwant.Builder()
        zero_tag = self._zero_tag()

        for sublat in self.atomic_lattice.sublattices:
            syst[sublat.shape(shape_fun, zero_tag)] = 0.0

        return syst

    def _build_wannier_builder(
        self,
        shape_fun: Callable[[np.ndarray], bool],
    ) -> kwant.Builder:
        """
        Build a Wannier builder from the parsed Hamiltonian and a shape function.
        """
        if self.lattice is None:
            raise RuntimeError("Wannier lattice has not been built.")

        syst = kwant.Builder()
        zero_tag = self._zero_tag()
        zero_R = self._zero_R()

        for sublat in self.lattice.sublattices:
            syst[sublat.shape(shape_fun, zero_tag)] = 0.0

        for R, full_mat in self.ham_dict.items():
            for i in range(self.num_wann):
                for j in range(self.num_wann):
                    val = full_mat[i, j]

                    if abs(val) < self.abs_hamiltonian_cutoff:
                        continue

                    if R == zero_R and i == j:
                        sublat = self.lattice.sublattices[i]
                        syst[sublat.shape(shape_fun, zero_tag)] = val
                        continue

                    hop_R = tuple(-x for x in R)
                    syst[
                        kwant.builder.HoppingKind(
                            hop_R,
                            self.lattice.sublattices[i],
                            self.lattice.sublattices[j],
                        )
                    ] = val

        return syst

    @staticmethod
    def _filter_builder_by_tags(
        builder: kwant.Builder,
        allowed_tags: set[tuple[int, ...]],
    ) -> kwant.Builder:
        """
        Keep only sites whose `site.tag` is in `allowed_tags`, and only hoppings
        whose endpoints both survive.
        """
        filtered = kwant.Builder()

        kept_sites = [site for site in builder.sites() if tuple(site.tag) in allowed_tags]
        kept_site_set = set(kept_sites)

        for site in kept_sites:
            filtered[site] = builder[site]

        for a, b in builder.hoppings():
            if a in kept_site_set and b in kept_site_set:
                filtered[a, b] = builder[a, b]

        return filtered

    def to_kwant_system(self, shape_fun: Callable[[np.ndarray], bool]):
        """
        Backward-compatible method: build a Wannier system directly using `shape_fun`.
        """
        syst = self._build_wannier_builder(shape_fun)
        syst.eradicate_dangling()
        self.system = syst.finalized()
        return self.system

    def to_kwant_systems(
        self,
        shape_fun: Callable[[np.ndarray], bool],
        larger_shape_fun: Callable[[np.ndarray], bool],
    ):
        """
        Build and return:

        1. atomic_system:
           Dummy atomic system cut with `shape_fun`
        2. system:
           Wannier system first cut with `larger_shape_fun`, then filtered so that
           only sites with `site.tag` present in `atomic_system` are kept

        Notes
        -----
        - `atomic_system` is meant only as a geometric helper for identifying
          atomic unit cells inside `shape_fun`.
        - Filtering is done by matching the Bravais-lattice tag `site.tag`.
        """
        atomic_builder = self._build_atomic_dummy_builder(shape_fun)
        allowed_tags = {tuple(site.tag) for site in atomic_builder.sites()}

        wannier_large_builder = self._build_wannier_builder(larger_shape_fun)
        wannier_filtered_builder = self._filter_builder_by_tags(
            wannier_large_builder,
            allowed_tags,
        )

        self.atomic_system = atomic_builder.finalized()
        self.system = wannier_filtered_builder.finalized()

        return self.atomic_system, self.system
