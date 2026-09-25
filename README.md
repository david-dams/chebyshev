# ML of Chebyshev Coefficients for 2D Nanoflakes

Predict Kernel Polynomial Method (KPM) Chebyshev coefficients directly
from the geometry of 2D nanoflakes. See [Weisse et al., Rev. Mod. Phys. 78,
275 (2006)](http://dx.doi.org/10.1103/RevModPhys.78.275) for KPM background.

## Setup

```bash
conda env create -f environment.yml
conda activate chebyshev
```

## Quick start

```bash
# 0. Download JARVIS Wannier90 data into data/wannier/
python scripts/download.py

# 1. Generate per-sample features + KPM moments from JARVIS Wannier90 data
python scripts/generate_data.py

# 2. Train all models on both holdout splits
python scripts/train.py radius_interval material

# 3. Reconstruct DOS plots from predicted coefficients
python scripts/plot_spectra.py
```

## Repository layout

```
scripts/
  w90.py            # Wannier90 .wout/.hr.dat -> Kwant system
  download.py       # fetch JARVIS bulk-wannier archive into data/wannier/
  generate_data.py  # build flakes, extract features + KPM moments
  train.py           # train linear/MLP/CNN, two holdout splits
  plot_spectra.py    # reconstruct DOS from predicted coefficients
tests/               # pytest: parsing, boundary, feature invariants
data/                # gitignored: wannier/ (JARVIS data), coefficients/ (intermediate results), training.npz (features and targets for training)
```

## Holdout splits

- **`radius_interval`** — hold out samples with radius in `HOLDOUT_RADIUS_RANGE`;
  tests within-material interpolation across flake size.
- **`material`** — hold out 20% of unique material IDs; tests cross-material
  generalization.

## Dependencies

Python >=3.13 · kwant · jax · flax · optax · numpy · scipy · shapely ·
matplotlib · requests
