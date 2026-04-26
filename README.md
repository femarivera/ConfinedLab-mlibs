# ConfinedLab mlibs

**MODFLOW 6 modelling utilities for synthetic multilayer groundwater systems.**

Developed at **Bordeaux INP, Lab EPOC, Université de Bordeaux**  
As part of the **ConfinedLab** project, funded by the OneWater PEPR DEESAC Project.  
Author: **MARIN RIVERA Carlos Felipe**

---

## What is this?

ConfinedLab is a Python package that provides modular, reusable utilities for building, parameterising, and visualising **MODFLOW 6** groundwater models. It is designed around synthetic multilayer confined aquifer systems, covering the full workflow from geometry generation to transient results analysis.

---

## Modules

| Module | Description |
|---|---|
| `modgeom6` | Generate structured grid geometries: idomain arrays, top/bottom elevations, thickness, recharge, layer subdivision |
| `modbound6` | Create boundary condition stress period data: rivers (RIV), general head (GHB), drains (DRN) |
| `modpar6` | Generate spatially correlated random fields of hydraulic parameters (K, Sy, Ss) using FFT-based simulation |
| `modplot6` | Visualise model grids, heads, cross-sections, boundary conditions, and budget summaries |
| `modpump6` | Analyse steady-state pumping scenarios: iterate pumping rates, capture rates, and water budgets |
| `modtransient6` | Process and visualise transient results: time-series heads, flows, storage release, zone budgets |

---

## Installation

### Requirements

- Python >= 3.8
- numpy
- scipy
- matplotlib
- flopy

### Local install (recommended for development)

Clone the repository and install in editable mode. Any changes you make to the files are immediately available — no reinstall needed.

```bash
git clone https://github.com/YOUR_USERNAME/confinedlab.git
cd confinedlab
pip install -e .
```

### Install directly from GitHub

```bash
pip install git+https://github.com/YOUR_USERNAME/confinedlab.git
```

---

## Quick start

```python
from confinedlab import modgeom6, modpar6

# --- Build a 3-layer right-dipping geometry ---
nlay, nrow, ncol = 3, 10, 50
outcrop_cells = [40, 30, 0]   # column index where each layer outcrops

idomain = modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells, direction="right")

# --- Generate a spatially correlated hydraulic conductivity field ---
geom_mean, mu, sigma2, sigma = modpar6.moments_from_percentiles(
    k1=1e-5, p1=0.05,   # 5th percentile: 1e-5 m/s
    k2=1e-3, p2=0.95    # 95th percentile: 1e-3 m/s
)

K_field = modpar6.generate_random_field(
    shape=(nrow, ncol),
    variogram_type="exponential",
    geom_mean=geom_mean,
    sill=sigma2,
    range_param=15.0,   # correlation length in model units
    seed=42
)
```

---

## Module overview

### `modgeom6` — Geometry

Functions to build the 3D grid structure of a synthetic multilayer system.

```python
modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells, direction)
modgeom6.compute_top(idomain, outcrop_z, transition, slope, direction, ...)
modgeom6.compute_thickness(idomain, base_thicknesses, transition, ...)
modgeom6.compute_bottom(ztop, thickness_array)
modgeom6.compute_irch(idomain)
modgeom6.compute_recharge(irch, R)
modgeom6.subdivide_layers(idomain, ztop, zbot, nsub_layers)
modgeom6.insert_soil_layer(ztop, zbot, idomain, soil_thickness)
```

### `modpar6` — Parameter fields

Generate random fields with realistic spatial structure from prior knowledge of hydraulic properties.

```python
# Convert prior knowledge to log-normal distribution parameters
modpar6.moments_from_percentiles(k1, p1, k2, p2)
modpar6.moments_from_arithmetic_mean_variance(arith_mean, arith_var)
modpar6.moments_from_log_mean_variance(log_mean, log_var, log_base)

# Generate 2D spatially correlated field
modpar6.generate_random_field(shape, variogram_type, geom_mean, sill, nugget, range_param, ...)

# Stack 2D fields into a 3D array (one field per layer)
modpar6.stack_fields_to_3D(field_list, nlay, nrow, ncol)
```

### `modbound6` — Boundary conditions

Create stress period data arrays for MODFLOW 6 boundary packages.

### `modplot6` — Plotting

Visualise model structure, results, and budget components for both steady-state and transient simulations.

### `modpump6` — Pumping analysis

Automate well pumping rate iteration and analyse captured discharge and induced recharge in steady-state models.

### `modtransient6` — Transient analysis

Extract and visualise time-series data, storage release proportions, and zone water budgets from transient MODFLOW 6 runs.

---

## Project structure

```
confinedlab/               ← project root
├── confinedlab/           ← installable package
│   ├── __init__.py
│   ├── modgeom6.py
│   ├── modbound6.py
│   ├── modpar6.py
│   ├── modplot6.py
│   ├── modpump6.py
│   └── modtransient6.py
├── pyproject.toml
└── README.md
```

---

## Acknowledgements

This work is part of the **ConfinedLab** project and was funded by the **OneWater PEPR DEESAC Project**.  