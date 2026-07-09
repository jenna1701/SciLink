---
name: exafs_simulation
description: >
  Router for EXAFS simulation skill bundles. Covers the full end-to-end
  pipeline: structure relaxation, molecular dynamics for thermal sampling,
  FEFF input generation, FEFF execution, chi(k) averaging, and plotting.
  Always co-load a simulation skill for the MD step.
---

# EXAFS Simulation — Skill Directory

This skill bundle provides the FEFF-specific stages of an EXAFS simulation
workflow. The MD stage (structure relaxation + molecular dynamics) is
intentionally delegated to whichever simulation skill is co-active.

## Skill index

| Subdirectory | Skill file | When to use |
|---|---|---|
| `exafs_workflow/` | `exafs_workflow.md` | End-to-end pipeline: relaxation → MD → FEFF input generation → FEFF execution → chi(k) averaging. The primary skill for running EXAFS simulations. |
| `generate_feff_input/` | `generate_feff_input.md` | FEFF card parameter selection (EDGE/HOLE, SCF, RMAX, CONTROL, CORRECTIONS) and batch generation from MD trajectories. Load alongside `exafs_workflow` or standalone when generating inputs from an existing trajectory. |

## Required co-loading — MD engine

The `exafs_workflow` skill owns the FEFF-specific stages only. For the
relaxation and molecular dynamics stages, co-load one simulation skill:

| Use case | Skills to co-load |
|---|---|
| Default inorganic / mixed systems | `exafs_simulation/exafs_workflow` + `machine_learning_potentials/mace` |
| Magnetic systems (Fe, Co, Ni, Mn, Cr) | `exafs_simulation/exafs_workflow` + `machine_learning_potentials/chgnet` |
| Organic / drug-like molecules | `exafs_simulation/exafs_workflow` + `machine_learning_potentials/mace` (mace-off23) |
| Large cells / MPI-scale MD | `exafs_simulation/exafs_workflow` + `machine_learning_potentials/mace` + `molecular_dynamics/lammps` |
| AIMD benchmark (highest accuracy) | `exafs_simulation/exafs_workflow` + `periodic_dft/vasp` |

Always also load `machine_learning_potentials/general` alongside any
MLIP backend skill for cross-backend selection guidance and acceptance
thresholds.

## Scope note

This is a simulate-mode skill bundle. EXAFS chi(k) fitting against
experimental data is analysis-mode work and belongs in a `curve_fitting/`
skill (or a future `xas` skill), not here.
