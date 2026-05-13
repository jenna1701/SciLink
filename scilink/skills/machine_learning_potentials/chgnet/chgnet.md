---
description: CHGNet (Crystal Hamiltonian Graph Neural Network) — universal pretrained ML interatomic potential, MPtrj-trained, ASE-only (no LAMMPS pair_style).
detect:
  binaries: []
  env_vars: []
  python_modules: [chgnet]
  guidance: |
    CHGNet is distributed as a pure-Python package (``pip install chgnet``).
    It does NOT provide a LAMMPS pair_style — MD runs go through ASE's
    Langevin / NPT / Verlet integrators with chgnet.model.dynamics
    .CHGNetCalculator attached as the calculator. A successful
    ``import chgnet`` means the backend is usable.
---
# CHGNet (Crystal Hamiltonian Graph Neural Network)

## overview

CHGNet is a universal ML interatomic potential pretrained on the MPtrj
dataset (~1.5M DFT relaxations harvested from Materials Project, much
broader element coverage than mace-mp-0's ~150k structures). It
predicts energies, forces, stresses, AND magnetic moments — the
last is unique among universal MLIPs and useful for magnetic-system
relaxations.

CHGNet is **ASE-only**. There is no LAMMPS `pair_style chgnet`. MD
runs therefore use the `runner="ase"` path through ``MLIPAgent`` (the
LAMMPS path raises `NotImplementedError` with an actionable message).
For systems where LAMMPS-side speed is required, prefer MACE via
`pair_style mliap unified`.

## planning

When to pick CHGNet over MACE:

- **Speed matters more than top-line accuracy.** CHGNet is markedly
  faster per atom than MACE-mp-0 medium on the same hardware; for
  rapid equilibration or large-cell screens it's the right default.
- **System contains magnetic elements (Fe, Co, Ni, Mn, Cr, ...).**
  CHGNet predicts per-atom magnetic moments alongside energies and
  forces, so spin-state characterization is built in.
- **Element coverage beyond MPtrj-150k is needed.** CHGNet's training
  set is ~10× larger and covers more chemistries.

When to prefer MACE-mp-0 instead:

- Accurate forces / NEB barriers / phonon-quality calculations.
- LAMMPS-side production runs (large system, MPI, long timescale).
- Systems where MACE has been benchmarked against DFT for the
  specific property of interest.

## implementation

Standard ASE-based MD with the CHGNet calculator:

```python
from ase.io.lammpsdata import read_lammps_data
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from chgnet.model.dynamics import CHGNetCalculator

atoms = read_lammps_data("system.data", style="atomic", sort_by_id=True)
atoms.calc = CHGNetCalculator()      # auto-loads the pretrained model

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

`CHGNetCalculator()` accepts a `use_device="cuda"` argument; let
`mlip_tools.generate_ase_script` set this from the agent's
`device` parameter.

## validation

CHGNet-specific accuracy thresholds (per the model card; checked
against MPtrj test split):

- **Energy MAE**: ~30 meV/atom (in-distribution)
- **Force MAE**: ~80 meV/Å (in-distribution)
- **Magnetic moment MAE**: ~0.3 μB/atom (transition metals)

Out-of-distribution warning signs:
- Forces > 5 eV/Å on equilibrium-looking structures suggest the
  system is outside MPtrj coverage — consider MACE-mp-0 or DFT
- Energy drift > 0.1 eV/atom/ps in NVE indicates the potential is
  not stable for that chemistry; tighten the timestep or switch
  potentials

For magnetic systems, sanity-check predicted moments against
known DFT values for at least one representative structure before
trusting long trajectories.

## analysis

Trajectories are standard ASE `.traj` files; use `ase.io.read`
or `ase.io.iread` with all the usual tooling. Magnetic moments
are accessible via `atoms.get_magnetic_moments()` if the
calculator was attached when the snapshot was made.

## interpretation

Common failure modes:

- **Silent OOM on GPU**: CHGNet on a 24 GB card handles up to
  ~3000 atoms in single-precision. Bigger systems force fp32
  and CPU fallback.
- **Spurious magnetic ordering**: predicted spins can lock into
  the wrong state for ambiguous magnetic systems; seed with
  initial_magmoms if you know the ground state.
- **Out-of-distribution drift**: if the system contains rare-earth
  or heavy elements, CHGNet may behave erratically; cross-check
  with MACE-mp-0 or fall back to DFT for short benchmark runs.
