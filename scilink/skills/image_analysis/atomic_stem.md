# STEM Atomic Resolution Image Analysis Skill

## overview

Atomic-resolution STEM image analysis (HAADF, MAADF). Individual atomic
columns are resolved as bright spots on a dark background. Applicable
to any crystalline material viewed along a zone axis. Covers column
detection, sublattice separation, lattice characterization, defect
identification, and structural variation analysis.

## planning

**Detection:** Choose a detection method that reliably finds atomic
columns across the full image despite intensity variations (different
species, thickness gradients). Background subtraction or bandpass
filtering before detection helps with non-uniform illumination.
Refine detected positions with 2D Gaussian fitting for sub-pixel
precision.

**Sublattice separation:** Intensity-based clustering alone is not
sufficient for complex structures. For layered materials (perovskites,
cuprates, etc.), columns of different species form distinct rows or
planes. Use both intensity AND position within the unit cell to assign
sublattices — identify which rows/layers correspond to which atomic
species based on the known crystal structure. Verify that the
assignment produces the correct stoichiometric ratios.

**Unit cell identification:** The nearest-neighbor distance is NOT the
lattice parameter for complex structures. For materials with multiple
sublattices (e.g., perovskites, layered compounds), the true unit cell
repeat may be 2x, 3x, or more of the shortest column spacing. Use
the FFT to identify the full periodicity, or count how many distinct
column rows/intensities repeat along each direction to determine the
true unit cell.

## analysis

Column detection typically involves: normalization, background
subtraction or bandpass filtering, blob/peak detection, and 2D
Gaussian refinement. Lattice parameters should be determined from
the FFT (which directly shows the periodicity) rather than solely
from nearest-neighbor distances (which may not capture the full unit
cell for complex structures).

**Sublattice separation:** Combine intensity clustering with
positional analysis. After detecting all columns and fitting lattice
vectors, project each column position onto the unit cell to determine
its fractional coordinates. Columns at the same fractional position
belong to the same sublattice. Verify by checking that each sublattice
has consistent intensity (bright = heavy atoms in HAADF).

**Defect identification:** Compare detected positions to ideal lattice
sites. Restrict vacancy search to the interior of the detected region
to avoid edge false positives. Verify vacancy candidates with forced
Gaussian fit at expected position.

**Structural anomalies:** After fitting the ideal lattice, examine the
spatial distribution of displacements, intensity variations, and local
lattice parameter changes across the image. Regions where these
quantities deviate systematically from the bulk may indicate structural
features worth reporting — let the data guide the interpretation rather
than searching for specific defect types.

## interpretation

**HAADF intensity:** Scales as ~Z^1.6-2. Brighter columns contain
heavier atoms. Use to assign chemical identity when composition is
known.

**Lattice parameters:** Compare against known bulk values. Report in
both pixels and physical units when calibration is available.

**Strain:** Only report distortions exceeding the position fit
uncertainty. Distinguish fitted-lattice residuals (local disorder)
from deviations against known ideal lattice (true strain).
Least-squares fitting absorbs mean strain — use known lattice
constants when available.

**Vacancy concentration:** In pristine crystals, typically 0.01-1%.
Above 5-10% usually indicates detection or fitting error, unless the
sample was intentionally modified (irradiation, beam damage, quenching).

## validation

**Detection completeness:** Detected vs expected column count (from
image area and unit cell) should be within 0.85-1.15.

**Sublattice populations:** Should match expected stoichiometry for
the material. Heavy doping can shift intensities between clusters.

**Lattice fit residual:** Below 0.3x the lattice spacing.

**NN distance consistency:** CV below 15% for well-ordered crystals.

**Unit cell sanity:** Measured lattice parameters should be close to
known bulk values when spatial calibration is available. If c/a or
a/b ratios differ greatly from expected, the unit cell identification
is likely wrong.
