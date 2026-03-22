# STEM Atomic Resolution Image Analysis Skill

## overview

Aberration-corrected scanning transmission electron microscopy (STEM)
atomic-resolution image analysis. Covers HAADF-STEM and MAADF-STEM
images where individual atomic columns are resolved as bright spots on
a dark background. Applicable to any crystalline material (perovskites,
2D materials, semiconductors, oxides, metals) viewed along a zone axis.
This skill covers atomic column detection, sub-pixel position refinement,
lattice parameter extraction, sublattice separation, point defect
identification, and local structural variation analysis via sliding
FFT/NMF decomposition.

## planning

**Detection method:** Use Laplacian of Gaussian (LoG) blob detection
to locate atomic columns. Do NOT use intensity thresholding with
`peak_local_max` — it fails when columns have varying brightness
(different atomic species, thickness gradients, detector non-uniformity).
LoG is inherently intensity-normalized and robust to these variations.

**LoG sigma range:** The sigma range MUST be adapted to the atomic
column size in the specific image. To estimate column width:
1. If metadata provides pixel size (nm/pixel) and the material is known,
   compute the expected lattice spacing in pixels from known lattice
   constants. Column FWHM is typically 1/4 to 1/3 of the lattice spacing.
2. Otherwise, compute the image FFT and identify the dominant spatial
   frequency to estimate lattice spacing in pixels.
3. As a last resort, use a broad sigma sweep and refine from detected
   blob sizes.
Set `min_sigma` and `max_sigma` to bracket the estimated column FWHM.
Do not hardcode sigma values.

**Sub-pixel refinement:** Refine LoG-detected positions with 2D
Gaussian fitting. Record fit uncertainty when strain analysis is
relevant.

**Background subtraction:** When intensity varies across the field of
view, apply background subtraction before detection. A large-sigma
Gaussian blur or rolling-ball filter removes slowly varying background.

**Sublattice separation:** For multi-element compounds with distinct
Z-contrast, separate sublattices by intensity-based clustering (k-means
or GMM on fitted peak amplitudes). Determine the number of clusters
from the known crystal structure and zone axis (if available from
metadata), or use a cluster validity metric (silhouette score, BIC)
if unknown. Don't use geometric offset computation from lattice
vectors — it fails most of the time.

**Sliding FFT + NMF for structural variation:** When the image contains
a sufficient number of atoms to produce a clear FFT signal, run the
sliding FFT + NMF tool to detect any symmetry-breaking defects, phase
boundaries, ordering transitions, or other structural inhomogeneities.
Run this early in the pipeline — its findings should inform column
detection and analysis. For example, if NMF reveals distinct structural
domains, analyze each domain's columns separately with appropriate
parameters. Usage:
```python
from scilink.tools.fft_nmf import SlidingFFTNMF
# Try a range of components and pick the best decomposition
for n in [2, 3, 4]:
    fft_nmf = SlidingFFTNMF(n_components=n)
    components, abundances = fft_nmf.fit_transform(image_2d)
    # Check if components are meaningfully different
```
Do not hardcode `n_components` — the number of structurally distinct
regions depends on the image. Try 2-4 and assess whether additional
components capture real spatial variation or just noise. If all
abundance maps look spatially uniform (low coefficient of variation),
NMF is not revealing anything useful — proceed with standard analysis
without FFT guidance. Only use NMF results to inform the analysis when
they show clear spatial structure (distinct domains, boundaries, bands).

Save components and abundance maps as .npy files. Render each component
(local FFT pattern) and its corresponding abundance map (spatial
distribution) as PNGs only if they reveal meaningful structural
variation. The component image shows *what* the local structure looks
like; the abundance map shows *where* it occurs.

## analysis

Column detection typically involves: normalization, background
subtraction (when needed), LoG blob detection, and 2D Gaussian
refinement for sub-pixel precision. Lattice parameters can be extracted
from nearest-neighbor distances of detected positions. Filter
detections by rejecting peaks with low amplitude or poor Gaussian fit
quality. If using sliding FFT + NMF, run it early so findings can
inform column detection (e.g., separate domains, adjust parameters
per region).

**Lattice fitting:**
Use the median nearest-neighbor distance (not mean — robust to
outliers) to estimate the lattice constant. Fit two lattice vectors
using the pairwise displacement vectors between detected columns.
The angle between vectors should be consistent with the expected
crystal symmetry.

**Sublattice separation:**
Apply k-means or GMM (k = number of expected sublattices for the
material and zone axis) on fitted peak amplitudes. If k is unknown,
try k=2,3,4 and select using silhouette score or BIC. After clustering,
verify that sublattice populations are reasonable for the crystal
structure.

**Defect identification:**
- Vacancies: ideal lattice site with no detected column within 0.4x
  the lattice spacing. Restrict search to the convex hull of detected
  positions to avoid false positives at image edges.
- Substitutional impurities: columns whose amplitude deviates
  significantly from their sublattice mean (beyond 2.5 sigma).
- Verify each vacancy candidate with a forced Gaussian fit at the
  expected position. If fitted amplitude exceeds 25% of the sublattice
  median, it is not a true vacancy.

**Sliding FFT + NMF (when used):**
Run on the full image early in the pipeline. Save as .npy:
`nmf_components.npy` and `nmf_abundances.npy`. Render as PNG any
component + abundance map pair that reveals spatial variation in
local structure. Include these visualizations alongside the main
atom detection visualization.

## interpretation

**HAADF intensity and atomic number:** In HAADF-STEM, image intensity
scales approximately as Z^1.6-2 (Z = atomic number). Brighter columns
contain heavier atoms. Use this to assign chemical identity to
sublattices when the sample composition is known.

**Lattice parameters:** Compare measured lattice parameters against
known bulk values for the material. Deviations may indicate strain
(epitaxial, compositional), non-stoichiometry, or incorrect zone axis
identification. Report lattice parameters in both pixels and physical
units when spatial calibration is available.

**Strain and distortion:** Only report lattice distortions when
displacements exceed 3x the position fit uncertainty. Distinguish
between: (a) deviation from the fitted lattice (local disorder relative
to the average structure) and (b) deviation from the known ideal
lattice (true strain relative to bulk). Least-squares lattice fitting
absorbs the mean strain into the fitted vectors — use known bulk lattice
constants when available to quantify absolute strain.

**NMF components:** Each NMF component represents a distinct local
FFT pattern. Interpret by examining the component image (which spatial
frequencies are present) and its abundance map (where this pattern
dominates). Possible interpretations include: different crystallographic
phases, domain orientations, stacking sequences, ordering variants, or
regions of varying crystalline quality.

**Vacancy concentration:** Typical vacancy concentrations in pristine
crystalline materials are 0.01-1%. Concentrations above 5-10% in an
apparently crystalline image usually indicate a detection or lattice
fitting error. However, if the sample metadata indicates modification
(e.g., ion irradiation, heavy electron beam exposure, high-temperature
quenching) that is known to create vacancies, higher concentrations
may be physically real — interpret in context of the sample history.

## validation

**Column detection completeness:** The number of detected columns
should be consistent with the image area divided by the unit cell area.
If the ratio of ideal lattice sites to detected columns is outside
0.85-1.15, the detection or lattice fit is unreliable.

**Sublattice balance:** For compounds with equal-multiplicity sublattices
sublattice populations should be approximately equal. A ratio below
0.5 or above 2.0 indicates failed sublattice separation. Exception: heavy substitutional doping can shift column
intensities between sublattice clusters — if the sample metadata
indicates significant doping, allow for less balanced populations and
consider using more than 2 clusters to separate doped sites.

**Lattice fit residual:** Mean residual (distance from each detected
column to its nearest ideal lattice site) should be below 0.3x the
lattice spacing. Higher residuals indicate poor lattice fitting or
significant real disorder.

**Gaussian fit quality:** Reject columns where the 2D Gaussian fit
R-squared is below 0.7 or where fitted sigma exceeds 2x the median
sigma (likely fitting two merged columns as one).

**Nearest-neighbor distance consistency:** The coefficient of variation
(std/mean) of nearest-neighbor distances should be below 15% for a
well-ordered crystal. Higher values suggest over- or under-detection,
or genuine disorder.

**Vacancy verification:** Every vacancy candidate must pass a forced
Gaussian fit check. Report the forced-fit amplitude as a fraction of
the sublattice median to quantify confidence.
