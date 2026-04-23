# STM & Conductive AFM (low-bias) Imaging Skill

## overview

Scanning tunneling microscopy (STM) and conductive atomic force
microscopy at low bias (cAFM) produce images whose features reflect
local density of states or local conductance. Apparent "atoms" can
shift with bias voltage, tip condition, or contact geometry. Use this
skill for LDOS heterogeneity, conductance domain mapping, and
pattern-level analysis of electronic imaging data.

## planning

### foundational
The primary signal is electronic. Set the plan's `quality_criteria`
to match electronic features — LDOS heterogeneity, conductance
variation, domain structure, coherence of abundance maps. Atomic-like
periodic patterns may still be present, but include lattice-related
criteria only when the objective explicitly calls for lattice
characterization.

The right Tier 1 tool depends on what the image shows. For
heterogeneous textures or phase domains (quantum materials,
crystalline patches, spatially varying electronic order),
`run_fft_nmf_analysis` with a feature-scale window is typically
appropriate. For discrete features on a surface (molecular
adsorbates, clusters, individual defects), peak/blob detection
(`skimage.feature`) or instance segmentation (SAM) is more natural —
count, measure, and map the spatial distribution of individual
features rather than looking for periodicity that isn't there.

Classical atom-detection tools (`detect_atoms`, `detect_atoms_dcnn`)
are designed for STEM-style atomic columns and typically not the
right default for electronic imaging — apparent atoms can shift with
bias, tip state, and contact geometry. Use them when the objective
explicitly calls for lattice characterization and the image is
genuinely atomic-resolution and stable.

## validation

### foundational
Validate against the electronic signal — spatial coherence of
detected domains, physical plausibility of conductance / LDOS values,
agreement between independent electronic descriptors.
