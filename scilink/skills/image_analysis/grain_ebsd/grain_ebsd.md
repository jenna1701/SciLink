---
description: "Polycrystalline GRAIN-MAP analysis — a space-filling tessellation of grains from an EBSD IPF colour map OR grayscale electron-channeling / BSE contrast. Segments the grains, reports the equivalent-diameter size distribution, the straight-grain-boundary (annealing-twin / Sigma3) fraction and twin-segment count per grain, and IPF texture (pole fractions). Use ONLY when the field is tiled by contiguous grains. NOT for discrete particles/precipitates/indents/loops on a matrix (use overlapping_objects), single-feature geometry (use region_morphometry), or atomic-lattice images (use atomic_stem)."
technique: "EBSD (IPF), SEM channeling/ECCI/BSE grain maps, optical grain maps"
---
# Grain / EBSD Microstructure Skill

## overview

Use for **polycrystalline grain analysis**: count and size grains, quantify the
annealing-twin (straight Sigma3 boundary) fraction, count twin segments per
grain, and — for IPF colour maps — report the crystallographic texture. Two
input types:
- **IPF colour map** (RGB; grains separated by orientation colour) → `mode="ipf"`;
- **grayscale channeling / BSE** image (grains by intensity contrast) →
  `mode="gray"`. (`mode="auto"` chooses by channel variance.)

> **MANDATORY TOOL.** Use `grain_analysis` (imported below). Do NOT hand-roll
> grain segmentation or use a discrete-object/blob detector — those cannot
> express grain-boundary topology, twin-boundary straightness, or IPF texture,
> which are the actual deliverables here. The whole segmentation+metrology step
> is a single `grain_analysis(...)` call; the rest is plotting/reporting.

## planning

### foundational — calibration
Read the pixel size from metadata (authoritative) and pass it as `pixel_size`
with `pixel_unit` (EBSD/SEM grain maps are usually µm/px). Note: a `step_size` /
`pixel_size` in metadata may refer to the ORIGINAL un-downscaled raster — if the
supplied image is smaller, compute pixel size from FOV / image-width so it
matches the pixels you actually have.

### foundational — pick the mode and deliverables
The objective usually asks for several of: grain size distribution, twin
(straight-boundary) fraction, twin count per grain, texture, EBSD-vs-channeling
detectability. `grain_analysis` returns all of them in one call — report each.

### foundational — tune `boundary_sensitivity` from the overlay (with a physical anchor)
Grain segmentation is image-dependent, **especially grayscale channeling
contrast** (faint, sub-grain-textured boundaries). Run once at the default
(`boundary_sensitivity=0.5`), overlay the grain boundaries on the image, and
adjust: raise it if obvious grains are merged, lower it if single grains are
split into sub-grains. IPF colour maps are far more reliable than channeling.

**Do NOT over-tune (the common failure).** Raising `boundary_sensitivity` until
intra-grain channeling texture fragments grains into sub-grains is wrong. Anchor
the choice physically:
- **Channeling/BSE UNDER-detects boundaries vs EBSD** — many true boundaries are
  weak/invisible in channeling, and coherent twins are merged. So a channeling
  grain COUNT should be an **underestimate** (LOWER), and median grain size
  **larger or equal**, relative to a co-registered EBSD/IPF map of the same
  region. A channeling count that EXCEEDS the EBSD count, or a median grain size
  far BELOW it, means you over-tuned into sub-grain fragments — **lower**
  `boundary_sensitivity`.
- If the objective references a co-registered EBSD map, use ITS grain count /
  median size as an upper-bound anchor; otherwise sanity-check the median grain
  size against physical expectation (annealed grains are typically many µm, not
  a flood of ~1-resolution fragments).
- When unsure, prefer **slight under-segmentation** to fragmentation, and report
  the count as a lower bound with the detectability caveat — that is the correct,
  honest answer for channeling contrast.

## analysis

```python
import numpy as np
from scilink.skills._shared.grain_analysis import grain_analysis, POLES

img = np.load("data.npy")          # RGB IPF map (H,W,3) or grayscale (H,W)
res = grain_analysis(
    img,
    pixel_size=0.4, pixel_unit="um",   # from metadata
    mode="auto",                        # "ipf" / "gray" / "auto"
    boundary_sensitivity=0.5,           # tune from the overlay (see above)
    contamination="auto",               # mask dark/bright debris (gray mode)
)

n          = res["n_grains"]
diam_um    = res["grain_diameters"]               # per-grain equiv diameter
twin_frac  = res["twin_boundary_fraction"]        # straight-boundary fraction
twin_per   = res["mean_twin_segments_per_grain"]
if res["mode"] == "ipf":
    texture = res["texture"]                      # per-pole count/area fractions
    dominant = res["dominant_texture"]
```

Overlay the grain boundaries (`res["label_map"]` contours) and the detected
straight segments (`res["straight_segments"]`) on the image, and plot the
`grain_diameters` histogram. Report: n grains, equivalent-diameter stats,
twin-boundary fraction + twin segments/grain, and (IPF) the texture pole
fractions + dominant component.

## interpretation

- **Twin-boundary fraction** = straight boundary length / total boundary length,
  a proxy for coherent annealing (Sigma3) twins, which are atomically flat and
  render as straight lines. It is a **proxy**, not a misorientation measurement
  — report it as such; a value near 1 means a strongly twinned, well-annealed
  microstructure.
- **IPF texture**: `<001>`=red, `<101>`=green, `<111>`=blue (Z direction). A
  dominant pole with high area fraction indicates a preferred orientation /
  texture component; near-equal fractions indicate a random texture.
- **EBSD (IPF) vs channeling**: IPF colour segmentation is reliable. Grayscale
  channeling/BSE segmentation is **inherently less reliable** — many true grain
  boundaries are weak or invisible in channeling contrast, and coherent twins
  (which share orientation colour in IPF too) are easily merged into the parent
  grain. State this when comparing; the channeling grain count is typically an
  underestimate and the twin fraction less certain.

## validation

### foundational
- **Grain count vs visual**: overlay boundaries; the count should match a visual
  estimate within ~±25 %. Far too few ⇒ raise `boundary_sensitivity`; a field of
  tiny fragments ⇒ lower it (or stronger pre-smoothing).
- **Size distribution**: physically plausible, right-skewed for annealed
  microstructures; a spike of sub-resolution fragments ⇒ over-segmentation.
- **Twin fraction sanity**: should sit in [0,1]; the straight-segment detector is
  validated to read ~1 for straight boundaries and ~0 for smoothly curved ones.
- **Texture**: pole fractions sum to ~1; a claimed dominant component should be
  visible as a colour cluster in the IPF map.
