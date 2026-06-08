---
description: Detection, counting, and per-object measurement of discrete objects — touching/overlapping OR dispersed/separated (grains, particles, nanoparticles, droplets, cells, bubbles, precipitates). Separates detection from instance partitioning when masks merge neighbors, and from per-object characterization (size/shape, lattice/FFT orientation, intensity) so detection is never gated on the property being measured.
---
# Overlapping / Touching Object Segmentation Skill

## overview

Segmentation of objects that touch, overlap, or are spatially connected.
Applies to any image where individual objects must be counted and measured
but appear merged in a binary mask — grains, particles, droplets, domains,
cells, bubbles, etc. The key principle is to separate detection (finding
where individual objects are) from assignment (labeling which pixels belong
to which object).

A second separation matters when the objective also asks to *characterize*
each object (a per-object property — crystalline lattice / FFT orientation,
fluorescence, composition, a size/shape class): **detection must not be
gated on that property.** Detect every object by morphology/contrast first;
measure the property per object afterward; and report the two counts
separately (objects detected vs. objects exhibiting the property). Folding
the property into detection (e.g. only "finding" particles that show lattice
fringes) silently under-counts the population.

## planning

### foundational
**Detect particles with the method that fits the image — choose by packing and
contrast (you can see it).** For **densely-packed or faint small cores**,
scale-space LoG blob detection is the right first choice — it finds each core as
its own maximum without merging. Use **`log_blob_detect`** for it: it is a
faithful SUPERSET of `skimage.feature.blob_log` (passes every native arg through,
so it equals raw `blob_log`) plus polarity handling, scale-bar masking, and
calibrated sizing for free — so prefer it over hand-calling `blob_log`. **Set
`polarity` from the image** (dark vs. bright objects) and **tune `threshold_rel`
from the overlay** (LoG over-detects at a default threshold; raise it until the
count matches what you see). For **well-separated** objects, Otsu + connected
components. For **touching** objects, SAM.

Two registered tools **add value in specific FAILURE MODES — reach for them when
your own detection mis-fires, not by default** (do not route a case to a tool
that `blob_log`/Otsu already handle well):
- **`scale_matched_blob_detect`** — use when detection **OVER-detects on a
  speckled / noisy background**: its scale-matched band-pass + band-pass-SNR gate
  reject the fine texture that watershed/Otsu/`blob_log` fragment into false
  particles (the case where naive thresholding craters). Best for SPARSE
  particles on a grainy background.
- **`log_blob_detect`** — a faithful SUPERSET of `skimage.feature.blob_log`: it
  passes every native `blob_log` arg straight through (so it equals `blob_log`
  with no annotation present) and adds opt-in conveniences — `polarity='dark'`
  inversion (`blob_log` finds only bright blobs), scale-bar masking, sigma-from-
  `object_diameter_nm`, and a calibrated nm diameter per blob. Use it instead of
  hand-calling `blob_log` to get those for free; tune `threshold_rel` and set
  `polarity` from the overlay exactly as you would with raw `blob_log`.

Both return per-object `bbox` for a per-object property step, take a `polarity`
you should **set from the image** (dark vs. bright; `auto` is only a fallback),
and expose their detection knobs to **tune from the overlay**: `log_blob_detect`
takes the native `blob_log` args directly (mainly `threshold_rel`, plus
`min_sigma`/`max_sigma` or `object_diameter_nm`+`pixel_size_nm` for the scale);
`scale_matched_blob_detect` takes a `params` dict (`k_thresh` / `snr_min`).
Re-run with adjusted values until the overlay matches the image. **A dense field
on a speckled background is the hard gap**
(band-pass merges, plain `blob_log` over-detects the speckle): there, prefer
`blob_log`/`log_blob_detect` with a raised `threshold_rel` (or a light
pre-smoothing), and verify the overlay. Use SAM / boundary routes below only for
genuinely touching objects or space-filling grains.

**Check next whether the problem actually needs instance segmentation.**
Several common cases resolve with simple classical methods before
reaching for a heavy model like SAM:

- **Well-separated objects** → Otsu + connected components is enough.
- **Objects separated by a visible boundary feature** (dark grain
  boundaries in etched metal, stained cell walls, bright domain
  edges) → use the boundary itself to separate the objects. Extract
  the boundary first — the method depends on how the boundary looks
  in the pixel data:
    - **Default: global Otsu thresholding** (`skimage.filters.threshold_otsu`)
      for clean dark/bright lines on a uniformly illuminated image.
      Simpler and usually sharper than adaptive variants; tune nothing.
    - **Adaptive thresholding** (`cv2.adaptiveThreshold` or
      `skimage.filters.threshold_local`) only when illumination varies
      noticeably across the field of view — block size and offset add
      tuning surface area that doesn't pay rent on evenly lit images.
    - **Edge detection** (Canny/Sobel) for gradient edges where there's
      no clean intensity step.
    - **Gradient magnitude or texture filters** for softer / textured
      boundaries.
    - Add **morphological closing** afterward if the boundary is broken.
  Once you have the boundary map, the two classical ways to turn it
  into labeled objects are (a) invert and run connected components on
  the interiors, or (b) use the boundary map as a watershed landscape.
  Usually sharper and faster than SAM for these images — SAM does not
  know to treat a thin boundary feature as an object separator.
- **Clean foreground-background intensity separation** → Otsu +
  connected components + morphological cleanup.

Only reach for SAM when objects genuinely touch or overlap AND no
visible boundary feature delineates them — the case where classical
approaches would merge adjacent objects into single blobs.

**Per-object characterization (decoupled from detection).** When the
objective is "detect the objects AND measure property X per object"
(e.g. *detect the nanoparticles and FFT each to get its lattice
orientation*), run it as two stages, never one:
1. **Detect ALL objects** by morphology/contrast using the decision
   tree above — independent of whether each object shows property X.
   For small, low-contrast, dispersed particles a band-pass +
   Laplacian-of-Gaussian blob detection (`skimage.feature.blob_log`)
   sized to the particle radius is usually more reliable than
   thresholding; de-duplicate detections within ~one radius.
2. **Measure X on each detected object**, and accept the per-object
   result only when it clears its own significance gate. For a
   per-particle *lattice orientation*, crop each object and take a
   windowed local FFT; report the orientation/d-spacing only when the
   first-order spot SNR exceeds a threshold, and flag the rest as
   "detected, property indeterminate" (off-zone / amorphous / too
   noisy). For a lattice/superstructure question, `fourier_reflection_map`
   (in `scilink.skills._shared.fourier_reflection`) can do the
   per-crop detection.
**Report N_detected and N_with_property as two separate numbers** — a
large gap is an expected, informative result (e.g. "12 particles
detected, 4 crystalline"), not a reason to drop the others.

### genuinely-overlapping case (no visible boundary)

When the foundational checks above rule out classical methods —
i.e. objects genuinely touch with no boundary feature you can extract —
connected component labeling on a binary mask will merge all touching
pixels into one object, so the pipeline must include a dedicated
splitting step. Within this case, in order of preference:

1. **SAM instance segmentation**: Use `run_sam_analysis` from
   `scilink.skills._shared.sam`. SAM detects individual object instances
   directly, even when they overlap, without requiring thresholding or
   binary masks. Works for any object shape. Tune via `sam_parameters`
   preset and `min_area` / `pruning_iou_threshold`. Avoid Gaussian blur
   before SAM unless noise is very high.

2. **Watershed splitting**: Create binary mask (any method) → distance
   transform → find markers (local maxima of distance transform) →
   watershed on inverted distance transform. Use when objects are
   roughly convex and SAM produces poor results (over-merged or
   over-split for the size scale). Key parameter: `min_distance` in
   `peak_local_max` should approximate the object radius.

3. **Instance detection**: Detect individual objects directly from
   the image without relying on a binary mask. For elliptical objects:
   Hough ellipse detection (`skimage.transform.hough_ellipse`) on an
   edge map (e.g., Canny). For circular objects: Hough circles
   (`cv2.HoughCircles`). Note: `cv2.fitEllipse` fits an ellipse to
   an existing contour — it cannot separate overlapping objects and
   should only be used after splitting.

4. **Contour decomposition**: Find contours of merged blobs → detect
   concavity points where objects touch (using convex hull defects) →
   split along concavity lines. Works when contact regions create
   visible indentations in the merged contour.

### advanced
When objects have different intensities (e.g., multi-phase domains),
cluster by intensity first (k-means, GMM), then apply splitting within
each cluster. Watershed markers can be improved by weighting the distance
transform with edge gradients (Sobel magnitude) so that watershed
boundaries follow real inter-object edges.

## analysis

Implementations match the cases laid out in `## planning`. Pick the
case first via the foundational decision tree there, then use the
matching implementation below.

### classical: visible boundary feature (etched grains, stained walls, …)

Extract the boundary, invert, and label the interiors. Default to
global Otsu unless illumination varies noticeably across the field of
view.

```
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk
from scipy.ndimage import label

# 1. Boundary mask. Dark boundaries → True where image < threshold.
thr = threshold_otsu(image_array)
boundary = image_array < thr          # flip the comparison if boundaries are bright

# 2. Close small gaps in the boundary so it fully encloses interiors.
boundary = binary_closing(boundary, disk(1))

# 3. binary_mask = inverse of the boundary; True where the objects are.
binary_mask = ~boundary
labels, n = label(binary_mask)
props = skimage.measure.regionprops(labels, intensity_image=image_array)
```

Use adaptive thresholding (`cv2.adaptiveThreshold` /
`skimage.filters.threshold_local`) only as a substitute for the Otsu
line above when illumination drifts across the image. Block size and
offset add tuning surface area that doesn't pay rent on uniformly lit
images.

### classical: well-separated or clean foreground/background

```
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk
from scipy.ndimage import label

thr = threshold_otsu(image_array)
binary_mask = image_array > thr       # flip if foreground is dark
binary_mask = binary_opening(binary_mask, disk(1))   # cleanup small specks
labels, n = label(binary_mask)
props = skimage.measure.regionprops(labels, intensity_image=image_array)
```

### genuinely-overlapping: SAM

Pass a 2D grayscale array or an HxWx3 RGB uint8 array. For multi-channel
images that are not RGB (e.g., 2-channel or 4-channel), pass a single
channel (e.g., `image[:,:,0]`). **Tune all numeric parameters to the
image** — the values below are syntax examples only, not recommended
defaults.

```
from scilink.skills._shared.sam import run_sam_analysis
result = run_sam_analysis(image_array, params={
    "sam_parameters": "default",     # always start from here; try "sensitive" if objects are missed
    "min_area": <tune>,              # minimum object area in pixels — set from image
    "max_area": <tune>,              # maximum object area in pixels — set from image
    "pruning_iou_threshold": <tune>  # masks with IoU above this are removed; lower = stricter, higher = keeps more overlapping objects
})
# Build labeled mask from SAM particles
labeled = np.zeros(image_array.shape[:2], dtype=np.int32)
for i, p in enumerate(result["particles"]):
    labeled[np.array(p["mask"], dtype=bool)] = i + 1
props = skimage.measure.regionprops(labeled, intensity_image=image_array)
```

Avoid Gaussian blur before SAM unless noise is very high.

### genuinely-overlapping: watershed

```
# binary_mask comes from one of the classical thresholding blocks above
distance = scipy.ndimage.distance_transform_edt(binary_mask)
markers = skimage.feature.peak_local_max(distance, min_distance=estimated_radius)
labeled_markers = scipy.ndimage.label(markers)[0]
labels = skimage.segmentation.watershed(-distance, labeled_markers, mask=binary_mask)
```

### genuinely-overlapping: ellipse detection (specific shapes only)

```
contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        # ellipse = ((cx, cy), (major, minor), angle)
```

After segmentation, filter out small fragments (area < threshold) and
border-touching objects if needed, then extract region properties.

### advanced
For highly elongated objects, distance-transform watershed tends to
over-segment because the distance ridge is flat. Use the Sobel gradient
as watershed landscape instead of inverted distance transform, with
markers still from distance transform peaks. This makes watershed
boundaries follow actual edges rather than geometric centers.

## interpretation

### foundational
After splitting, verify that the object count matches visual inspection.
Compare the size distribution to expected physical sizes — a bimodal
distribution with many small fragments suggests over-segmentation, while
too few large objects suggests under-segmentation.

## validation

### foundational
**Object count — judge recall by inspecting the raw image, both ways.**
Overlay the detections on the *original* image and check directly:
(a) visible objects with no detection mark → under-detection;
(b) detection marks on noise / background → over-detection. Both are
errors; report `N_detected` against this visual estimate (≈±20%). A
second automated detector is NOT a fix — it is just detection run again,
with the same biases (a permissive one over-detects on noise), so it adds
no independent ground truth. Where the population is small and
low-contrast the count is genuinely uncertain — **report that
uncertainty** (and the detection sensitivity used) rather than presenting
one number as exact. (The **size distribution** check below independently
guards over-detection: many fragments below ~1/4 the typical object area
signal over-segmentation.)

**Size distribution**: Should be unimodal or match expected physics.
Many fragments below 1/4 of the typical object area indicate
over-segmentation artifacts.

**Shape metrics**: Circularity and solidity should be physically
reasonable for the object type (e.g., >0.7 for droplets/bubbles,
variable for grains).

**Per-object characterization** (when the objective measures a property
per object): detection recall is judged on N_detected (the full
population), not on N_with_property — if the detected count tracks only
the objects that show the property (e.g. only the fringed particles),
detection was wrongly coupled to the property; re-detect on morphology
alone. Report N_detected and N_with_property separately.
