"""Scale-space (LoG) detection of DENSE, faint, small particles.

The companion to ``scale_matched_blob_detect``: where the band-pass + watershed
detector excels at sparse/medium particles on a noisy background, it *merges*
densely-packed touching cores (a distance transform of a dense cluster is one
plateau) and its precision gates reject faint cores — it caps out well below the
true count on a dense, low-contrast field of small cores.

This tool instead uses ``skimage.feature.blob_log`` — a multi-scale Laplacian-of-
Gaussian that finds EACH particle as its own scale-space maximum, so densely-
packed and faint cores are detected individually. It is sized from the known
object diameter, selects intensity polarity (set explicitly, or 'auto'
fallback), masks burned-in annotations,
and measures each diameter on the original image (shared with the band-pass tool).
"""
import numpy as np
from skimage.feature import blob_log
from skimage.filters import gaussian
from .blob_detect import _norm, _mask_annotation, _measure_diameter


def log_blob_detect(image_array, object_diameter_nm, pixel_size_nm,
                    polarity="auto", params=None):
    """Thin, faithful wrapper around skimage.feature.blob_log.

    It passes `blob_log` through unchanged (its `threshold_rel` / sigma range /
    overlap are exposed for the agent to tune), and adds ONLY the value-adds
    `blob_log` lacks: intensity-polarity selection, scale-bar/annotation masking,
    a loose object-scale size filter, and a calibrated diameter for each blob.
    It deliberately does NOT add a contrast/SNR gate — that gate is what made an
    earlier version under-detect faint cores and trail raw `blob_log`."""
    g = np.asarray(image_array, float)
    if g.ndim == 3:
        g = g[..., :3].mean(-1) if g.shape[2] in (3, 4) else g[..., 0]
    H, W = g.shape
    px = float(pixel_size_nm)
    radius_px = (float(object_diameter_nm) / px) / 2.0
    sig = radius_px / np.sqrt(2.0)               # LoG sigma for the object radius
    p = dict(sigma_lo=0.6, sigma_hi=1.6, num_sigma=5, threshold_rel=0.1,
             overlap=0.5, d_min_nm=0.3 * object_diameter_nm,
             d_max_nm=3.0 * object_diameter_nm)
    p.update(params or {})

    excl = _mask_annotation(g)
    n = _norm(g)
    order = [polarity] if polarity in ("bright", "dark") else ["bright", "dark"]

    best = None
    for pol in order:
        nn = n if pol == "bright" else 1.0 - n     # blob_log finds bright maxima
        img = nn.copy()
        if excl.any():
            img[excl] = np.median(nn[~excl])        # blank annotation region
        flat = nn - gaussian(nn, radius_px * 1.6)   # for calibrated sizing only
        locbg = np.median(flat[~excl]) if (~excl).any() else float(np.median(flat))
        blobs = blob_log(img, min_sigma=max(1.0, sig * p["sigma_lo"]),
                         max_sigma=sig * p["sigma_hi"], num_sigma=p["num_sigma"],
                         threshold=None, threshold_rel=p["threshold_rel"],
                         overlap=p["overlap"], exclude_border=False)
        objs, contrasts = [], []
        for y, x, s in blobs:
            y, x = int(y), int(x)
            if excl[y, x]:
                continue
            r_px = s * np.sqrt(2.0)
            d_meas = _measure_diameter(flat, y, x, r_px, px, nominal_r_px=radius_px)
            d_nm = d_meas if d_meas else 2.0 * r_px * px
            if d_nm < p["d_min_nm"] or d_nm > p["d_max_nm"]:
                continue
            border = x < radius_px or y < radius_px or x > W - radius_px or y > H - radius_px
            R = int(r_px)
            objs.append(dict(cy=float(y), cx=float(x), diameter_nm=round(float(d_nm), 3),
                             border=bool(border),
                             bbox=(max(0, y - R), max(0, x - R), min(H, y + R), min(W, x + R))))
            contrasts.append(float(flat[y, x] - locbg))
        # Polarity selection (NOT a per-blob gate): the CORRECT polarity has the
        # objects as genuinely high-contrast bright maxima, so its MEDIAN per-blob
        # contrast is higher. Median (not a sum) is robust to the wrong polarity
        # inflating its score by detecting many weak speckle blobs.
        pol_score = float(np.median(contrasts)) if contrasts else -1e9
        if best is None or pol_score > best[0]:
            best = (pol_score, pol, objs)
    _, pol_used, objs = best

    if not objs:
        return dict(n_detected=0, polarity_used=pol_used,
                    annotation_masked=bool(excl.any()), objects=[],
                    note="No blobs at the object scale; check object_diameter_nm / pixel_size_nm.")
    diam = [o["diameter_nm"] for o in objs]
    return dict(
        n_detected=len(objs),
        n_interior=sum(not o["border"] for o in objs),
        polarity_used=pol_used,
        annotation_masked=bool(excl.any()),
        diameters_nm=diam,
        diameter_median_nm=round(float(np.median(diam)), 3),
        diameter_mean_nm=round(float(np.mean(diam)), 3),
        diameter_std_nm=round(float(np.std(diam)), 3),
        objects=objs,
    )


from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="log_blob_detect",
    description=(
        "Scale-space (Laplacian-of-Gaussian) detection of DENSE, faint, small "
        "particles — the companion to scale_matched_blob_detect (band-pass + "
        "watershed), which merges densely-packed cores and under-detects them. "
        "blob_log finds each particle as its own scale-space maximum."
    ),
    import_line="from scilink.skills._shared.log_blob import log_blob_detect",
    signature=("log_blob_detect(image_array, object_diameter_nm, pixel_size_nm, "
               "polarity='auto', params=None) -> dict"),
    agents=["image_analysis"],
    when_to_use=(
        "Use to DETECT / COUNT / SIZE small particles that are DENSELY PACKED "
        "and/or FAINT (low contrast) — closely-spaced small cores on a crowded "
        "field — where a region-growing detector "
        "(scale_matched_blob_detect, watershed, SAM) merges touching cores and "
        "under-counts. Reach for scale_matched_blob_detect instead when particles "
        "are SPARSE/well-separated on a speckled background (it adds stronger "
        "speckle rejection); reach here when the field is crowded with many small "
        "low-contrast cores. Returns per-object bbox for a downstream property step."
    ),
    parameters={
        "object_diameter_nm": {"type": "number", "description":
            "Approximate particle diameter in nm — sets the LoG scale."},
        "pixel_size_nm": {"type": "number", "description": "nm per pixel (calibration)."},
        "polarity": {"type": "string", "description":
            "SET THIS from the image — you can see whether the particles are darker "
            "('dark') or brighter ('bright') than the background. 'auto' is only a "
            "fallback and can mis-pick on dense fields where both the objects and "
            "the gaps between them are strong features, so prefer setting it "
            "explicitly."},
        "params": {"type": "object", "description":
            "Tunable knobs — ADJUST these from what you see in the result vs the "
            "image; the defaults are a starting point, not a fixed answer: "
            "* threshold_rel (LoG sensitivity, default 0.06): LOWER it (e.g. 0.03, "
            "0.015) if faint/dense cores are clearly MISSED; RAISE it (e.g. 0.1) if "
            "background fluctuations are being detected as cores. * snr_min "
            "(per-blob contrast gate, default 2.0): raise to drop low-contrast "
            "false positives, lower to keep faint cores. * overlap (default 0.5): "
            "lower if touching cores are being merged into one. * d_min_nm / "
            "d_max_nm (size filter, default 0.3x/3x): tighten to a known size band. "
            "Re-run with adjusted values until the overlay matches the image — the "
            "'right' sensitivity is image-specific, so do not trust one default."},
    },
    required=["image_array", "object_diameter_nm", "pixel_size_nm"],
    returns=(
        "dict with 'n_detected', 'n_interior', 'polarity_used', 'annotation_masked', "
        "'diameters_nm' and 'diameter_median/mean/std_nm', and 'objects' (list of "
        "{cy, cx, diameter_nm, border, bbox=(y0,x0,y1,x1)}). If none: 'note'."
    ),
    example=(
        "res = log_blob_detect(image, object_diameter_nm=7.0, pixel_size_nm=0.11)\n"
        "print(res['n_detected'], 'median d =', res.get('diameter_median_nm'))\n"
        "# lower threshold_rel if faint cores are missed:\n"
        "# res = log_blob_detect(image, 7.0, 0.11, params={'threshold_rel': 0.03})"
    ),
)
