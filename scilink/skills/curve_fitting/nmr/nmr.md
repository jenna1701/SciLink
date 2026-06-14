---
description: 1D NMR curve fitting (solution and solid-state MAS) for ¹H, ¹³C, ¹⁹F, ³¹P, ²³Na, ²⁷Al, ¹⁷O, ⁶⁷Zn and related nuclei — peak deconvolution with Voigt/pseudo-Voigt lineshapes, rolling-baseline correction, phase/sign handling, MAS spinning-sideband awareness, and second-order quadrupolar central-transition lineshape fitting (C_Q, η_Q) for solid half-integer nuclei.
technique: [NMR, "nuclear magnetic resonance", "solid-state NMR", "ssNMR", "MAS NMR", "magic-angle spinning"]
quality_gate:
  metric: peak_region_r2
  accept_threshold: 0.85
  hard_reject_threshold: 0.30
  direction: higher_is_better
---
# NMR Curve Fitting Skill (1D, solution + solid-state MAS)

## overview

One-dimensional NMR spectra, processed and Fourier-transformed (a real
spectrum vs. chemical shift in ppm, decreasing left→right). Covers both
**solution-state** (narrow, near-Lorentzian lines) and **solid-state MAS**
(broader lines, spinning sidebands, and — for half-integer quadrupolar nuclei —
second-order quadrupolar lineshapes). Nuclei seen here: ¹H, ¹³C, ¹⁹F, ³¹P
(spin-½) and ²³Na, ²⁷Al, ¹⁷O, ⁶⁷Zn (half-integer quadrupolar).

**The single most important triage is solution vs. solid**, because it
dictates the lineshape:

- **Solution-state** (incl. quadrupolar nuclei like ¹⁷O/²³Na/⁶⁷Zn in liquids):
  fast molecular tumbling averages anisotropic interactions to zero, so every
  resonance is a near-perfect **Lorentzian** (or Voigt with a small Gaussian
  shim contribution). Fit a sum of Voigts. Do NOT use the quadrupolar lineshape.
- **Solid-state MAS**: lines are broadened by residual anisotropy; spin-½ nuclei
  give Voigt-like centrebands flanked by **spinning sidebands**; half-integer
  quadrupolar nuclei show an asymmetric **second-order quadrupolar
  central-transition** lineshape that carries C_Q and η_Q.

**Out of scope (v0):** 2D experiments; J-coupling multiplet structure (treat
resolved multiplets as separate peaks); satellite transitions, MQMAS, CSA
tensor extraction from sideband intensities; relaxation/T1–T2 series (a separate
"integral vs. delay" task, not a single-spectrum fit); raw-FID reprocessing
(this skill works on the processed spectrum).

## planning

**Hard prerequisite: the Larmor frequency of the observed nucleus (MHz).** It
sets the ppm↔Hz conversion (linewidths, sideband spacing) and the ν_Q²/ν_L
scaling of quadrupolar lineshapes. Take it from the spectrometer metadata
(`spectrometer_frequency_MHz`); never guess. The nucleus identity
(`nucleus`) and, for MAS data, the spinning rate (from the acquisition metadata
or the experiment title, in kHz) are the other key metadata.

**Step 1 — Check the phase / sign.** NMR spectra can come phased with a
negative-going peak (a 180° phase error, or the real part of an unphased
spectrum). If the dominant feature is a **dip** below baseline, fit a *signed*
amplitude (allow negative) or re-phase — never let a baseline-fitter "explain"
an inverted peak as background.

**Distinguish a real resonance from an instrumental artifact by WIDTH, and do
not discard a real one.** A feature that rises a few × the noise level and spans
**multiple points** (finite width in ppm) is a genuine resonance — fit it
(re-phasing or signed-amplitude if inverted), even at low SNR. Only a
**single-point** spike sitting exactly at the carrier (x ≈ 0 ppm) is a DC/centre
glitch to be masked; an edge spike at the spectrum boundary is a fold artifact.
A weak, broad, or inverted line (common for low-γ quadrupolar nuclei like ⁶⁷Zn,
²⁵Mg, ³³S near their detection limit) is still a resonance — recover it rather
than declaring "no signal" and fitting baseline only.

**Step 2 — Correct a rolling baseline BEFORE fitting.** Processed spectra
(especially ¹H MAS) frequently carry a broad oscillating baseline distortion
(first-point / digital-filter / probe-background artifact — wide humps and deep
lobes far from any real resonance). This is *not* chemistry and a peak model
cannot absorb it. Identify signal-free regions (away from all peaks and
sidebands), fit a low-order polynomial **or** a smoothing spline through only
those regions, and subtract. Re-baselining is sample-dependent: some spectra are
flat and need none. (See Implementation for the recipe.)

**Step 3 — Choose the lineshape by regime:**

| Situation | Model |
|---|---|
| Solution-state, any nucleus | Sum of **Voigt** (Lorentzian-dominant); use a pure Lorentzian only if the Gaussian width refines to ~0 |
| Solid MAS, spin-½ (¹³C, ³¹P, ¹⁹F, ¹H) | Sum of **Voigt** centrebands; account for sidebands (Step 4) |
| Solid MAS, half-integer quadrupolar, **symmetric narrow** line | **pseudo-Voigt** — and report FWHM, *not* C_Q (see below) |
| Solid MAS, half-integer quadrupolar, **visibly asymmetric / broad** line | second-order quadrupolar central-transition lineshape via `fit_quad_ct` |

Default to **Voigt over pure Lorentzian** — real NMR lines carry a Gaussian
(shim/disorder) contribution, and a pure Lorentzian over-peaks the apex.

**Step 4 — Spinning sidebands (MAS only).** A regular comb of weak peaks spaced
by exactly the **MAS rate** (ν_r in Hz → ν_r/ν_L ppm) flanking each centreband
is spinning sidebands, *not* distinct chemical sites. Two valid strategies: (a)
crop to the centreband and fit only that, or (b) fit the centreband plus the
sidebands as satellites at ±k·ν_r sharing the centreband's shape. **Either way,
any integrated-intensity / quantification claim must include sideband intensity**
— excluding it undercounts. Verify the spacing equals the known MAS rate.

**Step 5 — When to use the quadrupolar lineshape (and when not).** The
central-transition second-order lineshape pins C_Q and η_Q **only when it is
resolved** — i.e. the quadrupolar broadening clearly exceeds the
instrumental/disorder broadening, giving a visibly asymmetric line with
"horn"+"foot" structure. For a narrow, near-symmetric quadrupolar line, C_Q and
the linewidth are **degenerate** (a smaller C_Q + more broadening fits equally
well); `fit_quad_ct` returns a `Cq_resolved=False` flag in that case — then
treat C_Q as an upper bound and report the pseudo-Voigt FWHM as the primary
descriptor, recommending MQMAS / satellite-transition data to pin C_Q.

**Series fits (composition / temperature / treatment ladders).** Lock the model
(number of sites, lineshape, baseline strategy) across the series; let
shift/width/amplitude (and C_Q/η_Q where resolved) float per spectrum. The
scientific signal is then a single per-sample column (e.g. δ_iso vs. x,
linewidth vs. T) read across the series.

## implementation

**CRITICAL ordering: metadata → phase/sign → baseline → lineshape fit.**

**Rolling-baseline correction recipe** (apply when Step 2 flags a distorted
baseline; skip for a flat one):

```python
import numpy as np
from scipy.interpolate import UnivariateSpline

# x = ppm (ascending), y = intensity. Identify signal-free points: those below a
# robust threshold over a median-smoothed |y|, EXCLUDING a window around every
# peak/sideband. A spline through only those points captures the broad roll.
ynorm = np.abs(y - np.median(y))
thr = np.median(ynorm) + 2.0 * 1.4826 * np.median(np.abs(ynorm - np.median(ynorm)))
free = ynorm < thr                      # signal-free mask (coarse)
# widen exclusion around contiguous signal regions if needed, then:
spl = UnivariateSpline(x[free], y[free], k=3, s=len(x[free]) * np.var(y[free]))
y_corr = y - spl(x)                      # baseline-subtracted spectrum
```

A low-order polynomial (`np.polyfit` over `x[free]`, degree 2–4) is the simpler
alternative when the roll is gentle. Validate by eye: the corrected baseline
should be flat and centred on zero away from peaks.

**Voigt multi-peak fitting** (solution, and solid spin-½ centrebands). When the
spectrum has two or more overlapping environments (a shoulder, a sharp+broad
pair, a multi-site cluster), do NOT guess the component count — let the data
choose it with the skill tool, which adds peaks at the largest residual and
keeps each only if it materially improves the peak-region R² and is physically
distinct:

```python
from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt
# pass mas_rate_ppm (= spinning_rate_Hz/ν_L) so sidebands aren't fit as sites;
# tune improve_thresh down to catch a faint shoulder, or min_amp_snr up if noisy.
res = fit_multipeak_voigt(x.tolist(), y.tolist(), baseline=baseline.tolist(),
                          mas_rate_ppm=MAS_PPM)   # see the tool's parameter docs
```

For a single, clearly isolated symmetric line a one-component Voigt is fine.
Allow signed amplitudes if Step 1 flagged an inverted peak. Convert fitted widths
to Hz with the Larmor frequency for reporting. (Edge case the tool does not
robustly separate: two peaks at the *same* shift differing only in width
— a sharp+broad co-centred pair; fit those manually if present.)

**Routing — which fitter:** one symmetric line → single Voigt; several distinct
chemical shifts → `fit_multipeak_voigt`; a single *asymmetric quadrupolar*
powder line → `fit_quad_ct` (do NOT mimic a quadrupolar lineshape with a stack
of Voigts).

**Let the residual drive escalation — don't stop at a single component while
structure remains.** After any fit, inspect the residual *within the peak
region* (not the noise-only wings). A systematic, above-noise residual there — a
shoulder, a one-sided lean, a secondary bump, a sharp cusp the broad model
undershoots — means the model is incomplete, even if the global R² looks high.
Escalate: for a peak whose residual shows an extra *chemical environment*, hand
it to `fit_multipeak_voigt`; for a half-integer quadrupolar line whose residual
shows the characteristic asymmetric foot, try `fit_quad_ct` (one quadrupolar
site) before adding Voigts. This is residual-driven, not nucleus-driven — apply
it whenever a single symmetric component leaves real structure behind.

**Second-order quadrupolar central transition** (solid half-integer, resolved
asymmetric line): call the skill tool — it does the orientation-averaged powder
lineshape, a multi-start (C_Q, η) search to avoid local minima, and returns a
reliability flag:

```python
from scilink.skills.curve_fitting.nmr.quadrupolar import fit_quad_ct
# crop to the central-transition region first; pass the nucleus Larmor freq.
res = fit_quad_ct(ppm_roi.tolist(), y_roi.tolist(), nu_L_MHz=NU_L, I=1.5, mas=True)
p, d = res["parameters"], res["derived"]
# Honor d["Cq_resolved"]: if False, report p["Cq_MHz"] as an upper bound only.
```

**Quality metric (MANDATORY — the gate scores by `peak_region_r2`).** NMR's
wide digitized window makes a whole-window R² meaningless for a narrow peak
(noise dominates the variance), so after fitting, compute the peak-region R²
over the signal region and report it — the run is hard-rejected if the metric is
missing:

```python
from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
# y is the (baseline-corrected or raw) data fit to; y_fit is the model on the
# same x; baseline is the fitted baseline array (or omit if already subtracted).
q = peak_region_r2(x.tolist(), y.tolist(), y_fit.tolist(), baseline=baseline.tolist())
# q["peak_region_r2"] is the gate metric; q["r_squared"] is the global value.
```

Emit `FIT_RESULTS_JSON:` with `fit_quality` containing **`peak_region_r2`** (the
gate metric) and the global `r_squared`, plus per-peak δ (ppm) and width
(ppm + Hz), the lineshape/model used, and — for quadrupolar fits — C_Q, η_Q,
P_Q, and the `Cq_resolved` flag.

## interpretation

**Chemical-shift reference ranges** (general, vs the IUPAC primary references:
¹H/¹³C vs TMS; ¹⁹F vs CFCl₃; ³¹P vs 85% H₃PO₄; ²³Na vs 1 M NaCl(aq); ¹⁷O vs
H₂O; ⁶⁷Zn vs 1 M Zn(NO₃)₂). The "common features" are illustrative, not
assumptions — assign from the actual sample chemistry, not this list:

| Nucleus | Typical range | Common features (illustrative) |
|---|---|---|
| ¹H | 0–15 ppm | aliphatic 0–3, OH/NH 1–6, H₂O/alcohol 1–5, aromatic 6–9, acid/aldehyde 9–12 |
| ¹³C | 0–220 ppm | aliphatic 0–60, C–O 50–90, aromatic/alkene 100–150, carbonyl/carboxyl 160–210 |
| ¹⁹F | −60 to −230 ppm | CF₃ ≈ −60 to −80, CF₂/CF higher field, aromatic-F ≈ −100 to −170 |
| ³¹P | −200 to +250 ppm | phosphate/organophosphate near 0 ± 30; speciation often via small δ shifts |
| ²³Na | −20 to +15 ppm | aqueous Na⁺ ≈ 0 (reference); oxide/coordinated Na typically 0 to −15 |
| ²⁷Al | −20 to +120 ppm | octahedral Al ≈ 0–15, tetrahedral Al ≈ 55–80 (coordination diagnostic) |
| ¹⁷O | −50 to +1100 ppm | H₂O/D₂O 0; M–O–M / carbonyl / oxo much higher |
| ⁶⁷Zn | −20 to +300 ppm | aqueous Zn²⁺ ≈ 0; shift tracks coordination (tetrahedral vs octahedral) |

**Linewidth.** Report FWHM in both ppm and Hz (Hz = ppm × ν_L). In solution,
homogeneous T₂* ≈ 1/(π·FWHM_Hz); a Gaussian contribution (Voigt) signals shim or
unresolved-coupling broadening rather than relaxation. In solids, linewidth
reflects disorder / dipolar coupling / residual CSA, not T₂.

**Quadrupolar parameters (when `Cq_resolved=True`).** C_Q (MHz) measures the EFG
magnitude at the nucleus — larger = more distorted coordination; η_Q ∈ [0,1] is
the EFG asymmetry (0 = axial). The observed centre of gravity is shifted from
δ_iso(CS) by the quadrupole-induced shift δ_QIS (negative, ∝ 1/ν_L²) returned in
`derived` — quote δ_iso(CS), not the apparent peak, as the chemical shift. If
`Cq_resolved=False`, do not report C_Q as a measurement.

**Sidebands → CSA / quadrupolar product.** When sidebands are fit, their
intensity envelope encodes the CSA (spin-½) or the satellite/CSA interplay
(quadrupolar); v0 reports the manifold but defers tensor extraction.

## validation

- **Quality is `peak_region_r2`, not the whole-window R².** NMR's enormous
  dynamic range and wide digitized axis make a whole-spectrum R² noise-dominated:
  a correct fit of a narrow peak scores near 0. The gate therefore scores the
  peak-region R² (signal region only) reported by `peak_region_r2`. A low *global*
  R² with a healthy `peak_region_r2` and flat, noise-level residuals across the
  peak is an *acceptable* fit, not a failure. A low `peak_region_r2` means the fit
  is genuinely wrong *where the signal is* (missed/extra peak, wrong shape) —
  that is a real failure.
- **Sideband spacing must equal the MAS rate.** If fitted "extra peaks" sit at
  ±k·(ν_r/ν_L) ppm, they are sidebands — relabel them and fold their intensity
  into any quantification. If the spacing does NOT match ν_r, they are real
  resonances.
- **Integrated intensity includes sidebands.** A quantification (site
  populations, relative concentrations) that sums only centrebands undercounts
  when sidebands carry appreciable intensity.
- **Phase sanity.** A fit that required a large negative baseline plus a positive
  peak (or vice versa) to match an inverted line indicates a phase problem;
  prefer a signed-amplitude fit and flag for re-phasing.
- **Quadrupolar reliability.** Honor `Cq_resolved` from `fit_quad_ct`: report C_Q
  as a measurement only when resolved; otherwise as an upper bound with an
  MQMAS/satellite recommendation. Watch for η railing to 0 or 1 (poorly
  determined) — reported in `reliability_flags`.
- **Cross-series consistency.** In a composition/temperature ladder, δ_iso and
  linewidth should vary smoothly; an outlier spectrum usually means a
  mis-assigned sideband, an uncorrected baseline, or a phase flip — re-inspect
  rather than accept a discontinuity.
