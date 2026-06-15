---
description: NMR relaxation fitting — spin-lattice T1 (inversion / saturation recovery) and spin-spin T2 (echo decay) from an integral-vs-delay curve. Auto-selects the recovery model from the experiment, fits mono- or stretched-exponential (β for disordered solids) and two-component recoveries, and interprets T1/T2 in terms of mobility / correlation time, including activation energy from a variable-temperature series.
technique: ["NMR relaxation", "T1", "T2", "spin-lattice relaxation", "spin-spin relaxation", "inversion recovery", "saturation recovery", "relaxometry"]
quality_gate:
  metric: r_squared
  accept_threshold: 0.97
  hard_reject_threshold: 0.80
  direction: higher_is_better
---
# NMR Relaxation Skill (T1 / T2)

## overview

Fits an NMR **relaxation curve** — peak integral vs relaxation delay (x = time,
**not** ppm) — to extract a relaxation time. This is a different modality from
spectral fitting: the whole curve is signal (no wide empty window), so the
quality metric is plain **R²**, not peak-region R².

Three experiments, identified by the **pulse program** — never guessed from the
curve shape. The pulse-program stem (standard across vendors) maps to the
`fit_relaxation` `model=` argument; an ingestion that records the model type in
metadata can supply it directly instead:

| Pulse program (typical stem) | `fit_relaxation` `model=` | Model |
|---|---|---|
| inversion recovery (`t1ir`, `*invrec*`) | `"inversion_recovery"` | `I(t) = I0·(1 − A·exp(−(t/T1)^β))` — signed, negative at short t; A≈2 (fit for imperfect inversion) |
| saturation recovery (`satrec*`, `*satr*`) | `"saturation_recovery"` | `I(t) = I0·(1 − exp(−(t/T1)^β))` |
| CPMG / Hahn-echo train | `"t2_decay"` | `I(t) = I0·exp(−(t/T2)^β)` |

β is the stretching exponent: **β = 1 mono-exponential; β < 1 = a distribution
of relaxation times**, the standard description for a disordered / glassy solid
or a quadrupolar nucleus in a solid electrolyte.

**Out of scope (v0):** T1ρ, 2D relaxation (T1–T2 correlation / ILT), diffusion
(PFG/DOSY — a different experiment), and BPP spectral-density modelling beyond a
single-τc estimate.

## planning

**Pick the model from the experiment (the pulse program), not from the data
shape.** Map the pulse-program stem to the model per the table above (or use a
model-type field if the metadata carries one) and pass it to `fit_relaxation`.
An inversion recovery is signed (negative at short t); never force its amplitude
positive.

**Decide mono vs stretched.** Start mono-exponential (β = 1). Use a **stretched
exponent (β < 1)** when a single τ underfits — a systematic residual that a mono-
exponential leaves, typical of a **disordered solid / glassy electrolyte / broad
quadrupolar line** (e.g. ²³Na, ⁷Li in a solid electrolyte). Solution-state and
clean crystalline sites are almost always mono-exponential.

**Decide one vs two components.** Two relaxation times (`n_components=2`) when
the residual of a single-component fit is systematic and the sample has two
known environments (e.g. mobile + bound, surface + bulk). Don't add a second
component to chase noise — keep it only if R² improves materially and both
populations are non-negligible.

**Variable-temperature (VT) series → activation energy.** When the same
relaxation experiment is run across temperature, fit T1/T2 at each T, then
extract the **activation energy** from the motional regime: in the fast-motion
limit 1/T1 ∝ τc and τc = τ0·exp(Ea/kT), so a plot of ln(1/T1) (or ln τc) vs 1/T
is linear with slope ±Ea/k. Report Ea (eV or kJ/mol) — for an ion conductor this
is the **hopping barrier**, the headline number.

## implementation

Call the skill tool with the recovery model from metadata:

```python
from scilink.skills.curve_fitting.nmr_relaxation.relaxation import fit_relaxation
# delays in seconds, integrals signed (negative at short t for inversion recovery)
res = fit_relaxation(delay.tolist(), integral.tolist(),
                     model=MODEL,        # from the pulse program (see table); never guess
                     stretched=False)    # set True for disordered solids
p = res["parameters"]   # T1_s (or T2_s), beta, I0, A_inversion / populations
```

If the mono-exponential residual is systematic, re-fit with `stretched=True`
(or `n_components=2`) and keep the better fit. Emit `FIT_RESULTS_JSON:` with
`fit_quality` (`r_squared`), the relaxation time in **seconds**, β, and — for a
VT series — the per-temperature T1 so the Arrhenius fit can run downstream.

## interpretation

**T1 (spin-lattice).** The time to restore equilibrium magnetisation. Physically
it tracks **motion / mobility**: T1 is minimised when the motional correlation
time τc ≈ 1/ω0 (the BPP minimum). On the **fast-motion side** (small molecules,
mobile ions, high T) shorter T1 = slower motion; on the **slow-motion side**
(rigid solids, low T) shorter T1 = faster motion — so state which regime before
reading a T1 as "more/less mobile." For ions in an electrolyte, the VT-T1
**activation energy** is the robust, regime-independent mobility metric.

**T2 (spin-spin).** Governs the homogeneous linewidth (Δν½ ≈ 1/πT2). Short T2 =
broad line = strong dipolar coupling / slow motion / rigid lattice.

**β (stretching).** β < 1 means a **distribution** of relaxation times — report a
mean/median relaxation time, not a single sharp value, and read β itself as a
disorder/heterogeneity measure (smaller β = broader distribution). `flags` in
the tool output calls out a strongly stretched fit.

**Quadrupolar nuclei.** For ²³Na/²⁷Al/⁷Li etc., T1 is usually dominated by the
**quadrupolar relaxation** mechanism (modulation of the EFG by motion), so T1
directly probes local dynamics — which is exactly why relaxation is the headline
measurement for solid-electrolyte ion mobility.

## validation

- **Metric is R², over the whole curve** (the curve is all signal). Accept ≥ 0.97;
  a recovery curve that fits worse usually means the wrong model (inversion vs
  saturation), an unconverged stretched fit, or a mis-extracted curve (e.g. an
  interleaved acquisition whose increments were not separated).
- **Inversion recovery must be signed.** If the fit forced a positive amplitude
  on data that dips negative at short delay, the model/sign is wrong.
- **T1/T2 must lie within the sampled delay window** (or be flagged as
  extrapolated). A fitted T1 longer than the longest delay, or shorter than the
  shortest, is unconstrained — widen the delay list or report it as a bound.
- **Stretched β ∈ (0, 1].** β pinned at 1 with a poor fit → try a second
  component; β ≪ 1 → quote a distribution, not a single time.
- **VT Arrhenius:** check the ln(1/T1)-vs-1/T points are linear and on one side
  of the BPP minimum before quoting Ea; a V-shape means the data straddle the
  minimum and the simple slope is invalid.
