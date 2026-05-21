---
description: XRD structure matching — query crystal-structure databases (Materials Project, local CIF), simulate kinematic patterns, score against experiment by R-factor and Rwp.
---
# XRD Structure Matching Skill

## overview

Identify a crystalline phase from an experimental X-ray diffraction (XRD)
pattern by matching against database structures. Three tools chain together
to give an end-to-end identification workflow:

- `search_structures` — multi-backend database query (Materials Project,
  local CIF directory, COD). Returns ranked candidates with materialized
  CIF paths.
- `simulate_xrd_pattern` — kinematic XRD pattern from a CIF via pymatgen's
  XRDCalculator (CuKa default; any wavelength).
- `score_xrd_match` — R-factor / Rwp / cosine similarity between simulated
  and experimental patterns, with an accept/marginal/reject verdict.

The workflow supports two invocation patterns: **pre-fit** (chemistry is
hypothesized up front — query DB, simulate, score) and **post-fit** (peak-
fit experimental first to extract lattice parameters, then use those to
filter the DB).

This skill requires `pymatgen-analysis-diffraction` (install via
`pip install scilink[structure-matching]`). The skill degrades gracefully
when the package is absent — `search_structures` still works, but
`simulate_xrd_pattern` raises a clear RuntimeError.

## planning

**Two invocation patterns. Pick one — never both in the same plan.**

**Pre-fit pattern** — use when the user/metadata names a chemistry
hypothesis (e.g. "this is a TiO2 sample"):

1. `search_structures(query={"chemistry": [...], "top_n": 5})` — pick
   top_n ≤ 10 to bound simulation cost.
2. For each candidate, `simulate_xrd_pattern(candidate.structure_path)`
   on the same wavelength as the experiment.
3. `score_xrd_match` each simulated pattern against the experimental data.
4. Rank candidates by R-factor; report the best plus runners-up.

**Post-fit pattern** — use when the user provides no chemistry hypothesis
and you need to identify the phase from peak positions alone:

1. Peak-fit the experimental pattern (Lorentzian or pseudo-Voigt peaks
   with a low-order polynomial background) to extract peak positions.
2. Estimate lattice parameters from the strongest peaks via Bragg's law
   for a guessed crystal system (cubic is the simplest first guess; for
   non-cubic add the relevant peak indexing).
3. `search_structures(query={"chemistry": [inferred from metadata or
   left broad], "lattice_param_ranges": {"a": (a_est-0.2, a_est+0.2)}})`
   — narrow the search with extracted parameters.
4. Steps 2-4 of the pre-fit pattern.

**Bounding the candidate count.** Always set `top_n` between 3 and 10.
More than 10 simulations per spectrum bloats runtime without
adding identification certainty — if the answer isn't in the top 5, it's
usually a sign the query chemistry is wrong, not that you need more
candidates.

**Wavelength selection.** Default to CuKa unless the experiment metadata
says otherwise. MoKa is common for high-2θ work. Mismatched wavelength
between simulation and experiment will yield uniformly bad R-factors for
all candidates — if every candidate scores "reject" with a large constant
offset in peak positions, suspect a wavelength mismatch first.

## analysis

**CRITICAL: structure-matching workflow.** The per-item analysis script
must follow this exact sequence for the pre-fit pattern:

1. Load experimental 2-theta, intensity arrays.
2. Call `search_structures` once with a bounded `top_n`.
3. For each candidate, call `simulate_xrd_pattern` and `score_xrd_match`.
4. Emit a `MATCH_RESULTS_JSON: {...}` line to stdout collecting the
   ranked match list, and a `FIT_RESULTS_JSON: {...}` line with
   identification verdict + best-match metadata.

The `search_structures` tool already prints its own `DB_MATCHES_JSON:`
marker; do not re-emit it. The framework parser picks it up automatically.

**Complete pre-fit template** — adapt for the active wavelength and
chemistry hypothesis:

```python
import json
import numpy as np

from scilink.skills.structure_matching.xrd.search_structures import search_structures
from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
from scilink.skills.structure_matching.xrd.score_match import score_xrd_match

# ---- Step 1: Load experimental pattern ----
data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
exp_2theta, exp_intensity = data[:, 0], data[:, 1]

# ---- Step 2: Query DB ----
hits = search_structures(
    query={"chemistry": CHEMISTRY_HINT, "top_n": 5},
    output_dir="./candidates",
)

# ---- Step 3 & 4: Simulate + score each candidate ----
WAVELENGTH = "CuKa"
results = []
for cand in hits["candidates"]:
    sim = simulate_xrd_pattern(
        cand["structure_path"],
        wavelength=WAVELENGTH,
        two_theta_range=(float(exp_2theta.min()), float(exp_2theta.max())),
    )
    score = score_xrd_match(
        exp_two_theta=exp_2theta.tolist(),
        exp_intensity=exp_intensity.tolist(),
        sim_two_theta=sim["two_theta"],
        sim_intensity=sim["intensities"],
    )
    results.append({
        "id": cand["id"],
        "source": cand["source"],
        "formula": cand["formula"],
        "space_group": cand["space_group"],
        "r_factor": score["r_factor"],
        "rwp": score["rwp"],
        "verdict": score["verdict"],
    })

results.sort(key=lambda r: r["r_factor"])
best = results[0] if results else None

# ---- Step 5: Emit ranked match list + fit results ----
print("MATCH_RESULTS_JSON: " + json.dumps(results))
print("FIT_RESULTS_JSON: " + json.dumps({
    "best_match": best,
    "candidates_considered": len(results),
    "fit_quality": {
        "r_factor": best["r_factor"] if best else None,
        "verdict": best["verdict"] if best else "no_candidates",
    },
}))
```

**Background handling.** `score_xrd_match` defaults to `background="subtract_min"`
which subtracts the minimum experimental intensity as a flat offset. For
patterns with significant continuous background (amorphous halo,
fluorescence), fit and subtract a low-order polynomial background BEFORE
calling `score_xrd_match` and pass `background="none"`.

**Wavelength consistency.** Pass the same `wavelength` to every
`simulate_xrd_pattern` call. Pull from experiment metadata when present;
otherwise default CuKa.

**Peak broadening.** `score_xrd_match` uses a Lorentzian with FWHM=0.15°
by default. For low-resolution or strongly broadened experimental
patterns (e.g. nanocrystalline samples), increase `fwhm` to 0.3-0.5°
so peaks overlap as they do in the data.

**Do not over-interpret negative results.** If every candidate verdict is
"reject", the right next action is to broaden the chemistry hypothesis or
increase top_n — not to invent a new candidate.

## interpretation

**R-factor thresholds.** The verdict field in each score is derived from
the R-factor:

- R-factor ≤ 0.10 → "accept" — strong identification; report as the phase
- R-factor ≤ 0.20 → "marginal" — plausible match; list the top 3
- R-factor > 0.20 → "reject" — no confident match

Rwp (weighted profile R-factor) is a complementary metric; report it
alongside R-factor but use R-factor for the verdict.

**Reporting style for synthesis.** Phrase identification in proportion to
the R-factor margin:

- Best R-factor < 0.05 and runner-up > 2× best: declare identification
  with high confidence. Name the phase and space group.
- Best R-factor 0.05-0.10: declare identification with caveats. Note any
  alternative candidates within 1.5× of the best.
- Best R-factor 0.10-0.20: do not declare a single identification. List
  the top 3 candidates as "consistent with the experimental pattern" and
  recommend higher-resolution data or complementary techniques.
- All R-factors > 0.20: report no confident match. Suggest broadening the
  chemistry hypothesis, checking the experimental wavelength, or running
  a peak-fit-first workflow.

**What R-factor cannot tell you.** A good R-factor confirms the phase is
present; it does NOT confirm purity, sample preparation, or absence of
minor phases. Mention this caveat when reporting identifications.

**Multi-phase samples.** v1 of this skill assumes a single dominant phase.
For multi-phase identification, list the top 3 candidates with their
R-factors and note that quantitative phase fraction analysis (Rietveld
refinement) is the standard next step — that's outside this skill's
scope.

## validation

**Per-candidate quality checks.**

- The simulated pattern must cover the experimental 2-theta range. If
  `min(sim["two_theta"])` is more than 1° above `min(exp_2theta)`,
  expand the simulation range or note unobserved low-angle peaks.
- The number of simulated peaks should be reasonable for the candidate's
  symmetry (5-50 peaks typical in 10-90° 2θ; outliers indicate the
  simulation range was too narrow or too broad).
- R-factor and cosine similarity must agree on the verdict direction.
  When R-factor < 0.10 but cosine < 0.5, suspect a normalization issue —
  rerun with `background="subtract_min"` if not already.

**Cross-candidate sanity.** When 3+ candidates score "accept", the
chemistry hypothesis is too broad — narrow it. When 0 candidates score
above "reject", the chemistry hypothesis is wrong or the wavelength is
mismatched.

**FWHM tuning.** If the best candidate scores R-factor > 0.15 but visual
inspection (peak positions match) suggests it's the right phase, retry
`score_xrd_match` with `fwhm=0.3` or `fwhm=0.5`. Broad peaks in real
data require broader simulation profiles.

**Materials-Project authority.** Per the dedup rule, when MP and a local
CIF both return the same (formula, space_group), MP wins. Local CIFs
without space-group metadata cannot dedup against MP — they appear as
separate candidates.
