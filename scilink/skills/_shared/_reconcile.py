"""Technique-agnostic reconciliation of two complementary series analyses.

The recurring shape across in-situ / operando spectroscopy: the same series
of frames is analysed two ways that answer different questions —

* a **model-free** pass (profile / peak / band fitting) that gives per-frame
  FEATURES (position + weight) and works regardless of any database — it
  describes HOW the material evolves;
* an **identification** pass that gives per-frame LABELS (phase, species,
  element, …) — WHAT the material is, but only where a reference exists.

Neither alone is the whole picture. This module joins them: it tracks the
model-free features across frames, splits them into regimes, attributes each
to the identified label of its regime, and cross-checks the transition the
two passes find independently. Where identification produced no label, the
regime stays honestly ``None`` (unidentified) rather than being force-named.

This is the package-neutral CORE. Per-technique skills wrap it with their own
vocabulary and feature/label extraction (e.g. XRD's ``reconcile_series_phases``
maps 2θ peaks → features and phases → labels). The vocabulary here is generic:
``features`` (position, weight), ``labels`` (value, label), so NMR chemical
shifts, Raman/IR band positions, XPS binding energies, etc. all reuse it.

Transition model: v1 is a single low→high **crossover** — correct for a
one-step transformation ramp (dehydration, calcination, a polymorphic
transition). A titration or a multi-step A→B→C series needs a multi-regime
model; ``transition_model`` is the seam for that (only ``'single_crossover'``
is implemented today, and it is stated in the result).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _frame_features(fr: dict) -> list[tuple]:
    """(position, weight) list for one frame. Accepts 'features' with
    position/weight, or the leniency aliases center/area/amplitude so a
    caller can pass profile-fit peaks with minimal reshaping."""
    out = []
    for p in (fr.get("features") or fr.get("peaks") or []):
        pos = p.get("position", p.get("center"))
        w = p.get("weight", p.get("area", p.get("amplitude", 0.0)))
        try:
            out.append((float(pos), float(w)))
        except (TypeError, ValueError):
            continue
    return out


def reconcile_series(
    feature_frames: Sequence[dict],
    label_frames: Sequence[dict],
    tol: float = 0.25,
    min_presence_frac: float = 0.2,
    agreement_units: float = 15.0,
    transition_model: str = "single_crossover",
) -> dict[str, Any]:
    """Join a model-free feature series with an identification label series.

    ``feature_frames``: per-frame, in series order,
        ``[{'value': <series var>, 'features': [{'position': x, 'weight': w}, ...]}]``
        (``'peaks'`` / ``center`` / ``area`` accepted as aliases).
    ``label_frames``: aligned per-frame labels,
        ``[{'value': <series var>, 'label': <str or None>}]``.

    Returns tracked features (each attributed to a regime + label), the two
    transition estimates (feature-weight-share crossover vs label switch), and
    an agreement verdict. See module docstring for the transition-model note.
    """
    if transition_model != "single_crossover":
        raise ValueError(
            f"transition_model={transition_model!r} not implemented; only "
            "'single_crossover' is available (v1). Multi-regime is the seam.")
    n = min(len(feature_frames), len(label_frames))
    if n < 3:
        raise ValueError("reconcile_series needs at least 3 aligned frames.")
    feature_frames = list(feature_frames)[:n]
    label_frames = list(label_frames)[:n]
    V = np.array([float(f.get("value", i)) for i, f in enumerate(feature_frames)])

    # cluster feature positions across frames into tracked features
    allp = sorted(p for f in feature_frames for p, _ in _frame_features(f))
    refs: list[float] = []
    for p in allp:
        if not refs or abs(p - refs[-1]) > tol:
            refs.append(p)
        else:
            refs[-1] = 0.5 * (refs[-1] + p)

    def _seen(rp):
        return float(np.mean([any(abs(p - rp) <= tol for p, _ in _frame_features(f))
                              for f in feature_frames]))
    refs = [rp for rp in refs if _seen(rp) >= float(min_presence_frac)]
    if not refs:
        raise ValueError("no feature persisted across the series — raise tol or "
                         "lower min_presence_frac.")

    weight = np.zeros((n, len(refs)))
    for i, f in enumerate(feature_frames):
        ff = _frame_features(f)
        for j, rp in enumerate(refs):
            near = [(abs(p - rp), w) for p, w in ff if abs(p - rp) <= tol]
            if near:
                weight[i, j] = min(near)[1]

    third = max(1, n // 3)
    early, late = weight[:third].mean(axis=0), weight[-third:].mean(axis=0)
    low = np.where(early > late)[0]
    high = np.where(late >= early)[0]
    frac = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-9)
    low_share = frac[:, low].sum(axis=1) if len(low) else np.zeros(n)
    high_share = frac[:, high].sum(axis=1) if len(high) else np.zeros(n)

    t_model = None
    for i in range(1, n):
        if high_share[i - 1] < 0.5 <= high_share[i]:
            g = (0.5 - high_share[i - 1]) / (high_share[i] - high_share[i - 1] + 1e-9)
            t_model = float(V[i - 1] + g * (V[i] - V[i - 1]))
            break

    def _lab(i):
        lb = label_frames[i].get("label")
        return lb if lb else None
    lo_idx = [i for i in range(n) if low_share[i] >= high_share[i]]
    hi_idx = [i for i in range(n) if high_share[i] > low_share[i]]

    def _dom(idxs):
        names = [_lab(i) for i in idxs if _lab(i)]
        return max(set(names), key=names.count) if names else None
    lo_label, hi_label = _dom(lo_idx), _dom(hi_idx)

    t_id = None
    if lo_label and hi_label and lo_label != hi_label:
        last_lo = max([i for i in range(n) if _lab(i) == lo_label], default=None)
        first_hi = min([i for i in range(n) if _lab(i) == hi_label], default=None)
        if last_lo is not None and first_hi is not None and first_hi >= last_lo:
            t_id = float(0.5 * (V[last_lo] + V[first_hi]))

    if t_model is not None and t_id is not None:
        gap = abs(t_model - t_id)
        agree = {"units_apart": round(gap, 2),
                 "verdict": "consistent" if gap <= float(agreement_units) else "divergent"}
    else:
        agree = {"units_apart": None, "verdict": "one_sided"}

    tracked = []
    for j, rp in enumerate(refs):
        regime = "low" if j in low else "high"
        tracked.append({
            "position": round(float(rp), 3),
            "regime": regime,
            "label": lo_label if regime == "low" else hi_label,
            "weight_series": [round(float(v), 4) for v in weight[:, j]],
        })

    return {
        "values": [float(v) for v in V],
        "low_regime_label": lo_label,
        "high_regime_label": hi_label,
        "tracked_features": tracked,
        "transition_model_free": t_model,
        "transition_identification": t_id,
        "agreement": agree,
        "transition_model": transition_model,
        # arrays kept so a technique wrapper can plot without re-deriving
        "_low_idx": low.tolist(), "_high_idx": high.tolist(),
        "_weight": weight.tolist(), "_low_share": low_share.tolist(),
        "_high_share": high_share.tolist(), "_refs": [float(r) for r in refs],
    }


# --- generic extraction from stored analysis results (orchestrator seam) -------
#
# The orchestrator reconciles two PRIOR analyses (a model-free pass and an
# identification pass) it already ran and stored. Both write an
# ``analysis_results.json`` (per-frame ``individual_results``) and a
# ``series_fit_results.json`` (series values) in their output dir. Extraction
# is technique-agnostic: features are any per-frame fitted peaks (center/area),
# labels are the per-frame identity under one of the common label keys. So a
# single orchestrator step serves XRD today and NMR/Raman/XPS as their
# identification skills mature — no per-technique orchestration.

_LABEL_KEYS = ("identified_phase", "identified_species", "identified_compound",
               "identity", "phase", "species", "compound", "label")


def _read_json(path):
    import json
    from pathlib import Path
    p = Path(path)
    return __import__("json").loads(p.read_text()) if p.is_file() else None


def _series_values(output_dir, n):
    from pathlib import Path
    sfr = _read_json(Path(output_dir) / "series_fit_results.json") or {}
    vals = ((sfr.get("series_metadata") or {}).get("values")
            or sfr.get("values"))
    if vals and len(vals) >= n:
        return [float(v) for v in vals[:n]]
    return [float(i) for i in range(n)]


def _extract_features(analysis_dir) -> list[dict]:
    """Per-frame (position, weight) peaks from a model-free pass's stored
    result — any ``parameters`` entry that is a dict with a center/position."""
    from pathlib import Path
    d = _read_json(Path(analysis_dir) / "analysis_results.json") or {}
    res = d.get("individual_results") or []
    vals = _series_values(analysis_dir, len(res))
    frames = []
    for i, r in enumerate(res):
        feats = []
        for _, v in (r.get("parameters") or {}).items():
            if isinstance(v, dict) and ("center" in v or "position" in v):
                pos = v.get("position", v.get("center"))
                w = v.get("weight", v.get("area", v.get("amplitude", 0.0)))
                feats.append({"position": pos, "weight": w})
        frames.append({"value": vals[i], "features": feats})
    return frames


def _extract_labels(analysis_dir) -> list[dict]:
    """Per-frame identity label from an identification pass's stored result."""
    from pathlib import Path
    d = _read_json(Path(analysis_dir) / "analysis_results.json") or {}
    res = d.get("individual_results") or []
    vals = _series_values(analysis_dir, len(res))
    frames = []
    for i, r in enumerate(res):
        params = r.get("parameters") or {}
        label = next((params[k] for k in _LABEL_KEYS if params.get(k)), None)
        frames.append({"value": vals[i], "label": label})
    return frames


def reconcile_analysis_dirs(model_free_dir: str, identification_dir: str,
                            output_figure: Optional[str] = None,
                            tol: float = 0.25, min_presence_frac: float = 0.2,
                            agreement_units: float = 15.0) -> dict[str, Any]:
    """Reconcile two PRIOR analyses given their output directories: extract
    features from the model-free pass and labels from the identification pass,
    call :func:`reconcile_series`, and (optionally) plot a generic figure.
    Technique-agnostic — the orchestrator's coupled-series step calls this."""
    ff = _extract_features(model_free_dir)
    lf = _extract_labels(identification_dir)
    if not any(f.get("features") for f in ff):
        raise ValueError(
            f"no per-frame fitted peaks found in {model_free_dir} — the "
            "model-free pass must be a profile/peak-fitting run (its "
            "'parameters' carry peak center/area per frame). Check the "
            "model_free / identification arguments are not swapped.")
    r = reconcile_series(ff, lf, tol=tol, min_presence_frac=min_presence_frac,
                         agreement_units=agreement_units)
    if output_figure:
        try:
            _plot_generic(r, output_figure)
            r["figure"] = output_figure
        except Exception:
            r["figure"] = None
    return {k: v for k, v in r.items() if not k.startswith("_") or k == "figure"}


def _plot_generic(r, path):
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = _np.array(r["values"]); w = _np.array(r["_weight"])
    low, high = r["_low_idx"], r["_high_idx"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})
    for j in low:
        ax[0].plot(V, w[:, j], "-", color="tab:blue", alpha=0.6, lw=1)
    for j in high:
        ax[0].plot(V, w[:, j], "-", color="tab:red", alpha=0.6, lw=1)
    ax[0].plot([], [], color="tab:blue", label=f"low regime: {r['low_regime_label'] or 'unidentified'}")
    ax[0].plot([], [], color="tab:red", label=f"high regime: {r['high_regime_label'] or 'unidentified'}")
    ax[0].set_ylabel("feature weight (area)")
    ax[0].set_title("Model-free + Identification, reconciled\n"
                    "feature evolution (model-free), labels (identification)")
    ax[0].legend(fontsize=9, loc="upper right")
    ax[1].plot(V, r["_low_share"], "o-", color="tab:blue", label="low-regime weight share")
    ax[1].plot(V, r["_high_share"], "s-", color="tab:red", label="high-regime weight share")
    if r["transition_model_free"] is not None:
        ax[1].axvline(r["transition_model_free"], color="k", ls="--",
                      label=f"model-free transition ≈ {r['transition_model_free']:.1f}")
    if r["transition_identification"] is not None:
        ax[1].axvline(r["transition_identification"], color="tab:green", ls=":",
                      label=f"identification transition ≈ {r['transition_identification']:.1f}")
    ax[1].set_xlabel("series variable"); ax[1].set_ylabel("weight share")
    ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
