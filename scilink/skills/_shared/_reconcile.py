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
