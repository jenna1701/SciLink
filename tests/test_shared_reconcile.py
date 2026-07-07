"""Tests for the technique-agnostic reconcile core (skills/_shared/_reconcile).

Uses NON-XRD vocabulary (NMR chemical shifts as feature positions, species as
labels) to prove the core is not XRD-bound — it operates on generic
(position, weight) features and (value, label) identifications, so any
spectroscopy with a model-free and an identification pass reuses it.
"""

from __future__ import annotations

import pytest

from scilink.skills._shared._reconcile import reconcile_series


def _nmr_titration(n=21):
    """Reactant peaks (1.2, 3.4 ppm) convert to product (1.5, 2.9 ppm) across
    a titration; identification names 'reactant' early, 'product' late with a
    null gap through the mid-point."""
    ffr, lfr = [], []
    for i in range(n):
        x = i / (n - 1)                          # extent of reaction
        pH = 3.0 + 8.0 * x                        # the series variable
        feats = [{"position": 1.2, "weight": 100 * (1 - x)},
                 {"position": 3.4, "weight": 60 * (1 - x)},
                 {"position": 1.5, "weight": 100 * x},
                 {"position": 2.9, "weight": 60 * x}]
        ffr.append({"value": pH, "features": feats})
        lab = None if 9 <= i <= 11 else ("reactant" if i < 10 else "product")
        lfr.append({"value": pH, "label": lab})
    return ffr, lfr


def test_generic_reconcile_labels_and_transitions():
    ffr, lfr = _nmr_titration()
    r = reconcile_series(ffr, lfr)
    assert r["low_regime_label"] == "reactant"
    assert r["high_regime_label"] == "product"
    # both transitions near the reaction midpoint (pH 7)
    assert abs(r["transition_model_free"] - 7.0) < 1.5
    assert abs(r["transition_identification"] - 7.0) < 2.0
    assert r["agreement"]["verdict"] == "consistent"
    by_pos = {t["position"]: t for t in r["tracked_features"]}
    assert by_pos[1.2]["regime"] == "low" and by_pos[1.2]["label"] == "reactant"
    assert by_pos[1.5]["regime"] == "high" and by_pos[1.5]["label"] == "product"


def test_center_area_aliases_accepted():
    # a caller may pass profile-fit peaks as center/area instead of
    # position/weight — the core accepts the aliases.
    ffr, lfr = _nmr_titration()
    ffr2 = [{"value": f["value"],
             "peaks": [{"center": p["position"], "area": p["weight"]}
                       for p in f["features"]]} for f in ffr]
    r = reconcile_series(ffr2, lfr)
    assert r["low_regime_label"] == "reactant"
    assert r["high_regime_label"] == "product"


def test_unidentified_regime_stays_none():
    ffr, lfr = _nmr_titration()
    for i in range(10, len(lfr)):
        lfr[i]["label"] = None                   # product never named
    r = reconcile_series(ffr, lfr)
    assert r["low_regime_label"] == "reactant"
    assert r["high_regime_label"] is None
    assert r["transition_model_free"] is not None
    assert r["agreement"]["verdict"] == "one_sided"


def test_multiregime_model_is_a_named_seam():
    ffr, lfr = _nmr_titration()
    with pytest.raises(ValueError, match="single_crossover"):
        reconcile_series(ffr, lfr, transition_model="multi_regime")


def test_guards():
    with pytest.raises(ValueError):
        reconcile_series([{"value": 0, "features": []}], [{"value": 0, "label": None}])
