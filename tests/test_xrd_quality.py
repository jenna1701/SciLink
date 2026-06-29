"""Offline tests for the XRD peak-region quality metric and its wiring.

The metric is the shared 1D-spectroscopy ``peak_region_r2`` (hoisted to
``_shared``). For XRD it must (a) RESCUE a correct fit of a weak/noisy powder
pattern that a whole-pattern R² scores low, (b) still FAIL a fit that genuinely
misses reflections, (c) leave a high-SNR fit unchanged (no regression), and be
correctly skill-gated to the ``xrd_profile`` skill. ``fit_pattern`` must report
it. The NMR skill must keep re-exporting the same implementation.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml
from pathlib import Path

from scilink.skills._shared._quality_metrics import peak_region_r2

N = 8000
X = np.linspace(10.0, 80.0, N)  # 2-theta degrees


def _pv(A, x0, w):
    return A * (w / 2) ** 2 / ((X - x0) ** 2 + (w / 2) ** 2)


def _pattern(amp, seed=0):
    y = np.zeros(N)
    for c in (22.0, 28.0, 35.0, 48.0, 61.0):
        y += _pv(amp, c, 0.3)
    return y + np.random.default_rng(seed).normal(0, 1.0, N)


def _model(amp):
    y = np.zeros(N)
    for c in (22.0, 28.0, 35.0, 48.0, 61.0):
        y += _pv(amp, c, 0.3)
    return y


def test_rescues_correct_low_snr_pattern():
    # Weak peaks (5σ) on a noisy background: global R² low, region R² rescues.
    y = _pattern(amp=5)
    q = peak_region_r2(X, y, _model(5), baseline=np.zeros(N))
    assert q["r_squared"] < q["peak_region_r2"]
    assert q["peak_region_r2"] > 0.3


def test_high_snr_pattern_unchanged():
    # Strong peaks: region R² ≈ global R², both high → no regression on clean data.
    y = _pattern(amp=200)
    q = peak_region_r2(X, y, _model(200), baseline=np.zeros(N))
    assert q["peak_region_r2"] > 0.9
    assert abs(q["peak_region_r2"] - q["r_squared"]) < 0.05


def test_missed_reflections_not_rescued():
    # Model omits two of the five reflections: region R² must drop (genuine miss).
    y = _pattern(amp=40)
    partial = _pv(40, 22.0, 0.3) + _pv(40, 28.0, 0.3) + _pv(40, 35.0, 0.3)
    q = peak_region_r2(X, y, partial, baseline=np.zeros(N))
    full = peak_region_r2(X, y, _model(40), baseline=np.zeros(N))
    assert q["peak_region_r2"] < full["peak_region_r2"] - 0.2


def test_reexport_identity():
    # NMR and XRD bundles must re-export the SAME shared implementation.
    from scilink.skills.curve_fitting.nmr.quality import peak_region_r2 as nmr
    from scilink.skills.curve_fitting.xrd_profile.quality import peak_region_r2 as xrd
    assert nmr is xrd is peak_region_r2


def test_tool_registered_and_skill_gated():
    from scilink.skills._shared._registry import get_tools_for, get_tool_function
    names = {t.name for t in get_tools_for("curve_fitting", active_skills=["xrd_profile"])}
    assert "peak_region_r2" in names
    assert "peak_region_r2" not in {
        t.name for t in get_tools_for("curve_fitting", active_skills=[])}
    assert callable(get_tool_function("peak_region_r2", active_skills=["xrd_profile"]))


def test_fit_pattern_reports_peak_region_r2():
    from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
    y = _pattern(amp=200)
    res = fit_pattern(X.tolist(), y.tolist())
    assert "peak_region_r2" in res and "n_signal_points" in res
    assert 0.0 <= res["peak_region_r2"] <= 1.0


def test_frontmatter_gate_is_peak_region_r2():
    md = Path("scilink/skills/curve_fitting/xrd_profile/xrd_profile.md").read_text()
    fm = yaml.safe_load(md.split("---")[1])
    gate = fm["quality_gate"]
    assert gate["metric"] == "peak_region_r2"
    from scilink.agents.exp_agents.quality_gate import _coerce
    g = _coerce(gate)
    assert g.metric == "peak_region_r2"
    assert g.hard_reject_threshold <= g.accept_threshold
    # the gate reads the metric out of a fit_quality dict
    assert g.extract({"peak_region_r2": 0.89, "r_squared": 0.55}) == pytest.approx(0.89)
