"""Deterministic-layer tests for VaspQualityAgent.

Verifies that `_deterministic_issues` (the rules-based guardrails
applied before the LLM) flags the right problems given a structured
facts dict shaped like what `post_run_analysis.analyze_run_directory`
produces. No LLM, no VASP run, no API key required.
"""

import copy

import pytest

from scilink.agents.sim_agents.vasp_quality import _deterministic_issues


# ── Fixture facts ─────────────────────────────────────────────

CLEAN_STATIC = {
    "converged_electronic": True,
    "converged_ionic": True,
    "converged": True,
    "final_energy": -10.85,
    "n_ionic_steps": 1,
    "n_electronic_steps_last_ionic": 12,
    "incar_snapshot": {"NSW": 0, "NELM": 100, "ISMEAR": 1, "ISPIN": 2},
}

CLEAN_RELAX = {
    "converged_electronic": True,
    "converged_ionic": True,
    "converged": True,
    "final_energy": -78.30,
    "n_ionic_steps": 5,
    "n_electronic_steps_last_ionic": 8,
    "max_force_eV_per_A": 0.005,
    "incar_snapshot": {"NSW": 100, "NELM": 100, "EDIFFG": "-0.01", "IBRION": 2},
}


# ── Negative cases (should produce no issues) ────────────────

class TestNoFalsePositives:
    """The clean cases should produce no flags."""

    def test_clean_static_run_has_no_issues(self):
        assert _deterministic_issues(CLEAN_STATIC) == []

    def test_clean_relaxation_has_no_issues(self):
        assert _deterministic_issues(CLEAN_RELAX) == []


# ── Positive cases (should flag specific issues) ─────────────

class TestConvergenceFlags:
    """Vasprun-level convergence flags translate to critical issues."""

    def test_unconverged_electronic_is_critical(self):
        facts = copy.deepcopy(CLEAN_STATIC)
        facts["converged_electronic"] = False
        issues = _deterministic_issues(facts)
        assert any(
            i["severity"] == "critical" and "Electronic SCF" in i["description"]
            for i in issues
        )

    def test_unconverged_ionic_is_critical(self):
        facts = copy.deepcopy(CLEAN_RELAX)
        facts["converged_ionic"] = False
        issues = _deterministic_issues(facts)
        assert any(
            i["severity"] == "critical" and "Ionic relaxation" in i["description"]
            for i in issues
        )


class TestForceVsEdiffg:
    """Max-force-vs-EDIFFG only flags when EDIFFG is force-based (negative)
    and a relaxation was actually requested (NSW > 0)."""

    def test_force_above_negative_ediffg_threshold_warns(self):
        facts = copy.deepcopy(CLEAN_RELAX)
        facts["max_force_eV_per_A"] = 0.5  # well above |EDIFFG| = 0.01
        issues = _deterministic_issues(facts)
        force_issues = [i for i in issues if "Max force" in i["description"]]
        assert len(force_issues) == 1
        assert force_issues[0]["severity"] == "warning"
        assert force_issues[0]["metric"]["max_force_eV_per_A"] == 0.5
        assert force_issues[0]["metric"]["threshold_eV_per_A"] == 0.01

    def test_force_below_threshold_no_warn(self):
        facts = copy.deepcopy(CLEAN_RELAX)
        facts["max_force_eV_per_A"] = 0.005
        issues = _deterministic_issues(facts)
        assert not any("Max force" in i["description"] for i in issues)

    def test_static_calc_no_force_flag_even_with_force(self):
        """NSW = 0 means no relaxation was requested; force-vs-EDIFFG is
        meaningless and shouldn't produce a flag."""
        facts = copy.deepcopy(CLEAN_STATIC)
        facts["max_force_eV_per_A"] = 5.0  # very large
        issues = _deterministic_issues(facts)
        assert not any("Max force" in i["description"] for i in issues)

    def test_positive_ediffg_no_force_flag(self):
        """A positive EDIFFG is energy-based; force comparison doesn't apply."""
        facts = copy.deepcopy(CLEAN_RELAX)
        facts["incar_snapshot"]["EDIFFG"] = "1E-4"  # positive = energy-based
        facts["max_force_eV_per_A"] = 0.5
        issues = _deterministic_issues(facts)
        assert not any("Max force" in i["description"] for i in issues)


class TestNelmSaturation:
    """If the last ionic step's SCF reached NELM, the run may be at the
    iteration ceiling and not fully converged."""

    def test_last_step_at_nelm_warns(self):
        facts = copy.deepcopy(CLEAN_STATIC)
        facts["n_electronic_steps_last_ionic"] = 100  # == NELM
        issues = _deterministic_issues(facts)
        assert any(
            i["severity"] == "warning" and "NELM" in i["description"]
            for i in issues
        )

    def test_last_step_well_below_nelm_no_warn(self):
        facts = copy.deepcopy(CLEAN_STATIC)
        facts["n_electronic_steps_last_ionic"] = 12
        issues = _deterministic_issues(facts)
        assert not any("NELM" in i["description"] for i in issues)


class TestErrorHints:
    """Classified log-pattern hints from post_run_analysis are surfaced as
    warnings."""

    def test_error_hints_become_warnings(self):
        facts = copy.deepcopy(CLEAN_STATIC)
        facts["error_hints"] = [
            "ZBRENT: bracketing failure",
            "BRMIX: very serious problems",
        ]
        issues = _deterministic_issues(facts)
        log_issues = [i for i in issues if "Log pattern flagged" in i["description"]]
        assert len(log_issues) == 2
        for i in log_issues:
            assert i["severity"] == "warning"


class TestCombinations:
    """Multiple problems should all be flagged independently."""

    def test_unconverged_with_high_force_and_nelm_saturation(self):
        facts = copy.deepcopy(CLEAN_RELAX)
        facts["converged_electronic"] = False
        facts["converged_ionic"] = False
        facts["max_force_eV_per_A"] = 0.5
        facts["n_electronic_steps_last_ionic"] = 100  # == NELM
        issues = _deterministic_issues(facts)

        severities = [i["severity"] for i in issues]
        assert severities.count("critical") >= 2  # both convergence flags
        assert severities.count("warning") >= 2  # force + NELM saturation
