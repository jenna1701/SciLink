"""
Offline tests for parallel best-of-N anchor analysis.

Hermetic: `_execute_and_verify` is stubbed with canned results, the LLM
judge is a MagicMock returning fixed JSON. Verifies the fan-out wiring,
the quality gate, judge selection + fallbacks, winner config propagation,
and the strict N=1 / reuse-script no-op paths.
"""

import json
import logging
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
    UnifiedImageProcessingController,
)
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)
from scilink.agents.exp_agents.analysis_orchestrator_tools import (
    _resolve_n_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _judge_model(payload: str) -> MagicMock:
    model = MagicMock()
    resp = MagicMock()
    resp.text = payload
    model.generate_content.return_value = resp
    return model


def _parse_fn(response):
    try:
        return json.loads(response.text), None
    except Exception as e:  # pragma: no cover
        return None, {"error": str(e)}


def _controller(tmp_path, model=None) -> UnifiedImageProcessingController:
    return UnifiedImageProcessingController(
        model=model or MagicMock(),
        logger=logging.getLogger("test_best_of_n"),
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse_fn,
        executor=MagicMock(),
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir=str(tmp_path),
        image_to_bytes_fn=lambda arr: b"img",
    )


def _canned_result(score: float, approved: bool, success: bool = True,
                   tag: str = "") -> dict:
    return {
        "index": 0,
        "name": "image_0000",
        "success": success,
        "analysis_type": "test_analysis",
        "extracted_features": {"tag": tag},
        "quality_metrics": {},
        "visualization_bytes": b"\x89PNG" + tag.encode(),
        "visualization_path": f"/tmp/{tag or 'viz'}.png",
        "script": "print('hi')",
        "quality_history": {
            "final_score": score,
            "approved": approved,
            "verification_iterations": [
                {"score": score, "annealing_level": 0, "issues": []}
            ],
        },
    }


def _stub_execute(controller, results_by_tag: dict, record: list):
    """Install a stub _execute_and_verify returning per-candidate results."""

    def stub(state, image_data, data_path, image_name, image_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        record.append(state)
        tag = state.get("_candidate_tag", "_direct")
        result = results_by_tag[tag]
        # Simulate the QC loop refining the locked config in place.
        state["locked_analysis_config"] = {"refined_by": tag}
        return result

    controller._execute_and_verify = stub
    return stub


def _base_state(n: int) -> dict:
    return {
        "n_candidates": n,
        "locked_analysis_config": {"processing_pipeline": "original"},
        "original_image_bytes": b"\x89PNGoriginal",
    }


def _run(controller, state):
    return controller._execute_and_verify_best_of_n(
        state=state,
        image_data=np.zeros((4, 4)),
        data_path="img.npy",
        image_name="image_0000",
        image_idx=0,
    )


# ---------------------------------------------------------------------------
# Fan-out mechanics
# ---------------------------------------------------------------------------

def test_fanout_launches_n_isolated_attempts(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
        "cand_02": _canned_result(0.7, True, tag="cand_02"),
    }, seen)

    state = _base_state(3)
    result = _run(c, state)

    assert len(seen) == 3
    subdirs = {s["_candidate_subdir"] for s in seen}
    tags = {s["_candidate_tag"] for s in seen}
    assert subdirs == {f"_candidates/cand_{i:02d}" for i in range(3)}
    assert tags == {f"cand_{i:02d}" for i in range(3)}
    assert all(s["_suppress_human_feedback"] for s in seen)
    # Each attempt got its own config object (deep-copied), not the original.
    configs = [id(s["locked_analysis_config"]) for s in seen]
    assert len(set(configs)) == 3
    assert result["anchor_candidates"]
    assert len(result["anchor_candidates"]) == 3


def test_n1_is_strict_noop(tmp_path):
    c = _controller(tmp_path)
    seen = []
    _stub_execute(c, {"_direct": _canned_result(0.9, True, tag="direct")}, seen)

    state = _base_state(1)
    result = _run(c, state)

    assert len(seen) == 1
    # State passed by identity: no copy, no candidate keys injected.
    assert seen[0] is state
    assert "_candidate_subdir" not in state
    assert "anchor_candidates" not in result


def test_reuse_script_bypasses_fanout(tmp_path):
    c = _controller(tmp_path)
    seen = []

    def stub(state, image_data, data_path, image_name, image_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        seen.append((state, reuse_script))
        return _canned_result(0.9, True, tag="reuse")

    c._execute_and_verify = stub
    state = _base_state(3)
    result = c._execute_and_verify_best_of_n(
        state=state, image_data=np.zeros((4, 4)), data_path="img.npy",
        image_name="image_0000", image_idx=0,
        reuse_script="print('prior')", reuse_source="prior_run",
    )

    assert len(seen) == 1
    assert seen[0][0] is state
    assert seen[0][1] == "print('prior')"
    assert "anchor_candidates" not in result


def test_auxiliary_operands_force_single_attempt(tmp_path):
    c = _controller(tmp_path)
    seen = []
    _stub_execute(c, {"_direct": _canned_result(0.9, True, tag="direct")}, seen)

    state = _base_state(3)
    state["auxiliary_items"] = [{"label": "I0", "path": "ref.npy"}]
    result = _run(c, state)

    assert len(seen) == 1
    assert seen[0] is state
    assert "anchor_candidates" not in result


# ---------------------------------------------------------------------------
# Gating + judge selection
# ---------------------------------------------------------------------------

def test_gate_drops_unapproved_before_judge(tmp_path):
    # Judge picks survivor index 1; the unapproved candidate must not be
    # part of the survivor list the index refers to.
    model = _judge_model('{"selected_index": 1, "reasoning": "cleaner"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.95, False, tag="cand_00"),  # unapproved
        "cand_01": _canned_result(0.80, True, tag="cand_01"),
        "cand_02": _canned_result(0.75, True, tag="cand_02"),
    }, [])

    result = _run(c, _base_state(3))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert len(selected) == 1
    # Survivors (sorted by attempt) are cand_01, cand_02 -> index 1 = cand_02.
    assert selected[0]["attempt"] == 2
    assert result["extracted_features"]["tag"] == "cand_02"
    assert result["anchor_judge"]["reasoning"] == "cleaner"
    assert result["anchor_judge"]["fallback"] is False


def test_single_survivor_skips_judge(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.5, False, tag="cand_01"),
    }, [])

    state = _base_state(2)
    result = _run(c, state)

    model.generate_content.assert_not_called()
    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0


def test_judge_garbage_falls_back_to_argmax_score(tmp_path):
    model = _judge_model("not json at all {{{")
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.81, True, tag="cand_00"),
        "cand_01": _canned_result(0.92, True, tag="cand_01"),
    }, [])

    result = _run(c, _base_state(2))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 1  # highest score
    assert result["anchor_judge"]["fallback"] is True


def test_judge_out_of_range_index_falls_back(tmp_path):
    model = _judge_model('{"selected_index": 7, "reasoning": "??"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.7, True, tag="cand_01"),
    }, [])

    result = _run(c, _base_state(2))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0
    assert result["anchor_judge"]["fallback"] is True


def test_all_unapproved_keeps_best_score(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    low = _canned_result(0.45, False, tag="cand_00")
    low["quality_warning"] = "below threshold"
    _stub_execute(c, {
        "cand_00": low,
        "cand_01": _canned_result(0.30, False, tag="cand_01"),
    }, [])

    result = _run(c, _base_state(2))

    model.generate_content.assert_not_called()
    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0
    assert result["quality_warning"] == "below threshold"
    assert result["anchor_judge"]["fallback"] is True


def test_all_failed_returns_failure(tmp_path):
    c = _controller(tmp_path)
    fail = _canned_result(0.0, False, success=False, tag="cand_00")
    fail2 = _canned_result(0.0, False, success=False, tag="cand_01")
    _stub_execute(c, {"cand_00": fail, "cand_01": fail2}, [])

    result = _run(c, _base_state(2))

    assert result["success"] is False


def test_winner_config_propagates_to_outer_state(tmp_path):
    model = _judge_model('{"selected_index": 1, "reasoning": "better"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
    }, [])

    state = _base_state(2)
    _run(c, state)

    # Stub stamps each attempt's refined config; the winner's must win.
    assert state["locked_analysis_config"] == {"refined_by": "cand_01"}


def test_judge_evidence_includes_all_survivor_visualizations(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
        "cand_02": _canned_result(0.7, True, tag="cand_02"),
    }, [])

    _run(c, _base_state(3))

    (_, kwargs) = model.generate_content.call_args
    parts = kwargs.get("contents") or model.generate_content.call_args[0][0]
    image_parts = [p for p in parts if isinstance(p, dict)]
    # original + 3 candidate visualizations
    assert len(image_parts) == 4
    text = parts[0]
    assert "Candidate 0" in text and "Candidate 2" in text


# ---------------------------------------------------------------------------
# Curve fitting: same wrapper, R²-gated
# ---------------------------------------------------------------------------

def _curve_controller(tmp_path, model=None) -> UnifiedSeriesProcessingController:
    return UnifiedSeriesProcessingController(
        model=model or MagicMock(),
        logger=logging.getLogger("test_best_of_n_curve"),
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse_fn,
        executor=MagicMock(),
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir=str(tmp_path),
        plot_fn=lambda data, info: b"plot",
    )


def _canned_fit(r2: float, approved: bool, success: bool = True,
                tag: str = "") -> dict:
    return {
        "index": 0,
        "name": "spectrum_0000",
        "success": success,
        "model_type": "exp_decay",
        "parameters": {"tau": 1.0},
        "fit_quality": {"r_squared": r2},
        "visualization_bytes": b"\x89PNG" + tag.encode(),
        "visualization_path": f"/tmp/{tag or 'fit'}.png",
        "script": "print('fit')",
        "quality_history": {
            "final_r2": r2,
            "approved": approved,
            "verification_iterations": [
                {"r_squared": r2, "annealing_level": 0, "issues": []}
            ],
        },
    }


def _stub_fit(controller, results_by_tag: dict, record: list):
    def stub(state, curve_data, data_path, spectrum_name, spectrum_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        record.append(state)
        tag = state.get("_candidate_tag", "_direct")
        state["locked_fitting_config"] = {"refined_by": tag}
        return results_by_tag[tag]

    controller._fit_with_quality_control = stub
    return stub


def _curve_state(n: int) -> dict:
    return {
        "n_candidates": n,
        "locked_fitting_config": {"physical_model": "original"},
        "original_plot_bytes": b"\x89PNGoriginal",
    }


def _run_curve(controller, state, **kwargs):
    return controller._fit_with_quality_control_best_of_n(
        state=state,
        curve_data=np.zeros(64),
        data_path="spec.npy",
        spectrum_name="spectrum_0000",
        spectrum_idx=0,
        **kwargs,
    )


def test_curve_fanout_launches_n_isolated_attempts(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00"),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.97, True, tag="cand_02"),
    }, seen)

    state = _curve_state(3)
    result = _run_curve(c, state)

    assert len(seen) == 3
    assert {s["_candidate_subdir"] for s in seen} == {
        f"_candidates/cand_{i:02d}" for i in range(3)
    }
    assert all(s["_suppress_human_feedback"] for s in seen)
    configs = [id(s["locked_fitting_config"]) for s in seen]
    assert len(set(configs)) == 3
    assert len(result["anchor_candidates"]) == 3
    assert sum(1 for r in result["anchor_candidates"] if r["selected"]) == 1


def test_curve_n1_is_strict_noop(tmp_path):
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"_direct": _canned_fit(0.99, True, tag="direct")}, seen)

    state = _curve_state(1)
    result = _run_curve(c, state)

    assert len(seen) == 1
    assert seen[0] is state
    assert "anchor_candidates" not in result


def test_curve_reuse_script_bypasses_fanout(tmp_path):
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"_direct": _canned_fit(0.99, True, tag="direct")}, seen)

    state = _curve_state(3)
    result = _run_curve(c, state, reuse_script="print('prior')",
                        reuse_source="prior_run")

    assert len(seen) == 1
    assert seen[0] is state
    assert "anchor_candidates" not in result


def test_curve_auxiliary_operands_force_single_attempt(tmp_path):
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"_direct": _canned_fit(0.99, True, tag="direct")}, seen)

    state = _curve_state(3)
    state["auxiliary_items"] = [{"label": "I0", "path": "ref.npy"}]
    result = _run_curve(c, state)

    assert len(seen) == 1
    assert "anchor_candidates" not in result


def test_curve_gate_drops_unapproved_then_judge_picks(tmp_path):
    model = _judge_model('{"selected_index": 1, "reasoning": "more physical"}')
    c = _curve_controller(tmp_path, model)
    _stub_fit(c, {
        "cand_00": _canned_fit(0.999, False, tag="cand_00"),  # unapproved
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.97, True, tag="cand_02"),
    }, [])

    result = _run_curve(c, _curve_state(3))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    # Survivors are cand_01, cand_02 -> judge index 1 = cand_02.
    assert selected[0]["attempt"] == 2
    assert result["anchor_judge"]["reasoning"] == "more physical"


def test_curve_judge_garbage_falls_back_to_argmax_r2(tmp_path):
    model = _judge_model("not json")
    c = _curve_controller(tmp_path, model)
    _stub_fit(c, {
        "cand_00": _canned_fit(0.97, True, tag="cand_00"),
        "cand_01": _canned_fit(0.99, True, tag="cand_01"),
    }, [])

    result = _run_curve(c, _curve_state(2))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 1
    assert result["anchor_judge"]["fallback"] is True


def test_curve_winner_config_propagates(tmp_path):
    model = _judge_model('{"selected_index": 1, "reasoning": "better"}')
    c = _curve_controller(tmp_path, model)
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00"),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
    }, [])

    state = _curve_state(2)
    _run_curve(c, state)

    assert state["locked_fitting_config"] == {"refined_by": "cand_01"}


# ---------------------------------------------------------------------------
# Tool-side default resolution
# ---------------------------------------------------------------------------

class ImageAnalysisAgent:  # name drives the default lookup
    def analyze(self, data, n_candidates: int = 1):
        pass


class CurveFittingAgent:
    def analyze(self, data, n_candidates: int = 1):
        pass


class _NoSupportAgent:
    def analyze(self, data):
        pass


def test_resolve_default_image_agent_gets_three():
    assert _resolve_n_candidates(ImageAnalysisAgent(), None) == 3


def test_resolve_default_other_agent_gets_one():
    assert _resolve_n_candidates(CurveFittingAgent(), None) == 1


def test_resolve_explicit_request_wins():
    assert _resolve_n_candidates(ImageAnalysisAgent(), 5) == 5
    assert _resolve_n_candidates(ImageAnalysisAgent(), 1) == 1


def test_resolve_unsupported_agent_returns_none():
    assert _resolve_n_candidates(_NoSupportAgent(), 3) is None
    assert _resolve_n_candidates(_NoSupportAgent(), None) is None
