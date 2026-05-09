"""Benchmark / ablation harness for the VaspUpdater deterministic fix system.

Two-phase design:

  Phase 1 — Pattern-matching tests against fixture VASP logs.
    No VASP runs, no LLM calls. For each known error class, feed a
    representative log + base INCAR through `_try_deterministic_fixes`
    and assert that the expected diagnosis + INCAR fixes are produced.
    Runs anywhere; intended as the cheap-evidence layer for the PR.

  Phase 2 — End-to-end ablation on real VASP cases.
    For each registered case (a directory containing a POSCAR/INCAR/
    KPOINTS that fails in a known way), run `refine_inputs` once per
    ablation configuration to see which deterministic fix is responsible
    for unblocking which case. Optionally resubmit the proposed inputs
    to VASP and check whether convergence is achieved.

    Phase 2 is intentionally light on assumptions about your cluster
    plumbing — you supply the case registry and (if you want to verify
    convergence) a `vasp_runner` callable that knows how to submit a
    job and return its log/exit code.

Usage:

    # Phase 1 only (fast, no cluster needed):
    python examples/benchmark_vasp_updater.py --phase 1

    # Phase 2 only, full ablation (skip the convergence check):
    python examples/benchmark_vasp_updater.py --phase 2 --no-resubmit

    # Both:
    python examples/benchmark_vasp_updater.py --phase 1 --phase 2

Output: prints a human-readable table for each phase, writes JSON
to ./benchmark_results/<timestamp>/.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ══════════════════════════════════════════════════════════════
# Phase 1 — fixture-based pattern matching tests
# ══════════════════════════════════════════════════════════════

# Each fixture is a representative VASP log fragment for a known error,
# paired with the diagnosis + INCAR fix keys we expect _try_deterministic_fixes
# to produce. These are intentionally minimal — just enough to trigger the
# regex.

FIXTURE_INCAR = """\
GGA = PE
ENCUT = 450
ISMEAR = 1
SIGMA = 0.1
IBRION = 2
NSW = 200
"""

PATTERN_FIXTURES: List[Dict[str, Any]] = [
    {
        "label": "Missing GGA tag (POTCAR not found)",
        "log": (
            "Looking for PP for potpaw/Si\n"
            " No pseudopotential for the element Si found in the POTCAR file\n"
        ),
        "expected_diagnosis_substr": "Missing GGA tag",
        "expected_fix_keys": {"GGA"},
    },
    {
        "label": "ZBRENT bracketing failure",
        "log": (
            "ZBRENT: fatal error in bracketing\n"
            " please rerun with smaller EDIFF, or copy CONTCAR to POSCAR and continue\n"
        ),
        "expected_diagnosis_substr": "Ionic step",
        "expected_fix_keys": {"POTIM", "IBRION"},
    },
    {
        "label": "Sub-Space-Matrix not hermitian",
        "log": "Sub-Space-Matrix is not hermitian in DAV  -1.234E-05\n",
        "expected_diagnosis_substr": "Electronic minimization instability",
        "expected_fix_keys": {"ALGO"},
    },
    {
        "label": "BRMIX charge mixing failure",
        "log": (
            "BRMIX: very serious problems\n"
            " the old and the new charge density differ\n"
        ),
        "expected_diagnosis_substr": "Charge density mixing",
        "expected_fix_keys": {"AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "NELM"},
    },
    {
        "label": "RSPHER projection overlap",
        "log": "ERROR RSPHER: internal error in RSHPER\n",
        "expected_diagnosis_substr": "Real-space projection",
        "expected_fix_keys": {"LREAL"},
    },
    {
        "label": "EDDDAV did not converge",
        "log": "EDDDAV: Call to ZHEGV failed. Returncode = 6\n EDDDAV did not converge\n",
        "expected_diagnosis_substr": "Electronic minimization not converging",
        "expected_fix_keys": {"ALGO", "NELM"},
    },
    {
        "label": "Highest band is occupied (NBANDS too small)",
        "log": (
            "Your highest band is occupied at some k-points\n"
            " NBANDS = 32\n"
        ),
        "expected_diagnosis_substr": "Not enough empty bands",
        "expected_fix_keys": {"NBANDS"},
    },
    {
        "label": "Electronic SCF reached NELM",
        "log": " number of electronic SC steps reached NELM = 60\n",
        "expected_diagnosis_substr": "Electronic SCF did not converge",
        "expected_fix_keys": {"NELM", "ALGO"},
    },
    {
        "label": "Clean log (no errors) — should produce no fixes",
        "log": (
            "1 F= -.10000000E+02 E0= -.10000000E+02 d E =-.000000E+00\n"
            " writing wavefunctions\n"
        ),
        "expected_diagnosis_substr": None,
        "expected_fix_keys": set(),
    },
]


def run_phase_1() -> List[Dict[str, Any]]:
    """Exercise _try_deterministic_fixes against fixture logs.

    Returns one result dict per fixture with status (pass / fail / error)
    and the actual diagnoses + fixes the updater proposed.
    """
    from scilink.agents.sim_agents.vasp_updater import VaspUpdater

    # Construct without an LLM key — we won't reach the LLM layer for
    # these tests, the deterministic layer alone is what we're checking.
    updater = VaspUpdater.__new__(VaspUpdater)
    updater.logger = logging.getLogger("vasp_updater_test")

    results = []
    for fx in PATTERN_FIXTURES:
        try:
            det = updater._try_deterministic_fixes(
                vasp_log=fx["log"],
                incar_txt=FIXTURE_INCAR,
            )
            actual_diagnoses = det["diagnoses"]
            actual_fix_keys = set(det["fixes"].keys())

            expected_substr = fx["expected_diagnosis_substr"]
            expected_keys = fx["expected_fix_keys"]

            diag_ok = (
                expected_substr is None
                and not actual_diagnoses
            ) or (
                expected_substr is not None
                and any(expected_substr in d for d in actual_diagnoses)
            )
            keys_ok = actual_fix_keys == expected_keys

            status = "pass" if (diag_ok and keys_ok) else "fail"
            results.append({
                "label": fx["label"],
                "status": status,
                "expected_diagnosis_substr": expected_substr,
                "actual_diagnoses": actual_diagnoses,
                "expected_fix_keys": sorted(expected_keys),
                "actual_fix_keys": sorted(actual_fix_keys),
                "remaining_errors": det["remaining_errors"],
            })
        except Exception as exc:
            results.append({
                "label": fx["label"],
                "status": "error",
                "error": str(exc),
            })

    return results


# ══════════════════════════════════════════════════════════════
# Phase 2 — end-to-end ablation on real cases
# ══════════════════════════════════════════════════════════════

@dataclasses.dataclass
class VaspCase:
    """A failing VASP case for ablation. The case_dir must contain a
    POSCAR, INCAR, KPOINTS, and a vasp_log (the captured stdout of the
    failing run).
    """
    label: str
    case_dir: Path
    expected_error_class: str  # e.g. "BRMIX", "ZBRENT", "NBANDS", ...

    @property
    def poscar(self) -> Path: return self.case_dir / "POSCAR"
    @property
    def incar(self) -> Path: return self.case_dir / "INCAR"
    @property
    def kpoints(self) -> Path: return self.case_dir / "KPOINTS"
    @property
    def log(self) -> Path: return self.case_dir / "vasp_log"


# Fill this in with paths to your real failing cases on the cluster.
# Each case_dir should be a directory holding POSCAR / INCAR / KPOINTS
# / vasp_log (the captured failure log).
CASE_REGISTRY: List[VaspCase] = [
    # VaspCase(
    #     label="Si bulk (BRMIX failure on 4x4x4 supercell)",
    #     case_dir=Path("/people/alle927/scilink_benchmarks/si_brmix"),
    #     expected_error_class="BRMIX",
    # ),
    # VaspCase(...),
]


# Each ablation entry names a subset of KNOWN_FIXES diagnosis substrings to
# DISABLE. The "all" config runs with everything enabled; "none" disables
# every deterministic fix (LLM-only baseline).
ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {"name": "all",          "disable_diagnoses": []},
    {"name": "no_nbands",    "disable_diagnoses": ["Not enough empty bands"]},
    {"name": "no_zbrent",    "disable_diagnoses": ["Ionic step", "bracketing"]},
    {"name": "no_brmix",     "disable_diagnoses": ["Charge density mixing"]},
    {"name": "no_scf_nelm",  "disable_diagnoses": ["Electronic SCF did not converge"]},
    {"name": "no_eddav",     "disable_diagnoses": ["Electronic minimization not converging"]},
    {"name": "none",         "disable_diagnoses": ["__ALL__"]},
]


@contextlib.contextmanager
def ablate_known_fixes(disable_diagnoses: List[str]):
    """Temporarily filter scilink.agents.sim_agents.vasp_updater.KNOWN_FIXES
    to remove any entry whose diagnosis matches a disabled substring.
    """
    import scilink.agents.sim_agents.vasp_updater as vu
    original = vu.KNOWN_FIXES
    if "__ALL__" in disable_diagnoses:
        vu.KNOWN_FIXES = []
    else:
        vu.KNOWN_FIXES = [
            entry for entry in original
            if not any(d in entry["diagnosis"] for d in disable_diagnoses)
        ]
    try:
        yield
    finally:
        vu.KNOWN_FIXES = original


def run_phase_2(
    cases: List[VaspCase],
    *,
    api_key: Optional[str] = None,
    resubmit_runner: Optional[Callable[[Path], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """For each case, run `refine_inputs` once per ablation config.

    If a `resubmit_runner` is provided, it's called with the case_dir
    (which by then contains the proposed corrected INCAR) and should
    return a dict like {"converged": bool, "exit_code": int, ...}.
    Without it, the harness only records what fixes were proposed.
    """
    if not cases:
        return [{
            "note": "CASE_REGISTRY is empty. Add VaspCase entries pointing at "
                    "real failing-run directories on the cluster, then re-run.",
        }]

    from scilink.agents.sim_agents.vasp_updater import VaspUpdater

    results = []
    for case in cases:
        for ablation in ABLATION_CONFIGS:
            entry = {
                "label": case.label,
                "expected_error_class": case.expected_error_class,
                "ablation": ablation["name"],
            }

            if not (case.incar.exists() and case.log.exists()):
                entry["status"] = "skip"
                entry["reason"] = (
                    f"Missing files in {case.case_dir} "
                    f"(need POSCAR / INCAR / KPOINTS / vasp_log)"
                )
                results.append(entry)
                continue

            t0 = time.perf_counter()
            try:
                with ablate_known_fixes(ablation["disable_diagnoses"]):
                    updater = VaspUpdater(api_key=api_key)
                    result = updater.refine_inputs(
                        poscar_path=str(case.poscar),
                        incar_path=str(case.incar),
                        kpoints_path=str(case.kpoints),
                        vasp_log=case.log.read_text(),
                        original_request=f"benchmark case: {case.label}",
                    )
                refine_secs = time.perf_counter() - t0

                entry["refine_status"] = result.get("status")
                entry["refine_method"] = result.get("method")
                entry["refine_explanation"] = result.get("explanation", {})
                entry["refine_seconds"] = round(refine_secs, 2)

                # Optional: actually re-run VASP with the proposed fix.
                if resubmit_runner is not None:
                    # Stage the corrected INCAR into a sibling dir so we
                    # don't clobber the original failure case.
                    staged = case.case_dir.parent / (
                        case.case_dir.name + f".ablation.{ablation['name']}"
                    )
                    staged.mkdir(parents=True, exist_ok=True)
                    for src in (case.poscar, case.kpoints):
                        (staged / src.name).write_bytes(src.read_bytes())
                    (staged / "INCAR").write_text(result["suggested_incar"])

                    rerun = resubmit_runner(staged)
                    entry["rerun"] = rerun

                entry["status"] = "ok"
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)

            results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════

def print_phase_1_table(results: List[Dict[str, Any]]) -> None:
    print("\n=== Phase 1: Pattern-matching tests ===")
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"{n_pass} pass · {n_fail} fail · {n_err} error  (of {len(results)})\n")

    for r in results:
        marker = {"pass": "✓", "fail": "✗", "error": "!"}[r["status"]]
        print(f"  {marker} {r['label']}")
        if r["status"] == "fail":
            print(f"      expected diagnosis ~ {r['expected_diagnosis_substr']!r}")
            print(f"      actual   diagnoses = {r['actual_diagnoses']}")
            print(f"      expected fix keys  = {r['expected_fix_keys']}")
            print(f"      actual   fix keys  = {r['actual_fix_keys']}")
        elif r["status"] == "error":
            print(f"      {r['error']}")


def print_phase_2_table(results: List[Dict[str, Any]]) -> None:
    print("\n=== Phase 2: End-to-end ablation ===")
    if results and results[0].get("note"):
        print(f"  (skipped) {results[0]['note']}")
        return
    # Group by case label
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_case.setdefault(r["label"], []).append(r)
    for label, group in by_case.items():
        print(f"\n  {label}  (expected: {group[0]['expected_error_class']})")
        for r in group:
            status = r["status"]
            ablation = r["ablation"]
            method = r.get("refine_method", "—")
            secs = r.get("refine_seconds", "—")
            rerun = r.get("rerun", {})
            converged = rerun.get("converged") if rerun else "—"
            print(
                f"    {ablation:14s}  status={status:5s}  "
                f"method={str(method):14s}  refine={secs}s  "
                f"converged={converged}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        action="append",
        choices=["1", "2"],
        help="Which phases to run. Repeatable. Default: 1",
    )
    parser.add_argument(
        "--no-resubmit",
        action="store_true",
        help="Phase 2 only: skip the VASP resubmission step (just record proposed fixes).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key for the VaspInputAgent fallback in Phase 2. "
             "If not provided, ablation configs that fall through to the "
             "LLM will fail; the deterministic-only path still works.",
    )
    args = parser.parse_args()

    phases = args.phase or ["1"]
    out_dir = Path("benchmark_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "phases_run": phases,
    }

    if "1" in phases:
        results = run_phase_1()
        print_phase_1_table(results)
        bundle["phase_1"] = results
        any_failed = any(r["status"] != "pass" for r in results)
    else:
        any_failed = False

    if "2" in phases:
        # Resubmission stub: replace this with a call into your own
        # cluster-side runner if you want convergence verification.
        runner = None if args.no_resubmit else None  # plug in your runner here
        results = run_phase_2(
            CASE_REGISTRY,
            api_key=args.api_key,
            resubmit_runner=runner,
        )
        print_phase_2_table(results)
        bundle["phase_2"] = results

    (out_dir / "results.json").write_text(json.dumps(bundle, indent=2, default=str))
    print(f"\nResults written to: {out_dir}/results.json")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
