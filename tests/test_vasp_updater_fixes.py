"""Pattern-matching tests for VaspUpdater._try_deterministic_fixes.

For each known VASP error class, feed a representative log fragment +
a base INCAR through the deterministic-fix layer and assert that the
right diagnosis substring + INCAR fix keys are produced. No LLM call
required (these tests don't reach the LLM fallback).
"""

import logging

import pytest

from scilink.agents.sim_agents.vasp_updater import VaspUpdater


FIXTURE_INCAR = """\
GGA = PE
ENCUT = 450
ISMEAR = 1
SIGMA = 0.1
IBRION = 2
NSW = 200
"""


@pytest.fixture(scope="module")
def updater():
    """VaspUpdater without an LLM key — the deterministic layer doesn't
    reach the LLM, so we bypass __init__ to avoid needing credentials."""
    obj = VaspUpdater.__new__(VaspUpdater)
    obj.logger = logging.getLogger("vasp_updater_test")
    return obj


# Each tuple: (label, log, expected_diagnosis_substring, expected_fix_keys).
# A None diagnosis + empty fix-key set = "no error, no fix proposed".
FIXTURES = [
    (
        "missing_gga",
        (
            "Looking for PP for potpaw/Si\n"
            " No pseudopotential for the element Si found in the POTCAR file\n"
        ),
        "Missing GGA tag",
        {"GGA"},
    ),
    (
        "zbrent_bracketing",
        (
            "ZBRENT: fatal error in bracketing\n"
            " please rerun with smaller EDIFF, or copy CONTCAR to POSCAR and continue\n"
        ),
        "Ionic step",
        {"POTIM", "IBRION"},
    ),
    (
        "subspace_matrix",
        "Sub-Space-Matrix is not hermitian in DAV  -1.234E-05\n",
        "Electronic minimization instability",
        {"ALGO"},
    ),
    (
        "brmix",
        (
            "BRMIX: very serious problems\n"
            " the old and the new charge density differ\n"
        ),
        "Charge density mixing",
        {"AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "NELM"},
    ),
    (
        "rspher",
        "ERROR RSPHER: internal error in RSHPER\n",
        "Real-space projection",
        {"LREAL"},
    ),
    (
        "edddav",
        "EDDDAV: Call to ZHEGV failed. Returncode = 6\n EDDDAV did not converge\n",
        "Electronic minimization not converging",
        {"ALGO", "NELM"},
    ),
    (
        "nbands",
        "Your highest band is occupied at some k-points\n NBANDS = 32\n",
        "Not enough empty bands",
        {"NBANDS"},
    ),
    (
        "scf_nelm",
        " number of electronic SC steps reached NELM = 60\n",
        "Electronic SCF did not converge",
        {"NELM", "ALGO"},
    ),
    (
        "clean_log",
        (
            "1 F= -.10000000E+02 E0= -.10000000E+02 d E =-.000000E+00\n"
            " writing wavefunctions\n"
        ),
        None,
        set(),
    ),
]


@pytest.mark.parametrize(
    "label,log,expected_diag,expected_keys",
    FIXTURES,
    ids=[fixture[0] for fixture in FIXTURES],
)
def test_deterministic_fix_pattern(updater, label, log, expected_diag, expected_keys):
    """Each known VASP error class should yield the right diagnosis + fixes."""
    det = updater._try_deterministic_fixes(vasp_log=log, incar_txt=FIXTURE_INCAR)

    if expected_diag is None:
        # Clean log: no fixes proposed, no false positives.
        assert det["diagnoses"] == [], (
            f"Expected no diagnoses on clean log; got {det['diagnoses']}"
        )
        assert det["fixes"] == {}, (
            f"Expected no fixes on clean log; got {det['fixes']}"
        )
        return

    assert any(expected_diag in d for d in det["diagnoses"]), (
        f"Expected diagnosis substring {expected_diag!r} not found in "
        f"{det['diagnoses']}"
    )
    assert set(det["fixes"].keys()) == expected_keys, (
        f"Expected fix keys {expected_keys}, got {set(det['fixes'].keys())}"
    )
