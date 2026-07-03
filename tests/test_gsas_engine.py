"""Tests for the optional GSAS-II full-profile XRD simulation engine.

Most assertions are GSAS-II-INDEPENDENT (wavelength-alias resolution, registry
wiring, the Bragg peak-pick helper, the actionable missing-dependency error) so
they run in CI without GSAS-II installed. The end-to-end physics test
(``test_full_simulation_physics``) is gated on ``gsas_available()`` and skipped
when GSAS-II is absent; it is exercised live in a GSAS-II environment against
independent crystallographic ground truth (Si positions, Fe BCC systematic
absences, NaCl structure-factor intensities, Mo/Cu wavelength scaling).
"""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd import _gsas_engine as ge
from scilink.skills.structure_matching.xrd import simulate_xrd as sx


def _pymatgen_available():
    try:
        import pymatgen.core  # noqa: F401
        import pymatgen.io.cif  # noqa: F401
        return True
    except Exception:
        return False


# An origin-choice-ambiguous anatase CIF: only the I4_1/amd H-M symbol, no
# explicit symmetry operators. GSAS-II mis-resolves the origin from this alone
# (Fcalc(200)=0, wrong), while the real phase has a strong (200) at 48.05 deg.
# _canonicalize_cif must expand it to explicit coordinates so GSAS agrees with
# pymatgen. This is the regression guard for that finding.
_ANATASE_AMBIGUOUS = (
    "data_anatase\n_cell_length_a 3.7845\n_cell_length_b 3.7845\n_cell_length_c 9.5143\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'I 41/a m d'\n_symmetry_Int_Tables_number 141\n"
    "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    "Ti 0.0 0.0 0.0\nO 0.0 0.0 0.2081\n"
)


def test_engine_registered():
    assert "gsas" in sx._ENGINES
    assert sx._ENGINES["gsas"] is sx._simulate_gsas


def test_tool_exposes_engine_knobs():
    # The gsas physical knobs must be surfaced in TOOL_SPEC (LLM-facing) with
    # tuning guidance, not hidden in the Python signature.
    params = sx.TOOL_SPEC.parameters
    assert "crystallite_um" in params and "peak_rel_height" in params
    assert "nanocrystalline" in params["crystallite_um"]["description"].lower()
    assert "LOWER" in params["crystallite_um"]["description"]


_SI_CIF = (
    "data_Si\n_cell_length_a 5.43088\n_cell_length_b 5.43088\n_cell_length_c 5.43088\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'F d -3 m'\n_symmetry_Int_Tables_number 227\n"
    "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    "Si 0.0 0.0 0.0\n"
)


@pytest.mark.skipif(not _pymatgen_available(), reason="pymatgen not installed")
def test_pymatgen_ignores_engine_knobs(tmp_path):
    # Passing gsas-only knobs to the pymatgen engine must be silently ignored,
    # not error — the dispatch forwards them but pymatgen absorbs **_ignored.
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)
    r = sx.simulate_xrd_pattern(str(cif), engine="pymatgen",
                                crystallite_um=0.05, peak_rel_height=0.2)
    assert r["engine"] == "pymatgen" and len(r["two_theta"]) > 0


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_gsas_knobs_forwarded_through_tool(tmp_path):
    # The crystallite_um knob must actually reach the gsas engine via the tool:
    # a nanocrystalline size broadens the (111) peak vs the sharp default.
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)

    def fwhm(res):
        x = np.asarray(res["profile_two_theta"]); y = np.asarray(res["profile_intensities"])
        half = y.max() / 2
        above = x[y >= half]
        return (above.max() - above.min()) if above.size else 0.0

    nano = sx.simulate_xrd_pattern(str(cif), engine="gsas",
                                   two_theta_range=(27.0, 30.0), crystallite_um=0.05)
    bulk = sx.simulate_xrd_pattern(str(cif), engine="gsas",
                                   two_theta_range=(27.0, 30.0), crystallite_um=10.0)
    assert fwhm(nano) > fwhm(bulk)


def test_wavelength_alias_resolution():
    assert ge._resolve_wavelength(1.2) == 1.2
    assert ge._resolve_wavelength("CuKa") == pytest.approx(1.5406)
    assert ge._resolve_wavelength("MoKa") == pytest.approx(0.71073)
    # tolerant to case / spacing / hyphen
    assert ge._resolve_wavelength("Mo-Ka") == pytest.approx(0.71073)
    assert ge._resolve_wavelength(" cu ka ") == pytest.approx(1.5406)
    with pytest.raises(ValueError):
        ge._resolve_wavelength("nonsense")


def test_peak_pick_bragg_dspacing():
    # A synthetic two-peak profile: peak-pick must recover positions, normalize
    # intensities to 100 at the max, and convert to correct Bragg d-spacings.
    x = np.linspace(20.0, 60.0, 4000)
    y = np.zeros_like(x)
    for c, a, w in [(28.44, 0.5, 0.1), (47.30, 1.0, 0.1)]:  # (111) weaker, (220) strongest
        y += a * (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)
    tt, inten, d = ge._peak_pick(x, y, lam=1.5406, rel_height=0.05)
    assert len(tt) == 2
    assert tt[0] == pytest.approx(28.44, abs=0.05)
    assert tt[1] == pytest.approx(47.30, abs=0.05)
    assert inten.max() == pytest.approx(100.0)          # normalized to max
    assert inten[0] < inten[1]                          # weaker peak stays weaker
    # d = lambda / (2 sin theta): Si(111) ~ 3.135 A, Si(220) ~ 1.920 A
    assert d[0] == pytest.approx(3.135, abs=0.01)
    assert d[1] == pytest.approx(1.920, abs=0.01)


def test_peak_pick_empty_profile():
    x = np.linspace(10, 90, 100)
    tt, inten, d = ge._peak_pick(x, np.zeros_like(x), lam=1.5406)
    assert len(tt) == len(inten) == len(d) == 0


def test_degenerate_range_rejected():
    # A reversed range is normalized via min/max (forgiving); only a degenerate
    # zero-width range is invalid. The check fires before any GSAS/CIF work.
    if not ge.gsas_available():
        pytest.skip("GSAS-II not installed")
    with pytest.raises(ValueError):
        ge.simulate_gsas("dummy.cif", "CuKa", (50.0, 50.0))


def test_actionable_error_without_gsas():
    if ge.gsas_available():
        pytest.skip("GSAS-II present; missing-dependency path not exercised here")
    with pytest.raises(RuntimeError) as exc:
        sx.simulate_xrd_pattern("x.cif", engine="gsas")
    msg = str(exc.value)
    assert "scilink[gsas]" in msg and "Fortran" in msg


@pytest.mark.skipif(not _pymatgen_available(), reason="pymatgen not installed")
def test_canonicalize_cif_expands_to_p1(tmp_path):
    # A symbol-only CIF must be rewritten with explicit atom sites (P1), removing
    # the origin-choice ambiguity before GSAS-II sees it.
    src = tmp_path / "amb.cif"
    src.write_text(_ANATASE_AMBIGUOUS)
    out = ge._canonicalize_cif(str(src), str(tmp_path))
    assert out != str(src)
    text = open(out).read()
    assert "P 1" in text or "'P 1'" in text
    # all 12 atoms of the conventional anatase cell explicit (4 Ti + 8 O)
    assert text.count(" Ti") >= 4 and text.count(" O") >= 8


def test_canonicalize_cif_fallback_on_unparseable(tmp_path):
    # If the CIF cannot be parsed (or pymatgen absent), fall back to the original.
    src = tmp_path / "bad.cif"
    src.write_text("not a cif")
    assert ge._canonicalize_cif(str(src), str(tmp_path)) == str(src)


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_ambiguous_cif_intensities_recovered(tmp_path):
    # End-to-end regression guard: the ambiguous anatase CIF must yield the strong
    # (200) reflection at ~48.05 deg once the engine canonicalizes it — the exact
    # case that gave Fcalc(200)=0 before the fix.
    cif = tmp_path / "amb.cif"
    cif.write_text(_ANATASE_AMBIGUOUS)
    r = ge.simulate_gsas(str(cif), "CuKa", (20.0, 55.0))
    tt = np.asarray(r["two_theta"])
    assert tt.size and np.min(np.abs(tt - 48.05)) < 0.2   # (200) present, correct position
    # (101) at 25.28 remains the strongest line
    I = np.asarray(r["intensities"])
    assert tt[int(np.argmax(I))] == pytest.approx(25.28, abs=0.15)


@pytest.mark.skipif(not ge.gsas_available(), reason="GSAS-II not installed")
def test_full_simulation_physics(tmp_path):
    # Minimal Si CIF (diamond cubic, a=5.43088) -> exact (111)/(220) at CuKa,
    # full contract (core keys + continuous profile), correct d-spacing.
    cif = tmp_path / "Si.cif"
    cif.write_text(
        "data_Si\n"
        "_cell_length_a 5.43088\n_cell_length_b 5.43088\n_cell_length_c 5.43088\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F d -3 m'\n_symmetry_Int_Tables_number 227\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si 0.0 0.0 0.0\n"
    )
    r = ge.simulate_gsas(str(cif), "CuKa", (20.0, 60.0))
    for k in ("two_theta", "intensities", "hkls", "d_spacings", "wavelength",
              "profile_two_theta", "profile_intensities", "engine_note"):
        assert k in r
    assert r["hkls"] == []
    assert r["wavelength"] == pytest.approx(1.5406)
    assert len(r["profile_two_theta"]) == len(r["profile_intensities"]) > 100
    tt = np.asarray(r["two_theta"])
    assert np.min(np.abs(tt - 28.44)) < 0.15    # (111)
    assert np.min(np.abs(tt - 47.30)) < 0.15    # (220)
    i111 = int(np.argmin(np.abs(tt - 28.44)))
    assert r["d_spacings"][i111] == pytest.approx(3.1355, abs=0.01)
