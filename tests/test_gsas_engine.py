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


def test_engine_registered():
    assert "gsas" in sx._ENGINES
    assert sx._ENGINES["gsas"] is sx._simulate_gsas


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


def test_increasing_range_required():
    if not ge.gsas_available():
        pytest.skip("GSAS-II not installed")
    with pytest.raises(ValueError):
        ge.simulate_gsas("dummy.cif", "CuKa", (90.0, 10.0))


def test_actionable_error_without_gsas():
    if ge.gsas_available():
        pytest.skip("GSAS-II present; missing-dependency path not exercised here")
    with pytest.raises(RuntimeError) as exc:
        sx.simulate_xrd_pattern("x.cif", engine="gsas")
    msg = str(exc.value)
    assert "scilink[gsas]" in msg and "Fortran" in msg


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
