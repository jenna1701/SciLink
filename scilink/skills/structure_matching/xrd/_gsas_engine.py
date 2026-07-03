"""GSAS-II powder-XRD simulation engine (optional backend for ``simulate_xrd``).

Where the default pymatgen engine returns a *kinematic peak list* (positions +
structure-factor intensities, sufficient for peak-based phase identification),
GSAS-II computes a *full continuous profile* with realistic instrument and
sample broadening. That profile is what full-pattern matching and Rietveld-style
work need, so GSAS-II is offered as a drop-in alternative engine.

To keep the ``_ENGINES`` contract a true drop-in, this engine returns the same
core keys as the pymatgen engine (``two_theta``, ``intensities``, ``d_spacings``,
``wavelength``, ``structure_path``) — peak-picked from the profile — and *adds*
the continuous profile (``profile_two_theta`` / ``profile_intensities``) that
GSAS-II uniquely provides. Peak-based consumers are unaffected; profile-based
consumers use the extra keys.

This module is deliberately self-contained (numpy + a lazy GSAS-II import; no
scilink imports) so it can be exercised in a GSAS-II-only environment. pymatgen,
if present, is used only to canonicalize an ambiguous input CIF (see
``_canonicalize_cif``); it is optional and the engine degrades gracefully without
it.

Installation
------------
GSAS-II is an optional dependency built from source, so a Fortran compiler must
be present *at install time*. The verified recipe (conda provides the compiler
toolchain):

    conda create -n <env> python=3.12
    conda activate <env>
    conda install -c conda-forge fortran-compiler meson ninja
    pip install "scilink[gsas]"        # or: pip install -e ".[gsas]" from a checkout

The ``gsas`` extra pulls GSAS-II from its git repository
(``git+https://github.com/AdvancedPhotonSource/GSAS-II.git``) and compiles its
Fortran extensions against the environment's numpy. Install scilink/pymatgen
*before* GSAS-II so the compile links against the final numpy ABI. Note: the
compiler activation only happens inside an activated conda env — building via a
bare ``conda run`` (without activation) can fail with "gfortran cannot compile
programs" because ``CONDA_BUILD_SYSROOT`` is unset; use ``conda activate`` (or
source the env's activate scripts) for the build.

The simulation recipe is adapted, with permission, from the collaborator project
PhaseTransitionGUI (``phase_analysis.pipeline.simulation.gsas2powdersim``);
its physical correctness (peak positions, systematic absences, structure-factor
intensities, wavelength scaling) was independently verified before absorption.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

# Characteristic Kα1 wavelengths (Å) for the common lab anodes, so the engine
# accepts the same string aliases as pymatgen's XRDCalculator ('CuKa', ...).
_ANODE_WAVELENGTHS = {
    "cuka": 1.5406, "cuka1": 1.54056, "cuka2": 1.54439,
    "moka": 0.71073, "moka1": 0.70930,
    "coka": 1.78897, "feka": 1.93604, "crka": 2.28970, "agka": 0.55941,
}

# GSAS-II instrument-parameter template (constant-wavelength lab diffractometer).
# Only ``Lam`` varies with the requested wavelength; the U/V/W/X/Y/Z/SH-L profile
# terms are GSAS-II's standard reasonable lab defaults (same values the
# collaborator's Cu/Mo .instprm fixtures carried).
_INSTPRM_TEMPLATE = (
    "#GSAS-II instrument parameter file; do not add/delete items!\n"
    "Type:PXC\nBank:1.0\nLam:{lam:.6f}\nPolariz.:0.7\nAzimuth:0.0\nZero:0.0\n"
    "U:2.0\nV:-2.0\nW:5.0\nX:0.0\nY:0.0\nZ:0.0\nSH/L:0.002\n"
)


def _resolve_wavelength(wavelength: Any) -> float:
    """Resolve a wavelength spec to Å. Accepts a float/int or an anode alias
    ('CuKa', 'MoKa', …) matching pymatgen's XRDCalculator vocabulary."""
    if isinstance(wavelength, (int, float)):
        return float(wavelength)
    key = str(wavelength).strip().lower().replace(" ", "").replace("-", "")
    if key in _ANODE_WAVELENGTHS:
        return _ANODE_WAVELENGTHS[key]
    raise ValueError(
        f"Unrecognized wavelength {wavelength!r}. Pass a numeric wavelength in Å "
        f"or one of {sorted(_ANODE_WAVELENGTHS)}."
    )


def gsas_available() -> bool:
    """True if GSAS-II (GSASIIscriptable) can be imported in this environment."""
    try:
        import GSASII.GSASIIscriptable  # noqa: F401
        return True
    except Exception:
        return False


def _canonicalize_cif(structure_path: str, workdir: str) -> str:
    """Re-express a CIF with explicit atom positions (P1) via pymatgen so GSAS-II's
    structure-factor calculation is unambiguous and consistent with the pymatgen
    engine.

    An origin-choice-ambiguous CIF — only an H-M space-group symbol, no explicit
    ``_symmetry_equiv_pos_as_xyz`` operators (common for terse minimal CIFs) —
    can be resolved to a different origin by GSAS-II than by pymatgen, misplacing
    atoms and producing physically wrong intensities (verified: anatase I4_1/amd
    gave Fcalc(200)=0 from a symbol-only CIF, vs the correct strong (200) once
    coordinates were explicit). Expanding to P1 first removes the ambiguity and
    makes the two engines agree on the same input — the drop-in contract.

    Real database CIFs (COD, Materials Project) carry full symmetry operators and
    are already unambiguous; this only guards the terse-CIF tail. Falls back to
    the original path if pymatgen is unavailable or the CIF cannot be parsed, in
    which case correctness relies on the CIF carrying explicit symmetry."""
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
    except Exception:
        return structure_path
    try:
        structure = Structure.from_file(structure_path)
        out = os.path.join(workdir, "canonical.cif")
        CifWriter(structure, symprec=None).write_file(out)  # symprec=None -> P1, explicit sites
        return out
    except Exception:
        return structure_path


def _load_g2sc():
    """Lazily import GSASIIscriptable, raising an actionable error if absent."""
    try:
        from GSASII import GSASIIscriptable as G2sc
        return G2sc
    except Exception as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "The 'gsas' XRD engine requires GSAS-II (GSASIIscriptable), which is "
            "not importable in this environment. Install the optional extra: "
            "pip install 'scilink[gsas]' (builds GSAS-II from source; needs a "
            "Fortran compiler, e.g. `conda install -c conda-forge fortran-compiler "
            "meson ninja`)."
        ) from exc


def simulate_gsas(
    structure_path: str,
    wavelength: Any = "CuKa",
    two_theta_range: tuple = (10.0, 90.0),
    two_theta_step: float = 0.02,
    crystallite_um: float = 10.0,
    peak_rel_height: float = 0.01,
) -> dict[str, Any]:
    """Simulate a powder-XRD profile from a CIF with GSAS-II.

    Returns the pymatgen-engine contract keys (peak-picked from the profile) plus
    the continuous profile. See module docstring for the design rationale.

    Parameters
    ----------
    structure_path : str
        Path to a CIF file.
    wavelength : float | str
        Wavelength in Å, or an anode alias ('CuKa', 'MoKa', …).
    two_theta_range : (float, float)
        2θ window (degrees).
    two_theta_step : float
        Profile step (degrees). Sets the number of simulated profile points.
    crystallite_um : float
        Isotropic crystallite size (μm) for the sample size-broadening profile.
        LOWER it (e.g. 0.05) to broaden peaks toward a nanocrystalline sample;
        RAISE it for sharper lines. GSAS-II's collaborator default was 10 μm.
    peak_rel_height : float
        Peak-pick threshold as a fraction of the max profile intensity (default
        0.01). LOWER to report weaker reflections in the peak list; RAISE to keep
        only strong peaks. Only affects the discrete peak list, not the profile.
    """
    G2sc = _load_g2sc()
    lam = _resolve_wavelength(wavelength)
    tmin, tmax = float(min(two_theta_range)), float(max(two_theta_range))
    if tmax <= tmin:
        raise ValueError(f"two_theta_range must be increasing, got {two_theta_range}.")
    npts = max(int(round((tmax - tmin) / float(two_theta_step))) + 1, 2)

    # Work inside a scratch dir: GSAS-II writes a .gpx project + temp files.
    with tempfile.TemporaryDirectory() as tmpd:
        instprm = os.path.join(tmpd, "auto.instprm")
        with open(instprm, "w") as fh:
            fh.write(_INSTPRM_TEMPLATE.format(lam=lam))

        # Normalize the CIF to explicit P1 (if pymatgen is available) so an
        # origin-choice-ambiguous symbol-only CIF cannot be mis-resolved by
        # GSAS-II relative to the pymatgen engine — see _canonicalize_cif.
        cif_for_gsas = _canonicalize_cif(structure_path, tmpd)

        gpx = G2sc.G2Project(newgpx=os.path.join(tmpd, "sim.gpx"))
        phase = gpx.add_phase(phasefile=cif_for_gsas, phasename="phase")
        hist = gpx.add_simulated_powder_histogram(
            histname="sim",
            iparams=instprm,
            Tmin=tmin,
            Tmax=tmax,
            Tstep=(tmax - tmin) / (npts - 1),
            phases=[phase],
        )
        gpx.link_histogram_phase(hist, phase)
        phase.setSampleProfile(hist, "size", "isotropic", crystallite_um)
        gpx.do_refinements([{}])  # evaluate Ycalc (no parameters varied)

        x = np.asarray(hist.getdata("X"), dtype=float)
        ycalc = np.asarray(hist.getdata("Ycalc"), dtype=float)

    two_theta, intensities, d_spacings = _peak_pick(x, ycalc, lam, peak_rel_height)

    return {
        "two_theta": [float(v) for v in two_theta],
        "intensities": [float(v) for v in intensities],
        "hkls": [],  # GSAS profile engine does not emit per-peak hkl; peaks are picked
        "d_spacings": [float(v) for v in d_spacings],
        "wavelength": float(lam),
        "structure_path": structure_path,
        # GSAS-II's unique contribution: the full instrument/sample-broadened profile.
        "profile_two_theta": [float(v) for v in x],
        "profile_intensities": [float(v) for v in ycalc],
        "engine_note": (
            "GSAS-II full-profile simulation; 'two_theta'/'intensities' are peak-"
            "picked from the profile, 'profile_*' hold the continuous pattern."
        ),
    }


def _peak_pick(x: np.ndarray, y: np.ndarray, lam: float, rel_height: float = 0.01):
    """Pick peaks from the continuous profile and convert to d-spacings (Bragg).

    Returns (two_theta_peaks, intensities_0_100, d_spacings). Falls back to a
    plain local-maximum scan if scipy is unavailable.
    """
    y = np.asarray(y, dtype=float)
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax <= 0.0:
        return np.array([]), np.array([]), np.array([])
    thresh = rel_height * ymax
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(y, height=thresh)
    except Exception:  # pragma: no cover - scipy is a GSAS-II dep in practice
        idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > thresh))[0] + 1

    tt = x[idx]
    inten = 100.0 * y[idx] / ymax
    theta = np.radians(tt / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = lam / (2.0 * np.sin(theta))
    return tt, inten, d
