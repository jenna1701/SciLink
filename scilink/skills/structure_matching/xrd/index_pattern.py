"""``index_pattern`` tool — powder autoindexing: peak positions → candidate unit
cells (GSAS-II ``gsas`` backend).

The entry point to *blind* phase identification — when the sample chemistry is
unknown, ``search_structures``' chemistry-keyed query cannot be formed, but a
unit cell recovered from the peak positions alone can drive a lattice-parameter
search instead. Wraps GSAS-II's ITO-style ``DoIndexPeaks`` (random-cell volume
sweep per Bravais lattice + least-squares refinement, ranked by the de Wolff
M20 figure of merit).

The heavy lifting lives in the self-contained GSAS-II wrapper
(``_gsas_engine.index_pattern``); this module is the thin, registry-visible
TOOL_SPEC surface. GSAS-II is the optional ``gsas`` extra (see ``_gsas_engine``
for the install recipe)."""

from __future__ import annotations

from typing import Any, Union

from ..._shared._spec import ToolSpec


TOOL_SPEC = ToolSpec(
    name="index_pattern",
    description=(
        "Autoindex a powder XRD pattern: recover ranked candidate UNIT CELLS "
        "(lattice parameters + Bravais lattice + de Wolff M20 figure of merit) "
        "from peak positions alone — no chemistry needed. THE tool for blind "
        "identification of an unknown sample: index first, then search structure "
        "databases by the recovered lattice parameters instead of by elements. "
        "Needs >=10 clean peaks to be reliable. Requires the optional 'gsas' "
        "extra (GSAS-II)."
    ),
    import_line="from scilink.skills.structure_matching.xrd.index_pattern import index_pattern",
    signature=(
        "index_pattern(two_theta_peaks, wavelength='CuKa', crystal_systems=None, "
        "zero_offset=0.0, max_nc_no=4, start_volume=25.0, "
        "timeout_per_lattice=30.0, m20_min=2.0, top_n=10) -> dict"
    ),
    parameters={
        "two_theta_peaks": {"type": "list[float]", "description": "Peak positions in 2θ degrees (use extract_peaks' 'positions'). Feed ALL confident peaks — autoindexing fails with few peaks (>=10 needed for reliability; <5 raises)."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or wavelength in Å. Must match how the pattern was measured — a wrong wavelength scales every d-spacing and yields a wrong (but plausible-looking) cell."},
        "crystal_systems": {"type": "list[str]", "description": "Crystal systems to search, from ['cubic','trigonal','hexagonal','tetragonal','orthorhombic','monoclinic','triclinic']. Default: all EXCEPT triclinic (slow and unreliable — opt in explicitly). NARROW it when the pattern suggests high symmetry (few, sharp, well-separated peaks → try ['cubic','tetragonal','hexagonal'] first: much faster and fewer false cells)."},
        "zero_offset": {"type": "float", "description": "2θ zero correction in degrees applied during indexing (default 0). Set if the diffractometer zero error is known; an uncorrected zero shifts all d-spacings and degrades M20."},
        "max_nc_no": {"type": "int", "description": "de Wolff cap on calculated/observed reflection ratio (default 4). RAISE to allow larger unit cells (more calculated lines per observed peak); LOWER to force parsimony on simple patterns."},
        "start_volume": {"type": "float", "description": "Starting cell volume in Å^3 for the search sweep (default 25). The sweep grows volume automatically — RAISE only to skip small cells when the cell is known to be large (speeds the search)."},
        "timeout_per_lattice": {"type": "float", "description": "Seconds per Bravais lattice before moving on (default 30). RAISE for a thorough low-symmetry (monoclinic) search; LOWER for quick triage."},
        "m20_min": {"type": "float", "description": "Minimum M20 figure of merit to keep a candidate (default 2). RAISE (10–20) to return only convincing solutions."},
        "top_n": {"type": "int", "description": "Maximum candidate cells returned (default 10)."},
    },
    required=["two_theta_peaks"],
    returns=(
        "dict with 'candidate_cells' (ranked list of {M20, X20, bravais, "
        "crystal_system, a, b, c, alpha, beta, gamma, volume}), "
        "'lattice_param_ranges' (±2% a/b/c ranges around the best cell — pass "
        "directly to search_structures' lattice filter), 'n_peaks_used', "
        "'dmin_angstrom', 'wavelength', 'warnings', 'note'. Ranking: M20 "
        "descending with the SMALLEST volume first among near-equal M20 (a "
        "larger equal-M20 cell is usually a superlattice alias). M20 > ~10 with "
        "X20 = 0 is convincing; M20 near the cutoff is a guess. Solutions come "
        "in FAMILIES related by common factors (×1/2 subcells from lattice "
        "centering, ×√2/×√3 supercells) — if the top cell finds no database "
        "match, try other candidates and volume-related multiples. The "
        "simulate+score loop is the arbiter of a candidate cell, not M20 alone; "
        "iterate the knobs (more peaks, narrower crystal_systems, zero_offset) "
        "when scoring rejects."
    ),
    when_to_use=(
        "When the sample chemistry is UNKNOWN (blind identification): "
        "extract_peaks → index_pattern → search structure databases by the "
        "recovered lattice parameters → simulate + score the candidates as "
        "usual. Also useful to corroborate a chemistry-led identification: the "
        "indexed cell should match the identified phase's cell. Not reliable "
        "for triclinic phases or patterns with <10 clean peaks."
    ),
)


def index_pattern(
    two_theta_peaks,
    wavelength: Union[str, float] = "CuKa",
    crystal_systems: list = None,
    zero_offset: float = 0.0,
    max_nc_no: int = 4,
    start_volume: float = 25.0,
    timeout_per_lattice: float = 30.0,
    m20_min: float = 2.0,
    top_n: int = 10,
) -> dict[str, Any]:
    """Autoindex a powder pattern into candidate unit cells. See ``TOOL_SPEC``.

    Thin wrapper over the GSAS-II engine; lazy-imported so the module stays
    importable without the optional ``gsas`` dependency."""
    from ._gsas_engine import index_pattern as _impl
    return _impl(
        two_theta_peaks, wavelength=wavelength, crystal_systems=crystal_systems,
        zero_offset=zero_offset, max_nc_no=max_nc_no, start_volume=start_volume,
        timeout_per_lattice=timeout_per_lattice, m20_min=m20_min, top_n=top_n,
    )
