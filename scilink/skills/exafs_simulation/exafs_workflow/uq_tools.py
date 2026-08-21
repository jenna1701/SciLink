"""UQ-to-EXAFS glue: qualify MD frames by per-atom MLIP uncertainty.

Couples the per-atom uncertainty produced by UQ-MLIP's ``run-gbm.py``
(``UQ_<sample>.csv.gz`` with columns sample_idx, atom_idx, element, U_upper,
U_lower) to the EXAFS workflow. It maps each frame's uncertainty back to the
absorber and its scattering shell, then flags frames where the MLIP was
extrapolating in the region that actually shapes chi(k).

Important scope note
--------------------
This does NOT convert an energy-prediction interval into an error bar on
chi(k) — there is no calibrated map from a per-atom energy quantile interval
(eV) to a positional error (Angstrom). What it provides is a *confidence
gate*: which frames' geometries are trustworthy enough to include in the
average. Combined with the per-k sampling band from ``average_chi``
(chi_std / chi_sem), the result is a confidence-qualified spectrum, not a
first-principles MLIP-error band.

Assumes UQ embeddings were extracted on the SAME trajectory file used for
FEFF generation with ``--index ":"`` so that ``sample_idx`` equals the
trajectory frame index (which is also the frame index encoded in FEFF
subdir names).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from scilink.skills._shared._spec import ToolSpec

# UQ frame-gating is an OPTIONAL extension of the EXAFS skill: it couples the
# UQ-MLIP output to the FEFF averaging step. Its only extra dependency beyond
# the core workflow is pandas (for reading the UQ CSV). We import it lazily so
# this module always loads and ``select_frames_by_uq`` stays discoverable in
# the tool inventory even when pandas is absent — the caller then gets a clear,
# actionable error instead of the tool silently disappearing.


def _require_pandas():
    """Import pandas or raise an actionable error naming the optional extra."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "select_frames_by_uq requires pandas to read the UQ-MLIP CSV. "
            "This is an optional part of the EXAFS skill; install it with "
            "`pip install pandas` (or `pip install scilink[uq]`)."
        ) from exc
    return pd


def _per_atom_uncertainty(df):
    """Add a half-width ``U`` column: (U_upper - U_lower) / 2.

    Matches ``GBMRegressor.uncertainty`` in UQ-MLIP's ``gbm.py``.
    """
    df = df.copy()
    df["U"] = np.abs(df["U_upper"] - df["U_lower"]) / 2.0
    return df


def _neighbor_indices(atoms, target_atom: int, shell_radius: float) -> np.ndarray:
    """Indices of atoms within ``shell_radius`` of the absorber (mic), incl. absorber."""
    dists = atoms.get_distances(target_atom, range(len(atoms)), mic=True)
    within = np.where(dists <= shell_radius)[0]
    # Ensure the absorber itself is included even at shell_radius 0.
    return np.union1d(within, np.array([target_atom]))


def select_frames_by_uq(
    uq_csv: str,
    trajectory_path: str,
    target_atom: int,
    threshold: float,
    shell_radius: float = 6.0,
    aggregate: str = "max",
) -> dict[str, Any]:
    """Select MD frames where the MLIP is reliable near the absorber.

    For each frame, aggregates the per-atom uncertainty over the absorber and
    every atom within ``shell_radius`` of it, then keeps the frame if that
    aggregate is at or below ``threshold``.

    Parameters
    ----------
    uq_csv : str
        Path to UQ-MLIP output (``UQ_<sample>.csv.gz``). Must contain
        sample_idx, atom_idx, U_upper, U_lower. ``sample_idx`` must equal the
        trajectory frame index (extract embeddings with ``--index ":"`` on the
        same trajectory used for FEFF generation).
    trajectory_path : str
        Trajectory used both for FEFF generation and UQ embedding extraction.
    target_atom : int
        Index of the absorbing atom (same convention as the FEFF tools).
    threshold : float
        Maximum acceptable aggregated shell uncertainty (same units as the
        UQ model's per-atom energy interval, typically eV/atom).
    shell_radius : float
        Radius in Angstroms defining the scattering shell to inspect. Should
        roughly match the FEFF RMAX (default 6.0).
    aggregate : {"max", "mean"}
        How to combine per-atom uncertainties within the shell. "max" is
        conservative (one bad neighbor fails the frame); "mean" is lenient.

    Returns
    -------
    dict
        Keys: passing_frames (list[int]), failing_frames (list[int]),
        frame_uncertainty (dict[int, float]), n_pass (int), n_fail (int),
        threshold (float), aggregate (str), shell_radius (float).
    """
    if aggregate not in ("max", "mean"):
        raise ValueError("aggregate must be 'max' or 'mean'")

    pd = _require_pandas()
    df = _per_atom_uncertainty(pd.read_csv(uq_csv))
    trajectory = read(str(Path(trajectory_path)), ":")

    reduce = np.max if aggregate == "max" else np.mean

    frame_uncertainty: dict[int, float] = {}
    for frame, group in df.groupby("sample_idx"):
        frame = int(frame)
        if frame >= len(trajectory):
            # UQ has more frames than the trajectory we were handed; skip.
            continue
        atoms = trajectory[frame]
        shell = _neighbor_indices(atoms, target_atom, shell_radius)

        # atom_idx within this frame -> U; select the shell atoms.
        u_by_atom = group.set_index("atom_idx")["U"]
        shell_u = u_by_atom.reindex(shell).dropna()
        if shell_u.empty:
            continue
        frame_uncertainty[frame] = float(reduce(shell_u.to_numpy()))

    passing = sorted(f for f, u in frame_uncertainty.items() if u <= threshold)
    failing = sorted(f for f, u in frame_uncertainty.items() if u > threshold)

    return {
        "passing_frames": passing,
        "failing_frames": failing,
        "frame_uncertainty": frame_uncertainty,
        "n_pass": len(passing),
        "n_fail": len(failing),
        "threshold": threshold,
        "aggregate": aggregate,
        "shell_radius": shell_radius,
    }


# ---------------------------------------------------------------------------
# TOOL_SPECS
# ---------------------------------------------------------------------------

TOOL_SPECS = [
    ToolSpec(
        name="select_frames_by_uq",
        description=(
            "Select MD frames for EXAFS averaging by per-atom MLIP "
            "uncertainty. Aggregates UQ-MLIP per-atom uncertainty over the "
            "absorber and its scattering shell and keeps frames at or below a "
            "threshold. Produces a confidence gate for average_chi, not a "
            "chi(k) error bar."
        ),
        parameters={
            "uq_csv": {
                "type": "string",
                "description": (
                    "UQ-MLIP output CSV (UQ_<sample>.csv.gz) with sample_idx, "
                    "atom_idx, U_upper, U_lower. sample_idx must equal the "
                    "trajectory frame index."
                ),
            },
            "trajectory_path": {
                "type": "string",
                "description": (
                    "Trajectory used for both FEFF generation and UQ "
                    "embedding extraction."
                ),
            },
            "target_atom": {
                "type": "integer",
                "description": "Index of the absorbing atom.",
            },
            "threshold": {
                "type": "number",
                "description": (
                    "Max acceptable aggregated shell uncertainty (UQ model "
                    "units, typically eV/atom)."
                ),
            },
            "shell_radius": {
                "type": "number",
                "description": (
                    "Scattering shell radius in Angstroms to inspect "
                    "(default 6.0; match FEFF RMAX)."
                ),
            },
            "aggregate": {
                "type": "string",
                "description": '"max" (conservative) or "mean" (lenient).',
            },
        },
        required=["uq_csv", "trajectory_path", "target_atom", "threshold"],
        import_line=(
            "from scilink.skills.exafs_simulation.exafs_workflow.uq_tools "
            "import select_frames_by_uq"
        ),
        signature=(
            "select_frames_by_uq(uq_csv, trajectory_path, target_atom, "
            "threshold, shell_radius=6.0, aggregate='max') -> dict"
        ),
        agents=["simulation"],
        when_to_use=(
            "Between UQ-MLIP prediction and average_chi, to restrict the "
            "EXAFS average to snapshots where the MLIP was reliable in the "
            "absorber's neighborhood."
        ),
        returns=(
            "Dict with passing_frames / failing_frames lists, per-frame shell "
            "uncertainty, counts, and the threshold settings. Pass "
            "passing_frames to average_chi(include_frames=...)."
        ),
        example=(
            "from scilink.skills.exafs_simulation.exafs_workflow.uq_tools "
            "import select_frames_by_uq\n"
            "from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
            "import average_chi\n\n"
            "sel = select_frames_by_uq(\n"
            "    uq_csv='results/gbm_mace/UQ_md_trajectory.csv.gz',\n"
            "    trajectory_path='md_trajectory.xyz',\n"
            "    target_atom=0, threshold=0.05, shell_radius=6.0)\n"
            "avg = average_chi(sel_dir, 'exafs_uq',\n"
            "                  include_frames=sel['passing_frames'])"
        ),
    ),
]
