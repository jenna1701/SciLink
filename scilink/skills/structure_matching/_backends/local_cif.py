"""Local CIF directory backend — walks a directory of CIF files."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ._base import QuerySpec, StructureCandidate

try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Structure = None  # type: ignore

_logger = logging.getLogger(__name__)


class LocalCIFBackend:
    """Backend that walks a directory of CIF files.

    Root directory comes from the ``root_dir`` constructor arg or the
    ``SCILINK_LOCAL_CIF_DIR`` env var. Subdirectories are walked
    recursively. Files are parsed lazily via pymatgen on each query.
    """

    name = "local"

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        if self.root_dir is None:
            env = os.getenv("SCILINK_LOCAL_CIF_DIR")
            if env:
                self.root_dir = Path(env)

    def is_available(self) -> bool:
        return (
            PYMATGEN_AVAILABLE
            and self.root_dir is not None
            and self.root_dir.is_dir()
        )

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        if not self.is_available():
            return []

        target_elements = set(spec.chemistry)
        candidates: list[StructureCandidate] = []

        for cif_path in sorted(self.root_dir.rglob("*.cif")):  # type: ignore[union-attr]
            try:
                struct = Structure.from_file(str(cif_path))
            except Exception as e:
                _logger.debug("Skipping %s: parse failed (%s)", cif_path, e)
                continue

            present_elements = {str(el) for el in struct.composition.elements}
            if not target_elements.issubset(present_elements):
                continue
            if target_elements != present_elements:
                continue

            sg_symbol = None
            sg_number = None
            try:
                sg = struct.get_space_group_info()
                sg_symbol, sg_number = sg[0], sg[1]
            except Exception:
                pass

            if spec.space_group_hints and sg_number not in spec.space_group_hints:
                continue

            lattice = struct.lattice
            if spec.lattice_param_ranges and not _lattice_within_ranges(
                lattice, spec.lattice_param_ranges,
            ):
                continue

            candidates.append(StructureCandidate(
                id=cif_path.stem,
                source=self.name,
                formula=str(struct.composition.reduced_formula),
                space_group=sg_symbol,
                structure_path=str(cif_path),
                metadata={
                    "spacegroup_number": sg_number,
                    "lattice_a": lattice.a,
                    "lattice_b": lattice.b,
                    "lattice_c": lattice.c,
                },
                rank_score=1.0,
            ))
            if len(candidates) >= spec.top_n * 3:  # collect headroom before final dedup/rank
                break

        candidates.sort(key=lambda c: c.id)
        return candidates[:spec.top_n]


def _lattice_within_ranges(lattice, ranges: dict) -> bool:
    """Check lattice parameters against ``{a: (min, max), b: ..., c: ...}`` ranges."""
    for key in ("a", "b", "c"):
        if key in ranges:
            lo, hi = ranges[key]
            val = getattr(lattice, key)
            if not (lo <= val <= hi):
                return False
    for key in ("alpha", "beta", "gamma"):
        if key in ranges:
            lo, hi = ranges[key]
            val = getattr(lattice, key)
            if not (lo <= val <= hi):
                return False
    return True
