"""Materials Project backend.

Stub for the scaffolding commit — concrete query logic lands in commit 2.
"""

from __future__ import annotations

from ._base import QuerySpec, StructureCandidate


class MaterialsProjectBackend:
    name = "mp"

    def is_available(self) -> bool:
        return False  # placeholder

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        raise NotImplementedError("MaterialsProjectBackend.query not yet implemented")
