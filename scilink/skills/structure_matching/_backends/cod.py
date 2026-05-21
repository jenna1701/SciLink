"""Crystallography Open Database backend (stub).

COD has a REST API and no authentication requirement. Implementation is
deferred — the stub exists so the dispatch loop and registry behave the
same way they will once COD is wired in.
"""

from __future__ import annotations

from ._base import QuerySpec, StructureCandidate


class CODBackend:
    name = "cod"

    def is_available(self) -> bool:
        return False

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        raise NotImplementedError(
            "CODBackend is a v1 stub. See plan: implementation deferred."
        )
