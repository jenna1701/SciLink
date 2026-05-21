"""Local CIF directory backend.

Stub for the scaffolding commit — concrete query logic lands in commit 2.
"""

from __future__ import annotations

from ._base import QuerySpec, StructureCandidate


class LocalCIFBackend:
    name = "local"

    def __init__(self, root_dir: str | None = None) -> None:
        self.root_dir = root_dir

    def is_available(self) -> bool:
        return False  # placeholder

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        raise NotImplementedError("LocalCIFBackend.query not yet implemented")
