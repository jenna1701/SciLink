"""Protocol + dataclasses for structure-database backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class QuerySpec:
    """Specification for a structure-database query.

    Fields are interpreted independently by each backend; a backend may
    ignore any field it does not support (e.g. ``LocalCIFBackend`` ignores
    ``max_e_above_hull`` since CIFs on disk carry no thermodynamic data).
    """

    chemistry: list[str]
    space_group_hints: Optional[list[int]] = None
    lattice_param_ranges: Optional[dict[str, tuple[float, float]]] = None
    max_e_above_hull: Optional[float] = None
    top_n: int = 10

    def __post_init__(self) -> None:
        if not self.chemistry:
            raise ValueError("QuerySpec.chemistry must be non-empty")
        if self.top_n <= 0:
            raise ValueError("QuerySpec.top_n must be positive")


@dataclass
class StructureCandidate:
    """A single candidate structure returned by a backend.

    ``structure_path`` is populated by the dispatching tool after CIFs are
    materialized into the session directory — backends construct the
    candidate first and the dispatcher writes the file.
    """

    id: str
    source: str
    formula: str
    space_group: Optional[str] = None
    structure_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0


@runtime_checkable
class StructureBackend(Protocol):
    """Protocol every backend must satisfy.

    Backends are stateless once constructed (any caching is internal). The
    dispatching tool calls ``is_available()`` to filter the registered list,
    then ``query()`` on the survivors.
    """

    name: str

    def is_available(self) -> bool:
        """True when this backend can answer queries.

        A backend that needs an API key returns False when the key is
        missing; a backend that needs a local directory returns False when
        the directory does not exist. Must never raise.
        """
        ...

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        """Return candidates matching ``spec``.

        Returned candidates are unsorted; ranking and dedup are the
        dispatcher's responsibility. Implementation may raise on real
        errors (network failure, malformed response) — the dispatcher
        catches and continues with other backends.
        """
        ...
