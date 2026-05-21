"""Backend dispatch for structure-database queries.

This subpackage is private to the ``structure_matching`` skill domain. It is
not visible to the skill discovery walker (the leading underscore in
``_backends`` excludes it; see ``scilink/skills/loader.py:117-118``).

Backends implement the :class:`StructureBackend` protocol declared in
``_base``. Skill tools (e.g. ``xrd/search_structures.py``) dispatch over the
registered backend list, filtered by ``is_available()``.
"""

from ._base import StructureBackend, StructureCandidate, QuerySpec
from .materials_project import MaterialsProjectBackend
from .local_cif import LocalCIFBackend
from .cod import CODBackend

__all__ = [
    "StructureBackend",
    "StructureCandidate",
    "QuerySpec",
    "MaterialsProjectBackend",
    "LocalCIFBackend",
    "CODBackend",
]
