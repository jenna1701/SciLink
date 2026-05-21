"""LLM-friendly forwarding shim for skill-bundled tools.

Live testing showed Claude routinely synthesizes ``from scilink.agents.skills.<skill>.<tool>``
imports even though the canonical path is ``scilink.skills.<domain>.<skill>.<tool>``
— a natural hallucination given the model's "skills live under the agent
namespace" intuition. Rather than fight that pattern through prompt
hardening (which kept failing in trials), this package mirrors the
canonical locations under the guessed path via ``sys.modules`` entries.

Tool implementations stay at their real homes under ``scilink/skills/``;
this module only re-exposes them at the alias path so both imports
resolve to the same Python objects.

To add a skill: register its sub-packages in ``_ALIASES`` below.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
import types as _types


_ALIASES = {
    # alias path  →  canonical module path
    "scilink.agents.skills.xrd": "scilink.skills.structure_matching.xrd",
    "scilink.agents.skills.xrd.search_structures": "scilink.skills.structure_matching.xrd.search_structures",
    "scilink.agents.skills.xrd.simulate_xrd": "scilink.skills.structure_matching.xrd.simulate_xrd",
    "scilink.agents.skills.xrd.score_match_fast": "scilink.skills.structure_matching.xrd.score_match_fast",
    "scilink.agents.skills.xrd.score_match_robust": "scilink.skills.structure_matching.xrd.score_match_robust",
    "scilink.agents.skills.xrd.extract_peaks": "scilink.skills.structure_matching.xrd.extract_peaks",
}


def _install_aliases() -> None:
    for alias, canonical in _ALIASES.items():
        try:
            mod = _importlib.import_module(canonical)
        except ImportError:
            # Skip aliases whose underlying canonical module has optional
            # heavy deps that aren't installed (e.g. pymatgen XRD module).
            continue
        _sys.modules[alias] = mod


_install_aliases()
