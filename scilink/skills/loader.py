"""
Skill loader for domain-specific knowledge files.

Skills are markdown files with structured sections (## headings) that provide
domain-specific guidance for LLM-driven analysis pipelines. They can be
built-in (shipped with the package) or user-provided (custom .md files).

A skill file may optionally begin with a YAML frontmatter block delimited by
``---`` lines, e.g.::

    ---
    description: One-line LLM-facing blurb
    domain: force_field
    applies_to: [amber, gaff2, ff14sb]
    ---
    ## overview
    ...

Frontmatter, when present, is exposed under the ``meta`` key of the parsed
skill. Sections whose heading isn't in the canonical vocabulary are preserved
under the ``extras`` key (lowercased heading → body) and a warning is logged.
"""

import logging
import re
from pathlib import Path
from typing import Dict

import yaml

_SKILLS_DIR = Path(__file__).parent

_KNOWN_SECTIONS = {"overview", "planning", "analysis", "interpretation", "validation", "implementation"}

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_logger = logging.getLogger(__name__)


def list_skills(domain: str = "curve_fitting") -> list:
    """Return names of available built-in skills for a domain.

    Args:
        domain: Skill domain subdirectory (default: "curve_fitting").

    Returns:
        Sorted list of skill name strings (file stems).
    """
    skills_dir = _SKILLS_DIR / domain
    if not skills_dir.is_dir():
        return []
    return sorted(p.stem for p in skills_dir.glob("*.md"))


def list_all_skills() -> dict:
    """Auto-discover all built-in skill domains and their skills.

    Scans subdirectories of the skills package directory for ``.md`` files.

    Returns:
        Dict mapping domain names to sorted lists of skill names,
        e.g. ``{"curve_fitting": ["xps"], "hyperspectral": ["eels"]}``.
        Empty domains are omitted.
    """
    result = {}
    for sub in sorted(_SKILLS_DIR.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        names = sorted(p.stem for p in sub.glob("*.md"))
        if names:
            result[sub.name] = names
    return result


def load_skill(skill: str, domain: str = "curve_fitting") -> Dict:
    """Load and parse a skill markdown file.

    Args:
        skill: Either a built-in skill name (e.g. "xps") which resolves to
            ``scilink/skills/{domain}/{name}.md``, or a path to a custom
            ``.md`` file.
        domain: Skill domain subdirectory (default: "curve_fitting").

    Returns:
        Dict with keys: ``name`` (file stem), ``meta`` (frontmatter dict,
        empty if no frontmatter), one entry per canonical section in
        :data:`_KNOWN_SECTIONS` (missing sections default to empty string),
        and ``extras`` (dict of non-canonical-heading → body).

    Raises:
        FileNotFoundError: If the skill file cannot be found.
        ValueError: If the file is empty or cannot be parsed.
    """
    path = _resolve_skill_path(skill, domain)
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"Skill file is empty: {path}")

    meta, body = _split_frontmatter(text, source=str(path))
    sections, extras = _parse_sections(body, source=str(path))

    sections["name"] = path.stem
    sections["meta"] = meta
    sections["extras"] = extras
    return sections


def _resolve_skill_path(skill: str, domain: str) -> Path:
    """Resolve a skill name or path to an actual file path."""
    candidate = Path(skill)
    if candidate.suffix.lower() == ".md":
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Skill file not found: {candidate}")

    built_in = _SKILLS_DIR / domain / f"{skill}.md"
    if built_in.exists():
        return built_in

    available_dir = _SKILLS_DIR / domain
    available = sorted(p.stem for p in available_dir.glob("*.md")) if available_dir.is_dir() else []
    raise FileNotFoundError(
        f"Skill '{skill}' not found. Available built-in skills for '{domain}': {available}"
    )


def _split_frontmatter(text: str, source: str = "<skill>") -> tuple[dict, str]:
    """Strip an optional leading ``---``-fenced YAML frontmatter block.

    Returns ``(meta_dict, remaining_text)``. If no frontmatter is present,
    returns ``({}, text)``. Malformed YAML logs a warning and yields an
    empty meta dict but does not raise — the body is returned with the
    frontmatter stripped so section parsing can proceed.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    raw = m.group(1)
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        _logger.warning("Malformed frontmatter in %s: %s", source, e)
        return {}, text[m.end():]

    if not isinstance(parsed, dict):
        _logger.warning(
            "Frontmatter in %s did not parse to a mapping (got %s); ignoring.",
            source, type(parsed).__name__,
        )
        return {}, text[m.end():]

    return parsed, text[m.end():]


def _parse_sections(text: str, source: str = "<skill>") -> tuple[Dict[str, str], Dict[str, str]]:
    """Parse markdown into sections keyed by ``## heading`` name.

    Returns ``(known_sections, extras)`` where ``known_sections`` covers
    the canonical vocabulary (missing entries default to empty string) and
    ``extras`` captures any other ``## ...`` sections (lowercased heading
    → body). Unknown headings emit a warning so authors get feedback when
    content would otherwise be silently dropped.
    """
    sections = {k: "" for k in _KNOWN_SECTIONS}
    extras: Dict[str, str] = {}

    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)

    for part in parts[1:]:
        lines = part.split("\n", 1)
        heading = lines[0].strip().lower()
        body = lines[1].strip() if len(lines) > 1 else ""
        if heading in _KNOWN_SECTIONS:
            sections[heading] = body
        else:
            extras[heading] = body
            _logger.warning(
                "Skill %s: section '## %s' is not in the canonical vocabulary "
                "(%s); preserved under 'extras'.",
                source, heading, sorted(_KNOWN_SECTIONS),
            )

    return sections, extras
