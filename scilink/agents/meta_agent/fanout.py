"""Parallel multi-dataset analysis (fan-out) + complementarity gating + fusion.

See CLAUDE.md "The meta agent". This is the meta's **fan-out primitive**: run
several analysis branches concurrently over GENUINELY COMPLEMENTARY datasets —
each branch sees the others as full-mesh auxiliary operands — then fuse their
findings into one cross-dataset narrative.

Two guards bracket the fan-out, because the failure mode here is not a crash
but a *plausible fabrication*:

1. **Entry gate (complementarity).** Before any branch launches, the datasets
   are assessed and PARTITIONED. The fan-out runs only over the complementary
   subset that shares a join axis; redundant duplicates and unrelated outliers
   are pruned out. Forcing the fusion template over unrelated data would
   manufacture a correlation that isn't there — the gate is what prevents that.
2. **Exit guard (anti-spurious-fusion).** Even on a clean gate, the fusion
   prompt states that "no correlation found" is a valid, valuable conclusion,
   so the synthesis reconciles the evidence rather than inventing a link.

Branches run AUTONOMOUS regardless of the meta's mode: concurrent `input()`
human-feedback prompts cannot interleave across threads, so a parallel branch
cannot pause for approval. The single up-front user confirmation (AUTOPILOT)
compensates for the per-branch approval the user would otherwise get.

The logic lives here as free functions taking the orchestrator instance; thin
``MetaOrchestratorAgent`` methods wrap them, matching how ``telemetry.py`` is a
sibling helper to the orchestrator.
"""

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("meta_agent.fanout")

# Concurrency + sizing. The complementary SET (post-gate) is what these bound,
# not the raw input: the gate prunes first, so a 6-upload request with one
# complementary pair runs a 2-way mesh, not a 6-way one.
FANOUT_MAX_WORKERS = 4          # peak concurrent branches (rate-limit ceiling)
FANOUT_SOFT_CAP = 5             # warn / confirm beyond this many fused branches
FANOUT_HARD_CAP = 8             # refuse beyond this — cost/quality cliff
# In AUTONOMOUS mode there is no human to confirm, so the verdict IS the gate:
# proceed only on a confident 'complementary' read.
AUTONOMOUS_CONFIDENCE_THRESHOLD = 0.6


# ======================================================================
# Complementarity gate
# ======================================================================

COMPLEMENTARITY_ASSESSMENT_INSTRUCTIONS = """You are a measurement scientist deciding whether several datasets are \
GENUINELY COMPLEMENTARY — i.e. whether fusing their analyses into one \
cross-dataset narrative is scientifically meaningful, or would instead \
manufacture a correlation that the data do not support.

Datasets are complementary only when ALL THREE hold:
1. SAME SUBJECT — they measure the same physical system / sample / region.
2. NON-REDUNDANT — they carry different information (different modality, \
observable, or condition); two measurements of the same thing the same way \
are redundant, not complementary.
3. JOINABLE — a concrete axis exists to reconcile them ON: spatial \
co-registration, a shared energy/time/parameter axis, or a shared \
sample/condition. A join does NOT require pixel-level co-registration: \
reconciling one modality's spatially-resolved or local measurement against \
another's bulk / area-averaged measurement of the SAME sample (e.g. microscopy \
phase fractions vs XPS/XRD/EDX composition) is itself a valid join — this \
bulk-vs-local reconciliation is a canonical multimodal case, not a \
manufactured correlation. Without any join there is nothing to fuse.

Partition the datasets accordingly. Put into `fanout_set` ONLY a subset that \
is mutually complementary on all three criteria and shares ONE join axis \
(>= 2 members to be worth running in parallel). Cluster exact-duplicate / \
same-information datasets in `redundant_clusters`. List datasets that belong \
to a different system or have no join axis in `unrelated`.

Be conservative: if you are not confident the datasets share a system and a \
join axis, prefer `uncertain` over `complementary`. A wrong `complementary` \
call produces a fabricated cross-dataset claim, which is worse than declining.

Respond in valid JSON with EXACTLY these keys:
{
  "verdict": "complementary" | "partially_complementary" | "redundant" | "unrelated" | "uncertain",
  "confidence": <float 0..1>,
  "rationale": "<one or two sentences: what the datasets are and why this verdict>",
  "join_axis": "<the shared axis the fanout_set reconciles on, or null>",
  "fanout_set": ["<path>", ...],
  "redundant_clusters": [["<path>", "<path>"], ...],
  "unrelated": ["<path>", ...],
  "excluded_notes": "<why anything was left out of fanout_set, or empty>"
}
"""

_ANTI_SPURIOUS_FUSION_GUARD = """
IMPORTANT — do not manufacture correlations. These datasets were screened as \
plausibly complementary, but the evidence is the final authority. If the \
findings do NOT actually correlate or reconcile, say so plainly: "no \
significant cross-dataset correlation found" is a valid and valuable \
conclusion. Assert a correlation ONLY where the provided findings support it; \
never invent one to satisfy the synthesis.
"""


def _slug(text: str, maxlen: int = 32) -> str:
    """Filesystem-safe short slug from a label."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (s[:maxlen] or "branch")


_LLM_JSON_ATTEMPTS = 3


def _structured_model(orch):
    """A TOOL-FREE generative model for the gate/fusion JSON calls.

    The meta's chat model is built WITH the delegation tool schemas, so a
    structured-output prompt about datasets intermittently comes back as an
    attempted tool-call with empty text instead of the JSON we asked for.
    A dedicated tool-less model (same provider routing / credentials) removes
    that failure mode. Cached on the orchestrator.
    """
    m = getattr(orch, "_fanout_structured_model", None)
    if m is not None:
        return m
    if orch.base_url:
        from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
        m = OpenAIAsGenerativeModel(
            model=orch.model_name, api_key=orch.api_key, base_url=orch.base_url)
    else:
        from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
        m = LiteLLMGenerativeModel(
            model=orch.model_name, api_key=orch.api_key,
            system_instruction="You output only valid JSON exactly as instructed.",
            tools=None)
    orch._fanout_structured_model = m
    return m


def _parse_json_block(text: str) -> Optional[dict]:
    """Parse a JSON object from raw model text — tolerant of ```json fences and
    surrounding prose (falls back to the first balanced ``{...}``)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except Exception:  # noqa: BLE001 - fall back to first balanced {...}
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None


def _llm_json(orch, prompt: str) -> Optional[dict]:
    """LLM call returning parsed JSON, or None after retries.

    Retries on an empty completion or an unparseable body — Bedrock
    intermittently returns an empty content block, and silently fail-closing
    the gate on that transient would wrongly decline a complementary set.
    Only a persistent failure returns None (which callers fail closed on).
    """
    model = _structured_model(orch)
    for attempt in range(_LLM_JSON_ATTEMPTS):
        try:
            resp = model.generate_content(contents=[prompt])
            text = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"complementarity/fusion LLM call failed "
                           f"(attempt {attempt + 1}): {e}")
            continue
        parsed = _parse_json_block(text)
        if parsed is not None:
            return parsed
        logger.warning("complementarity/fusion LLM returned "
                       f"{'empty' if not text else 'unparseable'} response "
                       f"(attempt {attempt + 1}/{_LLM_JSON_ATTEMPTS}); retrying")
    return None


def _dataset_descriptor(path: str, role: Optional[str],
                        metadata: Optional[str]) -> dict:
    """Lightweight, router-tier descriptor of one dataset for the gate.

    Reuses the meta's content probe (shape/dtype, table columns, document /
    image dims) — the same evidence the meta routes on — plus any user-stated
    role and metadata. Deliberately does NOT load full arrays: the gate is a
    judgement over descriptors, consistent with the meta being a router.
    """
    from .meta_orchestrator_tools import _probe_file

    p = Path(path)
    desc: Dict[str, Any] = {"path": str(path)}
    if not p.exists():
        desc["note"] = "file not found"
        return desc
    try:
        desc["probe"] = _probe_file(p)
    except Exception as e:  # noqa: BLE001 - probe must not break the gate
        desc["note"] = f"probe failed: {e}"
    if role:
        desc["stated_role"] = role
    if metadata:
        mp = Path(metadata)
        if mp.exists() and mp.suffix.lower() == ".json":
            try:
                with open(mp, "r", errors="replace") as fh:
                    desc["metadata"] = json.load(fh)
            except Exception:  # noqa: BLE001
                desc["metadata"] = str(metadata)
        else:
            desc["metadata"] = str(metadata)
    return desc


def assess_complementarity(orch, datasets: List[dict]) -> dict:
    """Partition datasets into complementary / redundant / unrelated.

    `datasets` is a list of ``{"path", "role"?, "metadata"?}``. Returns the
    verdict dict (see COMPLEMENTARITY_ASSESSMENT_INSTRUCTIONS). Cached on the
    orchestrator by the frozenset of paths so the standalone tool and the
    internal gate in run_fanout don't double-spend the LLM call.
    """
    paths = [d.get("path") for d in datasets if d.get("path")]
    if len(paths) < 2:
        return {"verdict": "uncertain", "confidence": 0.0,
                "rationale": "Need at least two datasets to assess complementarity.",
                "join_axis": None, "fanout_set": [], "redundant_clusters": [],
                "unrelated": list(paths), "excluded_notes": ""}

    key = frozenset(paths)
    cached = orch._complementarity_cache.get(key)
    if cached is not None:
        return cached

    descriptors = [
        _dataset_descriptor(d["path"], d.get("role"), d.get("metadata"))
        for d in datasets if d.get("path")
    ]
    prompt = (
        COMPLEMENTARITY_ASSESSMENT_INSTRUCTIONS
        + "\n\n--- DATASETS ---\n"
        + json.dumps(descriptors, indent=2, default=str)
    )
    verdict = _llm_json(orch, prompt)
    if not verdict or "verdict" not in verdict:
        # Fail closed: an unparseable assessment must not green-light a fusion.
        verdict = {
            "verdict": "uncertain", "confidence": 0.0,
            "rationale": "Complementarity assessment did not return a usable verdict.",
            "join_axis": None, "fanout_set": [], "redundant_clusters": [],
            "unrelated": list(paths), "excluded_notes": "",
        }
    # Constrain the model's fanout_set to the actually-requested paths.
    requested = set(paths)
    verdict["fanout_set"] = [p for p in (verdict.get("fanout_set") or [])
                             if p in requested]
    # Defensive: a clearly-negative verdict must never carry a runnable set,
    # even if the model inconsistently populated one — it means "do not fuse".
    if (verdict.get("verdict") or "").lower() in ("unrelated", "redundant"):
        verdict["fanout_set"] = []
    orch._complementarity_cache[key] = verdict
    return verdict


# ======================================================================
# Confirmation
# ======================================================================

def _confirm_fanout(orch, verdict: dict, fanout_set: List[str],
                    branches_by_path: Dict[str, dict]) -> tuple:
    """Decide whether to fire the fan-out. Returns (proceed: bool, reason: str).

    AUTOPILOT (human attached): show the verdict + the exact plan and ask the
    user to confirm. AUTONOMOUS (no human): the verdict is the gate — proceed
    only on a confident 'complementary' read within the soft cap.
    """
    n = len(fanout_set)
    n_aux = n * (n - 1)  # full-mesh: each branch sees the other n-1

    if n > FANOUT_HARD_CAP:
        return False, (f"Complementary set has {n} datasets (> hard cap "
                       f"{FANOUT_HARD_CAP}); refuse to fan out. Run them in "
                       "smaller complementary groups.")

    if not orch._enable_human_feedback:
        # AUTONOMOUS: verdict-gated, conservative.
        v = (verdict.get("verdict") or "").lower()
        conf = float(verdict.get("confidence") or 0.0)
        if v != "complementary" or conf < AUTONOMOUS_CONFIDENCE_THRESHOLD:
            return False, (f"Autonomous mode declines fan-out: verdict='{v}' "
                           f"confidence={conf:.2f} (needs 'complementary' >= "
                           f"{AUTONOMOUS_CONFIDENCE_THRESHOLD}). "
                           f"{verdict.get('rationale', '')}")
        if n > FANOUT_SOFT_CAP:
            return False, (f"Autonomous mode declines a {n}-way mesh (> soft cap "
                           f"{FANOUT_SOFT_CAP}); too expensive to fire unattended.")
        return True, "autonomous: confident complementary verdict"

    # AUTOPILOT: informed human confirmation.
    lines = [
        "",
        "=" * 78,
        "🔀 PARALLEL MULTI-DATASET ANALYSIS — confirm before launching",
        "=" * 78,
        f"  Complementarity verdict : {verdict.get('verdict')} "
        f"(confidence {verdict.get('confidence')})",
        f"  Join axis               : {verdict.get('join_axis')}",
        f"  Rationale               : {verdict.get('rationale')}",
        "",
        f"  Will run {n} branches concurrently, full-mesh "
        f"(~{n_aux} auxiliary loads):",
    ]
    for path in fanout_set:
        b = branches_by_path.get(path, {})
        lines.append(f"    • {b.get('label') or _slug(Path(path).name)}  ({Path(path).name})")
    if verdict.get("redundant_clusters"):
        lines.append(f"  Pruned as redundant     : {verdict['redundant_clusters']}")
    if verdict.get("unrelated"):
        lines.append(f"  Pruned as unrelated     : {verdict['unrelated']}")
    if n > FANOUT_SOFT_CAP:
        lines.append(f"  ⚠️  {n}-way mesh exceeds the soft cap ({FANOUT_SOFT_CAP}) "
                     "— this is expensive.")
    lines.append("  Branches run AUTONOMOUSLY (no per-branch approval pauses).")
    lines.append("=" * 78)
    print("\n".join(lines))

    try:
        ans = input("\n🤔 Launch this parallel analysis? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # No usable input channel in a mode that expects one → do not fire an
        # expensive parallel op on a guess.
        return False, "no confirmation received (declined)"
    if ans in ("y", "yes"):
        return True, "user confirmed"
    return False, "user declined"


# ======================================================================
# Branch execution
# ======================================================================

def _make_ephemeral_analysis_child(orch, base_dir: Path):
    """Build an isolated, one-shot analysis orchestrator for one branch.

    NOT registered in ``orch._children`` — these are ephemeral fan-out workers,
    not the persistent singleton, so they share no mutable state across threads
    and are never restored. Resting mode AUTONOMOUS; run_task pins it per call.
    """
    from ..exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode,
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    child = AnalysisOrchestratorAgent(
        base_dir=str(base_dir),
        api_key=orch.api_key,
        model_name=orch.model_name,
        base_url=orch.base_url,
        embedding_model=orch.embedding_model,
        embedding_api_key=orch.embedding_api_key,
        futurehouse_api_key=orch.futurehouse_api_key,
        restore_checkpoint=False,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )
    child._agent_label = "Analysis branch"
    # Share skills / custom tools / MCP servers registered on the meta.
    orch._propagate_extensions_to_child(child)
    return child


def _mesh_task(branch: dict, companions: List[dict]) -> str:
    """Compose a branch's self-contained task with its full-mesh companions.

    Companions are named as auxiliary datasets with distinct labels so the
    specialist passes them through ``run_analysis``'s ``auxiliary_data`` /
    ``auxiliary_label`` — the existing operand path — and the codegen may use
    a shape-aligned companion numerically (correlate / mask / normalize).
    """
    task = branch["task"].rstrip()
    block = []
    if companions:
        block += ["", "",
                  f"PRIMARY dataset for THIS analysis: {branch['data_path']} — pass "
                  "it as run_analysis's `data_path`. The companion(s) below are "
                  "AUXILIARY ONLY; do NOT analyze a companion as the primary."]
    # Forward the caller-supplied metadata so the branch USES it rather than
    # synthesizing metadata from the task prose (which loses technique-specific
    # fields the downstream skill needs, e.g. the EELS energy axis).
    meta = branch.get("metadata")
    if meta:
        mp = Path(str(meta))
        if mp.exists() and mp.suffix.lower() == ".json":
            block += ["", f"Metadata for the primary dataset is at {mp} — call "
                          "`load_metadata` on this path before `run_analysis`; do "
                          "NOT synthesize metadata when this file is provided."]
        else:
            block += ["", f"Metadata for the primary dataset: {meta}"]
    if companions:
        block += ["",
                  "COMPANION DATASETS (complementary measurements of the SAME "
                  "system — pass each as auxiliary_data with the given label so your "
                  "generated code may correlate/mask/normalize against it where the "
                  "method benefits; they are optional operands, never required):"]
        for c in companions:
            block.append(f"  - auxiliary_data: {c['data_path']}  "
                         f"(auxiliary_label: '{c['label']}')")
    if not block:
        return task
    return task + "\n".join(block)


def _run_one_branch(orch, branch: dict, companions: List[dict],
                    entry: dict) -> None:
    """Execute a single fan-out branch into its preallocated ledger slot.

    Each worker touches ONLY its own ``entry`` dict, so there is no shared
    mutation across threads. Never raises — failures are captured into the
    ledger slot like ``_delegate`` does.
    """
    index = entry["index"]
    slug = _slug(branch.get("label") or Path(branch["data_path"]).stem)
    base_dir = orch.fanout_dir / f"{index:02d}_{slug}"
    try:
        from ..exp_agents.analysis_orchestrator import AnalysisMode
        child = _make_ephemeral_analysis_child(orch, base_dir)
        result = child.run_task(
            _mesh_task(branch, companions),
            context=branch.get("context"),
            autonomy=AnalysisMode.AUTONOMOUS,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"fan-out branch {index} failed: {e}")
        result = {"status": "error", "error": str(e), "summary": "",
                  "key_findings": [], "files_produced": [],
                  "suggested_followups": [], "warnings": []}
    orch._close_delegation(entry, result)


def run_fanout(orch, branches: List[dict]) -> str:
    """Gate → confirm → run branches concurrently (full-mesh aux). Returns JSON.

    `branches` is a list of ``{"data_path", "task", "label", "metadata"?,
    "context"?}``. The complementarity gate prunes to the complementary subset;
    only that subset runs, each branch seeing the others as auxiliary operands.
    """
    # --- normalize input ---
    norm: List[dict] = []
    for b in (branches or []):
        if not isinstance(b, dict):
            continue
        dp, task = b.get("data_path"), b.get("task")
        if not dp or not task:
            continue
        norm.append({
            "data_path": dp, "task": task,
            "label": (b.get("label") or Path(dp).stem),
            "metadata": b.get("metadata"), "context": b.get("context"),
        })
    if len(norm) < 2:
        return json.dumps({"status": "error",
                           "message": "Fan-out needs at least two branches, each "
                                      "with a data_path and a task."})

    by_path = {b["data_path"]: b for b in norm}

    # --- entry gate (reuses cached verdict if assess_complementarity ran) ---
    datasets = [{"path": b["data_path"], "metadata": b.get("metadata")}
                for b in norm]
    verdict = assess_complementarity(orch, datasets)
    fanout_set = [p for p in (verdict.get("fanout_set") or []) if p in by_path]

    if len(fanout_set) < 2:
        return json.dumps({
            "status": "declined",
            "reason": "not_complementary",
            "verdict": verdict,
            "message": (
                "The datasets are not genuinely complementary (no 2+ that share "
                "a system and a join axis), so a parallel cross-analysis with "
                "fusion was NOT run. Consider analyzing them independently via "
                "delegate_to_analysis, or one with the other as a plain "
                "auxiliary. See the verdict for redundant/unrelated groupings."
            ),
        }, indent=2, default=str)

    # --- confirmation ---
    proceed, reason = _confirm_fanout(orch, verdict, fanout_set, by_path)
    if not proceed:
        return json.dumps({"status": "declined", "reason": reason,
                           "verdict": verdict,
                           "fanout_set": fanout_set}, indent=2, default=str)

    # --- preallocate ledger slots (sequential, under lock — no concurrent append) ---
    run_branches = [by_path[p] for p in fanout_set]
    with orch._fanout_lock:
        entries = []
        group_id = f"fanout_{len(orch._delegation_ledger) + 1}"
        for b in run_branches:
            entry = orch._open_delegation(
                "analysis", _mesh_task(b, []), b.get("context"), None, b["label"])
            entry["parallel_group"] = group_id
            entry["fanout"] = True
            entries.append(entry)

    # --- run concurrently; each branch sees all others (full mesh) ---
    print(f"  🔀 Launching {len(run_branches)} parallel analysis branches "
          f"(group {group_id}, full-mesh aux)...")

    def _companions_for(i):
        return [{"data_path": run_branches[j]["data_path"],
                 "label": f"companion_{_slug(run_branches[j]['label'])}"}
                for j in range(len(run_branches)) if j != i]

    max_workers = min(len(run_branches), FANOUT_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_one_branch, orch, run_branches[i],
                        _companions_for(i), entries[i])
            for i in range(len(run_branches))
        ]
        for f in futures:
            f.result()  # _run_one_branch never raises; this just joins

    def _productive(e):
        # A branch that reports success but yields neither findings nor files
        # did no usable work (e.g. codegen aborted with no sandbox). Mirrors the
        # 'empty_but_successful' guard in _summarize_delegation_result.
        return (e.get("status") == "success"
                and (bool(e.get("key_findings")) or bool(e.get("files_produced"))))

    results = [{
        "delegation_index": e["index"],
        "label": e["label"],
        "status": e.get("status"),
        "produced_output": _productive(e),
        "key_findings": e.get("key_findings", []),
        "files_produced": e.get("files_produced", []),
    } for e in entries]
    productive = [r for r in results if r["produced_output"]]
    # A branch is "degraded" if it produced no usable output — whether it
    # hard-errored or reported success with empty findings/files (e.g. codegen
    # could not run). Either way the meta must not treat it as a completed
    # analysis or fuse it.
    degraded = [r for r in results if not r["produced_output"]]

    out = {
        "status": "success",
        "parallel_group": group_id,
        "join_axis": verdict.get("join_axis"),
        "branches_run": len(results),
        "branches_with_output": len(productive),
        "results": results,
        "next_step": (
            "Call fuse_delegations with delegation_indices="
            f"{[r['delegation_index'] for r in productive]} to reconcile these "
            "complementary findings into one cross-dataset interpretation."
            if len(productive) >= 2 else
            "Fewer than two branches produced usable output — report what ran "
            "to the user; do NOT fuse empty branches into a synthesis."
        ),
    }
    if degraded:
        n_err = sum(1 for r in degraded if r["status"] != "success")
        out["warning"] = (
            f"{len(degraded)} branch(es) produced no usable output "
            f"({n_err} errored, {len(degraded) - n_err} succeeded-but-empty, "
            "e.g. analysis code could not execute). Do not treat these as "
            "completed analyses or fuse them; report the gap to the user."
        )
    return json.dumps(out, indent=2, default=str)


# ======================================================================
# Fusion
# ======================================================================

def fuse_delegations(orch, indices: List[int], focus: Optional[str] = None) -> str:
    """Reconcile finished branch findings into one cross-dataset narrative.

    Reuses the HOLISTIC multi-modal synthesis template with the
    anti-spurious-correlation guard so "no correlation found" is a valid
    outcome. Reads each branch's findings from its ledger entry (summary +
    key_findings). Records itself as a ``mode="fusion"`` ledger entry.
    """
    from ..exp_agents.instruct import HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS

    ledger = orch._delegation_ledger
    by_index = {e["index"]: e for e in ledger}
    try:
        idxs = sorted({int(i) for i in (indices or [])})
    except (TypeError, ValueError):
        return json.dumps({"status": "error",
                           "message": "delegation_indices must be integers."})
    entries = [by_index[i] for i in idxs if i in by_index]
    ok = [e for e in entries if e.get("status") == "success"
          and (e.get("key_findings") or (e.get("summary") or "").strip())]
    if len(ok) < 2:
        return json.dumps({
            "status": "error",
            "message": ("Need >= 2 successful delegations with findings to fuse. "
                        f"Got {len(ok)} usable of {len(idxs)} requested."),
        })

    blocks = []
    for e in ok:
        findings = e.get("key_findings") or []
        findings_str = "\n".join(f"- {k}" for k in findings) if findings else "- (none)"
        blocks.append(
            f"### Dataset: {e.get('label') or ('delegation ' + str(e['index']))} "
            f"(delegation #{e['index']})\n"
            f"Summary:\n{e.get('summary', '') or '(none)'}\n\n"
            f"Key findings:\n{findings_str}"
        )

    prompt = (
        HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS
        + _ANTI_SPURIOUS_FUSION_GUARD
        + (f"\n\nFUSION FOCUS (weight your synthesis toward this): {focus}\n"
           if focus else "")
        + "\n\n--- PER-DATASET FINDINGS TO RECONCILE ---\n\n"
        + "\n\n".join(blocks)
    )
    parsed = _llm_json(orch, prompt)
    if not parsed or "detailed_analysis" not in parsed:
        return json.dumps({"status": "error",
                           "message": "Fusion synthesis did not return a usable result."})

    # Persist the fused report + record a fusion ledger entry.
    from datetime import datetime
    with orch._fanout_lock:
        fusion_n = sum(1 for e in ledger if e.get("mode") == "fusion") + 1
    out_dir = orch.fusion_dir / f"{fusion_n:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "fusion_report.json"
    fused = {
        "fused_from": [e["index"] for e in ok],
        "labels": [e.get("label") for e in ok],
        "focus": focus,
        "detailed_analysis": parsed.get("detailed_analysis", ""),
        "scientific_claims": parsed.get("scientific_claims", []),
    }
    try:
        with open(report_path, "w") as fh:
            json.dump(fused, fh, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not write fusion report: {e}")

    with orch._fanout_lock:
        orch._delegation_ledger.append({
            "index": len(orch._delegation_ledger) + 1,
            "timestamp": datetime.now().isoformat(),
            "mode": "fusion",
            "task": f"Fuse delegations {[e['index'] for e in ok]}",
            "label": "cross-dataset fusion",
            "context_from": [e["index"] for e in ok],
            "status": "success",
            "summary": parsed.get("detailed_analysis", ""),
            "key_findings": [c.get("claim", "") for c in parsed.get("scientific_claims", [])
                             if isinstance(c, dict)],
            "files_produced": [str(report_path)],
            "warnings": [],
            "error": None,
        })

    return json.dumps({
        "status": "success",
        "fused_from": [e["index"] for e in ok],
        "detailed_analysis": parsed.get("detailed_analysis", ""),
        "scientific_claims": parsed.get("scientific_claims", []),
        "report_path": str(report_path),
    }, indent=2, default=str)
