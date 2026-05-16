"""Telemetry tab — live snapshot of meta + specialist + worker-agent state.

Explore (meta) mode only. A compact, status-colored dependency graph of the
delegation ledger (meta -> delegations, annotated with the sub-agent each mode
selected, plus the `context_from` provenance edges) sits on top; the delegation
ledger, a per-agent tool-call sequence, and a detailed per-action breakdown
(inputs, outputs, reasoning) back it underneath. Refreshes on the sidebar
delegation tree's cadence (2s while a chat task runs).
"""

import json

import streamlit as st

from scilink.agents.meta_agent.telemetry import collect_session_telemetry

# Same palette as the sidebar delegation tree, for consistency.
_STATUS_FILL = {
    "success": "#3fb950",   # green
    "error": "#f85149",     # red
    "running": "#d29922",   # amber
}
_DEFAULT_FILL = "#8893a5"   # grey — unknown / not-yet-finished


def render_telemetry_tab() -> None:
    """Render the Explore-mode Telemetry tab."""
    task = st.session_state.get("chat_task")
    _interval = "2s" if (task is not None
                         and getattr(task, "is_running", False)) else None

    @st.fragment(run_every=_interval)
    def _panel() -> None:
        agent = st.session_state.get("agent")
        if agent is None or not hasattr(agent, "_delegation_ledger"):
            st.info("Telemetry is available once an Explore session is running.")
            return

        tel = collect_session_telemetry(agent)
        meta = tel.get("meta", {})
        agents = tel.get("agents", [])
        delegations = meta.get("delegations", [])

        total_actions = sum(a.get("action_count", 0) for a in agents)
        st.markdown(
            f"**{str(meta.get('meta_mode', '—')).title()}**  ·  "
            f"{meta.get('delegations_total', 0)} delegations  ·  "
            f"{len(agents)} worker agents  ·  {total_actions} actions logged"
        )
        st.caption(f"Session: {meta.get('session_dir', '—')}")

        if not delegations:
            st.info("No delegations yet — describe a goal and the meta "
                    "routes it to a specialist.")
            return

        # ── Dependency graph ─────────────────────────────────────────
        st.graphviz_chart(
            _delegation_graph_dot(meta, tel.get("sub_agents", {})),
            width="content",
        )
        st.caption("Grey edge = meta dispatched the delegation.  "
                   "Blue edge = a delegation's result fed the next as context.")

        # ── Delegation ledger ────────────────────────────────────────
        st.subheader("Delegation ledger")
        _ledger_table(delegations)

        # ── Tool sequence ────────────────────────────────────────────
        st.subheader("Tool sequence")
        st.caption("Every tool call each agent's LLM made, in order.")
        _tool_sequence_section(tel.get("tool_sequence", {}))

        # ── Detailed breakdown ───────────────────────────────────────
        st.subheader("Detailed breakdown")
        st.caption("Per worker agent: every logged action with its inputs, "
                   "outputs and reasoning.")
        if not agents:
            st.caption("No worker actions recorded yet.")
        for ag in agents:
            _worker_breakdown(ag)

        reports = tel.get("analysis_reports", [])
        if reports:
            st.markdown("**Analysis reasoning**")
            for rep in reports:
                _analysis_report(rep)

    _panel()


# ── dependency graph ─────────────────────────────────────────────────

def _dot_escape(text) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _truncate(text, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _delegation_graph_dot(meta: dict, sub_agents: dict) -> str:
    """DOT string: meta-agent root -> delegation nodes (colored by status,
    annotated with the sub-agent(s) the mode used), with dispatch edges and
    `context_from` provenance edges. Kept compact via tight spacing + a size
    cap so it does not dominate the tab."""
    dels = meta.get("delegations", [])
    out = [
        "digraph telemetry {",
        '  rankdir=TB; bgcolor="transparent"; pad=0.15;',
        '  size="7,4.5"; ratio=compress; ranksep=0.32; nodesep=0.22;',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", '
        'fontsize=9, margin="0.11,0.05", color="#30363d", fontcolor="white"];',
        '  edge [fontname="Helvetica", fontsize=8, arrowsize=0.7];',
        f'  meta [label="Meta-agent ({_dot_escape(str(meta.get("meta_mode", "")).title())})", '
        'fillcolor="#30363d"];',
    ]
    for d in dels:
        idx = d.get("index")
        fill = _STATUS_FILL.get(d.get("status"), _DEFAULT_FILL)
        subs = sub_agents.get(d.get("mode"), [])
        sub_line = ("\\n↳ " + _dot_escape(_truncate(", ".join(subs), 32))
                    if subs else "")
        label = (f'#{idx} · {_dot_escape(d.get("mode", "?"))}\\n'
                 f'{_dot_escape(_truncate(d.get("label", ""), 24))}{sub_line}')
        out.append(f'  d{idx} [label="{label}", fillcolor="{fill}"];')
    for d in dels:                                   # dispatch edges
        out.append(f'  meta -> d{d.get("index")} [color="#8893a5"];')
    for d in dels:                                   # context-provenance edges
        for src in d.get("context_from", []):
            out.append(f'  d{src} -> d{d.get("index")} '
                       f'[color="#58a6ff", penwidth=2.0, label="ctx"];')
    out.append("}")
    return "\n".join(out)


# ── tables ───────────────────────────────────────────────────────────

def _short_time(ts) -> str:
    """ISO timestamp -> HH:MM:SS (keeps the table narrow)."""
    if not ts:
        return ""
    s = str(ts)
    return s.split("T", 1)[1][:8] if "T" in s else s


def _ledger_table(delegations: list) -> None:
    import pandas as pd

    rows = []
    for d in delegations:
        cf = d.get("context_from") or []
        rows.append({
            "#": d.get("index"),
            "specialist": d.get("mode"),
            "task": d.get("label"),
            "status": d.get("status"),
            "context from": ", ".join(f"#{n}" for n in cf) if cf else "",
            "files": d.get("files", 0),
            "feature tables": d.get("feature_tables", 0),
            "warnings": d.get("warnings", 0),
            "started": _short_time(d.get("timestamp")),
            "completed": _short_time(d.get("completed_at")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── tool sequence ────────────────────────────────────────────────────

_LAYER_LABELS = [
    ("meta", "Meta-agent"),
    ("analysis", "Analysis specialist"),
    ("planning", "Planning specialist"),
]


def _compact_args(args, limit: int = 120) -> str:
    """One-line, truncated rendering of a tool call's arguments."""
    if not args:
        return ""
    try:
        s = json.dumps(args, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = str(args)
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _tool_sequence_section(sequence: dict) -> None:
    """Per-agent ordered tool-call list — the full sequence of tools run."""
    import pandas as pd

    shown = False
    for key, label in _LAYER_LABELS:
        calls = sequence.get(key) or []
        if not calls:
            continue
        shown = True
        st.markdown(f"**{label}** — {len(calls)} tool call(s)")
        rows = [{"#": i, "tool": c.get("tool"),
                 "arguments": _compact_args(c.get("args"))}
                for i, c in enumerate(calls, 1)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    if not shown:
        st.caption("No tool calls recorded yet.")


# ── detailed breakdown ───────────────────────────────────────────────

def _worker_breakdown(ag: dict) -> None:
    """One expander per worker agent: every action with input/output/reason."""
    oc = ag.get("outcomes", {})
    title = (f"{ag.get('name', '?')}  ·  {ag.get('specialist', '—')}  ·  "
             f"{ag.get('action_count', 0)} action(s)  "
             f"({oc.get('success', 0)}✓ / {oc.get('error', 0)}✗)")
    with st.expander(title):
        actions = ag.get("actions", [])
        for i, ac in enumerate(actions, 1):
            st.markdown(
                f"**{i}. `{ac.get('action', '?')}`**  ·  "
                f"`{ac.get('status', '—')}`  ·  "
                f"{_short_time(ac.get('timestamp'))}"
            )
            rationale = ac.get("rationale")
            if rationale:
                st.caption("Reasoning")
                st.write(rationale)
            _input = ac.get("input")
            _result = ac.get("result")
            ci, co = st.columns(2)
            with ci:
                st.caption("Input")
                if _input:
                    st.json(_input, expanded=True)
                else:
                    st.caption("— none recorded —")
            with co:
                st.caption("Output")
                if _result:
                    st.json(_result, expanded=True)
                else:
                    st.caption("— none recorded —")
            feedback = ac.get("feedback")
            if feedback:
                st.caption("Feedback")
                st.write(feedback)
            if i < len(actions):
                st.divider()


def _analysis_report(rep: dict) -> None:
    """One expander per analysis run: its detailed reasoning + claims."""
    claims = rep.get("claims", [])
    title = (f"{rep.get('analysis_id', 'analysis')}  ·  "
             f"{len(claims)} claim(s)  ·  {rep.get('status', '—')}")
    with st.expander(title):
        detailed = rep.get("detailed_analysis")
        if detailed:
            st.markdown(detailed)
        if claims:
            st.caption("Scientific claims")
            for c in claims:
                st.markdown(f"- **{c.get('claim', '')}**")
                if c.get("impact"):
                    st.caption(c["impact"])
