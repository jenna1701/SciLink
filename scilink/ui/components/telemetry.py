"""Telemetry tab — live snapshot of meta + specialist + worker-agent state.

Explore (meta) mode only. A status-colored dependency graph of the delegation
ledger (meta -> delegations, with the `context_from` provenance edges) sits on
top; dense detail tables back it underneath. Refreshes on the same cadence as
the sidebar delegation tree (2s while a chat task runs).
"""

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
        st.graphviz_chart(_delegation_graph_dot(meta), use_container_width=True)
        st.caption("Grey edge = meta dispatched the delegation.  "
                   "Blue edge = a delegation's result fed the next as context.")

        # ── Specialists ──────────────────────────────────────────────
        st.subheader("Specialists")
        _specialists_table(tel.get("specialists", {}))

        # ── Worker agents ────────────────────────────────────────────
        st.subheader("Worker agents")
        _workers_table(agents)

        # ── Every logged action ──────────────────────────────────────
        st.subheader("Worker activity")
        _actions_table(agents)

        # ── Delegation ledger ────────────────────────────────────────
        st.subheader("Delegation ledger")
        _ledger_table(delegations)

    _panel()


# ── dependency graph ─────────────────────────────────────────────────

def _dot_escape(text) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _delegation_graph_dot(meta: dict) -> str:
    """DOT string: meta-agent root -> delegation nodes (colored by status),
    with dispatch edges and `context_from` provenance edges."""
    dels = meta.get("delegations", [])
    out = [
        "digraph telemetry {",
        '  rankdir=TB; bgcolor="transparent"; pad=0.2;',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", '
        'fontsize=10, color="#30363d", fontcolor="white"];',
        '  edge [fontname="Helvetica", fontsize=8];',
        f'  meta [label="Meta-agent\\n({_dot_escape(str(meta.get("meta_mode", "")).title())})", '
        'fillcolor="#30363d"];',
    ]
    for d in dels:
        idx = d.get("index")
        fill = _STATUS_FILL.get(d.get("status"), _DEFAULT_FILL)
        label = (f'#{idx} · {_dot_escape(d.get("mode", "?"))}\\n'
                 f'{_dot_escape(d.get("label", ""))}')
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


def _specialists_table(specs: dict) -> None:
    import pandas as pd

    rows = []
    for name in ("analysis", "planning"):
        info = specs.get(name, {})
        if not info.get("instantiated"):
            rows.append({"specialist": name, "messages": "—",
                         "work": "not engaged"})
            continue
        if name == "analysis":
            work = f"{info.get('analyses_run', 0)} analysis run(s)"
        else:
            targets = info.get("optimization_targets", [])
            work = (f"{info.get('bo_data_points', 0)} BO data points"
                    + (f" · targets: {', '.join(targets)}" if targets else ""))
        rows.append({"specialist": name,
                     "messages": info.get("message_count", 0), "work": work})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _workers_table(agents: list) -> None:
    import pandas as pd

    if not agents:
        st.caption("No worker agents have run yet.")
        return
    rows = [{
        "agent": a.get("name"),
        "specialist": a.get("specialist"),
        "actions": a.get("action_count", 0),
        "✓": a.get("outcomes", {}).get("success", 0),
        "✗": a.get("outcomes", {}).get("error", 0),
        "first": _short_time(a.get("first_timestamp")),
        "last": _short_time(a.get("last_timestamp")),
    } for a in agents]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _actions_table(agents: list) -> None:
    """Every logged action across all workers, newest first."""
    import pandas as pd

    rows = []
    for a in agents:
        for ac in a.get("actions", []):
            rows.append({
                "time": _short_time(ac.get("timestamp")),
                "specialist": a.get("specialist"),
                "agent": a.get("name"),
                "action": ac.get("action"),
                "status": ac.get("status"),
                "rationale": ac.get("rationale"),
                "_sort": ac.get("timestamp") or "",
            })
    if not rows:
        st.caption("No worker actions recorded yet.")
        return
    rows.sort(key=lambda r: r["_sort"], reverse=True)
    df = pd.DataFrame(rows).drop(columns=["_sort"])
    st.dataframe(df, use_container_width=True, hide_index=True)


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
