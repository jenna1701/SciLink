"""Worker-thread log attribution for best-of-N candidate fan-outs.

Two problems when anchor attempts run in ThreadPoolExecutor workers:

1. The Streamlit UI's verbose panel filters log records by thread id (the
   background chat thread), so candidate attempts' detailed narration
   (plan-deviation justifications, verification recommendations, refinement
   reasoning) silently disappears from the panel.
2. In the CLI console all workers' lines interleave without attribution —
   two candidates print identical "Verification 2/7" lines.

A fan-out worker registers itself with its parent (chat) thread id and a
candidate tag. ``effective_thread`` lets the UI filter treat worker records
as belonging to the parent thread, and a root-logger filter prefixes each
registered worker's messages with its candidate tag so the interleave is
attributable everywhere (UI, CLI, log files).
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Tuple

# worker thread id -> (parent thread id, candidate tag)
_WORKERS: Dict[int, Tuple[int, str]] = {}
_LOCK = threading.Lock()
_FILTER_INSTALLED = False


def register_worker(parent_thread_id: int, tag: str) -> None:
    """Register the CURRENT thread as a fan-out worker of ``parent_thread_id``."""
    _install_prefix_filter()
    with _LOCK:
        _WORKERS[threading.get_ident()] = (parent_thread_id, tag)


def unregister_worker() -> None:
    with _LOCK:
        _WORKERS.pop(threading.get_ident(), None)


def effective_thread(thread_id: int) -> int:
    """Parent thread id for registered workers; identity otherwise."""
    entry = _WORKERS.get(thread_id)
    return entry[0] if entry else thread_id


def _install_prefix_filter() -> None:
    """Install a log-record factory that prefixes worker records (idempotent).

    A record factory (rather than a logger/handler filter) is the one hook
    that sees every record regardless of which named logger emitted it and
    which handlers are attached later.
    """
    global _FILTER_INSTALLED
    if _FILTER_INSTALLED:
        return
    with _LOCK:
        if _FILTER_INSTALLED:
            return
        previous_factory = logging.getLogRecordFactory()

        def factory(*args, **kwargs):
            record = previous_factory(*args, **kwargs)
            entry = _WORKERS.get(record.thread)
            if entry:
                record.msg = f"[{entry[1]}] {record.msg}"
            return record

        logging.setLogRecordFactory(factory)
        _FILTER_INSTALLED = True
