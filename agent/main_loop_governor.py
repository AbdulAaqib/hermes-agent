"""Main-loop labels + cost recording for the call governor.

Main-loop attempts (as opposed to auxiliary ``call_llm`` work) are labelled from the agent's
runtime identity — live chat, cron job, e2e suite, delegated subagent — and their usage is
rolled into the governor's daily ledger. Hooked from ``agent.turn_api_call.perform_api_call``
(every attempt) and ``agent.chat_completion_helpers._managed_summary_call`` (the
max-iterations summary, which bypasses the normal loop accounting).

The label is also exposed as a ``llm_execution`` middleware callback (``llm_execution``) for
deployments that register it through the plugin middleware chain.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent.call_governor")

_CHAT_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal", "matrix", "teams",
    "google_chat", "qqbot", "yuanbao", "homeassistant", "cli", "tui", "desktop", "web",
})

_cron_names_cache: dict = {}


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def cron_job_name(job_id: str) -> str:
    """Resolve a cron job's display name from ``$HERMES_HOME/cron/jobs.json``; id on miss."""
    job_id = str(job_id or "").strip()
    if not job_id:
        return "unknown"
    if job_id in _cron_names_cache:
        return _cron_names_cache[job_id]
    name = job_id
    try:
        data = json.loads((_hermes_home() / "cron" / "jobs.json").read_text(encoding="utf-8"))
        jobs = data.get("jobs") if isinstance(data, dict) else data
        for job in jobs or []:
            if isinstance(job, dict) and str(job.get("id") or "") == job_id:
                name = str(job.get("name") or job_id)
                break
    except Exception:
        logger.debug("call_governor: cron jobs.json lookup failed", exc_info=True)
    _cron_names_cache[job_id] = name
    return name


def _cron_job_id(session_id: str) -> str:
    parts = str(session_id or "").split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "cron" else ""


def _session_source(agent: Any) -> str:
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if session_db is not None and session_id:
        try:
            row = session_db.get_session(session_id)
            if row:
                return str(row.get("source") or "").strip().lower()
        except Exception:
            logger.debug("call_governor: session source lookup failed", exc_info=True)
    return ""


def resolve_main_loop_task(agent: Any) -> str:
    """Label a main-loop attempt: ``delegate``, ``cron:<job>``, ``e2e`` or ``chat``.

    Memoised on the agent: the label is stable for the life of the run, and the source
    lookup would otherwise hit state.db on every API call of a tool loop.
    """
    cached = getattr(agent, "_governor_task_label", None)
    if cached:
        return str(cached)
    if getattr(agent, "is_subagent", False):
        label = "delegate"
    else:
        session_id = str(getattr(agent, "session_id", "") or "")
        platform = str(getattr(agent, "platform", "") or "").strip().lower()
        if platform == "cron" or session_id.startswith("cron_"):
            label = f"cron:{cron_job_name(_cron_job_id(session_id))}"
        else:
            source = _session_source(agent)
            label = "e2e" if (source == "e2e" or platform == "e2e") else "chat"
    try:
        agent._governor_task_label = label
    except Exception:
        pass
    return label


def record_main_loop_response(
    agent: Any, response: Any, task: str, *, write_session: bool = False,
) -> None:
    """Roll a main-loop response into the governor ledger.

    ``write_session`` is False for the normal loop (``conversation_loop`` already writes
    ``session_model_usage``) and True for calls outside that path (the max-iterations
    summary), where the governor is the only writer.
    """
    try:
        from agent import call_governor

        session_db = getattr(agent, "_session_db", None) if write_session else None
        call_governor.record_response(
            response, task, provider=getattr(agent, "provider", "") or None,
            base_url=getattr(agent, "base_url", "") or None,
            route=str(getattr(agent, "base_url", "") or ""),
            session_id=str(getattr(agent, "session_id", "") or ""),
            session_db=session_db,
        )
    except Exception:
        logger.debug("call_governor: main-loop response recording failed (non-fatal)", exc_info=True)


def llm_execution(request: Any, next_call: Any, **context: Any) -> Any:
    """``llm_execution`` middleware: label each attempt and record its usage (best-effort)."""
    agent = context.get("agent")
    task = resolve_main_loop_task(agent) if agent is not None else "chat"
    response = next_call(request)
    if agent is not None and response is not None:
        record_main_loop_response(agent, response, task)
    return response
