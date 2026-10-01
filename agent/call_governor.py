"""Central LLM call governor + cost ledger.

Every paid LLM call passes through one of two chokepoints — auxiliary calls through
``agent.auxiliary_client.call_llm``/``async_call_llm``, main-loop attempts through
``agent.turn_api_call.perform_api_call`` — and both are funnelled here so the whole system
shares one admission policy and one per-UTC-day cost ledger.

Admission (``admit``) answers "may this call happen, and if not why": per-task daily caps,
minimum intervals, input-hash dedupe, and a global daily-budget breaker. Live user chat
(``task='chat'``) is never denied; a denied non-chat call raises :class:`CallDenied`, which
existing callers already treat as a failure and degrade around.

Recording (``record``/``record_response``) appends to the locked per-day JSON roll-up
(``$HERMES_HOME/state/call_governor/<date>.json``) and, when a session handle is supplied,
to the existing ``session_model_usage`` table under the task label. Cost prefers the
provider-reported figure (OpenRouter ``usage.cost``); otherwise it is estimated from
``agent.usage_pricing``.

The read-modify-write is ``fcntl``-locked, generalising the ``DailyCallBudget`` pattern in
``skills/testing/asuna-e2e-verification/scripts/e2e_common.py``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("agent.call_governor")


class CallDenied(Exception):
    """Raised when the governor refuses a (non-chat) LLM call.

    Subclasses :class:`Exception` so existing ``except Exception`` soft-fail paths
    (title generation, compression, goal judging, ...) degrade around it unchanged.
    """

    def __init__(self, task: str, reason: str) -> None:
        self.task = task
        self.reason = reason
        super().__init__(f"LLM call denied (task={task or 'unlabelled'}, reason={reason})")


@dataclass
class Decision:
    """Result of an admission check; ``reason`` explains a denial (empty when allowed)."""

    allow: bool
    reason: str = ""
    task: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


# Tasks that represent a live human turn: NEVER denied, regardless of caps or the breaker.
# MoA reference/aggregator calls happen INSIDE a chat turn, so denying them would break
# live chat (the one thing the breaker must never touch).
_CHAT_TASKS = frozenset({"chat", "moa_reference", "moa_aggregator"})

# Default per-task policy for known auxiliary/maintenance tasks. Values are merged over by
# ``config.yaml`` ``call_governor.tasks`` so the operator can override any knob.
_DEFAULT_TASKS: Dict[str, Dict[str, Any]] = {
    "title_generation": {"daily_cap": 200},
    "compression": {"daily_cap": 200},
    "background_review": {"daily_cap": 2},
    "curator": {"daily_cap": 10},
    "goal_judge": {"daily_cap": 50},
    "memory_query_rewrite": {"daily_cap": 50},
    "scene_extract": {"daily_cap": 40},
    "recent_plans": {"daily_cap": 20},
}

_DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "daily_budget_usd": 0.20,
    "tasks": _DEFAULT_TASKS,
}

_policy_cache: Optional[Dict[str, Any]] = None
_policy_lock = threading.Lock()


# ── time / paths ─────────────────────────────────────────────────────────────


def utc_day(now: Optional[float] = None) -> str:
    ts = time.time() if now is None else now
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def state_dir() -> Path:
    return _hermes_home() / "state" / "call_governor"


def state_path(day: Optional[str] = None) -> Path:
    return state_dir() / f"{day or utc_day()}.json"


def _empty_day(day: str) -> Dict[str, Any]:
    return {
        "date": day,
        "tasks": {},
        "counts": {},
        "last_call_ts": {},
        "dedupe": {},
        "total": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        "breaker_logged": False,
    }


def _coerce_day(data: Any, day: str) -> Dict[str, Any]:
    if not isinstance(data, dict) or str(data.get("date") or "") != day:
        return _empty_day(day)
    base = _empty_day(day)
    base.update({k: data[k] for k in base if k in data})
    for key in ("tasks", "counts", "last_call_ts", "dedupe"):
        if not isinstance(base.get(key), dict):
            base[key] = {}
    if not isinstance(base.get("total"), dict):
        base["total"] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    base["breaker_logged"] = bool(data.get("breaker_logged"))
    return base


@contextmanager
def _locked_state(day: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Open the day's JSON under an exclusive lock, yield the mutable dict, write it back."""
    day = day or utc_day()
    path = state_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with open(path, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            try:
                data = json.loads(fh.read() or "{}")
            except Exception:
                data = {}
            state = _coerce_day(data, day)
            yield state
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(state, separators=(",", ":")))
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_day(day: Optional[str] = None) -> Dict[str, Any]:
    """Return the day's ledger (a fresh empty one when the file is absent/corrupt)."""
    day = day or utc_day()
    try:
        raw = json.loads(state_path(day).read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return _coerce_day(raw, day)


# ── policy ───────────────────────────────────────────────────────────────────


def _merge_tasks(configured: Any) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in _DEFAULT_TASKS.items()}
    if not isinstance(configured, dict):
        return merged
    for task, cfg in configured.items():
        if not isinstance(cfg, dict):
            continue
        entry = merged.setdefault(str(task), {})
        entry.update(cfg)
    return merged


def get_policy(force_reload: bool = False) -> Dict[str, Any]:
    """Load the ``call_governor:`` config section merged over the built-in defaults."""
    global _policy_cache
    with _policy_lock:
        if _policy_cache is not None and not force_reload:
            return _policy_cache
        policy: Dict[str, Any] = {
            "enabled": bool(_DEFAULT_POLICY["enabled"]),
            "daily_budget_usd": float(_DEFAULT_POLICY["daily_budget_usd"]),
            "tasks": _merge_tasks(None),
        }
        # Test isolation: an unconfigured temp HERMES_HOME must not accumulate mocked spend
        # until the real breaker trips mid-suite. An explicit config section still wins.
        _test_isolation = bool(os.environ.get("HERMES_TEST_ISOLATION"))
        try:
            from hermes_cli.config import load_config_readonly

            cfg = load_config_readonly() or {}
            section = cfg.get("call_governor")
            if isinstance(section, dict):
                if "enabled" in section:
                    policy["enabled"] = bool(section.get("enabled"))
                if section.get("daily_budget_usd") is not None:
                    try:
                        policy["daily_budget_usd"] = float(section["daily_budget_usd"])
                    except (TypeError, ValueError):
                        pass
                policy["tasks"] = _merge_tasks(section.get("tasks"))
            elif _test_isolation:
                policy["enabled"] = False
        except Exception:
            logger.debug("call_governor: config load failed; using defaults", exc_info=True)
        _policy_cache = policy
        return policy


def set_policy_for_tests(policy: Optional[Dict[str, Any]]) -> None:
    """Inject a policy (or clear the cache) in tests; not for production use."""
    global _policy_cache
    with _policy_lock:
        _policy_cache = policy


# ── classification ───────────────────────────────────────────────────────────


def is_chat_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _CHAT_TASKS


# ── admission ────────────────────────────────────────────────────────────────


def admit(
    task: Optional[str], *, session_id: str = "", platform: str = "",
    inputs_hash: Optional[str] = None,
) -> Decision:
    """Decide whether a call labelled *task* may proceed.

    Live chat is always allowed. Otherwise: minimum interval, then input-hash dedupe, then
    the global daily-budget breaker, then the per-task daily cap. Concurrency-safe via the
    locked day file; an allowed admission reserves a count so parallel callers cannot race
    past a cap.
    """
    label = str(task or "unlabelled").strip() or "unlabelled"
    policy = get_policy()
    if not policy.get("enabled", True):
        return Decision(True, "disabled", label)
    if is_chat_task(label):
        return Decision(True, "chat", label)

    now = time.time()
    day = utc_day(now)
    task_cfg = (policy.get("tasks") or {}).get(label) or {}
    with _locked_state(day) as state:
        min_interval = _as_float(task_cfg.get("min_interval_s"))
        last = state["last_call_ts"].get(label)
        if min_interval and last is not None and (now - float(last)) < min_interval:
            return Decision(False, "min_interval", label)

        dedupe_ttl = _as_float(task_cfg.get("dedupe_ttl_s"))
        if inputs_hash and dedupe_ttl:
            key = f"{label}\x00{inputs_hash}"
            seen = state["dedupe"].get(key)
            if seen is not None and (now - float(seen)) < dedupe_ttl:
                return Decision(False, "dedupe", label)
            state["dedupe"][key] = now
            _prune_dedupe(state, now)

        total_cost = _as_float((state.get("total") or {}).get("cost_usd"))
        if total_cost >= float(policy.get("daily_budget_usd") or 0.0):
            if not state.get("breaker_logged"):
                state["breaker_logged"] = True
                logger.warning(
                    "call_governor: daily budget reached (%.4f/%.4f USD); denying non-chat tasks",
                    total_cost, float(policy.get("daily_budget_usd") or 0.0),
                )
            return Decision(False, "breaker", label)

        cap = task_cfg.get("daily_cap")
        if cap is not None and int(state["counts"].get(label, 0)) >= int(cap):
            return Decision(False, "daily_cap", label)

        state["counts"][label] = int(state["counts"].get(label, 0)) + 1
        state["last_call_ts"][label] = now
        return Decision(True, "", label)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _prune_dedupe(state: Dict[str, Any], now: float) -> None:
    """Drop dedupe entries older than an hour so the day file cannot grow unbounded."""
    cutoff = now - 3600
    dedupe = state.get("dedupe") or {}
    for key in [k for k, ts in dedupe.items() if _as_float(ts) < cutoff]:
        dedupe.pop(key, None)


# ── recording ────────────────────────────────────────────────────────────────


def record(
    task: Optional[str], *, model: Optional[str] = None, provider: Optional[str] = None,
    prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: Optional[float] = None,
    route: str = "", session_id: str = "", session_db: Any = None,
    cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0,
    actual_cost_usd: Optional[float] = None,
) -> None:
    """Append one call to the day roll-up, and (when *session_db* is given) session usage.

    ``cost_usd`` is the billed/estimated figure used for the day total; ``actual_cost_usd``
    is the provider-reported figure when known (OpenRouter ``usage.cost``). Never raises:
    accounting must not break a call.
    """
    label = str(task or "unlabelled").strip() or "unlabelled"
    try:
        day = utc_day()
        cost = float(cost_usd) if cost_usd is not None else 0.0
        with _locked_state(day) as state:
            entry = state["tasks"].setdefault(label, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
            })
            entry["calls"] = int(entry.get("calls", 0)) + 1
            entry["prompt_tokens"] = int(entry.get("prompt_tokens", 0)) + int(prompt_tokens or 0)
            entry["completion_tokens"] = int(entry.get("completion_tokens", 0)) + int(completion_tokens or 0)
            entry["cost_usd"] = float(entry.get("cost_usd", 0.0)) + cost
            if model:
                entry["model"] = str(model)
            if route:
                entry["route"] = str(route)
            total = state["total"]
            total["calls"] = int(total.get("calls", 0)) + 1
            total["prompt_tokens"] = int(total.get("prompt_tokens", 0)) + int(prompt_tokens or 0)
            total["completion_tokens"] = int(total.get("completion_tokens", 0)) + int(completion_tokens or 0)
            total["cost_usd"] = float(total.get("cost_usd", 0.0)) + cost
    except Exception:
        logger.debug("call_governor: record failed (non-fatal)", exc_info=True)

    if session_db is not None and session_id and label:
        try:
            session_db.record_auxiliary_usage(
                session_id, label, model=model, billing_provider=provider,
                billing_base_url=route or None, input_tokens=int(prompt_tokens or 0),
                output_tokens=int(completion_tokens or 0), cache_read_tokens=int(cache_read_tokens or 0),
                cache_write_tokens=int(cache_write_tokens or 0), reasoning_tokens=int(reasoning_tokens or 0),
                estimated_cost_usd=cost if cost_usd is not None else None,
            )
        except Exception:
            logger.debug("call_governor: session usage write failed (non-fatal)", exc_info=True)


def _raw_cost(usage: Any) -> Optional[float]:
    """Provider-reported cost from a usage object/dict (OpenRouter ``usage.cost``)."""
    candidates = ("cost", "cost_usd", "total_cost", "actual_cost_usd")
    if isinstance(usage, dict):
        for key in candidates:
            if usage.get(key) is not None:
                try:
                    return float(usage[key])
                except (TypeError, ValueError):
                    continue
        return None
    for key in candidates:
        val = getattr(usage, key, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def extract_usage_and_cost(
    response: Any, *, provider: Optional[str] = None, base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Normalise a response's usage into ledger fields, incl. billed cost when available."""
    raw = getattr(response, "usage", None)
    if raw is None:
        return None
    try:
        from agent.usage_pricing import normalize_usage, estimate_usage_cost

        usage = normalize_usage(raw, provider=provider)
    except Exception:
        logger.debug("call_governor: usage normalization failed", exc_info=True)
        return None
    model = str(getattr(response, "model", "") or "") or "unknown"
    billed = _raw_cost(raw)
    cost: Optional[float] = billed
    if cost is None:
        try:
            result = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)
            if result.amount_usd is not None:
                cost = float(result.amount_usd)
        except Exception:
            logger.debug("call_governor: cost estimation failed", exc_info=True)
    return {
        "model": model,
        "provider": provider,
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cost_usd": cost,
        "actual_cost_usd": billed,
    }


def record_response(
    response: Any, task: Optional[str], *, provider: Optional[str] = None,
    base_url: Optional[str] = None, route: str = "", session_id: str = "",
    session_db: Any = None,
) -> None:
    """Record a completed LLM response (usage + cost) against *task*."""
    fields = extract_usage_and_cost(response, provider=provider, base_url=base_url)
    if not fields:
        return
    record(
        task, model=fields["model"], provider=provider, prompt_tokens=fields["prompt_tokens"],
        completion_tokens=fields["completion_tokens"], cost_usd=fields["cost_usd"],
        actual_cost_usd=fields["actual_cost_usd"], route=route, session_id=session_id,
        session_db=session_db, cache_read_tokens=fields["cache_read_tokens"],
        cache_write_tokens=fields["cache_write_tokens"], reasoning_tokens=fields["reasoning_tokens"],
    )
