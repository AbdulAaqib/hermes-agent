"""Heat-routed per-turn model selection ("rung" routing).

Most agent turns are affectionate/flirty-but-non-explicit and can run on a flat-rate
subscription provider (OpenCode Go). Explicit turns — either the persisted lewd_lens
escalation rung has climbed to the heat threshold, or the incoming user message itself
scores explicit — must run on the OpenRouter route, because the OpenCode Go model
refuses that content.

This module owns the policy and the in-place swap. It is invoked once per turn from
``agent.turn_context.build_turn_context`` (the same universal prologue every surface
runs: CLI, gateway/Telegram, cron), reusing the existing ``switch_model`` machinery so
the agent loop is not forked. The rung signal comes from the lewd_lens persisted state
(``$HERMES_HOME/plugins/lewd_lens/state.json``) and its classifier lexicon (loaded from
the plugin so the two can never drift); the cron allowlist comes from the lewd_lens
``cron.job_name_regex`` config.

Safety nets (all reuse existing engine fallback machinery, not new loop code):
  * errors / quota / rate-limit / empty responses -> ``try_activate_fallback`` (the chain
    is pinned to the heat route while a cheap turn is active)
  * provider-side ``finish_reason=content_filter`` -> ``handle_content_policy_refusal``
  * plain-text refusals (finish_reason=stop, refusal-shaped text) -> the explicit check in
    ``agent.turn_response_check.check_api_response`` via :func:`text_refusal_for_fallback`

Config (``model_routing`` in config.yaml)::

    model_routing:
      enabled: true
      cheap: {provider: opencode-go, model: deepseek-v4.1-flash}
      heat:  {provider: custom:openrouter-tuned, model: deepseek/deepseek-v4-pro}
      heat_rung_threshold: 3
      refusal_fallback: true

Fail-open everywhere: any error logs and leaves the current route untouched.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("agent.model_router")

STATE_RELPATH = Path("plugins") / "lewd_lens" / "state.json"
CLASSIFIER_RELPATH = Path("plugins") / "lewd_lens" / "classifier.py"
JOBS_RELPATH = Path("cron") / "jobs.json"

# Mirrors prompt_builder: the rung resets to the cozy base after an hour idle.
ESCALATION_TTL_S = 3600.0
ESCALATION_MIN = 1
ESCALATION_MAX = 4
# Mirrors lewd_lens DEFAULT_CRON_CONFIG.job_name_regex.
DEFAULT_CRON_JOB_NAME_REGEX = (
    r"(?i)wake|tease|suggestive|coffee|skincare|cool-?down|midday|check-?in"
)

DEFAULT_CONFIG: Dict[str, Any] = {
    # OFF unless config.yaml opts in, so a deployment that never sets ``model_routing`` keeps
    # the exact pre-existing single-provider behaviour. Enable with the YAML in the report:
    # cheap=opencode-go (flat subscription), heat=custom:openrouter-tuned (OpenRouter).
    "enabled": False,
    "cheap": {"provider": "opencode-go", "model": "deepseek-v4.1-flash"},
    "heat": {"provider": "custom:openrouter-tuned", "model": "deepseek/deepseek-v4-pro"},
    "heat_rung_threshold": 3,
    "refusal_fallback": True,
    "explicit_message_heat": True,
}

# Refusal markers must be refusal-SHAPED (a bare "i can't" false-positives on in-character
# lines like "breed me till i can't think straight"). Kept byte-for-byte in sync with
# skills/testing/asuna-e2e-verification/scripts/talk_test.py::REFUSAL.
REFUSAL_RE = re.compile(
    r"i can'?t (?:do|help|continue|engage|participate|write|generate|provide|say|answer|assist|go there)\b|"
    r"i cannot (?:do|help|continue|engage|participate|write|generate|provide|say|answer|assist)\b|"
    r"i won'?t (?:do|help|continue|engage|participate|write|generate|provide|say|answer|assist)\b|"
    r"i'?m sorry,? but\b|i don'?t feel comfortable|"
    r"\bboundaries\b|out of character|not able to (?:do|help|continue|engage|provide|assist)\b|"
    r"as an ai\b",
    re.IGNORECASE,
)

_CONFIG_CACHE: Tuple[float, int, Dict[str, Any]] = (0.0, 0, {})
_RESOLVE_CACHE: Dict[Tuple[str, str], Tuple[float, Optional[Dict[str, Any]]]] = {}
_RESOLVE_TTL_S = 300.0
_CLASSIFIER_CACHE: Tuple[float, Optional[Any]] = (0.0, None)
_FALLBACK_COUNTS: Dict[str, int] = {}


def _home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def get_config() -> Dict[str, Any]:
    """Merged ``model_routing`` config (defaults under the user's block). Cached on file sig."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        cfg = {}
    raw = cfg.get("model_routing")
    if not isinstance(raw, dict):
        return dict(DEFAULT_CONFIG)
    # mtime-keyed cache avoids re-merging every turn while still honoring live edits.
    try:
        home = _home()
        path = home / "config.yaml"
        sig = path.stat().st_mtime_ns if path.exists() else 0
    except Exception:
        sig = 0
    global _CONFIG_CACHE
    if _CONFIG_CACHE[0] == sig and _CONFIG_CACHE[1] == id(raw):
        return _CONFIG_CACHE[2]
    merged = _deep_merge(DEFAULT_CONFIG, raw)
    _CONFIG_CACHE = (sig, id(raw), merged)
    return merged


def enabled() -> bool:
    return bool(get_config().get("enabled"))


def _norm_model(provider: str, model: str) -> str:
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider

        return str(normalize_model_for_provider(str(model or ""), str(provider or "")) or "").strip().lower()
    except Exception:
        return str(model or "").strip().lower()


def _route_cfg(config: Dict[str, Any], name: str) -> Tuple[str, str]:
    block = config.get(name) or {}
    return (
        str(block.get("provider") or "").strip(),
        str(block.get("model") or "").strip(),
    )


def on_cheap_route(agent: Any) -> bool:
    """True when the agent's live provider/model is the configured cheap route."""
    if not enabled():
        return False
    provider, model = _route_cfg(get_config(), "cheap")
    if not provider or not model:
        return False
    agent_provider = str(getattr(agent, "provider", "") or "").strip().lower()
    agent_model = str(getattr(agent, "model", "") or "").strip()
    if agent_provider != provider.strip().lower():
        return False
    return _norm_model(provider, agent_model) == _norm_model(provider, model)


def is_refusal(text: str) -> bool:
    return bool(text) and bool(REFUSAL_RE.search(str(text)))


# --------------------------------------------------------------------------- signals


def current_rung(now: Optional[float] = None) -> int:
    """Persisted lewd_lens escalation rung (1-4), reset to 1 after an hour idle."""
    try:
        path = _home() / STATE_RELPATH
        if not path.exists():
            return ESCALATION_MIN
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ESCALATION_MIN
    if not isinstance(data, dict):
        return ESCALATION_MIN
    try:
        rung = int(data.get("escalation_level") or ESCALATION_MIN)
    except (TypeError, ValueError):
        rung = ESCALATION_MIN
    rung = max(ESCALATION_MIN, min(ESCALATION_MAX, rung))
    try:
        ts = float(data.get("escalation_ts") or 0)
    except (TypeError, ValueError):
        ts = 0.0
    now = time.time() if now is None else now
    if ts and now - ts > ESCALATION_TTL_S:
        return ESCALATION_MIN
    return rung


def _load_classifier():
    global _CLASSIFIER_CACHE
    try:
        path = _home() / CLASSIFIER_RELPATH
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
    except Exception:
        return None
    if _CLASSIFIER_CACHE[1] is not None and _CLASSIFIER_CACHE[0] == mtime:
        return _CLASSIFIER_CACHE[1]
    mod = None
    try:
        spec = importlib.util.spec_from_file_location("lewd_lens_model_router_classifier", path)
        if spec is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
    except Exception:
        mod = None
    _CLASSIFIER_CACHE = (mtime, mod)
    return mod


def message_is_heat(text: Any) -> bool:
    """True when the incoming user message itself scores explicit via the lewd_lens lexicon."""
    if not isinstance(text, str) or not text.strip():
        return False
    if not get_config().get("explicit_message_heat", True):
        return False
    mod = _load_classifier()
    if mod is None or not hasattr(mod, "is_lewd"):
        return False
    threshold = mod.DEFAULT_THRESHOLD
    try:
        from hermes_cli.config import load_config_readonly

        lw = (load_config_readonly() or {}).get("lewd_lens") or {}
        if isinstance(lw, dict) and lw.get("threshold") is not None:
            threshold = float(lw.get("threshold"))
    except Exception:
        pass
    try:
        return bool(mod.is_lewd(text, threshold))
    except Exception:
        return False


def cron_job_name(session_id: str) -> str:
    """Job name for a ``cron_<job_id>_<timestamp>`` session, or "" when unresolvable."""
    sid = str(session_id or "")
    if not sid.startswith("cron_"):
        return ""
    try:
        jobs = json.loads((_home() / JOBS_RELPATH).read_text(encoding="utf-8"))
    except Exception:
        return ""
    entries: Any = jobs.get("jobs") if isinstance(jobs, dict) else jobs
    if not isinstance(entries, list):
        return ""
    for job in entries:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("id") or job.get("job_id") or "").strip()
        if job_id and sid.startswith(f"cron_{job_id}_"):
            return str(job.get("name") or job.get("job_name") or "").strip()
    return ""


def _cron_allowlisted(job_name: str) -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        lw = (load_config_readonly() or {}).get("lewd_lens") or {}
        pattern = DEFAULT_CRON_JOB_NAME_REGEX
        if isinstance(lw, dict):
            cron = lw.get("cron") if isinstance(lw.get("cron"), dict) else {}
            pattern = str(cron.get("job_name_regex") or pattern)
        return bool(job_name) and bool(re.search(pattern, job_name))
    except Exception:
        return bool(job_name) and bool(re.search(DEFAULT_CRON_JOB_NAME_REGEX, job_name))


def decide_route(
    config: Dict[str, Any], *, rung: int, msg_heat: bool, session_kind: str, cron_allowlisted: bool
) -> str:
    """Pure policy: ``"cheap"`` or ``"heat"``. Cron non-allowlisted jobs are always cheap."""
    if session_kind == "cron":
        if cron_allowlisted and rung >= int(config.get("heat_rung_threshold", 3)):
            return "heat"
        return "cheap"
    if msg_heat or rung >= int(config.get("heat_rung_threshold", 3)):
        return "heat"
    return "cheap"


# --------------------------------------------------------------------------- runtime resolution


def _resolve_target(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Resolve provider credentials (cached briefly). Never logs/returns the secret itself."""
    key = (str(provider), str(model))
    now = time.time()
    cached = _RESOLVE_CACHE.get(key)
    if cached is not None and now - cached[0] < _RESOLVE_TTL_S:
        return cached[1]
    resolved: Optional[Dict[str, Any]] = None
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider(requested=provider, target_model=model or None)
        if rt and rt.get("base_url"):
            resolved = {
                "api_key": rt.get("api_key") or "",
                "base_url": rt.get("base_url") or "",
                "api_mode": rt.get("api_mode") or "chat_completions",
                "provider": provider,
                "model": model,
            }
    except Exception as exc:  # AuthError / config error: fail-open
        logger.info("model_router: runtime resolution failed for %s/%s: %s", provider, model, exc)
        resolved = None
    _RESOLVE_CACHE[key] = (now, resolved)
    return resolved


def _pin_heat_fallback(agent: Any, config: Dict[str, Any]) -> None:
    """While a cheap turn is active, pin the fallback chain to the heat route so errors,
    quota/rate limits, empty responses and refusals all retry on OpenRouter.

    The operator's configured ``fallback_providers`` chain is snapshotted on the first pin and
    restored by :func:`_restore_fallback_chain` on a heat turn, so enabling routing does not
    permanently replace the configured chain with the heat route."""
    provider, model = _route_cfg(config, "heat")
    if not provider or not model:
        return
    if not hasattr(agent, "_model_router_orig_fallback_chain"):
        try:
            agent._model_router_orig_fallback_chain = list(
                getattr(agent, "_fallback_chain", None) or []
            )
        except Exception:
            pass
    chain = [{"provider": provider, "model": model}]
    try:
        agent._fallback_chain = chain
        agent._fallback_model = chain[0]
        agent._fallback_index = 0
        agent._fallback_activated = False
        agent._provider_fallback_active = False
        agent._provider_fallback_route = None
    except Exception:
        pass


def _restore_fallback_chain(agent: Any) -> None:
    """Restore the operator's fallback chain after a cheap-route pin (heat turns / routing off)."""
    if not hasattr(agent, "_model_router_orig_fallback_chain"):
        return
    try:
        chain = list(getattr(agent, "_model_router_orig_fallback_chain", None) or [])
        agent._fallback_chain = chain
        agent._fallback_model = chain[0] if chain else None
        agent._fallback_index = 0
        agent._fallback_activated = False
        agent._provider_fallback_active = False
        agent._provider_fallback_route = None
    except Exception:
        pass


def _apply_swap(agent: Any, provider: str, model: str, resolved: Dict[str, Any]) -> bool:
    from agent.agent_runtime_helpers import switch_model

    switch_model(
        agent, model, provider,
        api_key=resolved.get("api_key") or "",
        base_url=resolved.get("base_url") or "",
        api_mode=resolved.get("api_mode") or "",
    )
    return True


def _ensure_provider_overrides(agent: Any, provider: str) -> None:
    """Re-derive ``request_overrides`` for the routed provider.

    Runs even when the route does not change: a gateway-reused agent carries the previous
    turn's per-provider ``extra_body`` merge (``_merge_turn_request_overrides``), and letting
    an OpenRouter ``extra_body`` ride along to OpenCode Go would be sent verbatim.
    """
    try:
        from agent.agent_runtime_helpers import _apply_switched_provider_request_overrides

        _apply_switched_provider_request_overrides(agent, provider)
    except Exception:
        logger.debug("model_router: request_overrides re-derivation skipped", exc_info=True)


def log_turn(rung: int, msg_heat: bool, route: str, fallback: str = "none") -> None:
    logger.info(
        "model_router: rung=%s msg_heat=%s route=%s fallback=%s",
        rung, bool(msg_heat), route, fallback,
    )


def record_fallback(kind: str) -> None:
    _FALLBACK_COUNTS[kind] = _FALLBACK_COUNTS.get(kind, 0) + 1


def fallback_counts() -> Dict[str, int]:
    return dict(_FALLBACK_COUNTS)


def maybe_route_turn(
    agent: Any, *, user_message: Any, platform: str = "", session_id: str = ""
) -> Optional[Dict[str, Any]]:
    """Decide and apply this turn's route. Returns the decision (or None when disabled).

    Idempotent: when the agent already runs the target route it only re-pins the cheap
    fallback chain. Any failure logs and leaves the current route untouched (fail-open).
    """
    config = get_config()
    if not config.get("enabled"):
        return None
    session_kind = "cron" if str(platform or "").strip().lower() == "cron" else "live"
    try:
        rung = current_rung()
    except Exception:
        rung = ESCALATION_MIN
    if session_kind == "cron":
        job_name = cron_job_name(session_id)
        cron_allow = _cron_allowlisted(job_name)
        msg_heat = False
    else:
        cron_allow = False
        msg_heat = message_is_heat(user_message)

    route = decide_route(
        config, rung=rung, msg_heat=msg_heat,
        session_kind=session_kind, cron_allowlisted=cron_allow,
    )
    provider, model = _route_cfg(config, route)
    if not provider or not model:
        log_turn(rung, msg_heat, route)
        return {"route": route, "rung": rung, "msg_heat": msg_heat, "applied": False, "reason": "unconfigured"}

    applied = False
    try:
        needed_change = not (
            str(getattr(agent, "provider", "") or "").strip().lower() == provider.strip().lower()
            and _norm_model(provider, getattr(agent, "model", "") or "") == _norm_model(provider, model)
        )
        if needed_change:
            resolved = _resolve_target(provider, model)
            if resolved:
                _apply_swap(agent, provider, model, resolved)
                applied = True
        _ensure_provider_overrides(agent, provider)
        if route == "cheap":
            _pin_heat_fallback(agent, config)
        else:
            _restore_fallback_chain(agent)
        if needed_change and not applied:
            logger.warning(
                "model_router: %s route unavailable (provider=%s); staying on %s/%s",
                route, provider, getattr(agent, "provider", ""), getattr(agent, "model", ""),
            )
    except Exception as exc:
        logger.warning("model_router: route application failed (route=%s): %s", route, exc)
        applied = False

    log_turn(rung, msg_heat, route)
    return {
        "route": route, "rung": rung, "msg_heat": msg_heat, "applied": applied,
        "provider": provider, "model": model,
    }


# --------------------------------------------------------------------------- refusal safety net


def text_refusal_for_fallback(agent: Any, assistant_text: str) -> Optional[str]:
    """Non-None when a cheap-route plain-text refusal should be retried on the heat route.

    Marks the turn so it only retries once. Does NOT activate the fallback itself — the
    caller reuses ``handle_content_policy_refusal`` which owns the restart contract.
    """
    config = get_config()
    if not config.get("enabled") or not config.get("refusal_fallback", True):
        return None
    if not on_cheap_route(agent):
        return None
    if getattr(agent, "_model_router_refusal_retried", False):
        return None
    if not agent._has_pending_fallback():
        return None
    if not is_refusal(assistant_text):
        return None
    try:
        agent._model_router_refusal_retried = True
    except Exception:
        pass
    record_fallback("refusal")
    log_turn(current_rung(), False, "cheap", fallback="refusal")
    return str(assistant_text)


def note_route_fallback(agent: Any, kind: str) -> None:
    """Count/log an error/quota/empty fallback observed while a cheap route was active."""
    try:
        if on_cheap_route(agent) or getattr(agent, "_model_router_last_route", "") == "cheap":
            record_fallback(kind)
            log_turn(current_rung(), False, "cheap", fallback=kind)
    except Exception:
        pass
