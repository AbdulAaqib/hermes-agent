"""Deferred media delivery for live chat.

A plugin that decides during an agent turn to attach generated media whose
production is slow (a RunPod image is ~15-20s) may register a *job* here instead
of generating synchronously and blocking the whole reply. The gateway delivers
the reply text immediately, then — only after the text has gone out — runs the
job in a background task and sends the produced local image to the same chat
through the live adapter, while a native "upload_photo" chat action is shown.

Live-chat only. Cron and other non-live surfaces keep their synchronous path
(``MEDIA:`` inline), so this registry is only ever consulted by the base
adapter's live-chat delivery lane.

Design constraints:

* Decisions (which tier fires, gates, rate limits, prompt) are made
  synchronously by the plugin; only generation + send is deferred.
* At most one deferred image in flight per session/chat. A second registration
  while one is in flight is dropped (logged) — the plugin is told so it can
  skip its synchronous fallback.
* Failure sends nothing extra: the producer returning ``None``/raising logs a
  WARNING and the job is abandoned; no error text reaches the user.
* A job older than ``STALE_AFTER_SECONDS`` is dropped (logged) rather than
  delivered late; a gateway restart mid-generation simply loses the job.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import quote as _quote

logger = logging.getLogger(__name__)

# A job decided more than this many seconds ago is stale: the turn it belongs to
# is long gone, so the photo is dropped instead of arriving out of context.
STALE_AFTER_SECONDS = 120.0

ProduceFn = Callable[[], Optional[str]]


@dataclass
class DeferredMediaJob:
    """A pending image whose generation is deferred until after the reply text.

    ``produce`` is a *blocking* callable run in a worker thread; it returns a
    local image path on success or ``None`` on failure."""
    session_key: str
    platform: str
    chat_id: str
    produce: ProduceFn
    caption: str = ""
    created_at: float = field(default_factory=time.time)


# session_key -> job awaiting the reply text to be delivered first.
_pending: Dict[str, DeferredMediaJob] = {}
# session_key -> asyncio task currently generating/sending that chat's image.
_inflight: Dict[str, "asyncio.Task[None]"] = {}


def _key(job_or_key: Any) -> str:
    if isinstance(job_or_key, DeferredMediaJob):
        return str(job_or_key.session_key or "")
    return str(job_or_key or "")


def register_deferred_media(job: DeferredMediaJob) -> bool:
    """Queue ``job`` to run after the current chat's reply text is delivered.

    Returns ``True`` when accepted, ``False`` when dropped because another
    deferred image is already in flight (or pending) for this session. Callers
    use the boolean to decide between "deferred" and "drop" — an import/auth
    problem is the caller's own concern (it never reaches here)."""
    key = _key(job)
    if not key or not callable(job.produce):
        return False
    if key in _inflight or key in _pending:
        logger.info(
            "deferred media: already in flight for %s; dropping new job (tier pending)",
            key,
        )
        return False
    _pending[key] = job
    return True


def has_deferred_media(session_key: str) -> bool:
    return _key(session_key) in _pending


def pop_deferred_media(session_key: str) -> Optional[DeferredMediaJob]:
    return _pending.pop(_key(session_key), None)


async def deliver_deferred_media(
    adapter: Any, session_key: str, *, metadata: Optional[Dict[str, Any]] = None,
) -> Optional["asyncio.Task[None]"]:
    """Spawn the background generation + send for ``session_key``'s queued job.

    Called by the live-chat delivery lane *after* the reply text has been sent,
    so the returned task never delays the text. Returns the created task (or
    ``None`` when there was nothing to do) — the caller does not await it; the
    task is fire-and-forget and its failure is contained and logged."""
    key = _key(session_key)
    if not key:
        return None
    job = pop_deferred_media(key)
    if job is None:
        return None
    if job.session_key in _inflight:
        logger.info("deferred media: job already in flight for %s; dropping", key)
        return None
    age = time.time() - float(job.created_at or 0)
    if age > STALE_AFTER_SECONDS:
        logger.info("deferred media: dropping stale job for %s (%.0fs old)", key, age)
        return None
    task = asyncio.create_task(_run_job(adapter, job, metadata))
    _inflight[job.session_key] = task
    task.add_done_callback(lambda _t: _inflight.pop(job.session_key, None))
    return task


async def _run_job(adapter: Any, job: DeferredMediaJob, metadata: Optional[Dict[str, Any]]) -> None:
    key = job.session_key
    try:
        await _announce_generation(adapter, job)
        try:
            path = await asyncio.to_thread(job.produce)
        except Exception:
            logger.warning("deferred media: generation failed for %s", key, exc_info=True)
            return
        if not path:
            logger.warning("deferred media: generation produced nothing for %s", key)
            return
        age = time.time() - float(job.created_at or 0)
        if age > STALE_AFTER_SECONDS:
            logger.info("deferred media: dropping stale image for %s (%.0fs old)", key, age)
            return
        local = Path(str(path))
        if not local.exists():
            logger.warning("deferred media: produced path missing for %s: %s", key, local)
            return
        uri = "file://" + _quote(str(local))
        sender = getattr(adapter, "send_multiple_images", None)
        if not callable(sender):
            logger.warning("deferred media: adapter %r cannot send images for %s",
                           getattr(adapter, "name", adapter), key)
            return
        await sender(chat_id=job.chat_id, images=[(uri, job.caption or "")], metadata=metadata)
    except Exception:
        logger.warning("deferred media: delivery failed for %s", key, exc_info=True)


async def _announce_generation(adapter: Any, job: DeferredMediaJob) -> None:
    """Best-effort native chat action while the image generates (Telegram
    ``upload_photo``). Never raises — a missing action is harmless."""
    sender = getattr(adapter, "send_media_chat_action", None)
    if not callable(sender):
        return
    try:
        result = sender(job.chat_id, "upload_photo")
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.debug("deferred media: chat action failed", exc_info=True)


def in_flight_count() -> int:
    return len(_inflight)


def reset_for_tests() -> None:
    """Clear module state. Tests only — never called in production."""
    _pending.clear()
    _inflight.clear()


__all__ = [
    "DeferredMediaJob",
    "STALE_AFTER_SECONDS",
    "deliver_deferred_media",
    "has_deferred_media",
    "in_flight_count",
    "pop_deferred_media",
    "register_deferred_media",
    "reset_for_tests",
]
