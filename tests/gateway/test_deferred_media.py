"""Deferred media delivery (text-first, photo-after) — offline, no network.

The plugin registers a job during the agent turn; the gateway delivers the
reply text, then runs the job in a background task and sends the photo to the
same chat. These tests pin the contract directly (registry + base-adapter
drain) without a live platform.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gateway.deferred_media import (
    STALE_AFTER_SECONDS,
    DeferredMediaJob,
    deliver_deferred_media,
    has_deferred_media,
    in_flight_count,
    register_deferred_media,
    reset_for_tests,
)
from gateway.platforms.base import BasePlatformAdapter


class FakeAdapter:
    """Records the ordered sends through the real adapter contract surface."""

    name = "fake"

    def __init__(self):
        self.order: list[str] = []
        self.photos: list = []

    async def send_media_chat_action(self, chat_id, action="upload_photo", metadata=None):
        self.order.append("action")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0):
        self.order.append("photo")
        self.photos.append((chat_id, images))

        class _R:
            success = True

        return _R()


class FakePlainAdapter:
    """No chat action, no media send — used for the 'missing surface' path."""

    name = "plain"

    def __init__(self):
        self.photos = []

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0):
        self.photos.append(images)

        class _R:
            success = True

        return _R()


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_for_tests()
    yield
    reset_for_tests()


def _job(tmp_path, key="sk", produce=None, created_at=None, caption=""):
    img = tmp_path / "gen.png"
    img.write_bytes(b"\x89PNG")
    return DeferredMediaJob(
        session_key=key, platform="telegram", chat_id="chat-1",
        produce=produce or (lambda: str(img)), caption=caption,
        created_at=time.time() if created_at is None else created_at,
    )


@pytest.mark.asyncio
async def test_text_ships_before_generation_completes(tmp_path):
    """Text is appended by the caller BEFORE deliver(); the registry spawns the
    job without waiting, so the photo only appears once generation returns."""
    adapter = FakeAdapter()
    gate = threading.Event()
    order: list[str] = []

    def produce():
        gate.wait(5)
        order.append("produce")
        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNG")
        return str(img)

    order.append("text")  # the live lane has already sent the reply
    assert register_deferred_media(_job(tmp_path, produce=produce)) is True
    task = await deliver_deferred_media(adapter, "sk")
    assert task is not None
    # deliver() returned immediately; generation is still blocked.
    assert "produce" not in order
    assert adapter.order == []

    gate.set()
    await task
    assert order == ["text", "produce"]
    assert adapter.order == ["action", "photo"]
    assert adapter.photos[0][0] == "chat-1"
    assert in_flight_count() == 0


@pytest.mark.asyncio
async def test_failure_sends_nothing_extra(tmp_path):
    adapter = FakeAdapter()

    def produce():
        return None

    assert register_deferred_media(_job(tmp_path, produce=produce)) is True
    task = await deliver_deferred_media(adapter, "sk")
    await task
    assert adapter.order == ["action"]  # no photo
    assert has_deferred_media("sk") is False


@pytest.mark.asyncio
async def test_produce_exception_sends_nothing(tmp_path):
    adapter = FakeAdapter()

    def produce():
        raise RuntimeError("runpod down")

    assert register_deferred_media(_job(tmp_path, produce=produce)) is True
    task = await deliver_deferred_media(adapter, "sk")
    await task
    assert adapter.photos == []


@pytest.mark.asyncio
async def test_inflight_dedupe_drops_second_job(tmp_path):
    adapter = FakeAdapter()
    gate = threading.Event()

    def slow():
        gate.wait(5)
        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNG")
        return str(img)

    assert register_deferred_media(_job(tmp_path, produce=slow)) is True
    task = await deliver_deferred_media(adapter, "sk")
    assert in_flight_count() == 1

    # A second registration while the first is generating is refused.
    assert register_deferred_media(_job(tmp_path, produce=lambda: None)) is False
    assert has_deferred_media("sk") is False

    gate.set()
    await task
    assert len(adapter.photos) == 1
    assert in_flight_count() == 0


@pytest.mark.asyncio
async def test_stale_job_dropped_fake_clock(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    called: list[bool] = []

    def produce():
        called.append(True)
        return str(tmp_path)

    import gateway.deferred_media as dm
    job = _job(tmp_path, produce=produce, created_at=1000.0)
    assert register_deferred_media(job) is True
    monkeypatch.setattr(dm.time, "time", lambda: 1000.0 + STALE_AFTER_SECONDS + 1)
    task = await deliver_deferred_media(adapter, "sk")
    assert task is None
    assert called == []
    assert adapter.photos == []
    assert has_deferred_media("sk") is False


@pytest.mark.asyncio
async def test_stale_after_generation_but_before_send_is_dropped(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    img = tmp_path / "p.png"
    img.write_bytes(b"\x89PNG")

    import gateway.deferred_media as dm
    job = _job(tmp_path, produce=lambda: str(img), created_at=1000.0)
    assert register_deferred_media(job) is True
    # deliver() sees a fresh job; the clock jumps past stale while generation runs.
    calls = {"n": 0}

    def _now():
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 1000.0 + STALE_AFTER_SECONDS + 1

    monkeypatch.setattr(dm.time, "time", _now)
    task = await deliver_deferred_media(adapter, "sk")
    await task
    assert adapter.photos == []


def test_register_rejects_malformed_jobs():
    assert register_deferred_media(DeferredMediaJob("", "telegram", "c", lambda: "x")) is False
    assert register_deferred_media(DeferredMediaJob("sk", "telegram", "c", None)) is False


@pytest.mark.asyncio
async def test_no_op_without_matching_job():
    adapter = FakeAdapter()
    assert await deliver_deferred_media(adapter, "nobody") is None
    assert adapter.order == []


def test_turn_finalizer_forwards_gateway_identity(monkeypatch):
    """The live turn's transform hook must receive the chat identity so a plugin
    can register the deferred photo against the exact chat the text went to."""
    import logging
    from types import SimpleNamespace

    import hermes_cli.lifecycle as lifecycle
    from agent import turn_finalizer

    seen: dict = {}

    def fake_invoke_hook(hook_name, **kwargs):
        if hook_name == "transform_llm_output":
            seen.update(kwargs)
        return []

    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke_hook)
    agent = SimpleNamespace(
        session_id="sess-1", model="m", _persist_disabled=True,
        _gateway_session_key="agent:main:telegram:dm:42", _chat_id="42",
    )
    turn_finalizer._apply_output_hooks(
        agent, "hello", logging.getLogger("test"), platform="telegram",
        effective_task_id="t1", turn_id="turn-1", original_user_message="hi", messages=[])
    assert seen.get("session_key") == "agent:main:telegram:dm:42"
    assert seen.get("chat_id") == "42"
    assert seen.get("platform") == "telegram"


@pytest.mark.asyncio
async def test_base_adapter_drain_runs_job_after_text(tmp_path):
    """The base adapter's live-chat finally-block drain spawns the job through
    the real adapter contract (chat action + send)."""
    adapter = FakeAdapter()
    assert register_deferred_media(_job(tmp_path)) is True
    # Unbound call: exercises exactly what _process_message_background invokes.
    await BasePlatformAdapter._deliver_deferred_media(adapter, "sk", {})
    # Give the detached task a tick to finish (it is fire-and-forget).
    for _ in range(50):
        if adapter.photos:
            break
        await asyncio.sleep(0.01)
    assert adapter.order == ["action", "photo"]
    assert in_flight_count() == 0
